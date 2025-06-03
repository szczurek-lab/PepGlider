import os
from torch import optim, nn, logsumexp, device, cuda, save, isinf, backends, manual_seed, LongTensor, zeros_like, ones_like, isnan, tensor
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.multiprocessing as tmp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model.model import EncoderRNN, DecoderRNN
# import pandas as pd
import numpy as np
# from torch.autograd import Variable
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import clearml
from typing import Optional, Literal
from torch.optim import Adam
import itertools
import random
from pathlib import Path
from tqdm import tqdm
import data.dataset as dataset_lib
from model.constants import MIN_LENGTH, MAX_LENGTH, VOCAB_SIZE
import json
import multiprocessing as mp
import ar_vae_metrics as m
import data.data_describe as d
import monitoring as mn
import regularization as r

def setup_ddp(rank, world_size):
    # local_rank = int(os.environ["LOCAL_RANK"])
#    os.environ["MASTER_ADDR"] = "0"
#    os.environ["MASTER_PORT"] = "12355"
    cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    init_method="env://"
    return device(f"cuda:{rank}")

def set_seed(seed: int = 42) -> None:
    """
    Source:
    https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    """
    np.random.seed(seed)
    random.seed(seed)
    manual_seed(seed)
    cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # logger.info(f"Random seed set to {seed}")
    return None

def get_model_arch_hash(model: nn.Module) -> int:
    return hash(";".join(sorted([str(v.shape) for v in model.state_dict().values()])))

def save_model(model: nn.Module, name: str, with_hash: bool = True) -> None:
    if with_hash:
        short_hash = str(get_model_arch_hash(model)).removeprefix("-")[:5]
        model_name = f"{short_hash}_{name}"
    else:
        model_name = name
    save(
        model.state_dict(), (MODELS_DIR / model_name).with_suffix(".pt")
    )

def run_epoch_iwae(
    mode: Literal["test", "train"],
    encoder: EncoderRNN,
    decoder: DecoderRNN,
    dataloader: DataLoader,
    device: device,
    epoch: int,
    kl_beta: float,
    logger: Optional[clearml.Logger],
    optimizer: Optional[optim.Optimizer],
    eval_mode: Literal["fast", "deep"],
    iwae_samples: int,
    reg_dim,
    gamma,
    pool
):
    ce_loss_fun = nn.CrossEntropyLoss(reduction="none")
    encoder.to(DEVICE)
    decoder.to(DEVICE)
    encoder= nn.parallel.DistributedDataParallel(encoder, device_ids=[DEVICE.index])
    decoder= nn.parallel.DistributedDataParallel(decoder, device_ids=[DEVICE.index])
    encoder.to(device), decoder.to(device)
    if mode == "train":
        encoder.train(), decoder.train()
    else:
        encoder.eval(), decoder.eval()

    stat_sum = {
        "kl_mean": 0.0,
        "kl_best": 0.0,
        "kl_worst": 0.0,
        "ce_mean": 0.0,
        "ce_best": 0.0,
        "ce_worst": 0.0,
        "std": 0.0,
        "total": 0.0,
    }
    seq_true, model_out, model_out_sampled = [], [], []
    len_data = len(dataloader.dataset)

    results_fp = os.path.join(
    os.path.dirname(ROOT_DIR),
        'results_dict.json'
    )
    latent_codes = []
    attributes = []
    ar_vae_metrics = {}

    K = iwae_samples
    C = VOCAB_SIZE + 1

    for batch, labels in dataloader:       
        # print(f"Inspecting batch shape: {batch.shape}")
        physchem_original_async = d.calculate_physchem(pool, dataset_lib.decoded(batch, ""),) 
        peptides = batch.permute(1, 0).type(LongTensor).to(device) # S x B
        S, B = peptides.shape
        if optimizer:
            optimizer.zero_grad()

        # autoencoding

        mu, std = encoder(peptides)
        assert not (isnan(mu).all() or isnan(std).all() ), f" contains all NaN values: {mu}, {std}"
        assert not (isinf(mu).all() or isinf(std).all()), f" contains all Inf values: {mu}, {std}"

        prior_distr = Normal(zeros_like(mu), ones_like(std))
        q_distr = Normal(mu, std)
        z = q_distr.rsample((K,)) # K, B, L
        if mode == 'test':
                    latent_codes.append(z.reshape(-1, z.shape[2]).cpu().detach().numpy())
                    physchem_original = d.gather_physchem_results(physchem_original_async) # Pobierz wynik jako dict

                    num_peptides = len(physchem_original[1][0])
                    physchem_expanded_list = []

                    for i in range(num_peptides):
                        peptide_features = np.array([
                            physchem_original[0][0][i],
                            physchem_original[1][0][i],
                            physchem_original[2][0][i],
                        ])
                        # Powielaj cechy peptydu K razy
                        expanded_peptide_features = np.repeat(peptide_features[np.newaxis, :], K, axis=0)
                        physchem_expanded_list.append(expanded_peptide_features)

                    physchem_expanded = np.concatenate(physchem_expanded_list, axis=0)

                    attributes.append(np.concatenate((physchem_expanded, labels.unsqueeze(0).expand(K, -1).reshape(-1, 1).numpy()), axis=1))        # Kullback Leibler divergence
        log_qzx = q_distr.log_prob(z).sum(dim=2)
        log_pz = prior_distr.log_prob(z).sum(dim=2)

        kl_div = log_qzx - log_pz

        # reconstruction - cross entropy
        sampled_peptide_logits = decoder(z.reshape(K*B,-1))
        sampled_peptide_logits = sampled_peptide_logits.view(S, K, B, C)
        src = sampled_peptide_logits.permute(1, 3, 2, 0)  # K x C x B x S
        src_decoded = src.reshape(-1, C, S).argmax(dim=1) # K*B x S
        tgt = peptides.permute(1, 0).reshape(1, B, S).repeat(K, 1, 1)  # K x B x S
        src_decoded = dataset_lib.decoded(src_decoded, "")
        indexes = [index for index, item in enumerate(src_decoded) if item.strip()]
        filtered_list = [item for item in src_decoded if item.strip()]
        physchem_decoded_async = d.calculate_physchem(pool, filtered_list)
        physchem_decoded = d.gather_physchem_results(physchem_decoded_async)
        # K x B
        cross_entropy = ce_loss_fun(
            src,
            tgt,
        ).sum(dim=2)

        reg_loss = 0
        for dim in reg_dim:
            reg_loss += r.compute_reg_loss(
            z.reshape(-1,z.shape[2])[indexes,:], physchem_decoded[dim], dim, gamma, DEVICE.index #gamma i delta z papera
        )

        loss = logsumexp(
            cross_entropy + kl_beta * (log_qzx - log_pz), dim=0
        ).mean(dim=0) + tensor(reg_loss).to(device)

        # stats
        stat_sum["kl_mean"] += kl_div.mean(dim=0).sum(dim=0).item()
        stat_sum["kl_best"] += kl_div.min(dim=0).values.sum(dim=0).item()
        stat_sum["kl_worst"] += kl_div.max(dim=0).values.sum(dim=0).item()
        stat_sum["ce_mean"] += cross_entropy.mean(dim=0).sum(dim=0).item()
        stat_sum["ce_best"] += cross_entropy.min(dim=0).values.sum(dim=0).item()
        stat_sum["ce_worst"] += cross_entropy.max(dim=0).values.sum(dim=0).item()
        stat_sum["total"] += loss.item() * len(batch)
        stat_sum["std"] += std.mean(dim=1).sum().item()

        if optimizer:
            loss.backward()
            nn.utils.clip_grad_norm_(
                itertools.chain(encoder.parameters(), decoder.parameters()), max_norm=1.0
            )
            optimizer.step()

        # reporting
        if eval_mode == "deep":
            seq_true.append(peptides.cpu().detach().numpy())
            model_out.append(decoder(mu).cpu().detach().numpy())
            model_out_sampled.append(
                sampled_peptide_logits.mean(dim=1).cpu().detach().numpy() #to ensure this is okay, mean across K for one batch sequence
            )
    if mode == 'test':
        latent_codes = np.concatenate(latent_codes, 0)
        attributes = np.concatenate(attributes, 0)
        attributes, attr_list = m._extract_relevant_attributes(attributes, reg_dim)
        async_metrics = m.compute_all_metrics_async(pool, latent_codes, attributes, attr_list)
        ar_vae_metrics = m.gather_metrics(async_metrics)
        with open(results_fp, 'w') as outfile:
            json.dump(ar_vae_metrics, outfile, indent=2)

    if logger is not None:
        mn.report_scalars(
            logger,
            mode,
            epoch,
            scalars=[
                ("Total Loss", stat_sum["total"] / len_data),
                    ("Posterior Standard Deviation [mean]", stat_sum["std"] / len_data),
                (
                    "Cross Entropy Loss",
                    "mean over samples",
                    stat_sum["ce_mean"] / len_data,
                ),
                ("Cross Entropy Loss", "best sample", stat_sum["ce_best"] / len_data),
                ("Cross Entropy Loss", "worst sample", stat_sum["ce_worst"] / len_data),
                (
                    "KL Divergence",
                    "mean over samples",
                    stat_sum["kl_mean"] / len_data,
                ),
                ("KL Divergence", "best sample", stat_sum["kl_best"] / len_data),
                ("KL Divergence", "worst sample", stat_sum["kl_worst"] / len_data),
                ("KL Beta", kl_beta),
                ("Regularization Loss", reg_loss/gamma),
                ("Regularization Loss with gamma", reg_loss),
            ],
        )
        if eval_mode == "deep":
            mn.report_sequence_char(
                logger,
                hue=f"{mode} - mu",
                epoch=epoch,
                seq_true=np.concatenate(seq_true, axis=1),
                model_out=np.concatenate(model_out, axis=1),
                metrics = None,
                pool = pool
            )
            mn.report_sequence_char(
                logger,
                hue=f"{mode} - z",
                epoch=epoch,
                seq_true=np.concatenate(seq_true, axis=1),
                model_out=np.concatenate(model_out_sampled, axis=1),
                metrics = None,
                pool = pool
            )
            mn.report_sequence_char(
                logger,
                hue=f"{mode} - ar-vae metrics",
                epoch=epoch,
                seq_true=np.concatenate(seq_true, axis=1),
                model_out=None,
                metrics = ar_vae_metrics,
                pool = pool
            )
    return stat_sum["total"] / len_data

def run(rank, world_size):
    print(f'rank:{rank}')
    global ROOT_DIR 
    ROOT_DIR = Path(__file__).parent#.parent
    DATA_DIR = ROOT_DIR / "data"
    global MODELS_DIR 
    MODELS_DIR = ROOT_DIR
    params = {
        "num_heads": 4,
        "num_layers": 6,
        "layer_norm": True,
        "latent_dim": 56,
        "encoding": "add",
        "dropout": 0.1,
        "batch_size": 512,
        "lr": 0.001,
        "kl_beta_schedule": (0.000001, 0.01, 8000),
        "train_size": None,
        "epochs": 10000,
        "iwae_samples": 10,
        "model_name": "basic",
        "use_clearml": True,
        "task_name": "ar_vae_with_ar_vae_metrics",
        "device": "cuda",
        "deeper_eval_every": 20,
        "save_model_every": 100,
        "reg_dim": [0], # [hydrophobicity_moment, length, charge]
        "gamma_schedule": (0.1, 20, 8000)
    }
    encoder = EncoderRNN(
        params["num_heads"],
        params["num_layers"],
        params["latent_dim"],
        params["encoding"],
        params["dropout"],
        params["layer_norm"],
    )
    decoder = DecoderRNN(
        params["num_heads"],
        params["num_layers"],
        params["latent_dim"],
        params["encoding"],
        params["dropout"],
        params["layer_norm"],
    )

    data_manager = dataset_lib.AMPDataManager(
        DATA_DIR / 'unlabelled_positive.csv',
        DATA_DIR / 'unlabelled_negative.csv',
        min_len=MIN_LENGTH,
        max_len=MAX_LENGTH)

    amp_x, amp_y, amp_x_raw = data_manager.get_merged_data()
    optimizer = Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()),
        lr=params["lr"],
        betas=(0.9, 0.999),
    )
    if params["use_clearml"]:
        task = clearml.Task.init(
            project_name="ar-vae-v4", task_name=params["task_name"]
        )
        task.set_parameters(params)
        logger = task.logger
    else:
        logger = None

    best_loss = 1e18
    num_processes = 8
    global DEVICE 
    DEVICE = setup_ddp(rank, world_size)

    with mp.Pool(processes=num_processes) as pool:
        dataset = TensorDataset(amp_x, tensor(amp_y))
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size

        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], sampler=DistributedSampler(train_dataset), pin_memory=True, shuffle=False)
        eval_loader = DataLoader(eval_dataset, batch_size=params["batch_size"], sampler=DistributedSampler(eval_dataset), pin_memory=True, shuffle=False)
        for epoch in tqdm(range(params["epochs"])):
            train_loader.sampler.set_epoch(epoch)
            eval_mode = "deep" if epoch % params["deeper_eval_every"] == 0 else "fast"
            beta_0, beta_1, t_1 = params["kl_beta_schedule"]
            kl_beta = min(beta_0 + (beta_1 - beta_0) / t_1 * epoch, beta_1)
            gamma_0, gamma_1, t_1 = params["gamma_schedule"]
            gamma = min(gamma_0 + (gamma_1 - gamma_0) / t_1 * epoch, gamma_1)
            run_epoch_iwae(
                mode="train",
                encoder=encoder,
                decoder=decoder,
                dataloader=train_loader,
                device=device(params["device"]),
                logger=logger,
                epoch=epoch,
                optimizer=optimizer,
                kl_beta=kl_beta,
                eval_mode=eval_mode,
                iwae_samples=params["iwae_samples"],
                reg_dim=params["reg_dim"],
                gamma = gamma,
                pool = pool
            )
            if eval_mode == "deep":
                eval_loader.sampler.set_epoch(epoch)
                loss = run_epoch_iwae(
                    mode="test",
                    encoder=encoder,
                    decoder=decoder,
                    dataloader=eval_loader,
                    device=device(params["device"]),
                    logger=logger,
                    epoch=epoch,
                    optimizer=None,
                    kl_beta=kl_beta,
                    eval_mode=eval_mode,
                    iwae_samples=params["iwae_samples"],
                    reg_dim=params["reg_dim"],
                    gamma=gamma,
                    pool = pool
                )

                if epoch > 0 and epoch % params["save_model_every"] == 0:
                    save_model(
                        encoder,
                        f"{params['task_name']}_{params['model_name']}_epoch{epoch}_encoder.pt",
                        with_hash=False,
                    )
                    save_model(
                        decoder,
                        f"{params['task_name']}_{params['model_name']}_epoch{epoch}_decoder.pt",
                        with_hash=False,
                    )
        eval_model()
    destroy_process_group()

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # Metoda uruchamiania została już ustawiona (np. w innym module)
        pass
    # os.environ["USE_DISTRIBUTED"] = "1"
    cuda.memory._set_allocator_settings("max_split_size_mb:128")
    set_seed()
    # Inicjalizacja DDP jest już na poziomie globalnym
    world_size = cuda.device_count()
    tmp.spawn(run, args=(world_size,), nprocs=world_size)
# autoencoder.load_state_dict(load('./gmm_model.pt'))
# autoencoder = autoencoder.to('cpu')  


# x = np.asarray(short['Sequence'].tolist())
# y = np.asarray(short['Label'].tolist())
# padded_tab = pad(x)
# tab = to_one_hot(padded_tab)
# # print(tab)
# x_tensor = tensor(tab)
# y_tensor = tensor(y)
# dataset = tensor(x_tensor).to(DEVICE)
# e = e.to(DEVICE)
# m,l,pca_input = e.forward(dataset,DEVICE)
# # print(pca_input.shape)
# pca = PCA(n_components=2)
# pca_inp = pca_input.to('cpu')
# pcaa = pca.fit_transform(pca_inp.detach().numpy())
# plt.scatter(pcaa[:,0],pcaa[:,1])
# plt.show()

# wandb.finish()
# d = d.to(DEVICE)
# seq, _ = d.generate(10)
# print(from_one_hot(transpose(seq,0,1)))
