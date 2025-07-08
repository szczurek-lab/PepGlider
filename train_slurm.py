import torch
import os
from torch import optim, nn, logsumexp, cuda, save, backends, manual_seed, LongTensor, zeros_like, ones_like, tensor, cat
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader, random_split
torch.autograd.set_detect_anomaly(True)
from model.model import EncoderRNN, DecoderRNN
import numpy as np
import clearml
from typing import Optional, Literal
from torch.optim import Adam
import itertools
import random
from pathlib import Path
from tqdm import tqdm
import data.dataset as dataset_lib
from model.constants import MIN_LENGTH, MAX_LENGTH, VOCAB_SIZE
import ar_vae_metrics as m
import monitoring as mn
import regularization as r

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
    device: torch.device,
    epoch: int,
    kl_beta: float,
    logger: Optional[clearml.Logger],
    optimizer: Optional[optim.Optimizer],
    eval_mode: Literal["fast", "deep"],
    iwae_samples: int,
    reg_dim,
    gamma
):
    print(f'Epoch {epoch}')
    ce_loss_fun = nn.CrossEntropyLoss(reduction="none")
    if mode == "train":
        encoder.train(), decoder.train()
    else:
        encoder.eval(), decoder.eval()

    stat_sum = {
        "kl_mean": 0.0,
        "ce_sum": 0.0,
        "total": 0.0,
        "reg_loss": 0.0
    }
    seq_true, model_out, model_out_sampled = [], [], []
    len_data = len(dataloader.dataset)
    latent_codes = []
    attributes = []
    ar_vae_metrics = {}

    K = iwae_samples
    C = VOCAB_SIZE
    for batch, labels, physchem, attributes_input in dataloader: 
        peptides = batch.permute(1, 0).type(LongTensor).to(device) # S x B
        physchem_torch = physchem.to(device)
        S, B = peptides.shape
        if optimizer:
            optimizer.zero_grad()

        mu, std = encoder(peptides)

        prior_distr = Normal(zeros_like(mu), ones_like(std))
        q_distr = Normal(mu, std)
        iwae_terms, all_kl_divs, all_srcs, all_tgts = [], [], [], []
        reg_losses_per_sample_list  = []
        for _ in range(K):
            z = q_distr.rsample().to(device) # B, L
            if mode == 'test':
                    latent_codes.append(z.reshape(-1, z.shape[1]).cpu().detach().numpy())
                    labels_torch = labels.to(attributes_input.dtype).unsqueeze(1)
                    attributes.append(cat(
                        (attributes_input, labels_torch), dim=1
                    ))
            log_qzx = q_distr.log_prob(z).sum(dim=1)
            log_pz = prior_distr.log_prob(z).sum(dim=1)

            kl_div = log_qzx - log_pz
            all_kl_divs.append(kl_div)

            # reconstruction - cross entropy
            sampled_peptide_logits = decoder(z)
            # print(f'sampled_peptide_logits shape = {sampled_peptide_logits.shape}')
            src = sampled_peptide_logits.permute(1, 2, 0)  # B x C x S
            all_srcs.append(src)
            # print(f'src shape = {src.shape}')
            tgt = peptides.permute(1, 0)
            all_tgts.append(tgt)
            # print(f'tgt shape = {tgt.shape}')
            # cross_entropy = ce_loss_fun(
            #     src,
            #     tgt,
            # ).sum(dim=1)
            # all_cross_entropies.append(cross_entropy)
            # print(f'cross_entropy shape = {cross_entropy.shape}')

            reg_loss = 0
            for dim in reg_dim:
                reg_loss += r.compute_reg_loss(
                z.reshape(-1,z.shape[1]), physchem_torch[:, dim], dim, gamma, device #gamma i delta z papera
            )
            reg_losses_per_sample_list.append(reg_loss)
            # iwae_sample_term = cross_entropy + kl_beta * kl_div # (B,)
            # iwae_terms.append(iwae_sample_term)

        # iwae_terms_stacked = logsumexp(torch.stack(iwae_terms, dim=0), dim=0)#K reduction
        # print(f'iwae_terms_stacked shape = {iwae_terms_stacked.shape}')
        total_reg_loss = torch.stack(reg_losses_per_sample_list, dim=0).mean(dim=0).sum()
        # loss = logsumexp(iwae_terms_stacked, dim=0) + total_reg_loss
        # torch.stack z listą BxSxN_C (sampled_peptide_logits) da KxBxCxS
        stacked_srcs = torch.stack(all_srcs, dim=0).permute(0,2,1,3)
        # print(f'stacked_srcs shape = {stacked_srcs.shape}')
        stacked_tgts = torch.stack(all_tgts, dim=0)
        # print(f'stacked_tgts shape = {stacked_tgts.shape}')
        cross_entropy = ce_loss_fun(
                stacked_srcs,
                stacked_tgts,
        ).sum(dim=2)
        # print(f'cross_entropy shape = {cross_entropy.shape}')
        stacked_kl_divs = torch.stack(all_kl_divs, dim=0)#.mean(dim=0)
        loss = logsumexp(cross_entropy + kl_beta * stacked_kl_divs, dim=0).mean(dim=0) + total_reg_loss
        stacked_kl_divs = stacked_kl_divs.mean(dim=0)
        # stacked_cross_entropies = torch.stack(all_cross_entropies, dim=0).mean(dim=0)
        # stats
        stat_sum["kl_mean"] += stacked_kl_divs.mean(dim=0).item()
        stat_sum["ce_sum"] += cross_entropy.mean(dim=0).sum(dim=0).item()
        stat_sum["reg_loss"] = total_reg_loss
        print(f'total_reg_loss_with_gamma = {total_reg_loss}')
        stat_sum["total"] += loss.item() * len(batch)   

        if optimizer:
            loss.backward()
            nn.utils.clip_grad_norm_(
                itertools.chain(encoder.parameters(), decoder.parameters()), max_norm=1.0
            )
            optimizer.step()

        # reporting
        if eval_mode == "deep":
            seq_true.append(peptides.cpu().detach().numpy())
            # model_out.append(decoder(mu).cpu().detach().numpy())
            model_out_sampled.append(
                sampled_peptide_logits.cpu().detach().numpy()
            )

    if mode == 'test':
        latent_codes = np.concatenate(latent_codes, 0)
        attributes = cat(attributes, dim=0).numpy()
        attributes, attr_list = m.extract_relevant_attributes(attributes, reg_dim)
        interp_metrics = m.compute_interpretability_metric(
            latent_codes, attributes, attr_list
        )
        ar_vae_metrics = {}
        ar_vae_metrics["Interpretability"] = interp_metrics
        ar_vae_metrics.update(m.compute_correlation_score(latent_codes, attributes))
        ar_vae_metrics.update(m.compute_modularity(latent_codes, attributes))
        ar_vae_metrics.update(m.compute_mig(latent_codes, attributes))
        ar_vae_metrics.update(m.compute_sap_score(latent_codes, attributes))

    if logger is not None:
        mn.report_scalars(
            logger,
            mode,
            epoch,
            scalars=[
                ("Total Loss", stat_sum["total"] / len_data),
                    # ("Posterior Standard Deviation [mean]", stat_sum["std"] / len_data),
                (
                    "Cross Entropy Loss",
                    "sum over samples",
                    stat_sum["ce_sum"] / len_data,
                ),
                # ("Cross Entropy Loss", "best sample", stat_sum["ce_best"] / len_data),
                # ("Cross Entropy Loss", "worst sample", stat_sum["ce_worst"] / len_data),
                (
                    "KL Divergence",
                    "mean over samples",
                    stat_sum["kl_mean"] / len_data,
                ),
                # ("KL Divergence", "best sample", stat_sum["kl_best"] / len_data),
                # ("KL Divergence", "worst sample", stat_sum["kl_worst"] / len_data),
                # ("KL Beta", kl_beta),
                ("Regularization Loss", stat_sum["reg_loss"]/gamma),
                # ("Regularization Loss with gamma", stat_sum["reg_loss"]),
            ],
        )
        if eval_mode == "deep": 
            # mn.report_sequence_char_test(
            #     logger,
            #     hue=f"{mode} - mu",
            #     epoch=epoch,
            #     seq_true=np.concatenate(seq_true, axis=1),
            #     model_out=np.concatenate(model_out, axis=1),
            #     metrics = None
            # )
            mn.report_sequence_char_test(
                logger,
                hue=f"{mode} - z",
                epoch=epoch,
                seq_true=np.concatenate(seq_true, axis=1),
                model_out=np.concatenate(model_out_sampled, axis=1),
                metrics = None
            )
            mn.report_sequence_char_test(
                logger,
                hue=f"{mode} - ar-vae metrics",
                epoch=epoch,
                seq_true=np.concatenate(seq_true, axis=1),
                model_out=None,
                metrics = ar_vae_metrics
            )
    return stat_sum["total"] / len_data

def run():
    global ROOT_DIR 
    ROOT_DIR = Path(__file__).parent
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
        "model_name": os.getenv("CLEARML_PROJECT_NAME", 'ar-vae-v4'),
        "use_clearml": True,
        "task_name": os.getenv("CLEARML_TASK_NAME", "ar-vae 3 dims"),
        "device": "cuda",
        "deeper_eval_every": 20,
        "save_model_every": 20,
        "reg_dim": [0,1,2], # [length, charge, hydrophobicity_moment]
        "gamma_schedule": (0.00001, 20, 8000)
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
    DEVICE = torch.device(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)

    data_manager = dataset_lib.AMPDataManager(
        DATA_DIR / 'unlabelled_positive.csv',
        DATA_DIR / 'unlabelled_negative.csv',
        min_len=MIN_LENGTH,
        max_len=MAX_LENGTH)

    amp_x, amp_y, attributes_input, _ = data_manager.get_merged_data()
    attributes = dataset_lib.normalize_attributes(attributes_input)

    optimizer = Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()),
        lr=params["lr"],
        betas=(0.9, 0.999),
    )
    if params["use_clearml"]:
        task = clearml.Task.init(
            project_name=os.getenv("CLEARML_PROJECT_NAME", 'ar-vae-v4'), task_name=os.getenv("CLEARML_TASK_NAME", "ar-vae 3 dims")
        )
        task.set_parameters(params)
        logger = task.logger
    else:
        logger = None

    dataset = TensorDataset(amp_x, tensor(amp_y), attributes, attributes_input)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=params["batch_size"], shuffle=True)

    for epoch in tqdm(range(params["epochs"])):
        eval_mode = "deep" if epoch % params["deeper_eval_every"] == 0 else "fast"
        beta_0, beta_1, t_1 = params["kl_beta_schedule"]
        kl_beta = min(beta_0 + (beta_1 - beta_0) / t_1 * epoch, beta_1)
        gamma_0, gamma_1, t_1 = params["gamma_schedule"]
        if epoch < 1000:
            gamma = min(gamma_0 + (gamma_1 - gamma_0) / t_1 * epoch, gamma_0)
        else:
            gamma = min(gamma_0 + (gamma_1 - gamma_0) / t_1 * epoch, gamma_1)
        run_epoch_iwae(
                mode="train",
                encoder=encoder,
                decoder=decoder,
                dataloader=train_loader,
                device=DEVICE,
                logger=logger,
                epoch=epoch,
                optimizer=optimizer,
                kl_beta=kl_beta,
                eval_mode=eval_mode,
                iwae_samples=params["iwae_samples"],
                reg_dim=params["reg_dim"],
                gamma = gamma
        )
        if eval_mode == "deep":
            loss = run_epoch_iwae(
                    mode="test",
                    encoder=encoder,
                    decoder=decoder,
                    dataloader=eval_loader,
                    device=DEVICE,
                    logger=logger,
                    epoch=epoch,
                    optimizer=None,
                    kl_beta=kl_beta,
                    eval_mode=eval_mode,
                    iwae_samples=params["iwae_samples"],
                    reg_dim=params["reg_dim"],
                    gamma=gamma
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
    # eval_model() -> probably to do

if __name__ == '__main__':
    set_seed()
    run()
