import os
from torch import optim, nn, utils, logsumexp, device, cuda, save, isinf, backends, manual_seed, LongTensor, zeros_like, ones_like, isnan
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from model.model import EncoderRNN, DecoderRNN#, VAE
import pandas as pd
import pandas as pd
import numpy as np
from  torch import tensor, long, tanh, sign, isnan, distributed
from torch.autograd import Variable
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import clearml
from typing import Optional, Literal, List, Tuple, Union
from torch.optim import Adam
import itertools
import random
from pathlib import Path
from tqdm import tqdm
import data.dataset as dataset_lib
from model.constants import MIN_LENGTH, MAX_LENGTH, VOCAB_SIZE
import json
import modlamp.descriptors
import modlamp.analysis
import modlamp.sequences
import multiprocessing as mp
import metrics as m
try:
    mp.set_start_method('spawn')
except RuntimeError:
    # Metoda uruchamiania została już ustawiona (np. w innym module)
    pass
os.environ["USE_DISTRIBUTED"] = "1"
def setup_ddp():

    distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    cuda.set_device(local_rank)
    return device(f"cuda:{local_rank}")
# DEVICE = setup_ddp()
cuda.memory._set_allocator_settings("max_split_size_mb:128")
ROOT_DIR = Path(__file__).parent#.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR
import torch.distributed as dist

def is_main_process():
    return not distributed.is_available() or not distributed.is_initialized() or distributed.get_rank() == 0

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
set_seed()

data_manager = dataset_lib.AMPDataManager(
    DATA_DIR / 'unlabelled_positive.csv',
    DATA_DIR / 'unlabelled_negative.csv',
    min_len=MIN_LENGTH,
    max_len=MAX_LENGTH)

amp_x, amp_y, amp_x_raw = data_manager.get_merged_data()

def calculate_length(data:list):
    lengths = [len(x) for x in data]
    return lengths

def calculate_charge(data:list):
    h = modlamp.analysis.GlobalAnalysis(data)
    h.calc_charge()
    # return h.charge
    return list(h.charge)

def calculate_isoelectricpoint(data:list):
    h = modlamp.analysis.GlobalDescriptor(data)
    h.isoelectric_point()
    return list(h.descriptor.flatten())

def calculate_aromaticity(data:list):
    h = modlamp.analysis.GlobalDescriptor(data)
    h.aromaticity()
    return list(h.descriptor.flatten())

def calculate_hydrophobicity(data:list):
    h = modlamp.analysis.GlobalAnalysis(data)
    # h.calc_H(scale='eisenberg')
    h.calc_uH()
    return list(h.uH)

def calculate_hydrophobicmoment(data:list):
    h = modlamp.descriptors.PeptideDescriptor(data, 'eisenberg')
    h.calculate_moment()
    return list(h.descriptor.flatten())

# def calculate_physchem(peptides):
#     physchem = {}
#     #physchem['dataset'] = []
#     physchem['length'] = []
#     physchem['charge'] = []
#     #physchem['pi'] = []
#     #physchem['aromacity'] = []
#     physchem['hydrophobicity_moment'] = []
#     #physchem['hm'] = []

#     # physchem['dataset'] = len(peptides)
#     physchem['length'] = calculate_length(peptides)
#     physchem['charge'] = calculate_charge(peptides)[0].tolist()
#     # physchem['pi'] = calculate_isoelectricpoint(peptides)
#     # physchem['aromacity'] = calculate_aromaticity(peptides)
#     physchem['hydrophobicity_moment'] = calculate_hydrophobicity(peptides)[0].tolist()
#     # physchem['hm'] = calculate_hydrophobicmoment(peptides)

#     return pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in physchem.items() ]))

def calculate_physchem(pool, peptides):
    """
    Oblicza właściwości fizykochemiczne dla listy peptydów równolegle,
    dzieląc obliczenia dla każdej właściwości.

    Args:
        peptides: Lista sekwencji peptydów (ciągów znaków).
        num_processes: Liczba procesów do użycia w puli.

    Returns:
        dict: Słownik, w którym kluczami są nazwy właściwości
              ('length', 'charge', 'hydrophobicity_moment'),
              a wartościami są listy tych właściwości dla wszystkich peptydów.
    """
    results = {}
    hydrophobicity_result = pool.apply_async(calculate_hydrophobicity, (peptides,))
    length_result = pool.apply_async(calculate_length, (peptides,))
    charge_result = pool.apply_async(calculate_charge, (peptides,))

    results['hydrophobicity_moment'] = hydrophobicity_result.get()
    results['length'] = length_result.get()
    results['charge'] = charge_result.get()

    return results

def gather_physchem_results(async_results):
    """Zbiera wyniki obliczone asynchronicznie dla właściwości fizykochemicznych."""
    return {
        'hydrophobicity_moment': async_results['hydrophobicity_moment'].get(),
        'length': async_results['length'].get(),
        'charge': async_results['charge'].get()
    }

# dataset = TensorDataset(amp_x, tensor(amp_y))
# train_size = int(0.8 * len(dataset))
# eval_size = len(dataset) - train_size

# train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

# train_sampler = DistributedSampler(train_dataset)
# eval_sampler = DistributedSampler(eval_dataset)
# train_loader = DataLoader(train_dataset, batch_size=512, sampler=train_sampler, num_workers=0)
# eval_loader = DataLoader(eval_dataset, batch_size=512, sampler=eval_sampler, num_workers=0)

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
    "reg_dim": [0,1,2], # [length, charge, hydrophobicity]
    "gamma_schedule": (1000000, 100000000, 8000)
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

def report_scalars(
    logger: clearml.Logger,
    hue: str,
    epoch: int,
    scalars: List[Union[Tuple[str, float], Tuple[str, str, float]]],
):
    if not is_main_process():
        return
    for tpl in scalars:
        if len(tpl) == 2:
            name, val = tpl
            current_hue = hue
        else:
            name, hue_suffix, val = tpl
            current_hue = f"{hue} - {hue_suffix}"
        logger.report_scalar(title=name, series=current_hue, value=val, iteration=epoch)

def report_sequence_char(
    logger: clearml.Logger,
    hue: str,
    epoch: int,
    seq_true: np.ndarray,
    model_out: np.ndarray,
    metrics: dict,
    pool
):
#    if not is_main_process():
#        return
    if metrics is None:
        seq_pred = model_out.argmax(axis=2)
        src_pred = dataset_lib.decoded(tensor(seq_pred).permute(1, 0), "")
        filtered_list = [item for item in src_pred if item.strip()]
        if not filtered_list:
            print('All predicted sequences are empty')
        else:
            physchem_decoded = calculate_physchem(pool, filtered_list)
        len_true = seq_true.argmin(axis=0)
        len_pred = seq_pred.argmin(axis=0)

        pred_len_acc = (len_true == len_pred).mean()
        pred_len_mae = np.abs(len_true - len_pred).mean()

        correct, overall = 0, 0
        amino_correct, amino_total = 0, 0
        empty_correct, empty_total = 0, 0

        for len_ in range(len_pred.max() + 1):
            idx = len_pred == len_
            true_sub = seq_true[:, idx]
            pred_sub = seq_pred[:, idx]

            # Overall token accuracy (within the predicted length)
            correct += (true_sub[:len_].reshape(-1) == pred_sub[:len_].reshape(-1)).sum()
            overall += len_ * idx.sum()

            # Amino acid accuracy (non-padding tokens)
            amino_mask = true_sub > 0
            amino_correct += (true_sub[amino_mask] == pred_sub[amino_mask]).sum()
            amino_total += amino_mask.sum()

            # Empty (padding) token accuracy
            empty_mask = true_sub == 0
            empty_correct += (true_sub[empty_mask] == pred_sub[empty_mask]).sum()
            empty_total += empty_mask.sum()

        on_predicted_acc = correct / overall if overall > 0 else 0
        amino_acc = amino_correct / amino_total if amino_total > 0 else 0
        empty_acc = empty_correct / empty_total if empty_total > 0 else 0

        logger.report_scalar(
            title="Length Prediction Accuracy",
            series=hue,
            value=pred_len_acc,
            iteration=epoch,
        )
        logger.report_scalar(
            title="Length Loss [mae]", series=hue, value=pred_len_mae, iteration=epoch
        )
        logger.report_scalar(
            title="Token Prediction Accuracy (on predicted length)",
            series=hue,
            value=on_predicted_acc,
            iteration=epoch,
        )
        logger.report_scalar(
            title="Amino Token Accuracy", series=hue, value=amino_acc, iteration=epoch
        )
        logger.report_scalar(
            title="Empty Token Accuracy", series=hue, value=empty_acc, iteration=epoch
        )
        if filtered_list:
            # logger.report_scalar(
            #     title="Average length metric from modlamp", series=hue, value=physchem_decoded.iloc[:,0].mean(), iteration=epoch
            # )
            # logger.report_scalar(
            #     title="Average charge metric from modlamp", series=hue, value=physchem_decoded.iloc[:,1].mean(), iteration=epoch
            # )
            # logger.report_scalar(
            #     title="Average hydrophobicity moment metric from modlamp", series=hue, value=physchem_decoded.iloc[:,2].mean(), iteration=epoch
            # )
            logger.report_scalar(
                title="Average length metric from modlamp",
                series=hue,
                value=np.mean(physchem_decoded['length']),
                iteration=epoch
            )
            logger.report_scalar(
                title="Average charge metric from modlamp",
                series=hue,
                value=np.mean(physchem_decoded['charge']),
                iteration=epoch
            )
            logger.report_scalar(
                title="Average hydrophobicity moment metric from modlamp",
                series=hue,
                value=np.mean(physchem_decoded['hydrophobicity_moment']),
                iteration=epoch
            )
    else:
        print(f'hue {str(hue)}')
        print(f'metrics {metrics}')
        print(f'epoch {epoch}')
        for attr in metrics.keys():
            if attr == 'Interpretability':
                for subattr in metrics[attr].keys():
                    logger.report_scalar(
                        title=f"Interpretability - {subattr} of latent space", series=hue, value=metrics["Interpretability"][subattr][1], iteration=epoch
                    )
            else:
                logger.report_scalar(
                    title=f"{attr} of latent space", series=hue, value=metrics[attr], iteration=epoch
                )

def compute_reg_loss(z, labels, reg_dim, gamma, factor=1.0):
    """
    Computes the regularization loss
    """
    x = z[:, reg_dim]
    reg_loss = reg_loss_sign(x, labels, factor=factor)
    return gamma * reg_loss

def reg_loss_sign(latent_code, attribute, factor=1.0):
    """
    Computes the regularization loss given the latent code and attribute
    Args:
        latent_code: torch Variable, (N,)
        attribute: torch Variable, (N,)
        factor: parameter for scaling the loss
    Returns
        scalar, loss
    """
    # compute latent distance matrix
    latent_code = latent_code.to(DEVICE).reshape(-1, 1)
    lc_dist_mat = latent_code - latent_code.T

    # compute attribute distance matrix
    attribute_tensor = tensor(attribute.values).to(DEVICE)
    attribute_tensor = attribute_tensor.reshape(-1, 1)
    attribute_dist_mat = attribute_tensor - attribute_tensor.T

    # compute regularization loss
    loss_fn = nn.L1Loss()
    lc_tanh = tanh(lc_dist_mat * factor)
    attribute_sign = sign(attribute_dist_mat)
    sign_loss = loss_fn(lc_tanh, attribute_sign.float())

    return sign_loss.to(DEVICE)

def compute_reg_loss_parallel(args):
    """Oblicza reg_loss równolegle dla podanych wymiarów."""
    z, indexes, physchem_decoded, reg_dim, gamma, factor = args
    reg_loss = 0
    z_reshaped_indexed = z.reshape(-1, z.shape[2])[indexes, :]
    physchem_keys = list(physchem_decoded.keys())  # Pobierz listę kluczy z physchem_decoded

    for i, dim in enumerate(reg_dim):
        if i < len(physchem_keys):
            attribute_column = physchem_decoded[physchem_keys[i]]
            reg_loss += compute_reg_loss(
                z_reshaped_indexed.numpy(),  # Przekazujemy numpy array do compute_reg_loss (zakładam, że tak działa)
                np.array(attribute_column),
                dim,
                gamma=gamma,
                factor=factor
            )
        else:
            print(f"Ostrzeżenie: Brak odpowiadającej kolumny physchem dla dim {dim}")
            continue  # Pominięcie, jeśli nie ma wystarczającej liczby właściwości physchem
    return reg_loss

def _extract_relevant_attributes(labels, reg_dim): 
    attr_list = ['Length', 'Charge', 'Hydrophobicity moment']
    attr_labels = labels[:, reg_dim]
    return attr_labels, attr_list #kiedys do zmiany na bardziej uniwersalne

def calculate_metric(metric_name, latent_codes, attributes, *args):
    """Oblicza daną metrykę i zwraca ją w formie słownika."""
    if metric_name == "interpretability":
        result = m.compute_interpretability_metric(latent_codes, attributes, *args)
        return {"Interpretability": result}
    elif metric_name == "correlation":
        result = m.compute_correlation_score(latent_codes, attributes)
        return result
    elif metric_name == "modularity":
        result = m.compute_modularity(latent_codes, attributes)
        return result
    elif metric_name == "mig":
        result = m.compute_mig(latent_codes, attributes)
        return result
    elif metric_name == "sap_score":
        result = m.compute_sap_score(latent_codes, attributes)
        return result
    else:
        return {}

def calculate_metric_async(pool, name, latent_codes, attributes, *args):
    """Asynchronicznie oblicza pojedynczą metrykę."""
    return pool.apply_async(calculate_metric, (name, latent_codes, attributes, *args))

def compute_all_metrics_async(pool, latent_codes, attributes, attr_list):
    """Wysyła zadania obliczenia wszystkich metryk AR-VAE do puli procesów asynchronicznie."""
    metrics_to_calculate = [
        ("interpretability", latent_codes, attributes, attr_list),
        ("correlation", latent_codes, attributes),
        ("modularity", latent_codes, attributes),
        ("mig", latent_codes, attributes),
        ("sap_score", latent_codes, attributes),
    ]

    async_results = {}
    for name, lc, attr, *args in metrics_to_calculate:
        async_results[name] = calculate_metric_async(pool, name, lc, attr, *args)

    return async_results

def gather_metrics(async_results):
    """Zbiera wyniki obliczonych asynchronicznie metryk."""
    ar_vae_metrics = {}
    for name, async_result in async_results.items():
        result = async_result.get()
        print(f"Przetworzono wynik dla {name}: {result}")
        ar_vae_metrics.update(result)
    return ar_vae_metrics

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
        physchem_original_async = calculate_physchem(pool, (dataset_lib.decoded(batch, ""),)) 
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
            latent_codes.append(z.reshape(-1,z.shape[2]).cpu().detach().numpy())
            physchem_original = physchem_original_async.get() # Pobierz wynik jako dict
            physchem_expanded_list = []
            for _ in range(K):
                # Powiel każdy element listy właściwości K razy
                expanded_batch_features = np.array([physchem_original[key] for key in ['length', 'charge', 'hydrophobicity_moment']]).T.flatten()
                physchem_expanded_list.append(expanded_batch_features)
            physchem_expanded = np.array(physchem_expanded_list)

            attributes.append(np.concatenate((labels.unsqueeze(0).expand(K, -1).reshape(-1,1).numpy(), physchem_expanded), axis =1))
        # Kullback Leibler divergence
        log_qzx = q_distr.log_prob(z).sum(dim=2)
        log_pz = prior_distr.log_prob(z).sum(dim=2)

        kl_div = log_qzx - log_pz

        # reconstruction - cross entropy
        # z = z.reshape(K * B, -1) 
        sampled_peptide_logits = decoder(z.reshape(K*B,-1))
        sampled_peptide_logits = sampled_peptide_logits.view(S, K, B, C)
        src = sampled_peptide_logits.permute(1, 3, 2, 0)  # K x C x B x S
        src_decoded = src.reshape(-1, C, S).argmax(dim=1) # K*B x S
        tgt = peptides.permute(1, 0).reshape(1, B, S).repeat(K, 1, 1)  # K x B x S
        src_decoded = dataset_lib.decoded(src_decoded, "")
        indexes = [index for index, item in enumerate(src_decoded) if item.strip()]
        filtered_list = [item for item in src_decoded if item.strip()]
        physchem_decoded_async = calculate_physchem(pool, filtered_list)
        physchem_decoded = gather_physchem_results(physchem_decoded_async)
        # K x B
        cross_entropy = ce_loss_fun(
            src,
            tgt,
        ).sum(dim=2)

        # reg_loss = 0
        # for dim in reg_dim:
        #     reg_loss += compute_reg_loss(
        #     z.reshape(-1,z.shape[2])[indexes,:], physchem_decoded.iloc[:, dim], dim, gamma=gamma, factor=1.0 #gamma i delta z papera
        # )
        reg_loss = pool.apply_async(compute_reg_loss_parallel, (
                    z.detach().cpu().numpy(),  # Przekazujemy odłączone numpy array
                    indexes,
                    physchem_decoded,
                    reg_dim,
                    gamma,
                    1.0
                )
            )

        loss = logsumexp(
            cross_entropy + kl_beta * (log_qzx - log_pz) + reg_loss.get(), dim=0
        ).mean(dim=0)

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
        attributes, attr_list = _extract_relevant_attributes(attributes, reg_dim)
        # interp_metrics = m.compute_interpretability_metric(
        #     latent_codes, attributes, attr_list
        # )
        # ar_vae_metrics["Interpretability"] = interp_metrics
        # ar_vae_metrics.update(m.compute_correlation_score(latent_codes, attributes))
        # ar_vae_metrics.update(m.compute_modularity(latent_codes, attributes))
        # ar_vae_metrics.update(m.compute_mig(latent_codes, attributes))
        # ar_vae_metrics.update(m.compute_sap_score(latent_codes, attributes))
        async_metrics = compute_all_metrics_async(pool, latent_codes, attributes, attr_list)
        ar_vae_metrics = gather_metrics(async_metrics)
        with open(results_fp, 'w') as outfile:
            json.dump(ar_vae_metrics, outfile, indent=2)
        # print("Interpretability metrics:", ar_vae_metrics)
    if logger is not None:
        report_scalars(
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
                ("Regularization Loss", reg_loss.get()/gamma),
                ("Regularization Loss with gamma", reg_loss.get()),
            ],
        )
        if eval_mode == "deep":
            report_sequence_char(
                logger,
                hue=f"{mode} - mu",
                epoch=epoch,
                seq_true=np.concatenate(seq_true, axis=1),
                model_out=np.concatenate(model_out, axis=1),
                metrics = None,
                pool = pool
            )
            report_sequence_char(
                logger,
                hue=f"{mode} - z",
                epoch=epoch,
                seq_true=np.concatenate(seq_true, axis=1),
                model_out=np.concatenate(model_out_sampled, axis=1),
                metrics = None,
                pool = pool
            )
            report_sequence_char(
                logger,
                hue=f"{mode} - ar-vae metrics",
                epoch=epoch,
                seq_true=np.concatenate(seq_true, axis=1),
                model_out=None,
                metrics = ar_vae_metrics,
                pool = pool
            )
    return stat_sum["total"] / len_data

optimizer = Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()),
    lr=params["lr"],
    betas=(0.9, 0.999),
)

if params["use_clearml"]:
    task = clearml.Task.init(
        project_name="ar-vae-v3_pooling_test", task_name=params["task_name"]
    )
    task.set_parameters(params)
    logger = task.logger
else:
    logger = None

def run():
    best_loss = 1e18
    num_processes = 8
    with mp.Pool(processes=num_processes) as pool:
        dataset = TensorDataset(amp_x, tensor(amp_y))
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size

        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
        global DEVICE 
        DEVICE = setup_ddp()
        world_size = distributed.get_world_size()

        train_sampler = DistributedSampler(train_dataset, rank=DEVICE.index, num_replicas=world_size, shuffle=True)
        eval_sampler = DistributedSampler(eval_dataset, rank=DEVICE.index, num_replicas=world_size, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=512 // world_size, sampler=train_sampler, num_workers=4)
        eval_loader = DataLoader(eval_dataset, batch_size=512 // world_size, sampler=eval_sampler, num_workers=4)
        for epoch in tqdm(range(params["epochs"])):
            train_loader.sampler.set_epoch(epoch)
            eval_mode = "deep" if epoch % params["deeper_eval_every"] == 0 else "fast"
            beta_0, beta_1, t_1 = params["kl_beta_schedule"]
            kl_beta = min(beta_0 + (beta_1 - beta_0) / t_1 * epoch, 0.01)
            gamma_0, gamma_1, t_1 = params["gamma_schedule"]
            gamma = min(gamma_0 + (gamma_1 - gamma_0) / t_1 * epoch, 0.01)
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

if __name__ == '__main__':
    # Inicjalizacja DDP jest już na poziomie globalnym
    run()
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
