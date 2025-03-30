import os
from torch import optim, nn, utils, logsumexp, device, cuda, save, isinf, backends, manual_seed, LongTensor, zeros_like, ones_like, isnan
from torch.distributions import Normal
from model.model import EncoderRNN, DecoderRNN#, VAE
# import lightning as L
import pandas as pd
# import wandb
# from pytorch_lightning.loggers import WandbLogger
DEVICE = device(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')
import pandas as pd
import numpy as np
from  torch import tensor, long, tanh, sign
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

ROOT_DIR = Path(__file__).parent#.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
    return h.charge

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
    h.calc_H(scale='eisenberg')
    return list(h.H)

def calculate_hydrophobicmoment(data:list):
    h = modlamp.descriptors.PeptideDescriptor(data, 'eisenberg')
    h.calculate_moment()
    return list(h.descriptor.flatten())


def calculate_physchem(peptides):
    physchem = {}
    #physchem['dataset'] = []
    physchem['length'] = []
    physchem['charge'] = []
    #physchem['pi'] = []
    #physchem['aromacity'] = []
    physchem['hydrophobicity'] = []
    #physchem['hm'] = []

    # physchem['dataset'] = len(peptides)
    physchem['length'] = calculate_length(peptides)
    physchem['charge'] = calculate_charge(peptides)[0].tolist()
    # physchem['pi'] = calculate_isoelectricpoint(peptides)
    # physchem['aromacity'] = calculate_aromaticity(peptides)
    physchem['hydrophobicity'] = calculate_hydrophobicity(peptides)[0].tolist()
    # physchem['hm'] = calculate_hydrophobicmoment(peptides)

    return pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in physchem.items() ]))

# physchem = calculate_physchem([amp_x_raw.tolist()], ['amp_training_data'])

dataset = utils.data.TensorDataset(amp_x)
train_loader = utils.data.DataLoader(amp_x, batch_size=512, shuffle=True)
# train_loader = utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
# e = EncoderRNN(25, 512, 100, 2, bidirectional=True).to(DEVICE)
# d = DecoderRNN(100, 512, 21, 2).to(DEVICE)
# autoencoder = VAE(e, d)
# wandb_logger = WandbLogger(project='my-awesome-project')
# wandb_logger.experiment.config["batch_size"] = 512
# trainer = L.Trainer(max_epochs=300, logger=wandb_logger)
# trainer.fit(model=autoencoder, train_dataloaders=train_loader)
# save(autoencoder.state_dict(), './gmm_model.pt')
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
    "iwae_samples": 16,
    "model_name": "basic",
    "use_clearml": True,
    "task_name": "iwae_progressive_beta_v7_looking_for_a_problem",
    "device": "cuda",
    "deeper_eval_every": 20,
    "save_model_every": 100,
    "reg_dim": [0,1,2] # [length, charge, hydrophobicity]
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
):
    seq_pred = model_out.argmax(axis=2)
    physchem_decoded = calculate_physchem(dataset_lib.decoded(tensor(seq_pred).permute(1, 0), ""))
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
    logger.report_scalar(
        title="Average length metric from modlamp", series=hue, value=physchem_decoded.iloc[:,0].mean(), iteration=epoch
    )
    logger.report_scalar(
        title="Average charge metric from modlamp", series=hue, value=physchem_decoded.iloc[:,1].mean(), iteration=epoch
    )
    logger.report_scalar(
        title="Average hydrophobicity metric from modlamp", series=hue, value=physchem_decoded.iloc[:,2].mean(), iteration=epoch
    )

    # @staticmethod
def compute_reg_loss(z, labels, reg_dim, gamma, factor=1.0):
    """
    Computes the regularization loss
    """
    x = z[:, reg_dim]
    reg_loss = reg_loss_sign(x, labels, factor=factor)
    return gamma * reg_loss

    # @staticmethod
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
    latent_code = latent_code.reshape(-1, 1)
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

    return sign_loss

def run_epoch_iwae(
    mode: Literal["test", "train"],
    encoder: EncoderRNN,
    decoder: DecoderRNN,
    dataloader: utils.data.DataLoader,
    device: device,
    epoch: int,
    kl_beta: float,
    logger: Optional[clearml.Logger],
    optimizer: Optional[optim.Optimizer],
    eval_mode: Literal["fast", "deep"],
    iwae_samples: int,
    reg_dim
):
    ce_loss_fun = nn.CrossEntropyLoss(reduction="none")
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

    K = iwae_samples
    C = VOCAB_SIZE + 1

    for batch in dataloader:
        # decoded = dataset_lib.decoded(batch)
        # for i in range(len(decoded)):
        #     # print(batch[i])
        #     physchem.append(calculate_physchem(decoded[i]))

        # physchem_original = calculate_physchem(dataset_lib.decoded(batch, ""))
        
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
        # z_prior = q_distr.rsample((K,))

        # Kullback Leibler divergence
        log_qzx = q_distr.log_prob(z).sum(dim=2)
        log_pz = prior_distr.log_prob(z).sum(dim=2)

        kl_div = log_qzx - log_pz

        # reconstruction - cross entropy
        sampled_peptide_logits = decoder(z.reshape(K * B, -1)).reshape(S, K, B, C)
        src = sampled_peptide_logits.permute(1, 3, 2, 0)  # K x C x B x S
        # src_avg_k = src.mean(dim=0) # C x B x S
        src_decoded = src.reshape(-1, C, S).argmax(dim=1) # K*B x S
        tgt = peptides.permute(1, 0).reshape(1, B, S).repeat(K, 1, 1)  # K x B x S
        
        physchem_decoded = calculate_physchem(dataset_lib.decoded(src_decoded, ""))

        # K x B
        cross_entropy = ce_loss_fun(
            src,
            tgt,
        ).sum(dim=2)

        reg_loss = 0
        for dim in reg_dim:
            reg_loss += compute_reg_loss(
            z.reshape(-1,z.shape[2]), physchem_decoded.iloc[:, dim], dim, gamma=10.0, factor=1.0 #gamma i delta z papera
        )

        loss = logsumexp(
            cross_entropy + kl_beta * (log_qzx - log_pz) + reg_loss, dim=0
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
            ],
        )
        if eval_mode == "deep":
            report_sequence_char(
                logger,
                hue=f"{mode} - mu",
                epoch=epoch,
                seq_true=np.concatenate(seq_true, axis=1),
                model_out=np.concatenate(model_out, axis=1),
            )
            report_sequence_char(
                logger,
                hue=f"{mode} - z",
                epoch=epoch,
                seq_true=np.concatenate(seq_true, axis=1),
                model_out=np.concatenate(model_out_sampled, axis=1),
            )
    return stat_sum["total"] / len_data

optimizer = Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()),
    lr=params["lr"],
    betas=(0.9, 0.999),
)

if params["use_clearml"]:
    task = clearml.Task.init(
        project_name="ar-vae", task_name=params["task_name"]
    )
    task.set_parameters(params)
    logger = task.logger
else:
    logger = None

def process_batch_data(self, batch):
    inputs, labels = batch
    inputs = Variable(inputs).cuda()
    labels = Variable(labels).cuda()
    return inputs, labels

def model_go(batch):
    K = params["iwae_samples"]
    peptides = batch.permute(1, 0).type(LongTensor).to(device)
    S, B = peptides.shape
    if optimizer:
        optimizer.zero_grad()

    # autoencoding
    mu, std = encoder(peptides)
    assert not (isnan(mu).all() or isnan(std).all() ), f" contains all NaN values: {mu}, {std}"
    assert not (isinf(mu).all() or isinf(std).all()), f" contains all Inf values: {mu}, {std}"

    q_distr = Normal(mu, std)
    z = q_distr.rsample((K,))
    return z

def _extract_relevant_attributes(attributes):
    attr_list = [
        attr for attr in attr_dict.keys() if attr != 'digit_identity' and attr != 'color'
    ]
    attr_idx_list = [
        self.attr_dict[attr] for attr in attr_list
    ]
    attr_labels = attributes[:, attr_idx_list]
    return attr_labels, attr_list

def compute_representations(data_loader):
    latent_codes = []
    attributes = []
    for sample_id, batch in tqdm(enumerate(data_loader)):
        inputs, labels = process_batch_data(batch)
        z_tilde = model_go(inputs)
        latent_codes.append(z_tilde.cpu().numpy())
        attributes.append(labels.numpy())
        if sample_id == 200:
            break
    latent_codes = np.concatenate(latent_codes, 0)
    attributes = np.concatenate(attributes, 0)
    attributes, attr_list = _extract_relevant_attributes(attributes)
    return latent_codes, attributes, attr_list

def eval_model():
    results_fp = os.path.join(
    os.path.dirname(ROOT_DIR),
        'results_dict.json'
    )
    if os.path.exists(results_fp):
        with open(results_fp, 'r') as infile:
            metrics = json.load(infile)
    else:
        data_loader = eval_loader
        latent_codes, attributes, attr_list = compute_representations(data_loader)
        interp_metrics = compute_interpretability_metric(
            latent_codes, attributes, attr_list
        )
        self.metrics = {
            "interpretability": interp_metrics
        }
        self.metrics.update(compute_correlation_score(latent_codes, attributes))
        self.metrics.update(compute_modularity(latent_codes, attributes))
        self.metrics.update(compute_mig(latent_codes, attributes))
        self.metrics.update(compute_sap_score(latent_codes, attributes))
        self.metrics.update(self.test_model(batch_size=batch_size))
        if self.dataset_type == 'mnist':
            self.metrics.update(self.get_resnet_accuracy())
        with open(results_fp, 'w') as outfile:
            json.dump(self.metrics, outfile, indent=2)
    return self.metrics

def run():
    best_loss = 1e18
    for epoch in tqdm(range(params["epochs"])):
        eval_mode = "deep" if epoch % params["deeper_eval_every"] == 0 else "fast"
        beta_0, beta_1, t_1 = params["kl_beta_schedule"]
        kl_beta = min(beta_0 + (beta_1 - beta_0) / t_1 * epoch, 0.01)
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
            reg_dim=params["reg_dim"]
        )
        if eval_mode == "deep":
            loss = run_epoch_iwae(
                mode="test",
                encoder=encoder,
                decoder=decoder,
                dataloader=train_loader,
                device=device(params["device"]),
                logger=logger,
                epoch=epoch,
                optimizer=None,
                kl_beta=kl_beta,
                eval_mode=eval_mode,
                iwae_samples=params["iwae_samples"],
                reg_dim=params["reg_dim"]
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
