import os
from torch import optim, nn, utils, logsumexp, device, cuda, save, zeros, float32, transpose, LongTensor, zeros_like, ones_like
from torch.distributions import Normal
from model.model import EncoderRNN, DecoderRNN#, VAE
# import lightning as L
import pandas as pd
# import wandb
# from pytorch_lightning.loggers import WandbLogger
DEVICE = device(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')
import pandas as pd
import numpy as np
from  torch import tensor, long
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import clearml
from typing import Optional, Literal, List, Tuple, Union
from torch.optim import Adam
import itertools
from pathlib import Path
from tqdm import tqdm
import data.dataset as dataset
from model.constants import MIN_LENGTH, MAX_LENGTH, VOCAB_SIZE

ROOT_DIR = Path(__file__).parent#.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR
from typing import Optional, Literal

# def to_one_hot(x):
#     alphabet = "ACDEFGHIKLMNPQRSTVWY0"
    # char_to_index = {char: idx for idx, char in enumerate(alphabet)}
    # x_size = len(x)
    # sequence_length = 25
    # num_classes = len(alphabet)
    # one_hot_sequence = zeros((x_size, sequence_length, num_classes), dtype=float32)
    # for i, seq in enumerate(x):
    #     for j, char in enumerate(seq):
    #         one_hot_sequence[i, j, char_to_index[char]] = 1
    # return one_hot_sequence
    # alphabet = list('ACDEFGHIKLMNPQRSTVWY')
#     classes = range(0, 21)
#     aa_encoding = dict(zip(alphabet, classes))
#     return [[aa_encoding[aa] for aa in seq] for seq in x]

# def from_one_hot(encoded_seqs):
#     alphabet = list('ACDEFGHIKLMNPQRSTVWY0')
#     return [''.join([alphabet[idx.item()] for idx in seq]) for seq in encoded_seqs]

# def pad(x, max_length: int = 25) -> np.ndarray:
#     # Pad sequences
#     padded_sequences = []
#     for seq in x:
#         padded_seq = list(seq)[:max_length] + ['0'] * (max_length - len(seq))
#         # print(padded_seq)
#         padded_sequences.append(padded_seq)
#     return padded_sequences
data_manager = dataset.AMPDataManager(
    DATA_DIR / 'unlabelled_positive.csv',
    DATA_DIR / 'unlabelled_negative.csv',
    min_len=MIN_LENGTH,
    max_len=MAX_LENGTH)

amp_x, amp_y = data_manager.get_merged_data()
# df1 = pd.read_csv("unlabelled_negative.csv")
# df1['Label'] = 0
# df2 = pd.read_csv("unlabelled_positive.csv")
# df2['Label'] = 1
# df = pd.concat([df1, df2], axis=0).sample(df1.shape[0]+df2.shape[0])
# filtered_df = df[df['Sequence'].str.len() <= 25]
# short = filtered_df.sample(500)
# # print(filtered_df)
# x = np.asarray(filtered_df['Sequence'].tolist())
# y = np.asarray(filtered_df['Label'].tolist())
# padded_tab = pad(x)
# tab = to_one_hot(padded_tab)
# # # print(tab.shape)
# x_tensor = tensor(tab)
# y_tensor = tensor(y)
dataset = utils.data.TensorDataset(amp_x, tensor(amp_y, dtype=long))
train_loader = utils.data.DataLoader(amp_x, batch_size=512, shuffle=True)
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
    "latent_dim": 64,
    "encoding": "add",
    "dropout": 0.1,
    "batch_size": 512,
    "lr": 0.001,
    "kl_beta_schedule": (0.1, 1.0, 2000),
    "train_size": None,
    "epochs": 10000,
    "iwae_samples": 16,
    "model_name": "basic",
    "use_clearml": True,
    "task_name": "iwae_progressive_beta_v7",
    "device": "cuda",
    "deeper_eval_every": 20,
    "save_model_every": 100,
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    len_true = seq_true.argmin(axis=0)
    len_pred = seq_pred.argmin(axis=0)

    pred_len_acc = (len_true == len_pred).mean()
    pred_len_mae = np.abs(len_true - len_pred).mean()

    correct, overall = 0, 0
    amino_correct, amino_total = 0, 0
    empty_correct, empty_total = 0, 0

    for len_ in range(len_pred.max() + 1):
        idx = len_pred == len_
        true_sub, pred_sub = seq_true[:len_, idx], seq_pred[:len_, idx]

        # Overall token accuracy
        correct += (true_sub == pred_sub).sum()
        overall += len_ * idx.sum()

        # Amino accuracy (non-padding tokens)
        amino_mask = true_sub > 0  # Assuming "amino" tokens are non-zero
        amino_correct += (true_sub[amino_mask] == pred_sub[amino_mask]).sum()
        amino_total += amino_mask.sum()

        # Empty accuracy (padding tokens)
        empty_mask = true_sub == 0  # Assuming "empty" tokens are zeros
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
        # S x B
        peptides = batch.permute(1, 0).type(LongTensor).to(device)
        S, B = peptides.shape
        if optimizer:
            optimizer.zero_grad()

        # autoencoding
        mu, std = encoder(peptides)

        prior_distr = Normal(zeros_like(mu), ones_like(std))
        q_distr = Normal(mu, std)
        z = q_distr.rsample((K,))

        # Kullback Leibler divergence
        log_qzx = q_distr.log_prob(z).sum(dim=2)
        log_pz = prior_distr.log_prob(z).sum(dim=2)

        kl_div = log_qzx - log_pz

        # reconstruction - cross entropy
        sampled_peptide_logits = decoder(z.reshape(K * B, -1)).reshape(S, K, B, C)
        src = sampled_peptide_logits.permute(1, 3, 2, 0)  # K x C x B x S
        tgt = peptides.permute(1, 0).reshape(1, B, S).repeat(K, 1, 1)  # K x B x S

        # K x B
        cross_entropy = ce_loss_fun(
            src,
            tgt,
        ).sum(dim=2)

        loss = logsumexp(
            cross_entropy + kl_beta * (log_qzx - log_pz), dim=0
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
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

        # reporting
        if eval_mode == "deep":
            seq_true.append(peptides.cpu().detach().numpy())
            model_out.append(decoder(mu).cpu().detach().numpy())
            model_out_sampled.append(
                sampled_peptide_logits[:, 0, :, :].cpu().detach().numpy()
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
        project_name="HYDRAMP - basic transformer", task_name=params["task_name"]
    )
    task.set_parameters(params)
    logger = task.logger
else:
    logger = None

def run():
    best_loss = 1e18
    for epoch in tqdm(range(params["epochs"])):
        eval_mode = "deep" if epoch % params["deeper_eval_every"] == 0 else "fast"
        beta_0, beta_1, t_1 = params["kl_beta_schedule"]
        kl_beta = min(beta_0 + (beta_1 - beta_0) / t_1 * epoch, 1.0)
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
