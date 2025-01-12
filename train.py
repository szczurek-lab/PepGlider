import os
from torch import optim, nn, utils, load, device, cuda, save, zeros, float32, transpose, LongTensor, zeros_like, ones_like
from torch.distributions import Normal
from model.model import EncoderRNN, DecoderRNN#, VAE
import lightning as L
import pandas as pd
# import wandb
from pytorch_lightning.loggers import WandbLogger
DEVICE = device(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')
import pandas as pd
import numpy as np
from  torch import tensor, long
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import clearml
from typing import Optional, Literal
from torch.optim import Adam
import itertools
from pathlib import Path
from tqdm import tqdm
import data.dataset as dataset

MIN_LENGTH = 0
MAX_LENGTH = 25
ROOT_DIR = Path(__file__).parent.parent
# DATA_DIR = ROOT_DIR / "data"
# MODELS_DIR = ROOT_DIR / "pytorch-text-vae"
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
    'c:/Users/olagw/ar-vae/AR-VAE/data/unlabelled_positive.csv',
    'c:/Users/olagw/ar-vae/AR-VAE/data/unlabelled_negative.csv',
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
train_loader = utils.data.DataLoader(amp_x, batch_size=512)
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
    "batch_size": 16,
    "lr": 0.001,
    "kl_beta": 0.1,
    "train_size": None,
    "epochs": 10000,
    "model_name": "basic",
    "use_clearml": True,
    # "use_clearml": False,
    "task_name": "vae",
    "device": "cuda",
    "deeper_eval_every": 20,
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
def report_scalars(
    logger: clearml.Logger,
    series: str,
    epoch: int,
    scalars: list[tuple[str, float]],
):
    for name, val in scalars:
        logger.report_scalar(title=name, series=series, value=val, iteration=epoch)

def report_sequence_char(
    logger: clearml.Logger,
    series: str,
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
    for len_ in range(len_pred.max() + 1):
        idx = len_pred == len_
        true_sub, pred_sub = seq_true[:len_, idx], seq_pred[:len_, idx]

        correct += (true_sub == pred_sub).sum()
        overall += len_ * idx.sum()

    on_predicted_acc = correct / overall if overall > 0 else 0

    logger.report_scalar(
        title="Length Prediction Accuracy",
        series=series,
        value=pred_len_acc,
        iteration=epoch,
    )
    logger.report_scalar(
        title="Length Loss [mae]", series=series, value=pred_len_mae, iteration=epoch
    )
    logger.report_scalar(
        title="Token Prediction Accuracy (on predicted length)",
        series=series,
        value=on_predicted_acc,
        iteration=epoch,
    )

def run_epoch_vae(
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
):
    ce_loss_fun = nn.CrossEntropyLoss(reduction="sum")
    encoder.to(device), decoder.to(device)
    if mode == "train":
        encoder.train(), decoder.train()
    else:
        encoder.eval(), decoder.eval()

    stat_sum = {"kl": 0.0, "ce": 0.0, "std": 0.0, "total": 0.0}
    seq_true, model_out = [], []
    len_data = len(dataloader.dataset)

    for batch in dataloader:
        # S x B
        peptides = batch.permute(1, 0).type(LongTensor).to(device)
        if optimizer:
            optimizer.zero_grad()

        # autoencoding
        mu, std = encoder(peptides)

        prior_distr = Normal(zeros_like(mu), ones_like(std))
        q_distr = Normal(mu, std)
        z = q_distr.rsample()

        # Kullback Leibler divergence
        log_qzx = q_distr.log_prob(z).sum(dim=1)
        log_pz = prior_distr.log_prob(z).sum(dim=1)

        kl_div = (log_qzx - log_pz).sum()

        # reconstruction - cross entropy
        sampled_peptide_logits = decoder(z)  # S x B x C
        S, B, C = sampled_peptide_logits.shape
        input_reshaped = sampled_peptide_logits.permute(1, 0, 2).reshape(B * S, C)
        target_reshaped = peptides.permute(1, 0).reshape(B * S)

        cross_entropy = ce_loss_fun(
            input_reshaped,
            target_reshaped,
        )

        loss = (cross_entropy + kl_beta * kl_div) / len(batch)

        stat_sum["ce"] += cross_entropy.item()
        stat_sum["kl"] += kl_div.item()
        stat_sum["total"] += loss.item() * len(batch)
        stat_sum["std"] += std.mean(axis=1).sum().item()

        if optimizer:
            loss.backward()
            optimizer.step()

        # reporting
        seq_true.append(peptides.cpu().detach().numpy())
        model_out.append(sampled_peptide_logits.cpu().detach().numpy())
    if logger is not None:
        report_scalars(
            logger,
            mode,
            epoch,
            scalars=[
                ("Cross Entropy Loss", stat_sum["ce"] / len_data),
                ("KL Divergence", stat_sum["kl"] / len_data),
                ("Total Loss", stat_sum["total"] / len_data),
                ("Posterior Standard Deviation [mean]", stat_sum["std"] / len_data),
            ],
        )
        if eval_mode == "deep":
            report_sequence_char(
                logger,
                mode,
                epoch,
                seq_true=np.concatenate(seq_true, axis=1),
                model_out=np.concatenate(model_out, axis=1),
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

def save_model(model, name):
    save(model.state_dict(), MODELS_DIR / name)

def run():
    best_loss = 1e18
    for epoch in tqdm(range(params["epochs"])):
        eval_mode = "deep" if epoch % params["deeper_eval_every"] == 0 else "fast"
        run_epoch_vae(
            mode="train",
            encoder=encoder,
            decoder=decoder,
            dataloader=train_loader,
            device=device(params["device"]),
            logger=logger,
            epoch=epoch,
            optimizer=optimizer,
            kl_beta=params["kl_beta"],
            eval_mode=eval_mode,
        )
        if eval_mode == "deep":
            loss = run_epoch_vae(
                mode="test",
                encoder=encoder,
                decoder=decoder,
                dataloader=train_loader,
                device=device(params["device"]),
                logger=logger,
                epoch=epoch,
                optimizer=None,
                kl_beta=params["kl_beta"],
                eval_mode=eval_mode,
            )
            if loss < best_loss:
                best_loss = loss
                save_model(
                    encoder, f"{params['task_name']}_{params['model_name']}_encoder.pt"
                )
                save_model(
                    decoder, f"{params['task_name']}_{params['model_name']}_decoder.pt"
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

d = d.to(DEVICE)
seq, _ = d.generate(10)
print(from_one_hot(transpose(seq,0,1)))