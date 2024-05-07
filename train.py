import os
from torch import optim, nn, utils, Tensor, device, cuda, transpose
from model import EncoderRNN, DecoderRNN, VAE
import lightning as L
import pandas as pd
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
DEVICE = device(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')
import pandas as pd
import numpy as np
from  torch import tensor

def to_one_hot(x):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    classes = range(1, 21)
    aa_encoding = dict(zip(alphabet, classes))
    return [[aa_encoding[aa] for aa in seq] for seq in x]

def from_one_hot(encoded_seqs):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    return [''.join([alphabet[idx.item()] for idx in seq]) for seq in encoded_seqs]

def pad(x, max_length: int = 25) -> np.ndarray:
    # Pad sequences
    padded_sequences = []
    for seq in x:
        padded_seq = seq[:max_length] + [0] * (max_length - len(seq))
        # print(padded_seq)
        padded_sequences.append(padded_seq)
    return padded_sequences

df1 = pd.read_csv("unlabelled_negative.csv")
df1['Label'] = 0
df2 = pd.read_csv("unlabelled_positive.csv")
df2['Label'] = 1
df = pd.concat([df1, df2], axis=0)
filtered_df = df[df['Sequence'].str.len() <= 25]
# print(filtered_df)
x = np.asarray(filtered_df['Sequence'].tolist())
y = np.asarray(filtered_df['Label'].tolist())
tab = to_one_hot(x)
# print(tab)
padded_tab = pad(tab)
x_tensor = tensor(padded_tab)
y_tensor = tensor(y)
dataset = utils.data.TensorDataset(x_tensor, y_tensor)
train_loader = utils.data.DataLoader(dataset, batch_size=512)
e = EncoderRNN(25, 512, 100, 2, bidirectional=True).to(DEVICE)
d = DecoderRNN(100, 512, 21, 2).to(DEVICE)
autoencoder = VAE(e, d)
wandb_logger = WandbLogger(project='my-awesome-project')
wandb_logger.experiment.config["batch_size"] = 512
checkpoint_callback = ModelCheckpoint(dirpath=r'C:\Users\olagw\vae\pytorch-text-vae\checkpoints', filename='model-{epoch}')
trainer = L.Trainer(max_epochs=100, callbacks=[checkpoint_callback], logger=wandb_logger)
model = trainer.fit(model=autoencoder, train_dataloaders=train_loader)
trainer.save_checkpoint(r'C:\Users\olagw\vae\pytorch-text-vae\checkpoints\model.ckpt')
# loaded_model = VAE.load_from_checkpoint('checkpoints/model.ckpt')
wandb.finish()
d = d.to(DEVICE)
seq, _ = d.generate(2)
print(from_one_hot(transpose(seq,0,1)))
