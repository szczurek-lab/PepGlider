import os
from torch import optim, nn, utils, load, device, cuda, save, transpose, backends, manual_seed, LongTensor, zeros_like, ones_like, isnan
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
# from latent_geometry.mapping import TorchModelMapping
from typing import Optional, Literal, List, Tuple, Union
from torch.optim import Adam
import csv
import random
from pathlib import Path
from tqdm import tqdm
import data.dataset as dataset
from model.constants import MIN_LENGTH, MAX_LENGTH, VOCAB_SIZE, SEQ_LEN

ROOT_DIR = Path(__file__).parent#.parent
# DATA_DIR = ROOT_DIR / "data"
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
    "iwae_samples": 16,
    "model_name": "basic",
    "use_clearml": True,
    "task_name": "iwae_progressive_beta_v7_looking_for_a_problem",
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
).eval()
decoder = DecoderRNN(
    params["num_heads"],
    params["num_layers"],
    params["latent_dim"],
    params["encoding"],
    params["dropout"],
    params["layer_norm"],
).eval()

def load_model(model: nn.Module, name: str) -> None:
    state_dict = load(MODELS_DIR / name)
    model.load_state_dict(state_dict)

MODEL_NAME = "iwae_progressive_beta_v7_looking_for_a_problem_basic_epoch9900_"
load_model(
    encoder.to(params["device"]),
    f"{MODEL_NAME}encoder.pt",
)
load_model(
    decoder.to(params["device"]),
    f"{MODEL_NAME}decoder.pt",
)

# decoder_mapping_argmax = TorchModelMapping(
#     decoder,
#     in_shape=(-1, params["latent_dim"]),
#     out_shape=(SEQ_LEN, -1),
#     batch_size=params["batch_size"],
#     call_fn=decoder.decode,
# )

# encoder_mapping = TorchModelMapping(
#     encoder,
#     in_shape=(SEQ_LEN, -1),
#     out_shape=(-1, params["latent_dim"]),
#     call_fn=encoder.encode,
#     batch_size=params["batch_size"],
# )
DEVICE = device(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')

decoder = decoder.to(DEVICE)
seq = decoder.generate(1000, params['latent_dim'])
generated_sequences = dataset.decoded(dataset.from_one_hot(transpose(seq,0,1)), "0")

with open("sequences.csv", "w", newline="") as file:
    writer = csv.writer(file)
    for seq in generated_sequences:
        writer.writerow([seq])