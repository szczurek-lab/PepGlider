import torch
import os
from torch import optim, nn, logsumexp, cuda, save, backends, manual_seed, LongTensor, zeros_like, ones_like, tensor, cat
from torch.distributions import Normal
torch.autograd.set_detect_anomaly(True)
from model.model import EncoderRNN, DecoderRNN
import numpy as np
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
import csv
from training_functions import set_seed, get_model_arch_hash, save_model, run_epoch_iwae, run
from params_setting import set_params

if __name__ == '__main__':
    set_seed()
    encoder_filepath = os.path.join(
        os.sep, "net","tscratch","people","plggwiazale", "AR-VAE", "first_working_models",
        # os.sep, "home","gwiazale", "AR-VAE", "first_working_models",
        "hyperparams_tuning_mic_only_delta_0.6_ar-vae_epoch900_encoder.pt"
    )
    decoder_filepath = os.path.join(
        os.sep, "net","tscratch","people","plggwiazale", "AR-VAE", "first_working_models",
        # os.sep, "home","gwiazale", "AR-VAE", "first_working_models",
        "hyperparams_tuning_mic_only_delta_0.6_ar-vae_epoch900_decoder.pt"
    )
    # print('AMPs/nonAMPs')
    # run(['positiv_negativ_AMPs'])
    run(['positiv_AMPs'], encoder_filepath, decoder_filepath)
    # run(['positiv_AMPs'])
    # run(['positiv_negativ_AMPs'], encoder_filepath, decoder_filepath)
    #run(['uniprot'], encoder_filepath, decoder_filepath)
    # print('merged data AMPs/nonAMPs + Uniprot')
    # run(['uniprot','positiv_negativ_AMPs'], encoder_filepath, decoder_filepath)
