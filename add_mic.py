import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import io
import csv
import glob
import re
import copy
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from model.model import EncoderRNN, DecoderRNN
import random
from pathlib import Path
from scipy import stats
from torch import optim, nn, logsumexp, cuda, save, backends, manual_seed, LongTensor, zeros_like, ones_like, tensor, cat, transpose
from torch.distributions import Normal
torch.autograd.set_detect_anomaly(True)
import itertools
import seaborn as sns
from tqdm import tqdm
import data.dataset as dataset_lib
from model.constants import MIN_LENGTH, MAX_LENGTH, VOCAB_SIZE
import ar_vae_metrics as m
from itertools import combinations
from math import ceil
import sys
new_path = '/raid/BattleAMP-apex'
sys.path.append(new_path)
from benchmark_utils import (
    extract_minimal_predictions, extract_species_predictions, read_fasta
)
from utils import *
from sklearn.preprocessing import QuantileTransformer
from params_setting import set_params

def MIC_calc(seq_list):
    col = ['E. coli ATCC11775', 'P. aeruginosa PAO1', 'P. aeruginosa PA14', 'S. aureus ATCC12600', 'E. coli AIG221',
        'E. coli AIG222', 'K. pneumoniae ATCC13883', 'A. baumannii ATCC19606', 'A. muciniphila ATCC BAA-835',
        'B. fragilis ATCC25285', 'B. vulgatus ATCC8482', 'C. aerofaciens ATCC25986', 'C. scindens ATCC35704',
        'B. thetaiotaomicron ATCC29148', 'B. thetaiotaomicron Complemmented', 'B. thetaiotaomicron Mutant',
        'B. uniformis ATCC8492', 'B. eggerthi ATCC27754', 'C. spiroforme ATCC29900', 'P. distasonis ATCC8503',
        'P. copri DSMZ18205', 'B. ovatus ATCC8483', 'E. rectale ATCC33656', 'C. symbiosum', 'R. obeum', 'R. torques',
        'S. aureus (ATCC BAA-1556) - MRSA', 'vancomycin-resistant E. faecalis ATCC700802',
        'vancomycin-resistant E. faecium ATCC700221', 'E. coli Nissle', 'Salmonella enterica ATCC 9150 (BEIRES NR-515)',
        'Salmonella enterica (BEIRES NR-170)', 'Salmonella enterica ATCC 9150 (BEIRES NR-174)',
        'L. monocytogenes ATCC 19111 (BEIRES NR-106)']
    ecoli_cols = ['E. coli ATCC11775'] # Nissle excluded as non-virulent
    saureus_cols = ['S. aureus ATCC12600']
    paeruginosa_cols = ['P. aeruginosa PAO1', 'P. aeruginosa PA14']
    abaumannii_cols = ['A. baumannii ATCC19606']
    kpneumoniae_cols = ['K. pneumoniae ATCC13883']

    # Define strain columns
    bact_columns = {
        'ecoli': ecoli_cols,
        'saureus': saureus_cols,
        'paeruginosa': paeruginosa_cols,
        'abaumannii': abaumannii_cols,
        'kpneumoniae': kpneumoniae_cols,
    }


    max_len = 52  # maximun peptide length

    word2idx, idx2word = make_vocab()
    emb, AAindex_dict = AAindex('/raid/BattleAMP-apex/aaindex1.csv', word2idx)
    vocab_size = len(word2idx)
    emb_size = np.shape(emb)[1]

    model_num = 8
    repeat_num = 5

    f = open('/raid/BattleAMP-apex/best_key_list', 'r')
    lines = f.readlines()
    f.close()

    model_list = []
    for line in lines:
        parsed = line.strip('\n').strip('\r')
        model_list.append(parsed)

    ensemble_num = model_num * repeat_num

    deep_model_list = []
    for a_model_name in model_list:
        for a_en in range(repeat_num):
            key = 'trained_all_model_' + a_model_name + '_ensemble_' + str(a_en)

            if torch.cuda.is_available():
                model = torch.load('/raid/BattleAMP-apex/trained_models/' + key, weights_only=False)
            else:
                model = torch.load('/raid/BattleAMP-apex/trained_models/' + key, map_location=torch.device('cpu'), weights_only=False)
            model.eval()
            deep_model_list.append(model)

    ensemble_counter = 0
    for ensemble_id in range(ensemble_num):

        if torch.cuda.is_available():
            AMP_model = deep_model_list[ensemble_id].cuda().eval()
        else:
            AMP_model = deep_model_list[ensemble_id].eval()

        data_len = len(seq_list)
        batch_size = 200
        for i in range(int(ceil(data_len / float(batch_size)))):

            seq_batch = seq_list[i * batch_size:(i + 1) * batch_size]
            seq_rep, _, _ = onehot_encoding(seq_batch, max_len, word2idx)

            if torch.cuda.is_available():
                X_seq = torch.LongTensor(seq_rep).cuda()
            else:
                X_seq = torch.LongTensor(seq_rep)

            AMP_pred_batch = AMP_model(X_seq).cpu().detach().numpy()
            AMP_pred_batch = 10 ** (6 - AMP_pred_batch)

            if i == 0:
                AMP_pred = AMP_pred_batch
            else:
                AMP_pred = np.vstack([AMP_pred, AMP_pred_batch])

        if ensemble_id == 0:
            AMP_sum = AMP_pred
        else:
            AMP_sum += AMP_pred
        ensemble_counter += 1

    AMP_pred = AMP_sum / float(ensemble_counter)
    modes = ['ecoli', 'saureus']
    processed_data = {}
    for mode in modes:
        mode_indices = [col.index(c) for c in bact_columns[mode]]
        processed_data[mode] = np.mean(AMP_pred[:, mode_indices], axis=1)
    combined_array = np.column_stack(list(processed_data.values()))
    return torch.from_numpy(combined_array)

global ROOT_DIR 
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
global MODELS_DIR 
MODELS_DIR = ROOT_DIR / "first_working_models"
params, train_log_file, eval_log_file, logger = set_params(ROOT_DIR)

params['mic_flg'] = False

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

data_manager = dataset_lib.AMPDataManager(
    positive_filepath = DATA_DIR / 'amp_clean.fasta',
    negative_filepath = None,
    min_len=MIN_LENGTH,
    max_len=MAX_LENGTH,
    mic_flg = False,
    toxicity_flg = False,
    data_dir = DATA_DIR)

amp_x, amp_y, attributes_input, raw_x = data_manager.get_positive_data()
tensor_with_mics = MIC_calc(raw_x)
df = pd.DataFrame({
    'MIC': tensor_with_mics[:, 0],  # All rows, first column of the tensor
    # 'mic': tensor_with_mics[:, 1],  # All rows, second column of the tensor
    'Sequence': raw_x
})
df.to_csv('./data/new_e_coli.tsv', sep="\t")

df = pd.DataFrame({
    # 'MIC': tensor_with_mics[:, 0],  # All rows, first column of the tensor
    'MIC': tensor_with_mics[:, 1],  # All rows, second column of the tensor
    'Sequence': raw_x
})
df.to_csv('./data/new_s_aureus.tsv', sep="\t")
