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
from toxicity_classifier import classifier as c
import joypy

def find_files_with_matching_epochs(grouped_files):
    epochs_to_files = {}

    for prefix, groups in grouped_files.items():
        for suffix, file_list in groups.items():
            for filename in file_list:
                match = re.search(r'epoch(\d+)', filename)
                if match:
                    epoch_number = int(match.group(1))
                    
                    if epoch_number not in epochs_to_files:
                        epochs_to_files[epoch_number] = {}
                    if prefix not in epochs_to_files[epoch_number]:
                        epochs_to_files[epoch_number][prefix] = {'_encoder': '', '_decoder': ''}
                    epochs_to_files[epoch_number][prefix][suffix] = filename
    
    return epochs_to_files

def find_and_group_model_files(prefixes_to_compare, suffixes_to_group=['_encoder', '_decoder'], directory="./first_working_models"):
    found_files = {}
    unique_prefixes = sorted(list(set(prefixes_to_compare)))

    for prefix in unique_prefixes:
        search_pattern = os.path.join(directory, f"{prefix}*.pt")
        all_matches = glob.glob(search_pattern)
        files_for_prefix = {suffix: [] for suffix in suffixes_to_group}

        for file_path in all_matches:
            file_name = os.path.basename(file_path)
            found = False
            for suffix in suffixes_to_group:
                if file_name.endswith(f"{suffix}.pt"):
                    files_for_prefix[suffix].append(file_name)
                    found = True
                    break
        
        found_files[prefix] = files_for_prefix
    
    return found_files

def clean_row(row):
    return [item.replace('[', '').replace(']', '') for item in row if isinstance(item, str)]


def read_and_fix_csv(file_path, all_expected_columns):
    fixed_rows = []
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        num_cols_in_header = len(header)
        num_expected_cols = len(all_expected_columns)
        
        for row in reader:
            cleaned_row = clean_row(row)
            if len(cleaned_row) != num_cols_in_header:
                
                if len(cleaned_row) > num_expected_cols:
                    cleaned_row = cleaned_row[:num_expected_cols]
                
                if len(cleaned_row) < num_expected_cols:
                    cleaned_row.extend([None] * (num_expected_cols - len(cleaned_row)))

            fixed_rows.append(cleaned_row)
    
    df = pd.DataFrame(fixed_rows, columns=all_expected_columns)
    if 'MAE length' in df:
        df['MAE length'] = pd.to_numeric(df['MAE length'], errors='coerce')
    if 'MAE charge' in df:
        df['MAE charge'] = pd.to_numeric(df['MAE charge'], errors='coerce')
    if 'MAE hydrophobicity moment' in df:
        df['MAE hydrophobicity moment'] = pd.to_numeric(df['MAE hydrophobicity moment'], errors='coerce')
    df = df.fillna(0)
    
    return df

def convert_rgba_to_rgb(rgba):
    row, col, ch = rgba.shape
    if rgba.dtype == 'uint8':
        rgba = rgba.astype('float32') / 255.0
    if ch == 3:
        return rgba
    assert ch == 4
    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype='float32')

    rgb[:, :, 0] = r * a + (1.0 - a)
    rgb[:, :, 1] = g * a + (1.0 - a)
    rgb[:, :, 2] = b * a + (1.0 - a)

    return np.asarray(rgb)

def truncate_to_shortest(list_of_arrays):
    if not list_of_arrays:
        return []
    shortest_dim = min(arr.shape[0] for arr in list_of_arrays)
    truncated_arrays = [arr[:shortest_dim] for arr in list_of_arrays]
    return truncated_arrays
    
def single_plot_dim(data, target, epoch_number, models_prefixs_to_compare, filename, dim2=1, attr = ['Length', 'Charge' , 'Hydrophobic moment'], xlim=None, ylim=None):
    min_row = []
    max_row = []
    for i in range(len(attr)):
        min_row.append(np.nanmin(target))
        max_row.append(np.nanmax(target))

    if 'Length' in attr:
        j = 0
        # ymin = 1
        # ymax = 25
    elif 'Charge' in attr:
        j = 1
        # ymin = -9
        # ymax = 18
    elif 'Hydrophobicity' in attr:
        j = 2
        # ymin = -2.5
        # ymax = 1.4
    elif 'MIC E.coli' in attr:
        j = 3
        # ymin = 1.66
        # ymax = 167.6
    elif 'MIC S.aureus' in attr:
        j = 4
        # ymin = 1.14
        # ymax = 207.6
    elif 'Nontoxicity' in attr:
        j = 5
        # ymin = 0
        # ymax = 1
    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(
        x=data[:, j],
        y=data[:, dim2],
        c=target,
        s=24,
        linewidths=0,
        cmap="viridis",
        alpha=0.5
        # vmin=ymin,
        # vmax=ymax
    )
    ax = plt.gca()
    ax.set_xlim(-2.0,2.0)
    # plt.title(f'{models_prefixs_to_compare[0].split("_ar-vae")[0]}', fontsize=16)
    plt.xlabel(f'dimension: {attr[0]}', fontsize=14)
    plt.ylabel(f'not regularized dimension', fontsize=14)
    
    # --- Colorbar for the single plot ---
    # Note: You need a single axes object to attach a colorbar.
    # ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(
        ax.collections[0],
        cax=cax,
        label=attr[0],
        shrink=0.8,
        aspect=20
    )
    cbar.ax.set_ylabel('')
    # --- Finalizing the figure ---
    plt.suptitle(f"Epoch {epoch_number}", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, format='png', dpi=300)
    plt.show()

def plot_one_dim(data, target, epoch_number, models_prefixs_to_compare, filename, 
                 dim2=1, attr=['Length', 'Charge', 'Hydrophobic moment'], 
                 attr_to_print=[0, 1, 2], xlim=None, ylim=None, 
                 color_limits=None, 
                 half_y_range=False,
                 lower_figure=False): # <--- 1. Dodano nowy argument logiczny
    
    n_rows = int(data.shape[0] / len(attr))
    n_cols = len(attr)
    n_real_cols = len(attr_to_print)
    if half_y_range:
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_real_cols,
            figsize=(5 * 3, 3),
            dpi=150
        )
    elif lower_figure:
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_real_cols,
            figsize=(5 * 3, 3),
            dpi=150
        )
    else:
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_real_cols,
            figsize=(5 * 3, 5),
            dpi=150
        )
    for ax in np.atleast_1d(axes).flatten():
        # Iterujemy po wszystkich czterech ramkach
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)  # Ustawienie cieńszej linii
            spine.set_color('gray')
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    elif not isinstance(axes, list):
        axes = [axes]

    z = 0
    for i in range(n_cols):
        if i in attr_to_print:
            current_vmin = None
            current_vmax = None
            if color_limits is not None and i < len(color_limits):
                if color_limits[i] is not None:
                    current_vmin, current_vmax = color_limits[i]

            for j in range(n_rows):
                if target.shape[2] >= j + 1:
                    im = axes[z].scatter(
                        x=data[(i * n_rows) + j, :, i],
                        y=data[(i * n_rows) + j, :, dim2],
                        c=target[(i * n_rows) + j, :, i],
                        s=24,
                        linewidths=0,
                        cmap="viridis",
                        alpha=0.5,
                        vmin=current_vmin,
                        vmax=current_vmax
                    )
                    
                    if xlim:
                        axes[z].set_xlim(xlim)
                    if ylim:
                        axes[z].set_ylim(ylim)

                    if half_y_range:
                        curr_xmin, curr_xmax = axes[z].get_xlim()
                        x_span = curr_xmax - curr_xmin
                        
                        target_y_span = x_span / 2.0
                        
                        curr_ymin, curr_ymax = axes[z].get_ylim()
                        y_center = (curr_ymax + curr_ymin) / 2.0
                        
                        axes[z].set_ylim(
                            y_center - (target_y_span / 2.0), 
                            y_center + (target_y_span / 2.0)
                        )

                    axes[z].set_xlabel(f'dimension: {attr[i]}', fontsize=14)
                    axes[z].tick_params(axis='both', which='major', labelsize=12)
                    
                    if z % n_real_cols == 0:
                        axes[z].set_ylabel(f'not regularized dimension', fontsize=12)
                    
                    divider1 = make_axes_locatable(axes[z])
                    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
                    
                    cbar = fig.colorbar(im, cax=cax1) 
                    cbar.ax.tick_params(labelsize=12)
                    # Po utworzeniu paska kolorów (i zapisaniu go do zmiennej 'cb'):
                    if cbar:
                        # Colorbar jest również obiektem Axes (axes to jego wewnętrzny atrybut)
                        for spine in cbar.ax.spines.values():
                            spine.set_linewidth(0.5)
                            spine.set_color('gray')
                    z += 1
    
    # fig.suptitle(f'{models_prefixs_to_compare}', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, format='png', dpi=150)
    plt.show()
    
def plot_dim(data, target, epoch_number, models_prefixs_to_compare, filename, dim2=1, attr = ['Length', 'Charge' , 'Hydrophobic moment'], xlim=None, ylim=None):
    n_rows = len(attr)
    n_cols = int(data.shape[0]/len(attr))
    n_plots = n_rows * n_cols
    n_sets = target.ndim
    
    min_row = []
    max_row = []
    for i in range(len(attr)):
        if target.shape[2] >= i+1:
            min_row.append(np.nanmin(target[i*n_cols:(i*n_cols)+n_cols,:,i]))
            max_row.append(np.nanmax(target[i*n_cols:(i*n_cols)+n_cols,:,i]))
        
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5*n_cols, 5*n_rows),
        dpi=150           
    )

    for i in range(n_cols):
        for j in range(n_rows):
            if target.shape[2] >= j+1:
                axes[j,i].scatter(
                            x=data[(j*n_cols)+i,:, j],
                            y=data[(j*n_cols)+i,:, dim2],
                            c=target[(j*n_cols)+i,:,j],
                            s=24,
                            linewidths=0,
                            cmap="viridis",
                            alpha=0.5,
                            vmin=min_row[j],  
                            vmax=max_row[j]  
                )
                axes[j,i].set_title(f'{models_prefixs_to_compare[i].split("_ar-vae")[0]}', fontsize = 16)
                axes[j,i].set_xlabel(f'dimension: {attr[j]}', fontsize=14)
                axes[j,i].set_ylabel(f'not regularized dimension', fontsize=14)
    for i in range(n_rows):
        divider = make_axes_locatable(axes[i, n_cols-1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar_ax_row = fig.colorbar(
            axes[i, n_cols-1].collections[0], 
            cax=cax,
            label='Length',
            shrink=0.8,
            aspect=20 
        )
        cbar_ax_row.ax.set_ylabel('')
    fig.suptitle(f"Epoch {epoch_number}", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, format='png', dpi=150)
    plt.show()

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
    ecoli_cols = ['E. coli ATCC11775', 'E. coli AIG222', 'E. coli AIG221'] # Nissle excluded as non-virulent
    saureus_cols = ['S. aureus ATCC12600', 'S. aureus (ATCC BAA-1556) - MRSA']
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
    # df_raw = pd.DataFrame(data=AMP_pred, columns=col, index=seq_list)
    # df = pd.DataFrame()
    # for mode in modes:
    #     df[mode] = df_raw[bact_columns[mode]].mean(axis=1).reset_index()
    # return df
    
def plot_latent_surface(train_loader, encoders_list, decoders_list, dim1, dim2=1, grid_res=0.05, z_dim = 56, params = {},attr = ['Length', 'Charge' , 'Hydrophobic moment'], mode = 'calc', range_value=5):
    all_final_z_points = []
    all_final_attr_labels = []
    all_final_mae = []
    n_compare = len(encoders_list)
    DEVICE = torch.device(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')
            
    for d in dim1:
        dim_z = [[] for _ in range(n_compare)]
        dim_attr = [[] for _ in range(n_compare)]
        dim_mae = [[] for _ in range(n_compare)]
        
        for i, (encoder_name, decoder_name) in enumerate(zip(encoders_list, decoders_list)):
            # print(decoder_name)
            if 'hyperparams_tuning_mic_only_one_strain_mic_higher_hyperparams' not in encoder_name and 'hyperparams_tuning_mic_only_one_strain_mic_higher_hyperparams' not in decoder_name:
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
            else:
                encoder = EncoderRNN(
                        params["num_heads"],
                        params["num_layers"],
                        64,
                        params["encoding"],
                        params["dropout"],
                        params["layer_norm"],
                )
                decoder = DecoderRNN(
                        params["num_heads"],
                        params["num_layers"],
                        64,
                        params["encoding"],
                        params["dropout"],
                        params["layer_norm"],
                )
            # print(encoder_name)
            encoder.load_state_dict(torch.load(f"./first_working_models/{encoder_name}", map_location=DEVICE))
            encoder = encoder.to(DEVICE)
            decoder.load_state_dict(torch.load(f"./first_working_models/{decoder_name}", map_location=DEVICE))
            decoder = decoder.to(DEVICE)

            if train_loader != None:
                for batch, labels, physchem, attributes_input in train_loader: 
                    peptides = batch.permute(1, 0).type(LongTensor).to(DEVICE) # S x B
                    mu, std = encoder(peptides)
                    z = mu.clone()
                    outputs = decoder(z)
                    src = outputs.permute(1, 2, 0)  # B x C x S
                    src_decoded = src.argmax(dim=1) # B x S
                    decoded = dataset_lib.decoded(src_decoded, "") 
                    filtered_list = [item for item in decoded if item.strip()]
                    indexes = [index for index, item in enumerate(decoded) if item.strip()]
                    if mode == 'real':
                        attrs = attributes_input
                    else:
                        labels = dataset_lib.calculate_physchem_test(filtered_list)
                        mics = MIC_calc(filtered_list)
                        hemolytic_classifier = c.HemolyticClassifier('new_hemolytic_model.xgb')
                        features = hemolytic_classifier.get_input_features(np.array(filtered_list))
                        nontoxicity = hemolytic_classifier.predict_from_features(features, proba=True)
                        nontoxicity_tensor = torch.from_numpy(nontoxicity).unsqueeze(1)
                        # print(nontoxicity_tensor.shape)
                        # attrs = torch.cat((mics, nontoxicity_tensor), dim=1) # torch.cat((labels, mics), dim=1)
                        attrs = torch.cat((labels, mics, nontoxicity_tensor), dim=1)
                        # print(attrs.shape)
                    # print(f'attributes_input shape = {attributes_input.shape}')
                    # print(f'labels shape = {labels.shape}')
                    if params['mae_flg']:
                        diff_tensors = []

                        for k in range(len(dim1)):
                            abs_diff = torch.abs(attributes_input[indexes, k] - labels[:, k]).unsqueeze(1)
                            diff_tensors.append(abs_diff)
                        mae_tensor = torch.cat(diff_tensors, dim=1)
                        dim_mae[i].append(mae_tensor.detach().cpu().numpy())
                        # print(f'mae_tensor shape = {mae_tensor.shape}')
                                
                    dim_z[i].append(z[indexes, :56].detach().cpu().numpy())
                    dim_attr[i].append(attrs.detach().cpu().numpy())
            else:                       
                x1 = torch.arange(-range_value, range_value, grid_res)
                x2 = torch.arange(-range_value, range_value, grid_res)
                z1, z2 = torch.meshgrid([x1, x2])
                num_points = z1.size(0) * z1.size(1)
                if 'hyperparams_tuning_mic_only_one_strain_mic_higher_hyperparams' not in encoder_name and 'hyperparams_tuning_mic_only_one_strain_mic_higher_hyperparams' not in decoder_name:
                    z = torch.randn(1, params["latent_dim"]).to(DEVICE)
                else:
                     z = torch.randn(1, 64).to(DEVICE)
                z = z.repeat(num_points, 1)
                z[:, d] = z1.contiguous().view(1, -1)
                z[:, dim2] = z2.contiguous().view(1, -1)                                   
                mini_batch_size = 500
                num_mini_batches = num_points // mini_batch_size
                for j in tqdm(range(num_mini_batches)):
                    z_batch = z[j * mini_batch_size:(j + 1) * mini_batch_size, :]
                    outputs = decoder(z_batch)
                    src = outputs.permute(1, 2, 0)  # B x C x S
                    src_decoded = src.argmax(dim=1) # B x S
                    decoded = dataset_lib.decoded(src_decoded, "") 
                    filtered_list = [item for item in decoded if item.strip()]
                    indexes = [index for index, item in enumerate(decoded) if item.strip()]
                    labels = dataset_lib.calculate_physchem_test(filtered_list)
                    mics = MIC_calc(filtered_list)
                    hemolytic_classifier = c.HemolyticClassifier('new_hemolytic_model.xgb')
                    features = hemolytic_classifier.get_input_features(np.array(filtered_list))
                    nontoxicity = hemolytic_classifier.predict_from_features(features, proba=True)
                    nontoxicity_tensor = torch.from_numpy(nontoxicity).unsqueeze(1)
                    # print(nontoxicity_tensor.shape)
                    # attrs = torch.cat((mics, nontoxicity_tensor), dim=1) # torch.cat((labels, mics), dim=1)
                    attrs = torch.cat((labels, mics, nontoxicity_tensor), dim=1)
                    if params['mae_flg']:
                        diff_tensors = []

                        for k in range(len(dim1)):
                            abs_diff = torch.abs(attributes_input[indexes, k] - labels[:, k]).unsqueeze(1)
                            diff_tensors.append(abs_diff)
                        mae_tensor = torch.cat(diff_tensors, dim=1)
                        dim_mae[i].append(mae_tensor.detach().cpu().numpy())
                        # print(f'mae_tensor shape = {mae_tensor.shape}')
                                
                    dim_z[i].append(z_batch[indexes, :56].detach().cpu().numpy())
                    dim_attr[i].append(attrs.detach().cpu().numpy())
        for i in range(n_compare):
            # print(dim_z[0].shape)
            aggregated_z = np.vstack(dim_z[i])
            aggregated_attr = np.vstack(dim_attr[i])
            if params['mae_flg']:
                aggregated_mae = np.vstack(dim_mae[i])
            
            all_final_z_points.append(aggregated_z)
            all_final_attr_labels.append(aggregated_attr)
            if params['mae_flg']:
                all_final_mae.append(aggregated_mae)
                # print(f'all_final_mae shapes = {all_final_mae[i].shape}')
        # print(f'final_z_points shape = {len(all_final_z_points)}')
        # print(f'final_attr_labels shape = {len(all_final_attr_labels)}')
    final_z_points = truncate_to_shortest(all_final_z_points)
    final_attr_labels = truncate_to_shortest(all_final_attr_labels)
    if params['mae_flg']:
        final_mae = truncate_to_shortest(all_final_mae)
    aggregated_z_points = np.stack(final_z_points)
    aggregated_attr_labels = np.stack(final_attr_labels)
    # print(f'aggregated_attr_labels shape = {aggregated_attr_labels.shape}')
    if params['mae_flg']:
        aggregated_mae = np.stack(final_mae)
        # print(f'aggregated_mae shape = {aggregated_mae.shape}')
        aggregated_attr_labels_and_mae = np.stack([aggregated_attr_labels, aggregated_mae], axis=3)
        print(f'aggregated_attr_labels_and_mae shape = {aggregated_attr_labels_and_mae.shape}')
    # print(f'aggregated_z_points shape = {aggregated_z_points.shape}')
    # print(f'aggregated_attr_labels shape = {aggregated_attr_labels.shape}')
    save_filename = os.path.join(
            os.getcwd(),
            f'latent_surface_{dim2}dim.png'
    )
    match = re.search(r'epoch(\d+)', encoder_name)
    if match:
        epoch_number = int(match.group(1))
    if params['mae_flg']:
        return aggregated_z_points, aggregated_attr_labels_and_mae, epoch_number, save_filename, dim2, 
    else:
        return aggregated_z_points, aggregated_attr_labels, epoch_number, save_filename, dim2
    plot_dim(aggregated_z_points, aggregated_attr_labels, save_filename, dim2=dim2)

def save_sequences(seqs, filename):
    with open(filename, "w", newline="") as file:
        csv.writer(file).writerows([[s] for s in seqs])

def clean_sequences(raw_sequences):
    decoded = [seq.strip().rstrip("0") for seq in raw_sequences]
    return [seq for seq in decoded if seq and '0' not in seq]

def calculate_metric_stats(sequences, attr_name, device, classifiers=None):
    if not sequences:
        return "nan ± nan"

    val = None
    try:
        if 'Length' in attr_name:
            val = dataset_lib.calculate_length_test(sequences)
        elif 'Charge' in attr_name:
            val = dataset_lib.calculate_charge(sequences)
        elif 'Hydrophobicity' in attr_name:
            val = dataset_lib.calculate_hydrophobicity(sequences)
        elif 'MIC E.coli' in attr_name:
            val = MIC_calc(sequences)[:, 0].cpu().numpy()
        elif 'MIC S.aureus' in attr_name:
            val = MIC_calc(sequences)[:, 1].cpu().numpy()
        elif 'Nontoxicity' in attr_name and classifiers:
            features = classifiers['hemolytic'].get_input_features(np.array(sequences))
            nontoxicity = classifiers['hemolytic'].predict_from_features(features, proba=True)
            val = torch.from_numpy(nontoxicity).cpu().numpy()
    except Exception as e:
        print(f"Error calculating {attr_name}: {e}")
        return "nan ± nan"

    if val is not None:
        return f"{np.mean(val):.2f} ± {np.std(val):.2f}"
    return "nan ± nan"

def load_model_pair(enc_name, dec_name, params, device):
    encoder = EncoderRNN(params["num_heads"],
                    params["num_layers"],
                    params["latent_dim"],
                    params["encoding"],
                    params["dropout"],
                    params["layer_norm"],).to(device)
    decoder = DecoderRNN(params["num_heads"],
                    params["num_layers"],
                    params["latent_dim"],
                    params["encoding"],
                    params["dropout"],
                    params["layer_norm"],).to(device)
    
    enc_path = f"./first_working_models/{enc_name}"
    dec_path = f"./first_working_models/{dec_name}"
    
    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    decoder.load_state_dict(torch.load(dec_path, map_location=device))
    return encoder.eval(), decoder.eval()
    
def latent_explore(encoders_list, decoders_list, shifts, data_loader, params, attr_dict, mode='', submode='', val=None):
    DEVICE = torch.device(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')
    classifiers = {}
    generated = {}
    generated_analog = {}
    tmp_dict = {}
    tmp_analog_dict = {}
    combinations_keys = []
    combinations_dims = []
    
    if any('Nontoxicity' in k for k in attr_dict.keys()):
        classifiers['hemolytic'] = c.HemolyticClassifier('new_hemolytic_model.xgb')
        
    shifts_list = sorted([0] + [s for i in shifts for s in (i, -i)])
    
    if mode == 'multi':
        if submode == 'chosen':
            combinations_keys.append(list(attr_dict.keys()))
            combinations_dims.append(list(attr_dict.values()))
        else:
            keys = list(attr_dict.keys())
            for i in range(2, len(keys) + 1):
                for combo in combinations(keys, i):
                    combinations_keys.append(combo)
                    combinations_dims.append([attr_dict[k] for k in combo])
    else:
        combinations_keys = [[k] for k in attr_dict.keys()]
        combinations_dims = [[v] for v in attr_dict.values()]

    for target_attrs, target_dims in zip(combinations_keys, combinations_dims):
        dfs_data = {attr: {} for attr in target_attrs}
        dfs_analog_data = {attr: {} for attr in target_attrs}
        models_list = []

        for enc_name, dec_name in zip(encoders_list, decoders_list):
            model_name = enc_name.split("_ar-vae")[0]
            models_list.append(model_name)
            encoder, decoder = load_model_pair(enc_name, dec_name, params, DEVICE)

            for attr in target_attrs:
                if model_name not in dfs_data[attr]: dfs_data[attr][model_name] = []
                if model_name not in dfs_analog_data[attr]: dfs_analog_data[attr][model_name] = []

            current_shifts = [shifts] if (mode == 'multi' and submode == 'chosen') else shifts_list
            for shift_val in current_shifts:
                if isinstance(shift_val, list):
                    s_arg = shift_val
                    suffix = str(target_attrs)
                else:
                    s_arg = [shift_val for _ in range(len(target_dims))]
                    suffix = f"{target_attrs[0]}_{shift_val}" if len(target_attrs) == 1 else f"{target_attrs}_{shift_val}"

                # UNCONSTRAINED
                if val is not None:
                    raw_seq = decoder.generate_from(1000, params["latent_dim"], target_dims, s_arg, dim, val)
                else:
                    raw_seq = decoder.generate_from(1000, params["latent_dim"], target_dims, s_arg)
                decoded_seq = dataset_lib.decoded(dataset_lib.from_one_hot(raw_seq.permute(1, 0, 2)), "0")
                clean_seq = clean_sequences(decoded_seq)
                generated[f"{model_name}_{suffix}"] = clean_seq
                for attr in target_attrs:
                    stats = calculate_metric_stats(clean_seq, attr, DEVICE, classifiers)
                    dfs_data[attr][model_name].append(stats)

                # ANALOG 
                batch = next(iter(data_loader))
                peptides = batch[0].permute(1, 0).type(torch.LongTensor).to(DEVICE)
                mu, _ = encoder(peptides)
                mod_mu = mu.clone().detach()
                for i, dim in enumerate(target_dims):
                    shift_increment = shift_val[i] if isinstance(shift_val, list) else shift_val
                    mod_mu[:, dim] += shift_increment
            
                outputs = decoder(mod_mu)
                seq_idx = outputs.permute(1, 2, 0).argmax(dim=1)
                mod_decoded_seq = dataset_lib.decoded(seq_idx, "")
                clean_mod_seq = clean_sequences(mod_decoded_seq)
                generated_analog[f"{model_name}_{suffix}"] = clean_mod_seq
                for attr in target_attrs:
                    stats = calculate_metric_stats(clean_mod_seq, attr, DEVICE, classifiers)
                    dfs_analog_data[attr][model_name].append(stats)

        col_names = shifts_list if not (mode == 'multi' and submode == 'chosen') else None
        for attr in target_attrs:
            data_rows = list(dfs_data[attr].values())
            df = pd.DataFrame(data_rows, index=models_list, columns=col_names)
            if mode == 'multi':
                combo_key = str(target_attrs)
                if combo_key not in tmp_dict:
                    tmp_dict[combo_key] = {}
                tmp_dict[combo_key][attr] = df
            else:
                tmp_dict[attr] = df
            analog_rows = list(dfs_analog_data[attr].values())
            df_analog = pd.DataFrame(analog_rows, index=models_list, columns=col_names)
            if mode == 'multi':
                combo_key = str(target_attrs)
                if combo_key not in tmp_analog_dict:
                    tmp_analog_dict[combo_key] = {}
                tmp_analog_dict[combo_key][attr] = df_analog
            else:
                tmp_analog_dict[attr] = df_analog

    return tmp_dict, tmp_analog_dict, generated, generated_analog
        
def levenshtein_distance(s1, s2):
    rows = len(s1) + 1
    cols = len(s2) + 1
    distance = np.zeros((rows, cols), dtype=int)

    for i in range(1, rows):
        distance[i][0] = i
    for j in range(1, cols):
        distance[0][j] = j
    for col in range(1, cols):
        for row in range(1, rows):
            if s1[row-1] == s2[col-1]:
                cost = 0
            else:
                cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,        
                                     distance[row][col-1] + 1,        
                                     distance[row-1][col-1] + cost)
    return distance[row][col]

def levenshtein_similarity(s1, s2):
    if not s1 or not s2:
        return 0
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    similarity = 1 - (distance / max_len)
    return similarity

def calculate_pairwise_similarity(list1, list2):
    results = []
    for i, s1 in enumerate(list1):
        # for s2 in list2:
        similarity = levenshtein_similarity(s1, list2[i])
        results.append((s1, list2[i], similarity))
    return results  
    
def hobbit(fitted_transformers, encoder_name, decoder_name, data_loader, params, attr_dict, shift_value = 0.2):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    manual_seed(seed)
    cuda.manual_seed(seed)
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    DEVICE = torch.device('cpu')
    generated_analog = {}
    tmp_dict = {}
    normalized_tmp_dict = {}
    hobbit_path = {shift_value: [0,1,2],
                   -shift_value: [0,1,2]}
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
    encoder.load_state_dict(torch.load(f"./first_working_models/{encoder_name}", map_location=DEVICE))
    encoder = encoder.to(DEVICE)
    decoder.load_state_dict(torch.load(f"./first_working_models/{decoder_name}", map_location=DEVICE))
    decoder = decoder.to(DEVICE)
    encoder = encoder.eval()
    decoder = decoder.eval()         

    attr_name = [k for k in attr_dict.keys()]
    hobbit_results = []
    hobbit_normalized_results = []
    hobbit_normalized_all_results = []
    hobbit_all_results = []
    model = encoder_name.split("_ar-vae")[0]

    z_sample = torch.randn(10000, 56).to(DEVICE)
    z_sample[:, :3] = 0.0
    outputs = decoder(z_sample)
    src = outputs.permute(1, 2, 0) 
    seq = src.argmax(dim=1)
    generated_sequences = dataset_lib.decoded(seq, "")
    peptides = seq.permute(1, 0)
    generated_sequences = [seq.strip().rstrip("0") for seq in generated_sequences]
    generated_sequences = [seq for seq in generated_sequences if '0' not in seq]
    cleaned_sequences = [seq for seq in generated_sequences if seq]
    similarity_scores = calculate_pairwise_similarity(cleaned_sequences, cleaned_sequences)
    scores_only = [score for s1, s2, score in similarity_scores]
    mean_score = np.mean(scores_only)
    base_sequences = cleaned_sequences
    generated_analog[model+"_"+str(0)] = peptides
    if 'Length' in attr_name:
        if len(peptides) == 0:
            # unconstrained_dfs_analog_dict_combo['Length'].append(f'nan ± nan')
            curr_len = f'nan ± nan'
            normalized_len = f'nan ± nan'
        else:
            attr = dataset_lib.calculate_length_test(cleaned_sequences)
            length = np.array(attr).reshape(-1, 1)
            # print(np.array(attr).reshape(-1, 1).shape)
            # transformed_length_np = fitted_transformers[0].transform(np.array(attr).reshape(-1, 1))
            data = np.array(attr).reshape(-1, 1)
            data_min = np.min(data)
            data_max = np.max(data)
            transformed_length_np = (data - data_min) / (data_max - data_min)
            curr_len = f'{np.mean(attr, dtype=np.float64):.2f} ± {np.std(attr, dtype=np.float64):.2f}'
            normalized_len = f'{np.mean(transformed_length_np):.5f} ± {np.std(transformed_length_np):.5f}'
            # unconstrained_dfs_analog_dict_combo['Length'].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
    if 'Charge' in attr_name:
        if len(peptides) == 0:
            # unconstrained_dfs_analog_dict_combo['Charge'].append(f'nan ± nan')
            curr_charge = f'nan ± nan'
            normalized_charge = f'nan ± nan'
        else:
            attr = dataset_lib.calculate_charge(cleaned_sequences)
            charge = np.array(attr).reshape(-1, 1)
            # transformed_charge_np = fitted_transformers[1].transform(np.array(attr).reshape(-1, 1))
            data = np.array(attr).reshape(-1, 1)
            data_min = np.min(data)
            data_max = np.max(data)
            transformed_charge_np = (data - data_min) / (data_max - data_min)
            curr_charge = f'{np.mean(attr):.2f} ± {np.std(attr):.2f}'
            normalized_charge = f'{np.mean(transformed_charge_np):.5f} ± {np.std(transformed_charge_np):.5f}'
            # unconstrained_dfs_analog_dict_combo['Charge'].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
    if 'Hydrophobicity' in attr_name:
        if len(peptides) == 0:
            # unconstrained_dfs_analog_dict_combo['Hydrophobicity'].append(f'nan ± nan')
            curr_hydr = f'nan ± nan'
            normalized_hydr = f'nan ± nan'
        else:
            attr = dataset_lib.calculate_hydrophobicity(cleaned_sequences)
            hydr = np.array(attr).reshape(-1, 1)
            # transformed_hydrophobicity_np = fitted_transformers[2].transform(np.array(attr).reshape(-1, 1))
            data = np.array(attr).reshape(-1, 1)
            data_min = np.min(data)
            data_max = np.max(data)
            transformed_hydrophobicity_np = (data - data_min) / (data_max - data_min)
            curr_hydr = f'{np.mean(attr):.2f} ± {np.std(attr):.2f}'
            normalized_hydr = f'{np.mean(transformed_hydrophobicity_np):.5f} ± {np.std(transformed_hydrophobicity_np):.5f}'
            # unconstrained_dfs_analog_dict_combo['Hydrophobicity'].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
    hobbit_results.append([0, curr_len, curr_charge, curr_hydr, mean_score])
    hobbit_normalized_results.append([0, normalized_len, normalized_charge, normalized_hydr, mean_score])
    hobbit_normalized_all_results.append([0, transformed_length_np, transformed_charge_np, transformed_hydrophobicity_np, scores_only])
    hobbit_all_results.append([0, length, charge, hydr, scores_only])
    for shift, dims in hobbit_path.items():    
        for dim in dims:
            x = dataset_lib.pad(dataset_lib.to_one_hot(cleaned_sequences)).reshape(25, -1)
            x = x.int()
            mu = encoder.encode(x)
            mu, std = encoder(peptides)
            mod_mu = mu.clone().detach()
            mod_mu[:, dim] = mod_mu[:, dim] + shift
            outputs = decoder(mod_mu)
            src = outputs.permute(1, 2, 0) 
            seq = src.argmax(dim=1)
            modified_sequences = dataset_lib.decoded(seq, "")
            peptides = seq.permute(1, 0)
            # save_sequences(modified_sequences, f"{model}_modified_{attr_name}_{shift_value}.csv")
    
            modified_sequences = [seq.strip().rstrip("0") for seq in modified_sequences]
            modified_sequences = [seq for seq in modified_sequences if '0' not in seq]
            cleaned_sequences = [seq for seq in modified_sequences if seq]
            generated_analog[model+'_'+str(dim)+"_"+str(shift)] = cleaned_sequences
            similarity_scores = calculate_pairwise_similarity(base_sequences, cleaned_sequences)
            scores_only = [score for s1, s2, score in similarity_scores]
            mean_score = np.mean(scores_only)
            if 'Length' in attr_name:
                if len(cleaned_sequences) == 0:
                    # unconstrained_dfs_analog_dict_combo['Length'].append(f'nan ± nan')
                    curr_len = f'nan ± nan'
                    normalized_len = f'nan ± nan'
                else:
                    attr = dataset_lib.calculate_length_test(cleaned_sequences)
                    length = np.array(attr).reshape(-1, 1)
                    # transformed_length_np = fitted_transformers[0].transform(np.array(attr).reshape(-1, 1))
                    data = np.array(attr).reshape(-1, 1)
                    data_min = np.min(data)
                    data_max = np.max(data)
                    transformed_length_np = (data - data_min) / (data_max - data_min)
                    curr_len = f'{np.mean(attr):.2f} ± {np.std(attr):.2f}'
                    normalized_len = f'{np.mean(transformed_length_np):.5f} ± {np.std(transformed_length_np):.5f}'
            if 'Charge' in attr_name:
                if len(cleaned_sequences) == 0:
                    # unconstrained_dfs_analog_dict_combo['Charge'].append(f'nan ± nan')
                    curr_charge = f'nan ± nan'
                    normalized_charge = f'nan ± nan'
                else:
                    attr = dataset_lib.calculate_charge(cleaned_sequences)
                    charge = np.array(attr).reshape(-1, 1)
                    # transformed_charge_np = fitted_transformers[1].transform(np.array(attr).reshape(-1, 1))
                    data = np.array(attr).reshape(-1, 1)
                    data_min = np.min(data)
                    data_max = np.max(data)
                    transformed_charge_np = (data - data_min) / (data_max - data_min)
                    curr_charge = f'{np.mean(attr):.2f} ± {np.std(attr):.2f}'
                    normalized_charge = f'{np.mean(transformed_charge_np):.5f} ± {np.std(transformed_charge_np):.5f}'
            if 'Hydrophobicity' in attr_name:
                if len(cleaned_sequences) == 0:
                    # unconstrained_dfs_analog_dict_combo['Hydrophobicity'].append(f'nan ± nan')
                    curr_hydr = f'nan ± nan'
                    normalized_hydr = f'nan ± nan'
                else:
                    attr = dataset_lib.calculate_hydrophobicity(cleaned_sequences)
                    hydr = np.array(attr).reshape(-1, 1)
                    # transformed_hydrophobicity_np = fitted_transformers[2].transform(np.array(attr).reshape(-1, 1))
                    data = np.array(attr).reshape(-1, 1)
                    data_min = np.min(data)
                    data_max = np.max(data)
                    transformed_hydrophobicity_np = (data - data_min) / (data_max - data_min)
                    curr_hydr = f'{np.mean(attr):.2f} ± {np.std(attr):.2f}'
                    normalized_hydr = f'{np.mean(transformed_hydrophobicity_np):.5f} ± {np.std(transformed_hydrophobicity_np):.5f}'
            hobbit_results.append([shift, curr_len, curr_charge, curr_hydr, mean_score])
            hobbit_normalized_results.append([shift, normalized_len, normalized_charge, normalized_hydr, mean_score])
            hobbit_normalized_all_results.append([shift, transformed_length_np, transformed_charge_np, transformed_hydrophobicity_np, scores_only])
            hobbit_all_results.append([0, length, charge, hydr, scores_only])
    tmp_dict[str(attr_name)] = pd.DataFrame(hobbit_results)
    normalized_tmp_dict[str(attr_name)] = pd.DataFrame(hobbit_normalized_results)
    all_data = []
    for i, (shift, transformed_length, transformed_charge, transformed_hydr, mean_score) in enumerate(hobbit_normalized_all_results):
        num_rows = transformed_length.shape[0]
    
        data_block = {
            'step': np.repeat('p'+str(i), num_rows),
            'shift': np.repeat(shift, num_rows),  
            'length': transformed_length.flatten(), 
            'charge': transformed_charge.flatten(),
            'hydrophobicity': transformed_hydr.flatten(),
            'similarity': np.array(mean_score)
        }
        all_data.append(pd.DataFrame(data_block))
    final_df = pd.concat(all_data, ignore_index=True)
    df_normalized_melted = final_df.melt(id_vars=['step'],
                              value_vars=['length', 'charge', 'hydrophobicity', 'similarity'],
                              var_name='Metric',
                              value_name='Value')
    all_data = []
    for i, (shift, length, charge, hydr, mean_score) in enumerate(hobbit_all_results):
        num_rows = length.shape[0]
    
        data_block = {
            'step': np.repeat('p'+str(i), num_rows),
            'shift': np.repeat(shift, num_rows),  
            'length': length.flatten(), 
            'charge': charge.flatten(),
            'hydrophobicity': hydr.flatten(),
            'similarity': np.array(mean_score)
        }
        all_data.append(pd.DataFrame(data_block))
    final_df = pd.concat(all_data, ignore_index=True)
    df_melted = final_df.melt(id_vars=['step'],
                              value_vars=['length', 'charge', 'hydrophobicity', 'similarity'],
                              var_name='Metric',
                              value_name='Value')
    return tmp_dict, normalized_tmp_dict, df_melted, df_normalized_melted, generated_analog  

def transform_string_to_filename(original_string, segment_name):
    replacement_with_shift = fr'\1_{segment_name}\2\3.csv'
    pattern_with_shift = r'(.*)(_[A-Za-z0-9\.\s]+)(_[-+]?\d+)$'
    transformed_string = re.sub(pattern_with_shift, replacement_with_shift, original_string)

    if transformed_string == original_string:
        replacement_only_attr = fr'\1_{segment_name}\2.csv'
        pattern_only_attr = r'(.*)(_[A-Za-z0-9\.\s]+)$'
        transformed_string = re.sub(pattern_only_attr, replacement_only_attr, original_string)
    return transformed_string

def generate_fixed_sequences(encoder_name, decoder_name, dim_to_shift, shifts, dim_to_const_val , const_val, params):
    DEVICE = torch.device(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')
    encoder, decoder = load_model_pair(encoder_name, decoder_name, params, DEVICE)
    raw_seq = decoder.generate_from(5000, params["latent_dim"], dim_to_shift, shifts, dim_to_const_val=dim_to_const_val, val=const_val)
    decoded_seq = dataset_lib.decoded(dataset_lib.from_one_hot(raw_seq.permute(1, 0, 2)), "0")
    clean_seq = clean_sequences(decoded_seq)
    return clean_seq

def create_ridgeline_plot(df, title, s1='low activity (high MIC)', s2='high activity (low MIC)', e=40, x_min=None, x_max = None, ylabel = 'non-toxic', flip_axis=False):
    min_x = df['Score'].min() if x_min is None else x_min
    max_x = df['Score'].max() if x_max is None else x_max
    x_range_margin = (max_x - min_x) * 0.05
    x_range_min = min_x - x_range_margin 
    x_range_max = max_x + x_range_margin 

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    keys = df['Key'].unique()
    keys.sort()
    num_keys = len(keys)
    
    fig, axes = plt.subplots(nrows=num_keys, ncols=1, figsize=(8,2/3 * num_keys), 
                             sharex=True)                      
    if num_keys == 1:
        axes = [axes]
    if num_keys > 0:
        axes[0].set_xlim(x_range_min, x_range_max)
    palette = sns.color_palette("viridis", num_keys)
    
    for i, key in enumerate(keys):
        subset = df[df['Key'] == key]
        ax = axes[i]
        
        sns.kdeplot(
            data=subset, 
            x="Score", 
            fill=True, 
            alpha=0.8, 
            linewidth=1.5, 
            color=palette[i],
            ax=ax
        )
        ax.axhline(0, color='black', linewidth=1, linestyle='-')
        start_index = key.find('=')
        new_key = key[start_index+1:]
        end_index = ylabel.find("\n")
        if end_index == -1:
            end_index = len(ylabel)
        ax.text(
            x=0.0,
            y=0.1,   
            s = rf'$\alpha_{{{ylabel[:end_index]}}} = {new_key}$',
            transform=ax.transAxes, 
            fontsize=12, 
            ha='left'
        )
        
        ax.set_ylabel('')
        ax.set_yticks([])  
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False) 
        ax.spines['top'].set_visible(False) 

        if i < num_keys - 1:
            ax.set_xlabel('')
            ax.tick_params(axis='x', length=0, width=0, labelbottom=False)
        else:
            ax.tick_params(axis='x', direction='in', length=5, width=1, 
                            labelbottom=False, color='black') 
            
            ax.set_xlabel('') 
            ax.set_xticks([])

            ax.text(
                x=x_range_min + e, 
                y=-0.002,
                s=s1, 
                fontsize=12,
                ha='left', 
                va='top',
                transform=ax.transData
            )
            ax.text(
                x=x_range_max - e, 
                y=-0.002, 
                s=s2, 
                fontsize=12,
                ha='right', 
                va='top',
                transform=ax.transData
            )

    fig.supylabel(ylabel + ' →', fontsize=14, fontweight='bold', x=0.95) 
    # fig.suptitle(title, fontsize=16, y=1.02)
    plt.subplots_adjust(hspace=-0.5) 

    if flip_axis:
        if num_keys > 0:
            axes[0].invert_xaxis()
            
    plt.show()
    
def autolabel(rects, ax, threshold=500):
    for rect in rects:
        height = rect.get_height()
        if np.isnan(height):
            continue
            
        bar_color = rect.get_facecolor()
        is_dark_bar = bar_color[0] < 0.5 
        
        if height <= threshold:
            y_position = height
            y_offset = 3
            v_align = 'bottom' 
            text_color = 'black'
        else:
            y_position = height * 0.98
            y_offset = -5
            v_align = 'top'
            text_color = 'white' if is_dark_bar else 'black'

        ax.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width() / 2, y_position),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    ha='center', 
                    va=v_align,  # Ustawienie warunkowe
                    color=text_color,
                    fontsize=14,
                    fontweight='bold')

def format_name_alpha(name):
    pattern = r'alpha_([a-zA-Z-]+)\s*=\s*(-?\d+)'
    
    def change(match):
        alpha_key = match.group(1)
        alpha_value = match.group(2)
        return rf'$\alpha_{{\text{{{alpha_key}}}}} = {alpha_value}$'
    match_start = re.search(pattern, name)
    
    if match_start:
        before = name[:match_start.start()].strip()
        formula = change(match_start)
        return f"{before} {formula}"
    return name 
    
def sequences_counts_bar_plots(df, cols, title = 'MIC E.coli for Nontoxicity const val'):
    labels = sorted(np.unique(df['Key'].values), reverse=True)
    labels = [format_name_alpha(name) for name in labels]
    x = np.arange(len(labels))
    width = 0.3
    shift_1 = -width   
    shift_2 = 0
    shift_3 = width
    
    fig, ax = plt.subplots(figsize=(12.5,3), dpi=300)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    rocket_colors = sns.color_palette("rocket", 9)
    
    rects1 = ax.bar(x + shift_1, df[cols[0]], width, 
                    label=cols[0], color=[rocket_colors[i] for i in [1,1,1]])
    rects2 = ax.bar(x + shift_2, df[cols[1]], width, 
                    label=cols[1], color=[rocket_colors[i] for i in [4,4,4]])
    rects3 = ax.bar(x + shift_3, df[cols[2]], width, 
                    label=cols[2],  color=[rocket_colors[i] for i in [7,7,7]])
    
    
    ax.set_ylabel('Sample count', fontsize = 16)
    # ax.set_title(title, pad=20, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=14) 
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(
            facecolor='white', 
            fontsize=12, 
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=4
        )    
    autolabel(rects1,ax)
    autolabel(rects2,ax)
    autolabel(rects3,ax)
    
    fig.tight_layout()
    plt.show()
    