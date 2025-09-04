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
import notebook_functions as n
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
# from sklearn.decomposition import IncrementalPCA

def find_files_with_matching_epochs(grouped_files):
    """
    Znajduje pliki z tym samym numerem epoki i grupuje je według sufiksu.

    Args:
        grouped_files (dict): Zagnieżdżony słownik z prefiksami, sufiksami i listą plików.
    
    Returns:
        dict: Słownik z kluczem epoki.
    """
    epochs_to_files = {}

    for prefix, groups in grouped_files.items():
        for suffix, file_list in groups.items():
            for filename in file_list:
                # Używamy wyrażenia regularnego, aby znaleźć numer epoki
                match = re.search(r'epoch(\d+)', filename)
                if match:
                    epoch_number = int(match.group(1))
                    
                    if epoch_number not in epochs_to_files:
                        epochs_to_files[epoch_number] = {}
                    if prefix not in epochs_to_files[epoch_number]:
                        epochs_to_files[epoch_number][prefix] = {'_encoder': '', '_decoder': ''}
                    # print(epochs_to_files)
                    # print(f'\n')
                    # print(epoch_number)
                    # print(prefix)
                    # print(f'\n')
                    # Dodajemy plik do odpowiedniej listy na podstawie sufiksu
                    epochs_to_files[epoch_number][prefix][suffix] = filename
    
    return epochs_to_files

def find_and_group_model_files(prefixes_to_compare, suffixes_to_group=['_encoder', '_decoder'], directory="./first_working_models"):
    """
    Wyszukuje pliki .pt dla danych prefiksów i grupuje je według sufiksów.

    Args:
        prefixes_to_compare (list): Lista prefiksów do porównania.
        suffixes_to_group (list): Lista sufiksów do grupowania (np. ['_encoder', '_decoder']).
        directory (str): Ścieżka do folderu, w którym szukamy plików.
    
    Returns:
        dict: Zagnieżdżony słownik z pogrupowanymi plikami.
              Przykład: {'prefix_name': {'_encoder': [...], '_decoder': [...]}}
    """
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
    """Removes '[' and ']' characters from a list of strings."""
    return [item.replace('[', '').replace(']', '') for item in row if isinstance(item, str)]


def read_and_fix_csv(file_path, all_expected_columns):
    """
    Wczytuje plik CSV, ręcznie uzupełnia wiersze o brakujące kolumny
    i zwraca DataFrame z danymi.
    
    Args:
        file_path (str): Ścieżka do pliku CSV.
        all_expected_columns (list): Lista nazw wszystkich oczekiwanych kolumn.
    
    Returns:
        pd.DataFrame: Naprawiony DataFrame z danymi.
    """
    if not isinstance(all_expected_columns, list):
        print("Błąd: 'all_expected_columns' musi być listą.")
        return None

    fixed_rows = []
    
    try:
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
                
    except FileNotFoundError:
        print(f"Błąd: Plik '{file_path}' nie został znaleziony.")
        return None
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas przetwarzania pliku '{file_path}': {e}")
        return None
    
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
    """
    Przycinanie wszystkich tablic NumPy w liście do wymiaru
    najkrótszej tablicy.
    
    Args:
        list_of_arrays (list): Lista tablic NumPy.

    Returns:
        list: Nowa lista tablic, wszystkie o tej samej długości.
    """
    if not list_of_arrays:
        return []

    # 1. Znajdź najkrótszy wymiar (np. liczbę wierszy)
    shortest_dim = min(arr.shape[0] for arr in list_of_arrays)

    # 2. Utwórz nową listę z przyciętymi tablicami
    truncated_arrays = [arr[:shortest_dim] for arr in list_of_arrays]

    return truncated_arrays

def plot_dim(data, target, epoch_number, models_prefixs_to_compare, filename, dim2=1, attr = ['Length', 'Charge' , 'Hydrophobic moment'], xlim=None, ylim=None):
    n_rows = len(attr)
    n_cols = int(data.shape[0]/len(attr))
    n_plots = n_rows * n_cols
    n_sets = target.ndim
    
    min_row = []
    max_row = []
    for i in range(len(attr)):
        if target.shape[2] >= i+1:
            min_row.append(np.min(target[i*n_cols:(i*n_cols)+n_cols,:,i]))
            max_row.append(np.max(target[i*n_cols:(i*n_cols)+n_cols,:,i]))
        
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5*n_cols, 5*n_rows),
        dpi=150           
    )
    axes_flat = axes.flatten()
    for i in range(n_cols):
        for j in range(n_rows):
            if target.shape[2] >= j+1:
                axes[j,i].scatter(
                            x=data[(j*n_cols)+i,:, j],
                            y=data[(j*n_cols)+i,:, dim2],
                            c=target[(j*n_cols)+i,:,j],
                            s=12,
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
    
def plot_latent_surface(train_loader, encoders_list, decoders_list, dim1, dim2=1, grid_res=0.05, z_dim = 56, params = {}, range_value=5):
    attr = ['Length', 'Charge' , 'Hydrophobic moment']
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
                    labels = dataset_lib.calculate_physchem_test(filtered_list)
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
                                
                    dim_z[i].append(z[indexes, :].detach().cpu().numpy())
                    dim_attr[i].append(labels.detach().cpu().numpy())
            else:                       
                x1 = torch.arange(-range_value, range_value, grid_res)
                x2 = torch.arange(-range_value, range_value, grid_res)
                z1, z2 = torch.meshgrid([x1, x2])
                num_points = z1.size(0) * z1.size(1)
                z = torch.randn(1, params["latent_dim"]).to(DEVICE)
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
                    if params['mae_flg']:
                        diff_tensors = []

                        for k in range(len(dim1)):
                            abs_diff = torch.abs(attributes_input[indexes, k] - labels[:, k]).unsqueeze(1)
                            diff_tensors.append(abs_diff)
                        mae_tensor = torch.cat(diff_tensors, dim=1)
                        dim_mae[i].append(mae_tensor.detach().cpu().numpy())
                        # print(f'mae_tensor shape = {mae_tensor.shape}')
                                
                    dim_z[i].append(z_batch[indexes, :].detach().cpu().numpy())
                    dim_attr[i].append(labels.detach().cpu().numpy())
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
    # plot_dim(aggregated_z_points, aggregated_attr_labels, save_filename, dim2=dim2)

def save_sequences(seqs, filename):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        for seq in seqs:
            writer.writerow([seq])


def latent_explore(encoders_list, decoders_list, shifts, data_loader, params, attr_dict, mode = '', submode = ''):
    DEVICE = torch.device(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')
    generated = {}
    generated_analog = {}
    shifts_list = [0]
    for i in shifts:
        shifts_list.append(i)
        shifts_list.append(-i)
    shifts_list = sorted(shifts_list)
    tmp_dict = {}
    tmp_analog_dict = {}
    if mode == 'multi':
        all_combinations_keys = []
        all_combinations_dims = []
        if submode == 'chosen':
            all_combinations_keys.append([k  for k in attr_dict.keys()])
            all_combinations_dims.append([attr_dict[k]  for k in attr_dict.keys()])
                
        else:
            keys = list(attr_dict.keys())
            for i in range(2, len(keys) + 1):
                all_combinations_keys.extend(list(combinations(keys, i)))
            
            for combo in all_combinations_keys:
                dims = [attr_dict[key] for key in combo]
                all_combinations_dims.append(dims)   
        
        for attr_name, attr_dim in zip(all_combinations_keys, all_combinations_dims):
            unconstrained_dfs_all_attrs = {}
            unconstrained_dfs_analog_all_attrs = {}
            unconstrained_dfs_dict_combo = {}
            unconstrained_dfs_analog_dict_combo = {}
            for attr in attr_name:
                unconstrained_dfs_dict_combo[attr] = {}
                unconstrained_dfs_analog_dict_combo[attr] = {}
            models_list = []

            for encoder_name, decoder_name in zip(encoders_list, decoders_list):
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
                # print(encoder_name)
                encoder.load_state_dict(torch.load(f"./first_working_models/{encoder_name}", map_location=DEVICE))
                encoder = encoder.to(DEVICE)
                decoder.load_state_dict(torch.load(f"./first_working_models/{decoder_name}", map_location=DEVICE))
                decoder = decoder.to(DEVICE)
                encoder = encoder.eval()
                decoder = decoder.eval()      
                model = encoder_name.split("_ar-vae")[0]
                models_list.append(model)
                for attr in attr_name:
                    unconstrained_dfs_dict_combo[attr][model] = []
                    unconstrained_dfs_analog_dict_combo[attr][model] = []

                    
                if submode == 'chosen':
                    seq = decoder.generate_from(1000, params["latent_dim"], attr_dim, shifts)
                    generated_sequences = dataset_lib.decoded(dataset_lib.from_one_hot(transpose(seq, 0,1)), "0")
                    generated_sequences = [seq.strip().rstrip("0") for seq in generated_sequences]
                    generated_sequences = [seq for seq in generated_sequences if '0' not in seq]
                    cleaned_sequences = [seq for seq in generated_sequences if seq]
                    generated[model+'_'+str(attr_name)] = cleaned_sequences
    
                    if 'Length' in attr_name:
                        if len(cleaned_sequences) == 0:
                            unconstrained_dfs_dict_combo['Length'][model].append(f'nan ± nan')
                        else:
                            attr = dataset_lib.calculate_length_test(cleaned_sequences)
                            unconstrained_dfs_dict_combo['Length'][model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
                    if 'Charge' in attr_name:
                        if len(cleaned_sequences) == 0:
                            unconstrained_dfs_dict_combo['Charge'][model].append(f'nan ± nan')
                        else:
                            attr = dataset_lib.calculate_charge(cleaned_sequences)
                            unconstrained_dfs_dict_combo['Charge'][model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
                    if 'Hydrophobicity' in attr_name:
                        if len(cleaned_sequences) == 0:
                            unconstrained_dfs_dict_combo['Hydrophobicity'][model].append(f'nan ± nan')
                        else:
                            attr = dataset_lib.calculate_hydrophobicity(cleaned_sequences)
                            unconstrained_dfs_dict_combo['Hydrophobicity'][model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
        
                    # Generate analog
                    batch, _, _, _ = next(iter(data_loader))
                    peptides = batch.permute(1, 0).type(LongTensor).to(DEVICE)
                    mu, std = encoder(peptides)
                    mod_mu = mu.clone().detach()
                    for i, dim in enumerate(attr_dim):
                        mod_mu[:, dim] = mod_mu[:, dim] + shifts[i]
                    outputs = decoder(mod_mu)
                    src = outputs.permute(1, 2, 0) 
                    seq = src.argmax(dim=1)
                    modified_sequences = dataset_lib.decoded(seq, "")
        
                    modified_sequences = [seq.strip().rstrip("0") for seq in modified_sequences]
                    modified_sequences = [seq for seq in modified_sequences if '0' not in seq]
                    cleaned_modified_sequences = [seq for seq in modified_sequences if seq]
                    generated_analog[model+'_'+str(attr_name)+"_"] = cleaned_modified_sequences
    
                    if 'Length' in attr_name:
                        if len(cleaned_modified_sequences) == 0:
                            unconstrained_dfs_analog_dict_combo['Length'][model].append(f'nan ± nan')
                        else:
                            attr = dataset_lib.calculate_length_test(cleaned_modified_sequences)
                            unconstrained_dfs_analog_dict_combo['Length'][model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
                    if 'Charge' in attr_name:
                        if len(cleaned_modified_sequences) == 0:
                            unconstrained_dfs_analog_dict_combo['Charge'][model].append(f'nan ± nan')
                        else:
                            attr = dataset_lib.calculate_charge(cleaned_modified_sequences)
                            unconstrained_dfs_analog_dict_combo['Charge'][model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
                    if 'Hydrophobicity' in attr_name:
                        if len(cleaned_modified_sequences) == 0:
                            unconstrained_dfs_analog_dict_combo['Hydrophobicity'][model].append(f'nan ± nan')
                        else:
                            attr = dataset_lib.calculate_hydrophobicity(cleaned_modified_sequences)
                            unconstrained_dfs_analog_dict_combo['Hydrophobicity'][model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
                else:
                    for shift_value in shifts_list:
                        # Generate unconstrained
                        seq = decoder.generate_from(1000, params["latent_dim"], attr_dim, [shift_value for _ in range(len(attr_dim))])
                        generated_sequences = dataset_lib.decoded(dataset_lib.from_one_hot(transpose(seq, 0,1)), "0")
                        # save_sequences(generated_sequences, f"generated_sequences/{model}_unconstrained_{attr_name}_{shift_value}.csv")
                        generated_sequences = [seq.strip().rstrip("0") for seq in generated_sequences]
                        generated_sequences = [seq for seq in generated_sequences if '0' not in seq]
                        cleaned_sequences = [seq for seq in generated_sequences if seq]
                        generated[model+'_'+str(attr_name)+"_"+str(shift_value)] = cleaned_sequences
        
                        if 'Length' in attr_name:
                            if len(cleaned_sequences) == 0:
                                unconstrained_dfs_dict_combo['Length'][model].append(f'nan ± nan')
                            else:
                                attr = dataset_lib.calculate_length_test(cleaned_sequences)
                                unconstrained_dfs_dict_combo['Length'][model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
                        if 'Charge' in attr_name:
                            if len(cleaned_sequences) == 0:
                                unconstrained_dfs_dict_combo['Charge'][model].append(f'nan ± nan')
                            else:
                                attr = dataset_lib.calculate_charge(cleaned_sequences)
                                unconstrained_dfs_dict_combo['Charge'][model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
                        if 'Hydrophobicity' in attr_name:
                            if len(cleaned_sequences) == 0:
                                unconstrained_dfs_dict_combo['Hydrophobicity'][model].append(f'nan ± nan')
                            else:
                                attr = dataset_lib.calculate_hydrophobicity(cleaned_sequences)
                                unconstrained_dfs_dict_combo['Hydrophobicity'][model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
            
                        # Generate analog
                        batch, _, _, _ = next(iter(data_loader))
                        peptides = batch.permute(1, 0).type(LongTensor).to(DEVICE)
                        mu, std = encoder(peptides)
                        mod_mu = mu.clone().detach()
                        for i, dim in enumerate(attr_dim):
                            mod_mu[:, dim] = mod_mu[:, dim] + shift_value
                        outputs = decoder(mod_mu)
                        src = outputs.permute(1, 2, 0) 
                        seq = src.argmax(dim=1)
                        modified_sequences = dataset_lib.decoded(seq, "")
                        # save_sequences(modified_sequences, f"{model}_modified_{attr_name}_{shift_value}.csv")
            
                        modified_sequences = [seq.strip().rstrip("0") for seq in modified_sequences]
                        modified_sequences = [seq for seq in modified_sequences if '0' not in seq]
                        cleaned_modified_sequences = [seq for seq in modified_sequences if seq]
                        generated_analog[model+'_'+str(attr_name)+"_"+str(shift_value)] = cleaned_modified_sequences
        
                        if 'Length' in attr_name:
                            if len(cleaned_modified_sequences) == 0:
                                unconstrained_dfs_analog_dict_combo['Length'][model].append(f'nan ± nan')
                            else:
                                attr = dataset_lib.calculate_length_test(cleaned_modified_sequences)
                                unconstrained_dfs_analog_dict_combo['Length'][model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
                        if 'Charge' in attr_name:
                            if len(cleaned_modified_sequences) == 0:
                                unconstrained_dfs_analog_dict_combo['Charge'][model].append(f'nan ± nan')
                            else:
                                attr = dataset_lib.calculate_charge(cleaned_modified_sequences)
                                unconstrained_dfs_analog_dict_combo['Charge'][model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
                        if 'Hydrophobicity' in attr_name:
                            if len(cleaned_modified_sequences) == 0:
                                unconstrained_dfs_analog_dict_combo['Hydrophobicity'][model].append(f'nan ± nan')
                            else:
                                attr = dataset_lib.calculate_hydrophobicity(cleaned_modified_sequences)
                                unconstrained_dfs_analog_dict_combo['Hydrophobicity'][model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
            # print(f'unconstrained_dfs_dict_combo = {unconstrained_dfs_dict_combo}')
            for attr in attr_name:
                row_data = np.array(list(unconstrained_dfs_dict_combo[attr].values()))
                if submode == 'chosen':
                    unconstrained_df = pd.DataFrame(row_data, index=models_list)
                else:
                    unconstrained_df = pd.DataFrame(row_data, columns=shifts_list, index=models_list)
                unconstrained_dfs_all_attrs[attr] = unconstrained_df
            tmp_dict[str(attr_name)] = unconstrained_dfs_all_attrs
            
            for attr in attr_name:
                row_data = np.array(list(unconstrained_dfs_analog_dict_combo[attr].values()))
                if submode == 'chosen':
                    unconstrained_df = pd.DataFrame(row_data, index=models_list)
                else:
                    unconstrained_df = pd.DataFrame(row_data, columns=shifts_list, index=models_list)
                unconstrained_dfs_analog_all_attrs[attr] = unconstrained_df
            tmp_analog_dict[str(attr_name)] = unconstrained_dfs_analog_all_attrs
        return tmp_dict, tmp_analog_dict, generated, generated_analog
    else:
        for attr_name, attr_dim in attr_dict.items():
            unconstrained_dfs_dict = {}
            unconstrained_dfs_analog_dict = {}
            models_list = []
            
            for i, (encoder_name, decoder_name) in enumerate(zip(encoders_list, decoders_list)):
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
                # print(encoder_name)
                encoder.load_state_dict(torch.load(f"./first_working_models/{encoder_name}", map_location=DEVICE))
                encoder = encoder.to(DEVICE)
                decoder.load_state_dict(torch.load(f"./first_working_models/{decoder_name}", map_location=DEVICE))
                decoder = decoder.to(DEVICE)
                encoder = encoder.eval()
                decoder = decoder.eval()      
                model = encoder_name.split("_ar-vae")[0]
                models_list.append(model)
    
                unconstrained_dfs_dict[model] = []
                unconstrained_dfs_analog_dict[model] = []
              
                for shift_value in shifts_list:
                    # Generate unconstrained
                    seq = decoder.generate_from(1000, params["latent_dim"], [attr_dim], [shift_value])
                    generated_sequences = dataset_lib.decoded(dataset_lib.from_one_hot(transpose(seq, 0,1)), "0")
                    # save_sequences(generated_sequences, f"generated_sequences/{model}_unconstrained_{attr_name}_{shift_value}.csv")
                    generated_sequences = [seq.strip().rstrip("0") for seq in generated_sequences]
                    generated_sequences = [seq for seq in generated_sequences if '0' not in seq]
                    cleaned_sequences = [seq for seq in generated_sequences if seq]
                    generated[model+'_'+attr_name+"_"+str(shift_value)] = cleaned_sequences
    
                    if attr_name == 'Length':
                        attr = dataset_lib.calculate_length_test(cleaned_sequences)
                    elif attr_name == 'Charge':
                        attr = dataset_lib.calculate_charge(cleaned_sequences)
                    elif attr_name == 'Hydrophobicity':
                        attr = dataset_lib.calculate_hydrophobicity(cleaned_sequences)
                    unconstrained_dfs_dict[model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
        
                    # Generate analog
                    batch, _, _, _ = next(iter(data_loader))
                    peptides = batch.permute(1, 0).type(LongTensor).to(DEVICE)
                    mu, std = encoder(peptides)
                    mod_mu = mu.clone().detach()
                    mod_mu[:, attr_dim] = mod_mu[:, attr_dim] + shift_value
                    outputs = decoder(mod_mu)
                    src = outputs.permute(1, 2, 0) 
                    seq = src.argmax(dim=1)
                    modified_sequences = dataset_lib.decoded(seq, "")
                    # save_sequences(modified_sequences, f"{model}_modified_{attr_name}_{shift_value}.csv")
        
                    modified_sequences = [seq.strip().rstrip("0") for seq in modified_sequences]
                    modified_sequences = [seq for seq in modified_sequences if '0' not in seq]
                    cleaned_modified_sequences = [seq for seq in modified_sequences if seq]
                    generated_analog[model+'_'+attr_name+"_"+str(shift_value)] = cleaned_modified_sequences
    
                    if attr_name == 'Length':
                        attr = dataset_lib.calculate_length_test(cleaned_modified_sequences)
                    elif attr_name == 'Charge':
                        attr = dataset_lib.calculate_charge(cleaned_modified_sequences)
                    elif attr_name == 'Hydrophobicity':
                        attr = dataset_lib.calculate_hydrophobicity(cleaned_modified_sequences)
                    unconstrained_dfs_analog_dict[model].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
    
            row_data = np.array(list(unconstrained_dfs_dict.values()))
            unconstrained_df = pd.DataFrame(row_data, columns= shifts_list, index = models_list)
            tmp_dict[attr_name] = unconstrained_df
            
            row_data = np.array(list(unconstrained_dfs_analog_dict.values()))
            unconstrained_df_analog = pd.DataFrame(row_data, columns= shifts_list, index =models_list)
            tmp_analog_dict[attr_name] = unconstrained_df_analog
        return tmp_dict, tmp_analog_dict, generated, generated_analog

def hobbit(encoder_name, decoder_name, data_loader, params, attr_dict, shift_value = 0.2):
    DEVICE = torch.device('cpu')
    generated_analog = {}
    tmp_analog_dict = {}
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
    # print(encoder_name)
    encoder.load_state_dict(torch.load(f"./first_working_models/{encoder_name}", map_location=DEVICE))
    encoder = encoder.to(DEVICE)
    decoder.load_state_dict(torch.load(f"./first_working_models/{decoder_name}", map_location=DEVICE))
    decoder = decoder.to(DEVICE)
    encoder = encoder.eval()
    decoder = decoder.eval()         

    attr_name = [k for k in attr_dict.keys()]
    unconstrained_dfs_analog_all_attrs = {}
    unconstrained_dfs_analog_dict_combo = {}
    for attr in attr_name:
        unconstrained_dfs_analog_dict_combo[attr] = []
    model = encoder_name.split("_ar-vae")[0]
    batch, _, _, _ = next(iter(data_loader))
    peptides = batch.permute(1, 0).type(LongTensor).to(DEVICE)
    generated_analog[model+"_"+str(0)] = peptides
    if 'Length' in attr_name:
        if len(peptides) == 0:
            unconstrained_dfs_analog_dict_combo['Length'].append(f'nan ± nan')
        else:
            attr = dataset_lib.calculate_length_test(dataset_lib.decoded(peptides.permute(1, 0), ""))
            unconstrained_dfs_analog_dict_combo['Length'].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
    if 'Charge' in attr_name:
        if len(peptides) == 0:
            unconstrained_dfs_analog_dict_combo['Charge'].append(f'nan ± nan')
        else:
            attr = dataset_lib.calculate_charge(dataset_lib.decoded(peptides.permute(1, 0), ""))
            unconstrained_dfs_analog_dict_combo['Charge'].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
    if 'Hydrophobicity' in attr_name:
        if len(peptides) == 0:
            unconstrained_dfs_analog_dict_combo['Hydrophobicity'].append(f'nan ± nan')
        else:
            attr = dataset_lib.calculate_hydrophobicity(dataset_lib.decoded(peptides.permute(1, 0), ""))
            unconstrained_dfs_analog_dict_combo['Hydrophobicity'].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
            
    for shift, dims in hobbit_path.items():    
        for dim in dims:
            # Generate analog
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
            cleaned_modified_sequences = [seq for seq in modified_sequences if seq]
            generated_analog[model+'_'+str(dim)+"_"+str(shift)] = cleaned_modified_sequences
    
            if 'Length' in attr_name:
                if len(cleaned_modified_sequences) == 0:
                    unconstrained_dfs_analog_dict_combo['Length'].append(f'nan ± nan')
                else:
                    attr = dataset_lib.calculate_length_test(cleaned_modified_sequences)
                    unconstrained_dfs_analog_dict_combo['Length'].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
            if 'Charge' in attr_name:
                if len(cleaned_modified_sequences) == 0:
                    unconstrained_dfs_analog_dict_combo['Charge'].append(f'nan ± nan')
                else:
                    attr = dataset_lib.calculate_charge(cleaned_modified_sequences)
                    unconstrained_dfs_analog_dict_combo['Charge'].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
            if 'Hydrophobicity' in attr_name:
                if len(cleaned_modified_sequences) == 0:
                    unconstrained_dfs_analog_dict_combo['Hydrophobicity'].append(f'nan ± nan')
                else:
                    attr = dataset_lib.calculate_hydrophobicity(cleaned_modified_sequences)
                    unconstrained_dfs_analog_dict_combo['Hydrophobicity'].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
    for attr in attr_name:
        row_data = np.array(list(unconstrained_dfs_analog_dict_combo[attr])).reshape(1, 7)
        unconstrained_df = pd.DataFrame(row_data, columns=['Baseline','Length','Charge','Hydrophobicity','Length','Charge','Hydrophobicity'])
        unconstrained_dfs_analog_all_attrs[attr] = unconstrained_df
    tmp_analog_dict[str(attr_name)] = unconstrained_dfs_analog_all_attrs
    return tmp_analog_dict, generated_analog
        