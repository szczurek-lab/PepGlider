import torch
import os
from torch import optim, nn, logsumexp, cuda, save, isinf, backends, manual_seed, LongTensor, zeros_like, ones_like, isnan, tensor, cat
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader, random_split
torch.autograd.set_detect_anomaly(True)
from torch.autograd import Variable
from model.model import EncoderRNN, DecoderRNN
# import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Literal
from torch.optim import Adam
import itertools
import random
from pathlib import Path
from tqdm import tqdm
import data.dataset as dataset_lib
from model.constants import MIN_LENGTH, MAX_LENGTH, VOCAB_SIZE
# import json
import ar_vae_metrics as m
# import time
import matplotlib.colors as mcolors

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

# Twoje stałe (upewnij się, że są zaimportowane lub zdefiniowane)
# from model.constants import SEQ_LEN, VOCAB_SIZE, PAD_TOKEN, CLS_TOKEN
# Przykładowe wartości - Użyj swoich rzeczywistych!
SEQ_LEN = 25 # lub 26, jeśli z CLS_TOKEN
VOCAB_SIZE_AMINO_ACIDS = 20 # Jeśli masz 20 aminokwasów
PAD_TOKEN = VOCAB_SIZE_AMINO_ACIDS # Np. 20
# CLS_TOKEN = VOCAB_SIZE_AMINO_ACIDS + 1 # Np. 21
# TOTAL_VOCAB_SIZE = VOCAB_SIZE_AMINO_ACIDS + 2 # Np. 22

# W Twoim przypadku z checkpointu VOCAB_SIZE + 1 = 112, czyli VOCAB_SIZE = 111.
# To sugeruje, że masz VOCAB_SIZE_AMINO_ACIDS + inne_specjalne_tokeny = 111.
# Upewnij się, że masz to poprawnie zmapowane.
# Przykład:
# TOTAL_VOCAB_SIZE = 112 # To jest num_embeddings w warstwie liniowej dekodera
# PAD_TOKEN = ... # Upewnij się, że masz poprawny indeks PAD_TOKEN
# CLS_TOKEN = ... # Upewnij się, że masz poprawny indeks CLS_TOKEN

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

def plot_dim(data, target, filename, dim1=0, dim2=1, xlim=None, ylim=None):
    if xlim is not None:
        plt.xlim(-xlim, xlim)
    if ylim is not None:
        plt.ylim(-ylim, ylim)
    plt.scatter(
        x=data[:, dim1],
        y=data[:, dim2],
        c=target,
        s=12,
        linewidths=0,
        cmap="viridis",
        alpha=0.5
    )
    plt.xlabel(f'dimension: {dim1}')
    plt.ylabel(f'dimension: {dim2}')
    plt.colorbar()
    plt.savefig(filename, format='png', dpi=300)
    plt.close()
    img = Image.open(filename)
    img_resized = img.resize((485, 360), Image.Resampling.LANCZOS)
    img = convert_rgba_to_rgb(np.array(img_resized))
    return img

def plot_latent_surface(decoder, attr_str, dim1=0, dim2=1, grid_res=0.05, z_dim = 56):
    # create the dataspace
    x1 = torch.arange(-5., 5., grid_res)
    x2 = torch.arange(-5., 5., grid_res)
    z1, z2 = torch.meshgrid([x1, x2])
    num_points = z1.size(0) * z1.size(1)
    z = torch.randn(1, z_dim)
    z = z.repeat(num_points, 1)
    z[:, dim1] = z1.contiguous().view(1, -1)
    z[:, dim2] = z2.contiguous().view(1, -1)
    z = Variable(z).cuda()
    filtered_z_points = []
    filtered_attr_labels = []
    mini_batch_size = 500
    num_mini_batches = num_points // mini_batch_size
    attr_labels_all = []
    for i in tqdm(range(num_mini_batches)):
        z_batch = z[i * mini_batch_size:(i + 1) * mini_batch_size, :]
        outputs = decoder(z_batch)
        src = outputs.permute(1, 2, 0)  # B x C x S
        src_decoded = src.argmax(dim=1) # B x S
        # Listy do przechowywania danych z bieżącej paczki
        current_batch_filtered_z = []
        current_batch_filtered_labels = []

        # Iterujemy przez każdą próbkę w bieżącej paczce
        for j in range(mini_batch_size):
            # Zdekodowana sekwencja dla pojedynczej próbki
            # Upewnij się, że dataset_lib.decoded wie, jak przetworzyć pojedynczą sekwencję
            single_src_decoded = dataset_lib.decoded(src_decoded[j:j+1], "") # Przekazujemy mini-batch o rozmiarze 1
                
            # Sprawdzamy, czy sekwencja jest pusta po strip()
            if single_src_decoded and single_src_decoded[0].strip(): # Sprawdza czy lista nie jest pusta i element nie jest pustym stringiem
                # Obliczamy etykiety tylko dla niepustych sekwencji
                labels = dataset_lib.calculate_physchem_test(single_src_decoded) # Labels dla pojedynczej sekwencji
                # normalized_physchem_torch = labels.clone() # Zakładam, że labels to już tensor PyTorch
                    
                # # Normalizacja - tylko dla 1 elementu na raz
                # for dim_idx in range(labels.shape[1]):
                #     column_to_normalize = labels[:, dim_idx].unsqueeze(1)
                #     normalized_column = dataset_lib.normalize_dimension_to_0_1(column_to_normalize, dim=0)
                #     normalized_physchem_torch[:, dim_idx] = normalized_column.squeeze(1)

                current_batch_filtered_z.append(z_batch[j:j+1]) # Dodajemy odpowiadający punkt z
                current_batch_filtered_labels.append(labels) # Dodajemy znormalizowane atrybuty
            
        # Po przetworzeniu całej paczki, dołączamy tylko te, które przeszły filtr
        if current_batch_filtered_z: # Sprawdzamy, czy coś zostało dodane
            filtered_z_points.append(torch.cat(current_batch_filtered_z, 0))
            filtered_attr_labels.append(torch.cat(current_batch_filtered_labels, 0))

    final_z_points = torch.cat(filtered_z_points, 0).cpu().numpy()
    final_attr_labels = torch.cat(filtered_attr_labels, 0).cpu().numpy()
    # print(f'final_z_points shape = {final_z_points.shape}')
    # print(f'final_attr_labels shape = {final_attr_labels.shape}')
    #     src_decoded = dataset_lib.decoded(src_decoded, "")
    #     filtered_list = [item for item in src_decoded if item.strip()]
    #     labels = dataset_lib.calculate_physchem_test(filtered_list)
    #     normalized_physchem_torch = labels.clone()
    #     for dim_idx in range(labels.shape[1]):
    #        column_to_normalize = labels[:, dim_idx].unsqueeze(1)
    #        normalized_column = dataset_lib.normalize_dimension_to_0_1(column_to_normalize, dim=0)
    #        normalized_physchem_torch[:, dim_idx] = normalized_column.squeeze(1)
    #     attr_labels_all.append(normalized_physchem_torch)
    # attr_labels_all = torch.cat(attr_labels_all, 0).cpu().numpy()
    save_filename = os.path.join(
           os.path.dirname(os.path.realpath(__file__)),
           f'latent_surface_{attr_str}.png'
    )
    z = z.cpu().numpy()[:num_mini_batches*mini_batch_size, :]
    plot_dim(final_z_points, final_attr_labels[:, dim1], save_filename, dim1=dim1, dim2=dim2)

def run():
    global ROOT_DIR 
    ROOT_DIR = Path(__file__).parent
    DATA_DIR = ROOT_DIR / "data"
    global MODELS_DIR 
    MODELS_DIR = ROOT_DIR
    params = {
        "num_heads": 4,
        "num_layers": 6,
        "layer_norm": True,
        "latent_dim": 56,
        "encoding": "add",
        "dropout": 0.1,
        "batch_size": 64,
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
        "reg_dim": [0,1,2], # [length, charge, hydrophobicity_moment]
        "gamma_schedule": (1, 200, 8000)
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

    attr_dict = {
        'Length': 0, 
        'Charge': 1, 
        'Hydrophobicity moment': 2
    }
    DEVICE = torch.device(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')
    is_cpu = False if torch.cuda.is_available() else True
    encoder_filepath = os.path.join(
        os.sep, "home","gwiazale", "AR-VAE",
        "first_working_models","ar_vae_continue_training_ar-vae-v4_epoch940_encoder.pt"
    )
    decoder_filepath = os.path.join(
        os.sep, "home","gwiazale", "AR-VAE",
        "first_working_models","ar_vae_continue_training_ar-vae-v4_epoch940_decoder.pt"
    )

    if is_cpu:
        encoder.load_state_dict(
            torch.load(
                encoder_filepath,
                map_location=DEVICE
            )
        )
        decoder.load_state_dict(
            torch.load(
                decoder_filepath,
                map_location=DEVICE
            )
        )
    else:
        encoder.load_state_dict(torch.load(encoder_filepath))
        decoder.load_state_dict(torch.load(decoder_filepath))
    # device_first = device(f"cuda:{int(os.environ["LOCAL_RANK"])}")
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)
    # encoder.to(device_first)
    # decoder.to(device_first)
    # encoder= DDP(encoder, device_ids=[int(os.environ["LOCAL_RANK"])])
    # decoder= DDP(decoder, device_ids=[int(os.environ["LOCAL_RANK"])])

    data_manager = dataset_lib.AMPDataManager(
        DATA_DIR / 'unlabelled_positive.csv',
        DATA_DIR / 'unlabelled_negative.csv',
        min_len=MIN_LENGTH,
        max_len=MAX_LENGTH)

    amp_x, amp_y, attributes_input, _ = data_manager.get_merged_data()
    attributes = dataset_lib.normalize_attributes(attributes_input)    
    dataset = TensorDataset(amp_x, tensor(amp_y), attributes, attributes_input)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size

    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=params["batch_size"], shuffle=True)

    latent_codes, attributes, attr_list = m.compute_representations(eval_loader, encoder, params["reg_dim"], DEVICE)
    interp_metrics = m.compute_interpretability_metric(
        latent_codes, attributes, attr_list
    )
    ar_vae_metrics = {}
    ar_vae_metrics["Interpretability"] = interp_metrics
    ar_vae_metrics.update(m.compute_correlation_score(latent_codes, attributes))
    ar_vae_metrics.update(m.compute_modularity(latent_codes, attributes))
    ar_vae_metrics.update(m.compute_mig(latent_codes, attributes))
    ar_vae_metrics.update(m.compute_sap_score(latent_codes, attributes))
    interp_dict = ar_vae_metrics['Interpretability']
    print(f'interp_dict = {interp_dict}')
    attr_dims = [interp_dict[attr][0] for attr in attr_dict.keys()]
    non_attr_dims = [a for a in range(params['latent_dim']) if a not in attr_dims]
    for attr in interp_dict.keys():
        dim1 = interp_dict[attr][0]
        if attr == 'mean':
            continue
        plot_latent_surface(
            decoder,
            attr,
            dim1=dim1,
            dim2=non_attr_dims[-2],
            grid_res=0.05,
            z_dim = params["latent_dim"]
        )
        # plot_latent_heatmap(
        #     decoder=decoder,
        #     filename="latent_heatmap_length_dim3_15.png",
        #     dim1=dim1,
        #     dim2=non_attr_dims[-1],
        #     latent_dim=params["latent_dim"], # Z Twoich parametrów
        #     grid_size=100, # Większa siatka, gładszy obraz
        #     x_range=(-5, 5),
        #     y_range=(-5, 5),
        #     attribute_to_plot=attr
        # )

if __name__ == '__main__':
    set_seed()
    run()
