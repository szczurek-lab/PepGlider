import torch
import os
from torch import nn, cuda, backends, manual_seed, tensor
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.autograd import Variable
from model.model import EncoderRNN, DecoderRNN
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from pathlib import Path
from tqdm import tqdm
import data.dataset as dataset_lib
from model.constants import MIN_LENGTH, MAX_LENGTH
import ar_vae_metrics as m
from scipy import stats

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

def plot_latent_surface(decoder, attr_str, dim1=0, dim2=[1], grid_res=0.05, z_dim = 56):
    x1 = torch.arange(-5., 5., grid_res)
    x2 = torch.arange(-5., 5., grid_res)
    z1, z2 = torch.meshgrid([x1, x2])
    num_points = z1.size(0) * z1.size(1)
    for dim in dim2:
        if dim == 3:
            z = torch.randn(1, z_dim)
            z = z.repeat(num_points, 1)
            z[:, dim1] = z1.contiguous().view(1, -1)
            z[:, dim] = z2.contiguous().view(1, -1)
    #        print(f'z = {z}')
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
                current_batch_filtered_z = []
                current_batch_filtered_labels = []

                for j in range(mini_batch_size):
                    single_src_decoded = dataset_lib.decoded(src_decoded[j:j+1], "") 
                        
                    if single_src_decoded and single_src_decoded[0].strip(): 
                        labels = dataset_lib.calculate_physchem_test(single_src_decoded)
                        current_batch_filtered_z.append(z_batch[j:j+1])
                        current_batch_filtered_labels.append(labels) 
                    
                if current_batch_filtered_z: 
                    filtered_z_points.append(torch.cat(current_batch_filtered_z, 0))
                    filtered_attr_labels.append(torch.cat(current_batch_filtered_labels, 0))

            final_z_points = torch.cat(filtered_z_points, 0).cpu().numpy()
            final_attr_labels = torch.cat(filtered_attr_labels, 0).cpu().numpy()
            save_filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                f'latent_surface_{attr_str}_{dim}dim.png'
            )
            z = z.cpu().numpy()[:num_mini_batches*mini_batch_size, :]
            plot_dim(final_z_points, final_attr_labels[:, dim1], save_filename, dim1=dim1, dim2=dim)

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
        "batch_size": 512,
        "lr": 0.001,
        "train_size": None,
        "iwae_samples": 10,
        "model_name": os.getenv("CLEARML_PROJECT_NAME", 'ar-vae-v4'),
        "use_clearml": True,
        "task_name": os.getenv("CLEARML_TASK_NAME", "ar-vae 3 dims"),
        "device": "cuda",
        "deeper_eval_every": 20,
        "save_model_every": 20,
        "ar_vae_flg": False,
        "reg_dim": [0,1,2], # [length, charge, hydrophobicity_moment]
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
        "first_working_models","iwae_continue_training_ar-vae-v4_epoch880_encoder.pt"
    )
    decoder_filepath = os.path.join(
        os.sep, "home","gwiazale", "AR-VAE",
        "first_working_models","iwae_continue_training_ar-vae-v4_epoch880_decoder.pt"
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
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)

    data_manager = dataset_lib.AMPDataManager(
        DATA_DIR / 'unlabelled_positive.csv',
        DATA_DIR / 'unlabelled_negative.csv',
        min_len=MIN_LENGTH,
        max_len=MAX_LENGTH)

    amp_x, amp_y, attributes_input, _ = data_manager.get_merged_data()

    plt.figure(figsize=(10, 8))
    plt.scatter(
        x=attributes_input[:, 1],
        y=attributes_input[:, 2],
        s=10,
        alpha=0.6
    )
    plt.xlabel(f'Charge')
    plt.ylabel(f'Hydrophobicity moment')
    plt.title(f'Charge - Hydrophobicity correlation')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    spearman_corr, p_value = stats.spearmanr(attributes_input[:,1], attributes_input[:,2])
    if not np.isnan(spearman_corr):
        plt.text(
            0.05, # Pozycja X (lewa strona)
            0.95, # Pozycja Y (góra)
            f'Spearman correlation coefficient: {spearman_corr:.4f}',
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8, alpha=0.8)
        )

    # plt.show()
    plt.savefig('Charge - Hydrophobicity correlation.png')
    plt.close()
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
    attr_dims = [attr_dict[attr] for attr in attr_dict.keys()]
    non_attr_dims = [a for a in range(params['latent_dim']) if a not in attr_dims]
    for attr in attr_dict.keys():
        dim1 = attr_dict[attr]
        if attr == 'mean' or attr == 'Charge' or attr == 'Hydrophobicity moment':
            continue
        plot_latent_surface(
            decoder,
            attr,
            dim1=dim1,
            dim2=non_attr_dims,
            grid_res=0.05,
            z_dim = params["latent_dim"]
        )

if __name__ == '__main__':
    set_seed()
    run()
