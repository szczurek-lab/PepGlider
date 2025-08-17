import torch
import os
from torch import optim, nn, logsumexp, cuda, save, backends, manual_seed, LongTensor, zeros_like, ones_like, tensor, cat
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader, random_split
torch.autograd.set_detect_anomaly(True)
from model.model import EncoderRNN, DecoderRNN
import numpy as np
import clearml
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
import datetime
import csv
from training_functions import set_seed, get_model_arch_hash, save_model, run_epoch_iwae

def run():
    global ROOT_DIR 
    ROOT_DIR = Path(__file__).parent
    DATA_DIR = ROOT_DIR / "data"
    global MODELS_DIR 
    MODELS_DIR = ROOT_DIR / "first_working_models"
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
        "epochs": 9100,
        "iwae_samples": 10,
        "model_name": os.getenv("CLEARML_PROJECT_NAME", 'ar-vae-v4'),
        "use_clearml": False,
        "task_name": os.getenv("CLEARML_TASK_NAME", "ar-vae 3 dims"),
        "device": "cuda",
        "deeper_eval_every": 20,
        "save_model_every": 100,
        "ar_vae_flg": True,
        "reg_dim": [0,1,2], # [length, charge, hydrophobicity_moment]
        "gamma_schedule": (0.00001, 20, 8000),
        "gamma_multiplier": [1,1,1],
        "factor_schedule": (0.1,10,8000)
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
    DEVICE = torch.device(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')
    is_cpu = False if torch.cuda.is_available() else True
    encoder_filepath = os.path.join(
        os.sep, "net","tscratch","people","plggwiazale", "AR-VAE",
        "hyperparams_tuning_factor_0.1_ar-vae_epoch900_encoder.pt"
    )
    decoder_filepath = os.path.join(
        os.sep, "net","tscratch","people","plggwiazale", "AR-VAE",
        "hyperparams_tuning_factor_0.1_ar-vae_epoch900_decoder.pt"
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
    attributes = dataset_lib.normalize_attributes(attributes_input)

    optimizer = Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()),
        lr=params["lr"],
        betas=(0.9, 0.999),
    )
    if params["use_clearml"]:
        task = clearml.Task.init(
            project_name=os.getenv("CLEARML_PROJECT_NAME", 'ar-vae-v4'), task_name=os.getenv("CLEARML_TASK_NAME", "ar-vae 3 dims")
        )
        task.set_parameters(params)
        logger = task.logger
        train_log_file = None
        eval_log_file = None
    else:
        logger = None
        train_log_file = f'training_log_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'.replace(' ', '_')
        with open(ROOT_DIR / train_log_file, 'a', newline='') as csvfile:
            header = ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss","KL Div","KL Div * Beta","Reg Loss", "Reg Loss * Gamma", "Delta"] if params["ar_vae_flg"] else ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss", "KL Div", "KL Div * Beta"]
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)
        eval_log_file = f'validation_log_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'.replace(' ', '_')
        with open(ROOT_DIR / eval_log_file, 'a', newline='') as csvfile:
            if params["ar_vae_flg"]:
                header = ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss","KL Div","KL Div * Beta","Reg Loss", "Reg Loss * Gamma", "Delta",
                          "Length Pred Acc", "Length Loss [mae]", "Token Pre Acc", "Amino Acc", "Empty Acc", 
                          "MAE length", "MAE charge", "MAE hydrophobicity moment", 
                          "Interpretability - length", "Interpretability - charge", "Interpretability - hydrophobicity moment",
                          "Corr_score - length", "Corr_score - charge", "Corr_score - hydrophobicity moment",
                          "Modularity - length", "Modularity - charge", "Modularity - hydrophobicity moment",
                          "MIG - length", "MIG - charge", "MIG - hydrophobicity moment",
                          "SAP_score - length", "SAP_score - charge", "SAP_score - hydrophobicity moment"
                          ] 
            else:
                header = ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss","KL Div","KL Div * Beta",
                          "Length Pred Acc", "Length Loss [mae]", "Token Pre Acc", "Amino Acc", "Empty Acc", 
                          "MAE length", "MAE charge", "MAE hydrophobicity moment", 
                          "Interpretability - length", "Interpretability - charge", "Interpretability - hydrophobicity moment",
                          "Corr_score - length", "Corr_score - charge", "Corr_score - hydrophobicity moment",
                          "Modularity - length", "Modularity - charge", "Modularity - hydrophobicity moment",
                          "MIG - length", "MIG - charge", "MIG - hydrophobicity moment",
                          "SAP_score - length", "SAP_score - charge", "SAP_score - hydrophobicity moment"
                          ] 
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)

    dataset = TensorDataset(amp_x, tensor(amp_y), attributes, attributes_input)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=params["batch_size"], shuffle=True)

    for epoch in tqdm(range(params["epochs"])):
        epoch = epoch + (10000-params["epochs"])
        eval_mode = "deep" if epoch % params["deeper_eval_every"] == 0 else "fast"
        beta_0, beta_1, t_1 = params["kl_beta_schedule"]
        kl_beta = min(beta_0 + (beta_1 - beta_0) / t_1 * epoch, beta_1)
        gamma_0, gamma_1, t_1 = params["gamma_schedule"]
        if epoch < 1000:
            gamma = min(gamma_0 + (gamma_1 - gamma_0) / t_1 * epoch, 0.0)
        else:
            gamma = min(gamma_0 + (gamma_1 - gamma_0) / t_1 * epoch, gamma_1)
        delta_0, delta_1, t_1  = params['factor_schedule']
        delta = min(max(delta_0 + (delta_1 - delta_0) / t_1 * epoch, delta_0), delta_1)
        run_epoch_iwae(
                mode="train",
                encoder=encoder,
                decoder=decoder,
                dataloader=train_loader,
                device=DEVICE,
                logger=logger,
                train_log_file = train_log_file,
                eval_log_file = eval_log_file,
                epoch=epoch,
                optimizer=optimizer,
                kl_beta=kl_beta,
                eval_mode=eval_mode,
                iwae_samples=params["iwae_samples"],
                ar_vae_flg=params["ar_vae_flg"],
                reg_dim=params["reg_dim"],
                gamma = gamma,
                gamma_multiplier = params['gamma_multiplier'],
                factor = delta
        )
        if eval_mode == "deep":
            loss = run_epoch_iwae(
                    mode="test",
                    encoder=encoder,
                    decoder=decoder,
                    dataloader=eval_loader,
                    device=DEVICE,
                    logger=logger,
                    train_log_file = train_log_file,
                    eval_log_file = eval_log_file,
                    epoch=epoch,
                    optimizer=None,
                    kl_beta=kl_beta,
                    eval_mode=eval_mode,
                    iwae_samples=params["iwae_samples"],
                    ar_vae_flg=params["ar_vae_flg"],
                    reg_dim=params["reg_dim"],
                    gamma=gamma,
                    gamma_multiplier = params['gamma_multiplier'],
                    factor = delta
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
    # eval_model() -> probably to do

if __name__ == '__main__':
    set_seed()
    run()
