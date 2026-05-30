import clearml
import os
import datetime
import csv
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_params(root_dir):
    parser = argparse.ArgumentParser(description="AR-VAE training parameters")

    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--layer_norm", type=str2bool, default=True)
    parser.add_argument("--latent_dim", type=int, default=56)
    parser.add_argument("--encoding", type=str, default="add")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--kl_beta_schedule", type=float, nargs=3, default=[0.00001, 0.1, 8000])
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--iwae_samples", type=int, default=10)
    parser.add_argument("--model_name", type=str, default=os.getenv("CLEARML_PROJECT_NAME", 'ar-vae-v4'))
    parser.add_argument("--task_name", type=str, default=os.getenv("CLEARML_TASK_NAME", "ar-vae 3 dims"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--deeper_eval_every", type=int, default=20)
    parser.add_argument("--save_model_every", type=int, default=100)
    parser.add_argument("--ar_vae_flg", type=str2bool, default=True)
    parser.add_argument("--reg_dim", type=int, nargs='+', default=[3, 4, 5])
    parser.add_argument("--gamma_schedule", type=float, nargs=3, default=[0.00001, 20, 8000])
    parser.add_argument("--gamma_multiplier", type=float, nargs='+', default=[1, 1, 1, 1, 1, 1])
    parser.add_argument("--factor_multiplier", type=float, nargs='+', default=[0.1, 0.1, 0.1, 0.6, 0.6, 0.6])
    parser.add_argument("--factor_schedule", type=float, nargs=3, default=[1, 1, 8000])
    parser.add_argument("--scale_factor_flg", type=str2bool, default=True)
    parser.add_argument("--mic_flg", type=str2bool, default=True)
    parser.add_argument("--toxicity_flg", type=str2bool, default=True)
    parser.add_argument("--normalize_properties_flg", type=str2bool, default=True)
    parser.add_argument("--signum_modification_of_dist_matrix_flg", type=str2bool, default=True)

    args, _ = parser.parse_known_args()

    params = vars(args)
    params["kl_beta_schedule"] = tuple(params["kl_beta_schedule"])
    params["gamma_schedule"] = tuple(params["gamma_schedule"])
    params["factor_schedule"] = tuple(params["factor_schedule"])

    # Build attribute names based on reg_dim (matching ar_vae_metrics.extract_relevant_attributes)
    all_attr_names = {0: 'length', 1: 'charge', 2: 'hydrophobicity', 3: 'mic_e_coli', 4: 'mic_s_aureus', 5: 'nontoxicity'}
    attr_names = [all_attr_names[i] for i in params["reg_dim"]]

    # Build training log header
    train_log_file = f'training_log_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'.replace(' ', '_')
    with open(root_dir / train_log_file, 'a', newline='') as csvfile:
        header = ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss", "KL Div", "KL Div * Beta"]
        if params["ar_vae_flg"]:
            header += ["Reg Loss", "Reg Loss * Gamma", "Delta"]
            if params["scale_factor_flg"]:
                header += ["Scale factor"]
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)

    # Build eval log header
    eval_log_file = f'validation_log_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'.replace(' ', '_')
    with open(root_dir / eval_log_file, 'a', newline='') as csvfile:
        header = ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss", "KL Div", "KL Div * Beta"]
        if params["ar_vae_flg"]:
            header += ["Reg Loss", "Reg Loss * Gamma", "Delta"]
            if params["scale_factor_flg"]:
                header += ["Scale factor"]
        header += ["Length Pred Acc", "Length Loss [mae]", "Token Pre Acc", "Amino Acc", "Empty Acc"]
        header += ["MAE length", "MAE charge", "MAE hydrophobicity moment"]
        for metric in ["Interpretability", "Corr_score", "Modularity", "MIG", "SAP_score"]:
            for attr in attr_names + ["mean"]:
                header.append(f"{metric} - {attr}")
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)

    return params, train_log_file, eval_log_file
