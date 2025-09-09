import clearml
import os
import datetime
import csv

def set_params(root_dir):
    params = {
        "num_heads": 4,
        "num_layers": 6,
        "layer_norm": True,
        "latent_dim": 56,
        "encoding": "add",
        "dropout": 0.1,
        "batch_size": 512,
        "lr": 0.001,
        "kl_beta_schedule": (0.00001, 0.1, 8000),
        "train_size": None,
        "epochs": 10000,
        "iwae_samples": 10,
        "model_name": os.getenv("CLEARML_PROJECT_NAME", 'ar-vae-v4'),
        "use_clearml": False,
        "task_name": os.getenv("CLEARML_TASK_NAME", "ar-vae 3 dims"),
        "device": "cuda",
        "deeper_eval_every": 20,
        "save_model_every": 100,
        "ar_vae_flg": True,
        "reg_dim": [0,1,2,3,4], # [length, charge, hydrophobicity_moment]
        "gamma_schedule": (0.00001, 20, 8000),
        "gamma_multiplier": [1,1,1,1,1],
        "factor_multiplier": [0.1,0.1,0.1,0.6,0.6],
        "factor_schedule": (1,1,8000),
        'scale_factor_flg': False,
        'mic_flg': True
    }

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
        with open(root_dir / train_log_file, 'a', newline='') as csvfile:
            if params["ar_vae_flg"]:
                header = ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss","KL Div","KL Div * Beta","Reg Loss", "Reg Loss * Gamma", "Delta",] 
                if params["scale_factor_flg"]:
                        header = ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss","KL Div","KL Div * Beta","Reg Loss", "Reg Loss * Gamma", "Delta","Scale factor",]
            else:
                header = ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss", "KL Div", "KL Div * Beta"]
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)
        eval_log_file = f'validation_log_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'.replace(' ', '_')
        with open(root_dir / eval_log_file, 'a', newline='') as csvfile:
            if params["ar_vae_flg"]:
                header = ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss","KL Div","KL Div * Beta","Reg Loss", "Reg Loss * Gamma", "Delta", 
                          "Length Pred Acc", "Length Loss [mae]", "Token Pre Acc", "Amino Acc", "Empty Acc", 
                          "MAE length", "MAE charge", "MAE hydrophobicity moment", 
                          "Interpretability - length", "Interpretability - charge", "Interpretability - hydrophobicity","Interpretability - mean",
                          "Corr_score - length", "Corr_score - charge", "Corr_score - hydrophobicity","Corr_score - mean",
                          "Modularity - length", "Modularity - charge", "Modularity - hydrophobicity","Modularity - mean",
                          "MIG - length", "MIG - charge", "MIG - hydrophobicity moment","MIG - mean",
                          "SAP_score - length", "SAP_score - charge", "SAP_score - hydrophobicity","SAP_score - mean"
                          ] 
                if params["scale_factor_flg"]:
                    header = ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss","KL Div","KL Div * Beta","Reg Loss", "Reg Loss * Gamma", "Delta", "Scale factor",
                            "Length Pred Acc", "Length Loss [mae]", "Token Pre Acc", "Amino Acc", "Empty Acc", 
                            "MAE length", "MAE charge", "MAE hydrophobicity moment", 
                            "Interpretability - length", "Interpretability - charge", "Interpretability - hydrophobicity","Interpretability - mean",
                            "Corr_score - length", "Corr_score - charge", "Corr_score - hydrophobicity","Corr_score - mean",
                            "Modularity - length", "Modularity - charge", "Modularity - hydrophobicity","Modularity - mean",
                            "MIG - length", "MIG - charge", "MIG - hydrophobicity moment","MIG - mean",
                            "SAP_score - length", "SAP_score - charge", "SAP_score - hydrophobicity","SAP_score - mean"
                        ] 
                if params['mic_flg']:
                    header = ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss","KL Div","KL Div * Beta","Reg Loss", "Reg Loss * Gamma", "Delta", 
                            "Length Pred Acc", "Length Loss [mae]", "Token Pre Acc", "Amino Acc", "Empty Acc", 
                            "MAE length", "MAE charge", "MAE hydrophobicity moment", 
                            "Interpretability - length", "Interpretability - charge", "Interpretability - hydrophobicity", "Interpretability - mic_e_cola", "Interpretability - mic_s_aureus","Interpretability - mean",
                            "Corr_score - length", "Corr_score - charge", "Corr_score - hydrophobicity", "Corr_score - mic_e_cola", "Corr_score - mic_s_aureus","Corr_score - mean",
                            "Modularity - length", "Modularity - charge", "Modularity - hydrophobicity", "Modularity - mic_e_cola", "Modularity - mic_s_aureus","Modularity - mean",
                            "MIG - length", "MIG - charge", "MIG - hydrophobicity moment", "MIG - mic_e_cola", "MIG - mic_s_aureus","MIG - mean",
                            "SAP_score - length", "SAP_score - charge", "SAP_score - hydrophobicity", "SAP_score - mic_e_cola", "SAP_score - mic_s_aureus","SAP_score - mean"
                            ] 
            else:
                header = ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss","KL Div","KL Div * Beta",
                          "Length Pred Acc", "Length Loss [mae]", "Token Pre Acc", "Amino Acc", "Empty Acc", 
                          "MAE length", "MAE charge", "MAE hydrophobicity moment", 
                          "Interpretability - length", "Interpretability - charge", "Interpretability - hydrophobicity","Interpretability - mean",
                          "Corr_score - length", "Corr_score - charge", "Corr_score - hydrophobicity","Corr_score - mean",
                          "Modularity - length", "Modularity - charge", "Modularity - hydrophobicity","Modularity - mean",
                          "MIG - length", "MIG - charge", "MIG - hydrophobicity moment","MIG - mean",
                          "SAP_score - length", "SAP_score - charge", "SAP_score - hydrophobicity","SAP_score - mean"
                          ] 
                if params['mic_flg']:
                    header = ["Mode", "Epoch", "Total Loss", "Cross Entropy Loss","KL Div","KL Div * Beta",
                            "Length Pred Acc", "Length Loss [mae]", "Token Pre Acc", "Amino Acc", "Empty Acc", 
                            "MAE length", "MAE charge", "MAE hydrophobicity moment", 
                            "Interpretability - length", "Interpretability - charge", "Interpretability - hydrophobicity", "Interpretability - mic_e_cola", "Interpretability - mic_s_aureus","Interpretability - mean",
                            "Corr_score - length", "Corr_score - charge", "Corr_score - hydrophobicity", "Corr_score - mic_e_cola", "Corr_score - mic_s_aureus","Corr_score - mean",
                            "Modularity - length", "Modularity - charge", "Modularity - hydrophobicity", "Modularity - mic_e_cola", "Modularity - mic_s_aureus","Modularity - mean",
                            "MIG - length", "MIG - charge", "MIG - hydrophobicity moment", "MIG - mic_e_cola", "MIG - mic_s_aureus","MIG - mean",
                            "SAP_score - length", "SAP_score - charge", "SAP_score - hydrophobicity", "SAP_score - mic_e_cola", "SAP_score - mic_s_aureus","SAP_score - mean"
                            ] 
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)
    return params, train_log_file, eval_log_file, logger