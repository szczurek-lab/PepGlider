# PepGlider
We present PepGlider, a continuous property regularization framework that enables direct control over their specific values. The method achieves structured latent space with superior disentanglement quality and displays smooth property gradients along regularized dimension. *In silico* experimental results demonstrate that PepGlider enables independent control of naturally correlated properties, and supports both *de novo* generation and targeted optimization of existing peptides. In application to antimicrobial peptide design, PepGlider  generated candidates with desired antibacterial activity profile and low toxicity profile. Unlike existing approaches, PepGlider provides precise control over continuous property distributions, while maintaining generation quality, thus offering a generalizable solution for peptide design.

## Packages requirements
All used packages and their versions are provided in `requirements.txt` file in repository. You can install them using:
```bash
pip install -r requirements.txt
```
## Model training
You can run a model training using `train.py` file. All default parameters are defined in `params_setting.py` file. Current configuration enables you to train PepGlider model with MIC E.coli, MIC S.aureus and Nontoxicity regularization. Feel free to change configuration to train your model!
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
        "epochs": 9100,
        "iwae_samples": 10,
        "model_name": os.getenv("CLEARML_PROJECT_NAME", 'ar-vae-v4'),
        "use_clearml": False,
        "task_name": os.getenv("CLEARML_TASK_NAME", "ar-vae 3 dims"),
        "device": "cuda",
        "deeper_eval_every": 20,
        "save_model_every": 20,
        "ar_vae_flg": True,
        "reg_dim": [3,4,5], 
        "gamma_schedule": (0.00001, 20, 8000),
        "gamma_multiplier": [1,1,1,1,1,1],
        "factor_multiplier": [0.1,0.1,0.1,0.6,0.6,0.6],
        "factor_schedule": (1,1,8000),
        'scale_factor_flg': False,
        'mic_flg': True,
        'toxicity_flg': True,
        'normalize_properties_flg':True,
        'signum_modification_of_dist_matrix_flg': False

## Model evaluation
All final numeric results, tables and figures were produced using two notebooks:
* `figures_notebook_1.ipynb`
* `figures_notebook_2.ipynb`

Feel free to duplicate them and analyse models on this repo! We leave model files to upload in `first_working_models` directory.