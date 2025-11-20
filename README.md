# PepGlider
We present PepGlider, a continuous property regularization framework that enables direct control over their specific values. The method achieves structured latent space with superior disentanglement quality and displays smooth property gradients along regularized dimension. *In silico* experimental results demonstrate that PepGlider enables independent control of naturally correlated properties, and supports both *de novo* generation and targeted optimization of existing peptides. In application to antimicrobial peptide design, PepGlider  generated candidates with desired antibacterial activity profile and low toxicity profile. Unlike existing approaches, PepGlider provides precise control over continuous property distributions, while maintaining generation quality, thus offering a generalizable solution for peptide design.

## Packages requirements
All used packages and their versions are provided in `requirements.txt` file in repository. You can install them using:
```bash
pip install -r requirements.txt
```
## Model training
You can run a model training using `train.py` file. All default parameters are defined in `params_setting.py` file. Current configuration enables you to train PepGlider model with MIC E.coli, MIC S.aureus and Nontoxicity regularization. Feel free to change configuration to train your model! Below there is a table with parameters description:

| Parameter | Default value | Description |
| :--- | :--- | :--- |
| `num_heads` | `4` | The number of **attention heads** in the transformer layers. |
| `num_layers` | `6` | The number of transformer layers in the model's encoder/decoder. |
| `layer_norm` | `True` | Flag specifying whether **Layer Normalization** should be used. |
| `latent_dim` | `56` | The dimension of the **latent space** of the VAE model. |
| `encoding` | `"add"` | The method of combining or add input embedding vectors with positional ones in the architecture. |
| `dropout` | `0.1` | The **dropout** coefficient used for regularization during training. |
| `batch_size` | `512` | The **batch size** used during training the model. |
| `lr` | `0.001` | The **learning rate** for the optimizer. |
| `kl_beta_schedule` | `(0.00001, 0.1, 8000)` | The weight growth schedule for the **KL-divergence** term (e.g., start, target, steps to reach the target). |
| `train_size` | `None` | Optional limit on the size of the training set. |
| `epochs` | `10000` | The number of training epochs. |
| `iwae_samples` | `10` | The number of decoder training iterations used in the **IWAE** (Implicitly Weighted Autoencoder) model. |
| `model_name` | `os.getenv(...)` | The name of the model for saving files; retrieved from the `CLEARML_PROJECT_NAME` environment variable. |
| `use_clearml` | `False` | Flag specifying whether to use the **ClearML** platform for experiment tracking. |
| `task_name` | `os.getenv(...)` | The task name in ClearML; retrieved from the `CLEARML_TASK_NAME` environment variable. |
| `device` | `"cuda"` | The device for computation (`cuda` for GPU or `cpu`). |
| `deeper_eval_every` | `20` | How often (in epochs) to perform a more detailed evaluation. |
| `save_model_every` | `100` | How often (in epochs) to save the model state. |
| `ar_vae_flg` | `True` | Flag enabling **AR-VAE** or **PepGlider** model trainig. |
| `reg_dim` | `[3, 4, 5]` | The dimensions (indices) in the latent space that should be subjected to regularization. |
| `gamma_schedule` | `(0.00001, 20, 8000)` | The weight growth schedule for $\gamma$ hyperparameter. |
| `gamma_multiplier` | `[1, 1, 1, 1, 1, 1]` | Multipliers applied to $\gamma$ for individual dimensions. |
| `factor_multiplier` | `[0.1, 0.1, 0.1, 0.6, 0.6, 0.6]` | Multipliers for $\factor$ hyperparameter. |
| `factor_schedule` | `(1, 1, 8000)` | The weight growth schedule for factor regularization. |
| `scale_factor_flg` | `False` | Flag specifying whether to scale the $\factor$ hyperparameter. |
| `mic_flg` | `True` | Flag enabling add **MIC** values to the train dataset. |
| `toxicity_flg` | `True` | Flag enabling generate **non-toxicity** scores for the train dataset. |
| `normalize_properties_flg` | `True` | Flag specifying whether to normalize the input data **properties**. |
| `signum_modification_of_dist_matrix_flg` | `False` | Flag enabling modification of the distance matrix with the `signum` function. |

## Model evaluation
All final numeric results, tables and figures were produced using two notebooks:
* `figures_notebook_1.ipynb`
* `figures_notebook_2.ipynb`

Feel free to duplicate them and analyse models on this repo! We leave model files to upload in `first_working_models` directory.