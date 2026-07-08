import random

def read_fasta_file(path: str) -> list[str]:
    sequences: list[str] = []
    current_seq: list[str] = []

    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            if line.startswith(">"):
                if current_seq:
                    sequence = "".join(current_seq)
                    if sequence:
                        sequences.append(sequence)
                    current_seq = []
            else:
                current_seq.append(line)

        # Add the last sequence if present
        if current_seq:
            sequence = "".join(current_seq)
            if sequence:
                sequences.append(sequence)

    return sequences

def write_to_fasta_file(
    sequences: list[str],
    path: str,
    headers: list[str] | None = None,
):
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            header = headers[i] if headers else f">sequence_{i + 1}"
            f.write(f"{header}\n")
            f.write(f"{seq}\n")

def random_subset(sequences: list[str], n_samples: int, seed: int = 42) -> list[str]:
    """
    Select a random subset of unique sequences with deterministic behavior.

    Args:
        sequences: The list of input sequences to sample from.
        n_samples: The number of sequences to sample.
        seed: The random seed for reproducibility.

    Returns:
        A list of `n_samples` randomly chosen, unique sequences.

    Raises:
        ValueError: If `n_samples` exceeds the number of available sequences.
    """
    if n_samples > len(sequences):
        raise ValueError(f"Cannot sample {n_samples} sequences from a list of length {len(sequences)}.")

    rng = random.Random(seed)
    return rng.sample(sequences, n_samples)

# def hobbit(fitted_transformers, encoder_name, decoder_name, data_loader, params, attr_dict, shift_value = 0.2):
#     seed = 42
#     np.random.seed(seed)
#     random.seed(seed)
#     manual_seed(seed)
#     cuda.manual_seed(seed)
#     backends.cudnn.deterministic = True
#     backends.cudnn.benchmark = False
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     DEVICE = torch.device('cpu')
#     generated_analog = {}
#     tmp_dict = {}
#     normalized_tmp_dict = {}
#     hobbit_path = {shift_value: [0,1,2],
#                    -shift_value: [0,1,2]}
#     encoder = EncoderRNN(
#         params["num_heads"],
#         params["num_layers"],
#         params["latent_dim"],
#         params["encoding"],
#         params["dropout"],
#         params["layer_norm"],
#     )
#     decoder = DecoderRNN(
#         params["num_heads"],
#         params["num_layers"],
#         params["latent_dim"],
#         params["encoding"],
#         params["dropout"],
#         params["layer_norm"],
#     )
#     encoder.load_state_dict(torch.load(f"./first_working_models/{encoder_name}", map_location=DEVICE))
#     encoder = encoder.to(DEVICE)
#     decoder.load_state_dict(torch.load(f"./first_working_models/{decoder_name}", map_location=DEVICE))
#     decoder = decoder.to(DEVICE)
#     encoder = encoder.eval()
#     decoder = decoder.eval()         

#     attr_name = [k for k in attr_dict.keys()]
#     hobbit_results = []
#     hobbit_normalized_results = []
#     hobbit_normalized_all_results = []
#     hobbit_all_results = []
#     model = encoder_name.split("_ar-vae")[0]

#     z_sample = torch.randn(10000, 56).to(DEVICE)
#     z_sample[:, :3] = 0.0
#     outputs = decoder(z_sample)
#     src = outputs.permute(1, 2, 0) 
#     seq = src.argmax(dim=1)
#     generated_sequences = data_describe.decoded(seq, "")
#     peptides = seq.permute(1, 0)
#     generated_sequences = [seq.strip().rstrip("0") for seq in generated_sequences]
#     generated_sequences = [seq for seq in generated_sequences if '0' not in seq]
#     cleaned_sequences = [seq for seq in generated_sequences if seq]
#     similarity_scores = calculate_pairwise_similarity(cleaned_sequences, cleaned_sequences)
#     scores_only = [score for s1, s2, score in similarity_scores]
#     mean_score = np.mean(scores_only)
#     base_sequences = cleaned_sequences
#     generated_analog[model+"_"+str(0)] = peptides
#     if 'Length' in attr_name:
#         if len(peptides) == 0:
#             # unconstrained_dfs_analog_dict_combo['Length'].append(f'nan ± nan')
#             curr_len = f'nan ± nan'
#             normalized_len = f'nan ± nan'
#         else:
#             attr = data_describe.calculate_length_test(cleaned_sequences)
#             length = np.array(attr).reshape(-1, 1)
#             # print(np.array(attr).reshape(-1, 1).shape)
#             # transformed_length_np = fitted_transformers[0].transform(np.array(attr).reshape(-1, 1))
#             data = np.array(attr).reshape(-1, 1)
#             data_min = np.min(data)
#             data_max = np.max(data)
#             transformed_length_np = (data - data_min) / (data_max - data_min)
#             curr_len = f'{np.mean(attr, dtype=np.float64):.2f} ± {np.std(attr, dtype=np.float64):.2f}'
#             normalized_len = f'{np.mean(transformed_length_np):.5f} ± {np.std(transformed_length_np):.5f}'
#             # unconstrained_dfs_analog_dict_combo['Length'].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
#     if 'Charge' in attr_name:
#         if len(peptides) == 0:
#             # unconstrained_dfs_analog_dict_combo['Charge'].append(f'nan ± nan')
#             curr_charge = f'nan ± nan'
#             normalized_charge = f'nan ± nan'
#         else:
#             attr = data_describe.calculate_charge(cleaned_sequences)
#             charge = np.array(attr).reshape(-1, 1)
#             # transformed_charge_np = fitted_transformers[1].transform(np.array(attr).reshape(-1, 1))
#             data = np.array(attr).reshape(-1, 1)
#             data_min = np.min(data)
#             data_max = np.max(data)
#             transformed_charge_np = (data - data_min) / (data_max - data_min)
#             curr_charge = f'{np.mean(attr):.2f} ± {np.std(attr):.2f}'
#             normalized_charge = f'{np.mean(transformed_charge_np):.5f} ± {np.std(transformed_charge_np):.5f}'
#             # unconstrained_dfs_analog_dict_combo['Charge'].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
#     if 'Hydrophobicity' in attr_name:
#         if len(peptides) == 0:
#             # unconstrained_dfs_analog_dict_combo['Hydrophobicity'].append(f'nan ± nan')
#             curr_hydr = f'nan ± nan'
#             normalized_hydr = f'nan ± nan'
#         else:
#             attr = data_describe.calculate_hydrophobicity(cleaned_sequences)
#             hydr = np.array(attr).reshape(-1, 1)
#             # transformed_hydrophobicity_np = fitted_transformers[2].transform(np.array(attr).reshape(-1, 1))
#             data = np.array(attr).reshape(-1, 1)
#             data_min = np.min(data)
#             data_max = np.max(data)
#             transformed_hydrophobicity_np = (data - data_min) / (data_max - data_min)
#             curr_hydr = f'{np.mean(attr):.2f} ± {np.std(attr):.2f}'
#             normalized_hydr = f'{np.mean(transformed_hydrophobicity_np):.5f} ± {np.std(transformed_hydrophobicity_np):.5f}'
#             # unconstrained_dfs_analog_dict_combo['Hydrophobicity'].append(f'{np.mean(attr):.2f} ± {np.std(attr):.2f}')
#     hobbit_results.append([0, curr_len, curr_charge, curr_hydr, mean_score])
#     hobbit_normalized_results.append([0, normalized_len, normalized_charge, normalized_hydr, mean_score])
#     hobbit_normalized_all_results.append([0, transformed_length_np, transformed_charge_np, transformed_hydrophobicity_np, scores_only])
#     hobbit_all_results.append([0, length, charge, hydr, scores_only])
#     for shift, dims in hobbit_path.items():    
#         for dim in dims:
#             x = dataset_lib.pad(data_describe.to_one_hot(cleaned_sequences)).reshape(25, -1)
#             x = x.int()
#             mu = encoder.encode(x)
#             mu, std = encoder(peptides)
#             mod_mu = mu.clone().detach()
#             mod_mu[:, dim] = mod_mu[:, dim] + shift
#             outputs = decoder(mod_mu)
#             src = outputs.permute(1, 2, 0) 
#             seq = src.argmax(dim=1)
#             modified_sequences = data_describe.decoded(seq, "")
#             peptides = seq.permute(1, 0)
#             # save_sequences(modified_sequences, f"{model}_modified_{attr_name}_{shift_value}.csv")
    
#             modified_sequences = [seq.strip().rstrip("0") for seq in modified_sequences]
#             modified_sequences = [seq for seq in modified_sequences if '0' not in seq]
#             cleaned_sequences = [seq for seq in modified_sequences if seq]
#             generated_analog[model+'_'+str(dim)+"_"+str(shift)] = cleaned_sequences
#             similarity_scores = calculate_pairwise_similarity(base_sequences, cleaned_sequences)
#             scores_only = [score for s1, s2, score in similarity_scores]
#             mean_score = np.mean(scores_only)
#             if 'Length' in attr_name:
#                 if len(cleaned_sequences) == 0:
#                     # unconstrained_dfs_analog_dict_combo['Length'].append(f'nan ± nan')
#                     curr_len = f'nan ± nan'
#                     normalized_len = f'nan ± nan'
#                 else:
#                     attr = data_describe.calculate_length_test(cleaned_sequences)
#                     length = np.array(attr).reshape(-1, 1)
#                     # transformed_length_np = fitted_transformers[0].transform(np.array(attr).reshape(-1, 1))
#                     data = np.array(attr).reshape(-1, 1)
#                     data_min = np.min(data)
#                     data_max = np.max(data)
#                     transformed_length_np = (data - data_min) / (data_max - data_min)
#                     curr_len = f'{np.mean(attr):.2f} ± {np.std(attr):.2f}'
#                     normalized_len = f'{np.mean(transformed_length_np):.5f} ± {np.std(transformed_length_np):.5f}'
#             if 'Charge' in attr_name:
#                 if len(cleaned_sequences) == 0:
#                     # unconstrained_dfs_analog_dict_combo['Charge'].append(f'nan ± nan')
#                     curr_charge = f'nan ± nan'
#                     normalized_charge = f'nan ± nan'
#                 else:
#                     attr = data_describe.calculate_charge(cleaned_sequences)
#                     charge = np.array(attr).reshape(-1, 1)
#                     # transformed_charge_np = fitted_transformers[1].transform(np.array(attr).reshape(-1, 1))
#                     data = np.array(attr).reshape(-1, 1)
#                     data_min = np.min(data)
#                     data_max = np.max(data)
#                     transformed_charge_np = (data - data_min) / (data_max - data_min)
#                     curr_charge = f'{np.mean(attr):.2f} ± {np.std(attr):.2f}'
#                     normalized_charge = f'{np.mean(transformed_charge_np):.5f} ± {np.std(transformed_charge_np):.5f}'
#             if 'Hydrophobicity' in attr_name:
#                 if len(cleaned_sequences) == 0:
#                     # unconstrained_dfs_analog_dict_combo['Hydrophobicity'].append(f'nan ± nan')
#                     curr_hydr = f'nan ± nan'
#                     normalized_hydr = f'nan ± nan'
#                 else:
#                     attr = data_describe.calculate_hydrophobicity(cleaned_sequences)
#                     hydr = np.array(attr).reshape(-1, 1)
#                     # transformed_hydrophobicity_np = fitted_transformers[2].transform(np.array(attr).reshape(-1, 1))
#                     data = np.array(attr).reshape(-1, 1)
#                     data_min = np.min(data)
#                     data_max = np.max(data)
#                     transformed_hydrophobicity_np = (data - data_min) / (data_max - data_min)
#                     curr_hydr = f'{np.mean(attr):.2f} ± {np.std(attr):.2f}'
#                     normalized_hydr = f'{np.mean(transformed_hydrophobicity_np):.5f} ± {np.std(transformed_hydrophobicity_np):.5f}'
#             hobbit_results.append([shift, curr_len, curr_charge, curr_hydr, mean_score])
#             hobbit_normalized_results.append([shift, normalized_len, normalized_charge, normalized_hydr, mean_score])
#             hobbit_normalized_all_results.append([shift, transformed_length_np, transformed_charge_np, transformed_hydrophobicity_np, scores_only])
#             hobbit_all_results.append([0, length, charge, hydr, scores_only])
#     tmp_dict[str(attr_name)] = pd.DataFrame(hobbit_results)
#     normalized_tmp_dict[str(attr_name)] = pd.DataFrame(hobbit_normalized_results)
#     all_data = []
#     for i, (shift, transformed_length, transformed_charge, transformed_hydr, mean_score) in enumerate(hobbit_normalized_all_results):
#         num_rows = transformed_length.shape[0]
    
#         data_block = {
#             'step': np.repeat('p'+str(i), num_rows),
#             'shift': np.repeat(shift, num_rows),  
#             'length': transformed_length.flatten(), 
#             'charge': transformed_charge.flatten(),
#             'hydrophobicity': transformed_hydr.flatten(),
#             'similarity': np.array(mean_score)
#         }
#         all_data.append(pd.DataFrame(data_block))
#     final_df = pd.concat(all_data, ignore_index=True)
#     df_normalized_melted = final_df.melt(id_vars=['step'],
#                               value_vars=['length', 'charge', 'hydrophobicity', 'similarity'],
#                               var_name='Metric',
#                               value_name='Value')
#     all_data = []
#     for i, (shift, length, charge, hydr, mean_score) in enumerate(hobbit_all_results):
#         num_rows = length.shape[0]
    
#         data_block = {
#             'step': np.repeat('p'+str(i), num_rows),
#             'shift': np.repeat(shift, num_rows),  
#             'length': length.flatten(), 
#             'charge': charge.flatten(),
#             'hydrophobicity': hydr.flatten(),
#             'similarity': np.array(mean_score)
#         }
#         all_data.append(pd.DataFrame(data_block))
#     final_df = pd.concat(all_data, ignore_index=True)
#     df_melted = final_df.melt(id_vars=['step'],
#                               value_vars=['length', 'charge', 'hydrophobicity', 'similarity'],
#                               var_name='Metric',
#                               value_name='Value')
#     return tmp_dict, normalized_tmp_dict, df_melted, df_normalized_melted, generated_analog  