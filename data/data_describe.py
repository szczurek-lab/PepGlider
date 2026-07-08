import numpy as np
# import modlamp.analysis
import pandas as pd
from typing import Tuple, Dict, List
import torch
from modlamp import analysis
from sklearn.preprocessing import QuantileTransformer

def from_one_hot(encoded_seqs):
    return encoded_seqs.argmax(dim=-1)

def to_one_hot(x):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    classes = range(1, 21)
    aa_encoding = dict(zip(alphabet, classes))
    return [[aa_encoding[aa] for aa in seq] for seq in x]

def decoded(encoded_seqs, mode):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    classes = range(1, 21)
    aa_encoding = dict(zip(classes, alphabet))
    decoded_seqs = []
    for i, seq in enumerate(encoded_seqs):
        decoded_seq = ''.join(mode if idx == 0 else str(aa_encoding[idx.item()]) for idx in seq)
        decoded_seqs.append(decoded_seq)
    return decoded_seqs

def pad(x: List[List[float]], max_length: int = 25) -> torch.Tensor:
    sequences = [torch.tensor(seq[:max_length]) for seq in x]
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0.0
    )
    if padded_sequences.size(1) > max_length:
        return padded_sequences[:, :max_length]
    elif padded_sequences.size(1) < max_length:
        padding = torch.zeros((len(x), max_length - padded_sequences.size(1)))
        return torch.cat((padded_sequences, padding), dim=1)
    else:
        return padded_sequences
        
# def calculate_length(data:list):
#     lengths = [len(x) for x in data]
#     return [np.array(lengths)]

def calculate_length_test(data:list):
    lengths = [len(x) for x in data]
    return lengths

def calculate_lengths(self, dataset):
    seqs = dataset['Sequence'].tolist()
    lengths = [len(seq) for seq in seqs]
    dataset.loc[:, "Sequence length"] = lengths
    return dataset, lengths
        
def calculate_charge(data:list):
    h = analysis.GlobalAnalysis(data)
    h.calc_charge()
    # return h.charge
    return list(h.charge)

# def calculate_isoelectricpoint(data:list):
#     h = modlamp.analysis.GlobalDescriptor(data)
#     h.isoelectric_point()
#     return list(h.descriptor.flatten())

# def calculate_aromaticity(data:list):
#     h = modlamp.analysis.GlobalDescriptor(data)
#     h.aromaticity()
#     return list(h.descriptor.flatten())

def calculate_hydrophobicity(data:list):
    h = analysis.GlobalAnalysis(data)
    h.calc_H()
    return list(h.H)

# def calculate_hydrophobicmoment(data:list):
#     h = modlamp.analysis.GlobalAnalysis(data)
#     h.calc_uH()
#     return list(h.uH)

# def calculate_physchem(pool, peptides):
#     """
#     Oblicza właściwości fizykochemiczne dla listy peptydów równolegle,
#     dzieląc obliczenia dla każdej właściwości.

#     Args:
#         peptides: Lista sekwencji peptydów (ciągów znaków).
#         num_processes: Liczba procesów do użycia w puli.

#     Returns:
#         dict: Słownik, w którym kluczami są nazwy właściwości
#               ('length', 'charge', 'hydrophobicity_moment'),
#               a wartościami są listy tych właściwości dla wszystkich peptydów.
#     """
#     results = {}
#     results['hydrophobicity_moment'] = pool.apply_async(calculate_hydrophobicity, (peptides,))
#     results['length'] = pool.apply_async(calculate_length, (peptides,))
#     results['charge'] = pool.apply_async(calculate_charge, (peptides,))
#     return results

# def gather_physchem_results(async_results):
#     """Zbiera wyniki obliczone asynchronicznie dla właściwości fizykochemicznych."""
#     return [
#         async_results['hydrophobicity_moment'].get(),  # index 0
#         async_results['length'].get(),                 # index 1
#         async_results['charge'].get()                  # index 2
#     ]

def calculate_physchem_test(peptides: List[str]) -> torch.Tensor:
    all_features_per_peptide = []
    global_analysis_obj = analysis.GlobalAnalysis(peptides)
    lengths = calculate_length_test(peptides)
    global_analysis_obj.calc_charge()
    charges_np_array = np.asarray(global_analysis_obj.charge).flatten()
    charges = charges_np_array.tolist()
    global_analysis_obj.calc_H()
    hydrophobicity_np_array = np.asarray(global_analysis_obj.H).flatten()
    hydrophobicity = hydrophobicity_np_array.tolist()

    for i in range(len(peptides)):
        peptide_features = [
            lengths[i],
            charges[i],
            hydrophobicity[i]
        ]
        all_features_per_peptide.append(peptide_features)

    physchem_tensor = torch.tensor(all_features_per_peptide, dtype=torch.float32)

    return physchem_tensor

def adaptive_range_normalize(data, roi_min=0, roi_max=32, roi_bins=7, out_bins=3):
    data = np.asarray(data)
    result = np.zeros_like(data)
    roi_mask = (data >= roi_min) & (data <= roi_max)
    out_mask = ~roi_mask
    
    if np.any(roi_mask):
        roi_data = data[roi_mask]
        hist, bins = np.histogram(roi_data, bins=roi_bins)
        cdf = np.zeros(len(bins))
        cdf[1:] = np.cumsum(hist) / np.sum(hist)
        roi_normalized = np.interp(roi_data, bins, cdf)
        # result[roi_mask] = -(-1 + 1.4 * roi_normalized)
        result[roi_mask] = -(0 + -0.7 * roi_normalized)
    if np.any(out_mask):
        out_data = data[out_mask]
        hist, bins = np.histogram(out_data, bins=out_bins)
        cdf = np.zeros(len(bins))
        cdf[1:] = np.cumsum(hist) / np.sum(hist)
        out_normalized = np.interp(out_data, bins, cdf)
        # result[out_mask] = -(0.4 + 0.6 * out_normalized)
        result[out_mask] = -(-0.3 + 0.3 * out_normalized)
    
    return result

def z_score_normalize(data: np.ndarray) -> np.ndarray:
    mean_val = np.nanmean(data)
    std_val = np.nanstd(data)
    
    # Avoid division by zero if all values are the same
    if std_val == 0:
        return np.zeros_like(data)
        
    return (data - mean_val) / std_val

def normalize_attributes(physchem_tensor_original, reg_dim):
    fitted_transformers: Dict[int, QuantileTransformer] = {}
    physchem_tensor_normalized = torch.empty_like(physchem_tensor_original) 

    for col_idx in reg_dim:
        column_tensor = physchem_tensor_original[:, col_idx]
        data_to_transform_np = column_tensor.cpu().numpy().reshape(-1, 1)
        if col_idx ==3 or col_idx==4:
            qt = QuantileTransformer(
                        output_distribution='uniform',
                        n_quantiles=10,                )
            non_nan_mask = ~np.isnan(data_to_transform_np)
            normalized_values_input = np.log2(data_to_transform_np)
            normalized_values = adaptive_range_normalize(normalized_values_input[non_nan_mask])
            # normalized_values = qt.fit_transform(normalized_values_input)
            # normalized_values = z_score_normalize(data_to_transform_np[non_nan_mask])
            transformed_data_np = np.full_like(data_to_transform_np, np.nan)
            transformed_data_np[non_nan_mask] = normalized_values
            # transformed_data_np = normalized_values
        else:
            qt = QuantileTransformer(
                        output_distribution='uniform',
                        n_quantiles=10,                )
            transformed_data_np = qt.fit_transform(data_to_transform_np)
            # transformed_data_np = (transformed_data_np * 2) - 1
            fitted_transformers[col_idx] = qt
            # non_nan_mask = ~np.isnan(data_to_transform_np)
            # normalized_values = z_score_normalize(data_to_transform_np[non_nan_mask])
            # transformed_data_np = np.full_like(data_to_transform_np, np.nan)
            # transformed_data_np[non_nan_mask] = normalized_values
        
        transformed_column_tensor_2d = torch.from_numpy(transformed_data_np).float()
        physchem_tensor_normalized[:, col_idx] = transformed_column_tensor_2d.squeeze(1) 

    return physchem_tensor_normalized


