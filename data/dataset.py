import random
from collections import defaultdict
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from Bio import SeqIO
import modlamp
from sklearn.preprocessing import QuantileTransformer

STD_AA = list('ACDEFGHIKLMNPQRSTVWY')


def check_if_std_aa(sequence):
    if all(aa in STD_AA for aa in sequence):
        return True
    return False

def check_length(sequence, min_length, max_length):
    if min_length <= len(sequence) <= max_length:
        return True
    return False

def to_one_hot(x):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    classes = range(1, 21)
    aa_encoding = dict(zip(alphabet, classes))
    return [[aa_encoding[aa] for aa in seq] for seq in x]

def from_one_hot(encoded_seqs):
    return encoded_seqs.argmax(dim=-1)

def decoded(encoded_seqs, mode):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    classes = range(1, 21)
    aa_encoding = dict(zip(classes, alphabet))
    decoded_seqs = []
    for i, seq in enumerate(encoded_seqs):
        decoded_seq = ''.join(mode if idx == 0 else str(aa_encoding[idx.item()]) for idx in seq)
        decoded_seqs.append(decoded_seq)
    return decoded_seqs

def calculate_length_test(data:list):
    lengths = [len(x) for x in data]
    return lengths

def calculate_charge(data:list):
    h = modlamp.analysis.GlobalAnalysis(data)
    h.calc_charge()
    # return h.charge
    return list(h.charge)

def calculate_hydrophobicmoment(data:list):
    h = modlamp.analysis.GlobalAnalysis(data)
    h.calc_uH()
    return list(h.uH)

def calculate_physchem_test(peptides: List[str]) -> torch.Tensor:
    """
    Calculates physicochemical properties of peptides and returns them as a PyTorch tensor.

    Args:
        peptides (List[str]): A list of peptide sequences.

    Returns:
        torch.Tensor: A tensor containing the physicochemical properties.
                      Shape will be (num_peptides, num_physchem_features).
    """
    # Initialize lists to store properties for each peptide
    # We'll collect properties for each peptide as sub-lists
    all_features_per_peptide = []

    # Calculate properties for the entire list of peptides
    # This is more efficient than calling GlobalAnalysis for each peptide
    global_analysis_obj = modlamp.analysis.GlobalAnalysis(peptides)

    # Length
    lengths = calculate_length_test(peptides)
    global_analysis_obj.calc_charge()
    charges_np_array = np.asarray(global_analysis_obj.charge).flatten() # <--- KLUCZOWA ZMIANA
    charges = charges_np_array.tolist()

    # print(f"DEBUG: charges (after asarray and flatten): {charges[:5]} (first 5 elements, len={len(charges)})")

    global_analysis_obj.calc_uH()
    # To samo dla hydrophobic_moments
    hydrophobic_moments_np_array = np.asarray(global_analysis_obj.uH).flatten() # <--- KLUCZOWA ZMIANA
    hydrophobic_moments = hydrophobic_moments_np_array.tolist()

    # print(f"DEBUG: hydrophobic_moments (after asarray and flatten): {hydrophobic_moments[:5]} (first 5 elements, len={len(hydrophobic_moments)})")


    # Consolidate features for each peptide
    # Assuming all lists (lengths, charges, hydrophobic_moments) have the same length
    # which should be len(peptides)
    for i in range(len(peptides)):
        peptide_features = [
            lengths[i],
            charges[i],
            hydrophobic_moments[i]
        ]
        all_features_per_peptide.append(peptide_features)

    # Convert the list of lists into a PyTorch tensor
    # It will automatically infer the shape (num_peptides, num_features)
    physchem_tensor = torch.tensor(all_features_per_peptide, dtype=torch.float32)

    return physchem_tensor

def pad(x: List[List[float]], max_length: int = 25) -> torch.Tensor:
    """
    Pads sequences to the same length using PyTorch. Sequences longer than `max_length` are truncated.
    Sequences shorter than `max_length` are padded with 0.0 at the end.

    Args:
        x (List[List[float]]): List of sequences to pad.
        max_length (int): Maximum length to pad or truncate the sequences to.

    Returns:
        torch.Tensor: Padded tensor of shape (len(x), max_length).
    """
    # Convert input to torch tensors
    sequences = [torch.tensor(seq[:max_length]) for seq in x]  # Truncate longer sequences
    # Pad sequences
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0.0
    )
    # Truncate or pad to ensure all sequences are exactly max_length
    if padded_sequences.size(1) > max_length:
        return padded_sequences[:, :max_length]
    elif padded_sequences.size(1) < max_length:
        padding = torch.zeros((len(x), max_length - padded_sequences.size(1)))
        return torch.cat((padded_sequences, padding), dim=1)
    else:
        return padded_sequences
    
def normalize_attributes(physchem_tensor_original, ):
    fitted_transformers: Dict[int, QuantileTransformer] = {}
    feature_names = ["Hydrophobic Moment", "Length", "Charge"]# [hydrophobicity_moment, length, charge] - correct order
    physchem_tensor_normalized = torch.empty_like(physchem_tensor_original) # Tworzy tensor o tym samym kształcie i dtype co original
    for col_idx in range(3):
        feature_name = feature_names[col_idx] if col_idx < len(feature_names) else f"Column_{col_idx}"
        # print(f"\nProcessing column {col_idx} ('{feature_name}') with Quantile Transformation...")

        column_tensor = physchem_tensor_original[:, col_idx]

        data_to_transform_np = column_tensor.cpu().numpy().reshape(-1, 1)

        qt = QuantileTransformer(output_distribution='normal', random_state=42)

        transformed_data_np = qt.fit_transform(data_to_transform_np)

        fitted_transformers[col_idx] = qt

        transformed_column_tensor_2d = torch.from_numpy(transformed_data_np).float() # to jest już (N, 1)

        physchem_tensor_normalized[:, col_idx] = transformed_column_tensor_2d.squeeze(1) # Squeeze, bo przypisujemy 1D do 1D slices

        # print(f"Applied Quantile Transformation to '{feature_name}'. Min: {transformed_column_tensor_2d.min():.4f}, Max: {transformed_column_tensor_2d.max():.4f}")


    # print("\nNormalized physchem_tensor (first 5 rows):")
    # print(physchem_tensor_normalized[:5])
    # print(f"Shape of normalized physchem_tensor: {physchem_tensor_normalized.shape}")
    return physchem_tensor_normalized

class AMPDataManager:

    def __init__(
            self,
            positive_filepath: str,
            negative_filepath: str,
            min_len: int,
            max_len: int,
    ):
        if str(positive_filepath).endswith(".csv"):
            self.positive_data = pd.read_csv(positive_filepath)
        else:
            with open(positive_filepath) as fasta_file:  # Will close handle cleanly
                identifiers = []
                sequences = []
                for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
                    seq_str = str(seq_record.seq)  # Convert Seq object to string
                    if seq_str:  # Only add non-empty sequences
                        sequences.append(seq_str)
                        identifiers.append(seq_record.id)
            
            #converting lists to pandas Series    
            s1 = pd.Series(identifiers, name='ID')
            s2 = pd.Series(sequences, name='Sequence')
            #Gathering Series into a pandas DataFrame and rename index as ID column
            self.positive_data = pd.DataFrame({'ID': s1, 'Sequence': s2})
        if str(negative_filepath).endswith(".csv"):
            self.negative_data = pd.read_csv(negative_filepath)
        else:
            with open(negative_filepath) as fasta_file:  # Will close handle cleanly
                identifiers = []
                sequences = []
                for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
                    seq_str = str(seq_record.seq)  # Convert Seq object to string
                    if seq_str:  # Only add non-empty sequences
                        sequences.append(seq_str)
                        identifiers.append(seq_record.id)

            #converting lists to pandas Series    
            s1 = pd.Series(identifiers, name='ID')
            s2 = pd.Series(sequences, name='Sequence')
            #Gathering Series into a pandas DataFrame and rename index as ID column
            self.negative_data = pd.DataFrame({'ID': s1, 'Sequence': s2})

        self.min_len = min_len
        self.max_len = max_len

    def _filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (df['Sequence'].str.len() >= self.min_len) & (df['Sequence'].str.len() <= self.max_len)
        return df.loc[mask]

    def _filter_data(self):
        return self._filter_by_length(self.positive_data), self.negative_data

    @staticmethod
    def _get_probs(peptide_lengths: List[int]) -> Dict[int, float]:
        probs = defaultdict(lambda: 1)
        for length in peptide_lengths:
            probs[length] += 1
        return {k: round(v / len(peptide_lengths), 4) for k, v in probs.items()}

    @staticmethod
    def _draw_subsequences(df, new_lengths):
        new_lengths.sort(reverse=True)
        df = df.sort_values(by="Sequence length", ascending=False)

        d = []
        for row, new_length in zip(df.itertuples(), new_lengths):
            seq = row[2]
            curr_length = row[3]
            if new_length > curr_length:
                new_seq = seq
            elif new_length == curr_length:
                new_seq = seq
            else:
                begin = random.randrange(0, int(curr_length) - new_length)
                new_seq = seq[begin:begin + new_length]
            d.append(
                {
                    'Name': row[1],
                    'Sequence': new_seq,
                }
            )
        new_df = pd.DataFrame(d)
        return new_df

    def _equalize_data(self, positive_data: pd.DataFrame, negative_data: pd.DataFrame, balanced_classes: bool = True):
        positive_seq = positive_data['Sequence'].tolist()
        positive_lengths = [len(seq) for seq in positive_seq]

        negative_seq = negative_data['Sequence'].tolist()
        negative_lengths = [len(seq) for seq in negative_seq]
        negative_data.loc[:, "Sequence length"] = negative_lengths

        probs = self._get_probs(positive_lengths)
        k = len(positive_lengths) if balanced_classes else len(negative_lengths)

        new_negative_lengths = random.choices(list(probs.keys()), probs.values(), k=k)
        negative_data_distributed = self._draw_subsequences(self.negative_data, new_negative_lengths)
        return positive_data, negative_data_distributed

    def plot_distributions(self, equalize: bool = True):
        if equalize:
            pos_dataset, neg_dataset = self.get_data()
        else:
            pos_dataset, neg_dataset = self.positive_data, self.negative_data
        sns.set(color_codes=True)
        # TODO: figure out where this functionality should be. Goal is to plot distribution before and after baladancing
        positive_seq = pos_dataset['Sequence'].tolist()
        positive_lengths = [len(seq) for seq in positive_seq]

        negative_seq = neg_dataset['Sequence'].tolist()
        negative_lengths = [len(seq) for seq in negative_seq]

        fig, (ax2, ax3) = plt.subplots(figsize=(12, 6), ncols=2)
        sns.distplot(positive_lengths, ax=ax2)
        sns.distplot(negative_lengths, ax=ax3)
        ax2.set_title("Positive")
        ax3.set_title("Negative")

        plt.show()

    def _join_datasets(self, pos_dataset: pd.DataFrame, neg_dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        pos_dataset.loc[:, 'Label'] = 1
        neg_dataset.loc[:, 'Label'] = 0
        merged = pd.concat([pos_dataset, neg_dataset])
        x = np.asarray(merged['Sequence'].tolist())
        y = np.asarray(merged['Label'].tolist())
        x_changed = pad(to_one_hot(x))
        attributes = calculate_physchem_test(decoded(x_changed, ""),) 
        return x_changed, y, attributes, x

    def get_data(self, balanced: bool = True):
        pos_dataset, neg_dataset = self._filter_data()
        return self._equalize_data(pos_dataset, neg_dataset, balanced_classes=balanced)

    def get_merged_data(self, balanced: bool = True):
        pos_dataset, neg_dataset = self.get_data(balanced=balanced)
        return self._join_datasets(pos_dataset, neg_dataset)