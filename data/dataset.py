import random
from collections import defaultdict
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from Bio import SeqIO
from modlamp import analysis
from sklearn.preprocessing import QuantileTransformer
import sys
sys.path.append('..')
from model.constants import MIN_LENGTH, MAX_LENGTH
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import tensor 

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
    h = analysis.GlobalAnalysis(data)
    h.calc_charge()
    # return h.charge
    return list(h.charge)

def calculate_hydrophobicity(data:list):
    h = analysis.GlobalAnalysis(data)
    h.calc_H()
    return list(h.H)

def calculate_hydrophobicmoment(data:list):
    h = analysis.GlobalAnalysis(data)
    h.calc_uH()
    return list(h.uH)

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

def adaptive_range_normalize(data, roi_min=0, roi_max=32, roi_bins=7, out_bins=3):
    """
    Adaptive range normalization with emphasis on 0-30 range.
    
    Args:
        data: Input array to normalize
        roi_min: Minimum value of region of interest (default: 0)
        roi_max: Maximum value of region of interest (default: 30)
        roi_bins: Number of bins for ROI values (default: 7)
        out_bins: Number of bins for outside values (default: 3)
    
    Returns:
        Normalized array with values in [-1, 1], where:
        - ROI values (0-30) are mapped to [-1, 0.4] (70% of range)
        - Outside values are mapped to [0.4, 1] (30% of range)
    """
    data = np.asarray(data)
    result = np.zeros_like(data)
    
    # Separate data into ROI and outside
    roi_mask = (data >= roi_min) & (data <= roi_max)
    out_mask = ~roi_mask
    
    # Process ROI values (map to [-1, 0.4])
    if np.any(roi_mask):
        roi_data = data[roi_mask]
        hist, bins = np.histogram(roi_data, bins=roi_bins)
        cdf = np.zeros(len(bins))
        cdf[1:] = np.cumsum(hist) / np.sum(hist)
        roi_normalized = np.interp(roi_data, bins, cdf)
        result[roi_mask] = -(-1 + 1.4 * roi_normalized)
    
    # Process outside values (map to [0.4, 1])
    if np.any(out_mask):
        out_data = data[out_mask]
        hist, bins = np.histogram(out_data, bins=out_bins)
        cdf = np.zeros(len(bins))
        cdf[1:] = np.cumsum(hist) / np.sum(hist)
        out_normalized = np.interp(out_data, bins, cdf)
        result[out_mask] = -(0.4 + 0.6 * out_normalized)
    
    return result

def normalize_attributes(physchem_tensor_original, reg_dim):
    fitted_transformers: Dict[int, QuantileTransformer] = {}
    # feature_names = ["Hydrophobic Moment", "Length", "Charge"]
    physchem_tensor_normalized = torch.empty_like(physchem_tensor_original) 

    for col_idx in reg_dim:
        # feature_name = feature_names[col_idx] if col_idx < len(feature_names) else f"Column_{col_idx}"
        column_tensor = physchem_tensor_original[:, col_idx]
        data_to_transform_np = column_tensor.cpu().numpy().reshape(-1, 1)
        # qt = QuantileTransformer(output_distribution='normal', random_state=42)
        if col_idx ==3 or col_idx==4:
            non_nan_mask = ~np.isnan(data_to_transform_np)
            normalized_values = adaptive_range_normalize(data_to_transform_np[non_nan_mask])
            transformed_data_np = np.full_like(data_to_transform_np, np.nan)
            print(normalized_values)
            transformed_data_np[non_nan_mask] = normalized_values
        else:
            qt = QuantileTransformer(
                        output_distribution='uniform',
                        n_quantiles=10,                )
            transformed_data_np = qt.fit_transform(data_to_transform_np)
            fitted_transformers[col_idx] = qt
        transformed_column_tensor_2d = torch.from_numpy(transformed_data_np).float()
        physchem_tensor_normalized[:, col_idx] = transformed_column_tensor_2d.squeeze(1) 

    return physchem_tensor_normalized

def normalize_dimension_to_0_1(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Wejście musi być tensorem PyTorch.")
    if dim >= tensor.ndim or dim < -tensor.ndim:
        raise ValueError(f"Wymiar {dim} jest poza zakresem tensora o {tensor.ndim} wymiarach.")

    min_vals = tensor.min(dim=dim, keepdim=True).values
    max_vals = tensor.max(dim=dim, keepdim=True).values
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    normalized_tensor = (tensor - min_vals) / range_vals
    return normalized_tensor

def prepare_data_for_training(data_dir, batch_size, data_type,mic_flg, reg_dim):
    if 'positiv_negativ_AMPs' in data_type and 'uniprot' in data_type:
        data_manager = AMPDataManager(
            positive_filepath = data_dir / 'unlabelled_positive.csv',
            negative_filepath = data_dir / 'unlabelled_negative.csv',
            uniprot_filepath = [data_dir / i for i in ['Uniprot_0_25_train.csv','Uniprot_0_25_test.csv','Uniprot_0_25_val.csv']],
            min_len=MIN_LENGTH,
            max_len=MAX_LENGTH,
            mic_flg = mic_flg,
            data_dir = data_dir)

        amp_x, amp_y, attributes_input, _ = data_manager.get_uniform_data()
    elif 'positiv_negativ_AMPs' in data_type:
        data_manager = AMPDataManager(
            positive_filepath = data_dir / 'unlabelled_positive.csv',
            negative_filepath = data_dir / 'unlabelled_negative.csv',
            min_len=MIN_LENGTH,
            max_len=MAX_LENGTH,
            mic_flg = mic_flg,
            data_dir = data_dir)

        amp_x, amp_y, attributes_input, _ = data_manager.get_merged_data()
    elif 'positiv_AMPs' in data_type:
        data_manager = AMPDataManager(
            positive_filepath = data_dir / 'amp_clean.fasta',
            negative_filepath = None,
            min_len=MIN_LENGTH,
            max_len=MAX_LENGTH,
            mic_flg = mic_flg,
            data_dir = data_dir)

        amp_x, amp_y, attributes_input, _ = data_manager.get_positive_data()
    elif 'uniprot' in data_type:
        data_manager = AMPDataManager(
            uniprot_filepath = [data_dir / i for i in ['Uniprot_0_25_train.csv','Uniprot_0_25_test.csv','Uniprot_0_25_val.csv']],
            min_len=MIN_LENGTH,
            max_len=MAX_LENGTH,
            mic_flg = mic_flg,
            data_dir = data_dir)
        amp_x, amp_y, attributes_input, _ = data_manager.get_uniprot_data()
    attributes = normalize_attributes(attributes_input, reg_dim)
    # print(f'attributes shape = {attributes.shape}')
    for i in range(attributes_input.shape[1]):
        print(f'{i} - min = {np.mean(attributes_input[:,i].cpu().numpy(), axis=1)}, max = {np.mean(attributes_input[:,i].cpu().numpy(), axis=1)}')
    #print(f'attributes_input shape = {attributes_input.shape}')
    # for i, attr_name in enumerate(['Length', 'Charge', 'Hydrophobic moment']):
        # plot_hist_lengths(attributes_input[:,i].cpu().numpy(), attr_name)
    dataset = TensorDataset(amp_x, tensor(amp_y), attributes, attributes_input)
    # print(f'dataset size = {len(dataset)}')
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, eval_loader

def plot_hist_lengths(data, attr_name):
    unique_values = np.unique(data)
    # print(f'unique_values = {unique_values}')
    bin_edges = np.concatenate([unique_values - 0.5, [unique_values[-1] + 0.5]])

    plt.hist(data, bins=40)
    #bin_edges, edgecolor='black')
    # plt.xticks(unique_values)  # Set the x-ticks to be the unique values

    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.title(f"Data {attr_name}s' sequency histogram")
    plt.savefig(f'{attr_name}_seq_hist.png')
    plt.close()

class AMPDataManager:
    def __init__(
            self,
            positive_filepath: str = None,
            negative_filepath: List[str] = None,
            uniprot_filepath: List[str] = [],
            min_len: int = 0,
            max_len: int = 25,
            mic_flg: bool = False,
            data_dir: str = ''
    ):
        if str(positive_filepath).endswith(".csv"):
            self.positive_data = pd.read_csv(positive_filepath)
            if mic_flg:
                new_data1 = pd.read_csv(data_dir / 'apex-ecoli.tsv', sep='\t')
                new_data2 = pd.read_csv(data_dir / 'apex-saureus.tsv', sep='\t')

                self.positive_data = self.update_and_add_sequences(self.positive_data, new_data1, new_label='mic_e_cola')
                self.positive_data = self.update_and_add_sequences(self.positive_data, new_data2, new_label='mic_s_aureus')
            # print(self.positive_data)
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
            s2 = pd.Series(sequences, name='Sequence')
            #Gathering Series into a pandas DataFrame and rename index as ID column
            self.positive_data = pd.DataFrame({'Sequence': s2})
            if mic_flg:
                new_data1 = pd.read_csv(data_dir / 'apex-ecoli.tsv', sep='\t')
                new_data2 = pd.read_csv(data_dir / 'apex-saureus.tsv', sep='\t')

                self.positive_data = self.update_and_add_sequences(self.positive_data, new_data1, new_label='mic_e_cola')
                self.positive_data = self.update_and_add_sequences(self.positive_data, new_data2, new_label='mic_s_aureus')

        if str(negative_filepath).endswith(".csv"):
            self.negative_data = pd.read_csv(negative_filepath)
        else:
            self.negative_data = None
            # with open(negative_filepath) as fasta_file:  # Will close handle cleanly
            #     identifiers = []
            #     sequences = []
            #     for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
            #         seq_str = str(seq_record.seq)  # Convert Seq object to string
            #         if seq_str:  # Only add non-empty sequences
            #             sequences.append(seq_str)
            #             identifiers.append(seq_record.id)
  
            # s1 = pd.Series(identifiers, name='ID')
            # s2 = pd.Series(sequences, name='Sequence')
            # self.negative_data = pd.DataFrame({'ID': s1, 'Sequence': s2})
        if len(uniprot_filepath) != 0:
            dfs = [pd.read_csv(f) for f in uniprot_filepath]
            self.uniprot_data = pd.concat(dfs, ignore_index=True)
        else:
            self.uniprot_data = None
        self.mic_flg = mic_flg
        self.min_len = min_len
        self.max_len = max_len
    @staticmethod
    def update_and_add_sequences(df_main: pd.DataFrame, new_df: pd.DataFrame, new_label: str = 'MIC') -> pd.DataFrame:
        """
        Updates existing sequences in the main DataFrame with activity values from a new DataFrame
        and appends new sequences with a specified label.
        
        Args:
            df_main (pd.DataFrame): The main DataFrame (e.g., self.positive_data).
            new_df (pd.DataFrame): The DataFrame containing new sequences and MIC values.
            new_label (str): The label to assign to sequences not present in df_main.
            
        Returns:
            pd.DataFrame: The updated main DataFrame.
        """
        # Create a Series to hold the activity values from the new DataFrame, using sequences as the index
        # CHANGE: Use 'Sequence' and 'MIC'
        # new_activities = new_df.set_index('Sequence')['MIC']
        
        # # 1. Update existing sequences
        # # Use .update() to add new activity values for matching sequences
        # df_main = df_main.set_index('Sequence')
        # df_main.update(new_activities)
        # df_main = df_main.reset_index()
        
        # # 2. Identify and append new sequences
        # # Find sequences that are in the new_df but not in df_main
        # # CHANGE: Use 'Sequence' here
        # new_sequences_df = new_df[~new_df['Sequence'].isin(df_main['Sequence'])].copy()
        
        # # Add the 'label' column with the specified value
        # # CHANGE: Use 'Sequence' and 'MIC' for renaming
        new_df = new_df.rename(columns={'MIC': new_label})
        
        # # Append the new sequences to the main DataFrame
        # updated_df = pd.concat([df_main, new_sequences_df], ignore_index=True)
        # print(updated_df)
        updated_df = pd.merge(df_main, new_df, on='Sequence', how='left')

        return updated_df
    
    def _filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (df['Sequence'].str.len() >= self.min_len) & (df['Sequence'].str.len() <= self.max_len)
        return df.loc[mask]

    def _filter_data(self):
        if self.positive_data is not None and self.negative_data is not None and self.uniprot_data is not None:
            return self._filter_by_length(self.positive_data), self.negative_data, self._filter_by_length(self.uniprot_data)
        elif self.positive_data is not None and self.negative_data is not None:
            return self._filter_by_length(self.positive_data), self.negative_data
        elif self.positive_data is not None and self.negative_data is None:
            return self._filter_by_length(self.positive_data)
        else: 
            return self._filter_by_length(self.uniprot_data)

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
        return merged
    
    def output_data(self, df):
        x = np.asarray(df['Sequence'].tolist())
        if 'Label' in df.columns:
            y = np.asarray(df['Label'].tolist())
        else:
            y = np.zeros((x.shape[0]))
        x_changed = pad(to_one_hot(x))
        attributes = calculate_physchem_test(decoded(x_changed, ""),) 
        if self.mic_flg:
            mic_e_cola = torch.Tensor(df['mic_e_cola'].tolist()).unsqueeze(1)
            # print(mic_e_cola)
            mic_s_aureus = torch.Tensor(df['mic_s_aureus'].tolist()).unsqueeze(1)
            result_tensor = torch.cat([attributes, mic_e_cola, mic_s_aureus], dim=1)
            return x_changed, y, result_tensor, x
        return x_changed, y, attributes, x

    def get_data(self, balanced: bool = True):
        pos_dataset, neg_dataset = self._filter_data()
        return self._equalize_data(pos_dataset, neg_dataset, balanced_classes=balanced)

    def get_merged_data(self, balanced: bool = True):
        pos_dataset, neg_dataset = self.get_data(balanced=balanced)
        return self.output_data(self._join_datasets(pos_dataset, neg_dataset))
    
    def calculate_lengths(self, dataset):
        seqs = dataset['Sequence'].tolist()
        lengths = [len(seq) for seq in seqs]
        dataset.loc[:, "Sequence length"] = lengths
        return dataset, lengths
  
    def get_uniprot_data(self):
        uniprot_dataset = self._filter_data()
        uniprot_dataset,uniprot_lengths = self.calculate_lengths(uniprot_dataset)

        probs = self._get_probs(uniprot_lengths)
        # plot_hist_lengths(uniprot_dataset['Sequence length'].to_numpy()
        return self.output_data(uniprot_dataset)

    def get_positive_data(self):
        positive_dataset = self._filter_data()
        positive_dataset.loc[:, 'Label'] = 1
        return self.output_data(positive_dataset)
    
    def get_uniform_data(self):
        pos_dataset, neg_dataset, uniprot_dataset = self._filter_data()
        pos_dataset, neg_dataset = self._equalize_data(pos_dataset, neg_dataset, balanced_classes=True)
        pos_dataset,_ = self.calculate_lengths(pos_dataset)
        neg_dataset,_ = self.calculate_lengths(neg_dataset)
        uniprot_dataset,_ = self.calculate_lengths(uniprot_dataset)

        pos_lengths = set(pos_dataset['Sequence length'].unique())
        neg_lengths = set(neg_dataset['Sequence length'].unique())
        uniprot_lengths = set(uniprot_dataset['Sequence length'].unique())
        common_lengths = pos_lengths & neg_lengths & uniprot_lengths
        if 7 in common_lengths:
           common_lengths.remove(7)
        df = pd.concat([pos_dataset[['Sequence', "Sequence length"]], neg_dataset[['Sequence', "Sequence length"]], uniprot_dataset], axis=0)
        filtered_df = df[df['Sequence length'].isin(common_lengths)]
        counts = filtered_df['Sequence length'].value_counts().min()
        final_df = pd.DataFrame()
        for i in pd.unique(filtered_df['Sequence length']):
            final_df = pd.concat([final_df, filtered_df[filtered_df['Sequence length'] == i].iloc[:counts,:]])
        return self.output_data(final_df)
