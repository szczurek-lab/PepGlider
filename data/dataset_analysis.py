import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from modlamp.descriptors import GlobalDescriptor
from modlamp.analysis import GlobalAnalysis
from scipy.stats import ttest_ind
import dataset as dataset_lib
import random
from torch import manual_seed, cuda, backends
import os

MIN_LENGTH = 0
MAX_LENGTH = 25

ROOT_DIR = Path(__file__).parent#.parent
DATA_DIR = ROOT_DIR

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
# set_seed()

data_manager = dataset_lib.AMPDataManager(
    DATA_DIR / 'unlabelled_positive.csv',
    DATA_DIR / 'unlabelled_negative.csv',
    min_len=MIN_LENGTH,
    max_len=MAX_LENGTH)

_, amp_y, amp_x = data_manager.get_merged_data()

df = pd.DataFrame({
    'Sequence': amp_x,
    'Label': amp_y
})

filtered_df = df[df['Sequence'].str.len() <= 200].copy()
filtered_df['Hydrophobic ratio'] = np.nan

for idx, seq in enumerate(filtered_df['Sequence']):
    try:
        desc = GlobalDescriptor([seq])
        analysis = GlobalAnalysis([seq])
        analysis.calc_H()
        hydro_ratio = analysis.H[0][0]
        filtered_df.at[filtered_df.index[idx], 'Hydrophobic ratio'] = hydro_ratio

        desc.charge_density()
        charge = desc.descriptor[0, 0] 
        filtered_df.at[filtered_df.index[idx], 'Charge'] = charge
    except Exception as e:
        print(f"Error processing sequence {seq}: {e}")

filtered_df.dropna(subset=['Hydrophobic ratio'], inplace=True)
filtered_df.dropna(subset=['Charge'], inplace=True)

# Plot the data
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=filtered_df, x='Label', y='Hydrophobic ratio', palette="Set2")
# plt.title('Hydrophobic Ratio by Label', fontsize=16)
# plt.xlabel('Label', fontsize=14)
# plt.ylabel('Hydrophobic Ratio', fontsize=14)
# plt.xticks(ticks=[0, 1], labels=['Negative (0)', 'Positive (1)'], fontsize=12)
# plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=filtered_df, x='Label', y='Hydrophobic ratio', palette="Set2")
plt.title('Hydrophobic Ratio by Label', fontsize=16)
plt.xlabel('Label', fontsize=14)
plt.ylabel('Hydrophobic Ratio', fontsize=14)
plt.xticks(ticks=[0, 1], labels=['Negative (0)', 'Positive (1)'], fontsize=12)
plt.show()

# plt.figure(figsize=(10, 6))
# sns.boxplot(data=filtered_df, x='Label', y='Charge', palette="Set2")
# plt.title('Charge by Label', fontsize=16)
# plt.xlabel('Label', fontsize=14)
# plt.ylabel('Charge', fontsize=14)
# plt.xticks(ticks=[0, 1], labels=['Negative (0)', 'Positive (1)'], fontsize=12)
# plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=filtered_df, x='Label', y='Charge', palette="Set2")
plt.title('Charge by Label', fontsize=16)
plt.xlabel('Label', fontsize=14)
plt.ylabel('Charge', fontsize=14)
plt.xticks(ticks=[0, 1], labels=['Negative (0)', 'Positive (1)'], fontsize=12)
plt.show()

print("Sequence lengths for Label 0:")
print(df[df['Label'] == 0]['Sequence'].str.len().describe())
print("Sequence lengths for Label 1:")
print(df[df['Label'] == 1]['Sequence'].str.len().describe())

label_0 = filtered_df[filtered_df['Label'] == 0]['Hydrophobic ratio']
label_1 = filtered_df[filtered_df['Label'] == 1]['Hydrophobic ratio']

t_stat, p_value = ttest_ind(label_0, label_1)
print(f"Hydrophobic ratio: T-statistic: {t_stat}, P-value: {p_value}")

label_0 = filtered_df[filtered_df['Label'] == 0]['Charge']
label_1 = filtered_df[filtered_df['Label'] == 1]['Charge']

t_stat, p_value = ttest_ind(label_0, label_1)
print(f"Charge: T-statistic: {t_stat}, P-value: {p_value}")


