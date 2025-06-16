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
from sklearn.preprocessing import QuantileTransformer

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

_, amp_y, _a, amp_x = data_manager.get_merged_data()

df = pd.DataFrame({
    'Sequence': amp_x,
    'Label': amp_y
})

# filtered_df = df[df['Sequence'].str.len() <= 200].copy()
df['Hydrophobic ratio'] = np.nan

for idx, seq in enumerate(df['Sequence']):
    try:
        desc = GlobalDescriptor([seq])
        analysis = GlobalAnalysis([seq])
        analysis.calc_H()
        hydro_ratio = analysis.H[0][0]
        df.at[df.index[idx], 'Hydrophobic ratio'] = hydro_ratio

        desc.charge_density()
        charge = desc.descriptor[0, 0] 
        df.at[df.index[idx], 'Charge'] = charge
        df.at[df.index[idx], 'Length'] = len(df.at[df.index[idx], 'Sequence'])
    except Exception as e:
        print(f"Error processing sequence {seq}: {e}")

df.dropna(subset=['Hydrophobic ratio'], inplace=True)
df.dropna(subset=['Charge'], inplace=True)

# Plot the data
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=filtered_df, x='Label', y='Hydrophobic ratio', palette="Set2")
# plt.title('Hydrophobic Ratio by Label', fontsize=16)
# plt.xlabel('Label', fontsize=14)
# plt.ylabel('Hydrophobic Ratio', fontsize=14)
# plt.xticks(ticks=[0, 1], labels=['Negative (0)', 'Positive (1)'], fontsize=12)
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.violinplot(data=df, x='Label', y='Hydrophobic ratio', palette="Set2")
# plt.title('Hydrophobic Ratio by Label', fontsize=16)
# plt.xlabel('Label', fontsize=14)
# plt.ylabel('Hydrophobic Ratio', fontsize=14)
# plt.xticks(ticks=[0, 1], labels=['Negative (0)', 'Positive (1)'], fontsize=12)
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.boxplot(data=filtered_df, x='Label', y='Charge', palette="Set2")
# plt.title('Charge by Label', fontsize=16)
# plt.xlabel('Label', fontsize=14)
# plt.ylabel('Charge', fontsize=14)
# plt.xticks(ticks=[0, 1], labels=['Negative (0)', 'Positive (1)'], fontsize=12)
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.violinplot(data=df, x='Label', y='Charge', palette="Set2")
# plt.title('Charge by Label', fontsize=16)
# plt.xlabel('Label', fontsize=14)
# plt.ylabel('Charge', fontsize=14)
# plt.xticks(ticks=[0, 1], labels=['Negative (0)', 'Positive (1)'], fontsize=12)
# plt.show()

df['Length'].hist()
plt.title('Histogram of Length')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()

df['Hydrophobic ratio'].hist()
plt.title('Histogram of Hydrophobic ratio')
plt.xlabel('Hydrophobic ratio')
plt.ylabel('Frequency')
plt.show()

df['Charge'].hist()
plt.title('Histogram of Charge')
plt.xlabel('Charge')
plt.ylabel('Frequency')
plt.show()

print("Hydrophobic ratio description:")
print(df['Hydrophobic ratio'].describe())
print("Charge description:")
print(df['Charge'].describe())

NORMALIZATION_FACTORS = {
    "Length": (2, 25),
    "Hydrophobic ratio": (-2.5, 1.1),
    "Charge": (-0.008416, 0.007726)
}

fitted_transformers={}
for attr, (min_attr, max_attr) in NORMALIZATION_FACTORS.items():
    # df[attr] = (df[attr] - min_attr) / (max_attr - min_attr)
    data_to_transform = df[attr].values.reshape(-1, 1)
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    df[attr] = qt.fit_transform(data_to_transform)
    fitted_transformers[attr] = qt # Store the fitted transformer
    print(f"Applied Quantile Transformation to '{attr}' (output=Normal).")

df['Length'].hist()
plt.title('Histogram of Length after normalization')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()

df['Hydrophobic ratio'].hist()
plt.title('Histogram of Hydrophobic ratio after normalization')
plt.xlabel('Hydrophobic ratio')
plt.ylabel('Frequency')
plt.show()

df['Charge'].hist()
plt.title('Histogram of Charge after normalization')
plt.xlabel('Charge')
plt.ylabel('Frequency')
plt.show()

print("Sequence lengths for Label 0:")
print(df[df['Label'] == 0]['Sequence'].str.len().describe())
print("Sequence lengths for Label 1:")
print(df[df['Label'] == 1]['Sequence'].str.len().describe())

label_0 = df[df['Label'] == 0]['Hydrophobic ratio']
label_1 = df[df['Label'] == 1]['Hydrophobic ratio']

t_stat, p_value = ttest_ind(label_0, label_1)
print(f"Hydrophobic ratio: T-statistic: {t_stat}, P-value: {p_value}")

label_0 = df[df['Label'] == 0]['Charge']
label_1 = df[df['Label'] == 1]['Charge']

t_stat, p_value = ttest_ind(label_0, label_1)
print(f"Charge: T-statistic: {t_stat}, P-value: {p_value}")


