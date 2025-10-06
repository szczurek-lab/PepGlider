from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef, precision_score
import classifier as c
import pandas as pd
from Bio import SeqIO
import io
from sklearn.model_selection import train_test_split

with open('./toxicity_classifier/hemolytic.fasta', 'r', encoding='utf-8') as f:
    amp_clean = f.read()
with open('./toxicity_classifier/nonhemolytic.fasta', 'r', encoding='utf-8') as f:
    nonhemolytic_content = f.read()
with open('./toxicity_classifier/Signal peptide.fasta', 'r', encoding='utf-8') as f:
    signal_hemolytic_content = f.read()
with open('./toxicity_classifier/Metabolic.fasta', 'r', encoding='utf-8') as f:
    metabolic_hemolytic_content = f.read()
with open('./toxicity_classifier/Hormone.fasta', 'r', encoding='utf-8') as f:
    hormone_hemolytic_content = f.read()

hemolytic_df = c.parse_fasta_to_df(amp_clean, 0, 0.8)
nonhemolytic_df = c.parse_fasta_to_df(nonhemolytic_content, 1, 0.2)
signal_nonhemolytic_df = c.parse_fasta_to_df(signal_hemolytic_content, 1, 0.5)
metabolic_nonhemolytic_df = c.parse_fasta_to_df(metabolic_hemolytic_content, 1, 0.5)
hormone_nonhemolytic_df = c.parse_fasta_to_df(hormone_hemolytic_content, 1, 0.5)
final_df = pd.concat([hemolytic_df, nonhemolytic_df, signal_nonhemolytic_df, metabolic_nonhemolytic_df, hormone_nonhemolytic_df], ignore_index=True)
# final_df = pd.read_csv('./toxicity_classifier/hydramp.csv')
print(hemolytic_df['sequence'].to_numpy().shape)
print(final_df['sequence'].to_numpy().shape)
hemolytic_classifier = c.HemolyticClassifier('/AR-VAE/new_hemolytic_model.xgb')
features = hemolytic_classifier.get_input_features(final_df['sequence'].to_numpy())
labels = final_df['nontoxicity'].to_numpy()
mask_high_quality_idxs = final_df['nontoxicity'].to_numpy()
train_input, eval_input, train_labels, eval_labels, train_mask_high_quality_idxs, eval_mask_high_quality_idxs = train_test_split(
   features, labels, mask_high_quality_idxs , test_size=0.03, random_state=42, stratify=labels
)
hemolytic_classifier.eval_trained_classifier(eval_input,eval_labels, mask_high_quality_idxs=eval_mask_high_quality_idxs)

