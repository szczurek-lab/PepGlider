import pandas as pd
from Bio import SeqIO
import io
import classifier as c
from sklearn.model_selection import train_test_split
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def parse_fasta_to_df(file_content, nontoxicity_label, weight_value):
    """
    Parses a FASTA file's content into a pandas DataFrame.

    Args:
        file_content (str): The content of the FASTA file.
        toxicity_label (int): The label (0 or 1) for the 'toxicity' column.

    Returns:
        pd.DataFrame: A DataFrame with 'sequence' and 'toxicity' columns.
    """
    sequences = []
    # Use SeqIO.parse for robust FASTA parsing
    for record in SeqIO.parse(io.StringIO(file_content), "fasta"):
        try:
            if all(char in AMINO_ACIDS for char in str(record.seq)):#set(str(record.seq)).issubset(AMINO_ACIDS):
                sequences.append(str(record.seq))
        except UnicodeDecodeError:
            continue

    # Create a DataFrame from the list of sequences
    df = pd.DataFrame(sequences, columns=['sequence'])
    # Add the toxicity column with the specified label
    df['nontoxicity'] = nontoxicity_label
    df['weight'] = weight_value
    return df

# --- Step 1: Create mock FASTA file contents for demonstration ---
# In your actual code, you would read the files from disk:
with open('./toxicity_classifier/hemolytic.fasta', 'r', encoding='utf-8') as f:
    hemolytic_content = f.read()
with open('./toxicity_classifier/nonhemolytic.fasta', 'r', encoding='utf-8') as f:
    nonhemolytic_content = f.read()
with open('./toxicity_classifier/Signal peptide.fasta', 'r', encoding='utf-8') as f:
    signal_hemolytic_content = f.read()
with open('./toxicity_classifier/Metabolic.fasta', 'r', encoding='utf-8') as f:
    metabolic_hemolytic_content = f.read()
with open('./toxicity_classifier/Hormone.fasta', 'r', encoding='utf-8') as f:
    hormone_hemolytic_content = f.read()

# --- Step 2: Parse each file and add the toxicity label ---
hemolytic_df = parse_fasta_to_df(hemolytic_content, 1, 0.8)
nonhemolytic_df = parse_fasta_to_df(nonhemolytic_content, 0, 0.2)
signal_hemolytic_df = parse_fasta_to_df(signal_hemolytic_content, 1, 0.5)
metabolic_nonhemolytic_df = parse_fasta_to_df(metabolic_hemolytic_content, 1, 0.5)
hormone_hemolytic_df = parse_fasta_to_df(hormone_hemolytic_content, 1, 0.5)
print('read all fastas')
# --- Step 3: Concatenate the two DataFrames ---
final_df = pd.concat([hemolytic_df, nonhemolytic_df, signal_hemolytic_df, metabolic_nonhemolytic_df, hormone_hemolytic_df], ignore_index=True)
hemolytic_classifier = c.HemolyticClassifier(None)
features = hemolytic_classifier.get_input_features(final_df['sequence'].to_numpy())
labels = final_df['nontoxicity'].to_numpy()
mask_high_quality_idxs = final_df['weight'].to_numpy()
print(f'features shape = {features.shape}, labels shape = {labels.shape}, mask_high_quality_idxs shape = {mask_high_quality_idxs.shape}')
train_input, eval_input, train_labels, eval_labels, train_mask_high_quality_idxs, eval_mask_high_quality_idxs = train_test_split(
    features, labels, mask_high_quality_idxs , test_size=0.03, random_state=42, stratify=labels
)

hemolytic_classifier.train_classifier(train_input, train_labels, train_mask_high_quality_idxs)
hemolytic_classifier.save('')
print('trained')
hemolytic_classifier.eval_with_k_fold_cross_validation(eval_input,labels=eval_labels,mask_high_quality_idxs = eval_mask_high_quality_idxs)
