import pandas as pd

# 1. Load the dataset
# Assuming 'data.csv' is your filename
# df = pd.read_csv('new_e_coli.tsv',sep = '\t') 
df = pd.read_csv('new_s_aureus.tsv',sep = '\t') 

# --- STEP 1: Extract Peptides ---
# Extract the first column (index 0) which has no name
peptides = df['Sequence']
peptides.to_csv('peptides.txt', index=False, header=False)

# --- STEP 2: Grouping and Processing ---
# We assign the first column a name 'seq' internally for the grouping logic
#df.columns.values[0] = 'seq'

#ecoli_cols = ['E. coli ATCC 11775']
#saureus_cols = ['S. aureus ATCC 12600']
# ecoli_cols = ['MIC']
s_aureus_cols = ['MIC']
# Grouping logic as requested
#ecoli_means_df = df.groupby('seq')[ecoli_cols].mean()
#saureus_means_df = df.groupby('seq')[saureus_cols].mean()
#grouped_df = pd.concat([ecoli_means_df, saureus_means_df], axis=1)

# Fill nulls with 64
#grouped_df = grouped_df.fillna(64)

# --- STEP 3: Binary Coding ---
# Rule: 1 if value <= 32, else 0
def binary_code(val):
    return 1 if val <= 32 else 0

# Apply coding to each column
#e_coli_results = grouped_df['E. coli ATCC 11775'].apply(binary_code)
#s_aureus_results = grouped_df['S. aureus ATCC 12600'].apply(binary_code)
# e_coli_results = df[ecoli_cols].applymap(binary_code)
s_aureus_results = df[s_aureus_cols].applymap(binary_code)
# --- STEP 4: Save Results ---
# e_coli_results.to_csv('E_coli.txt', index=False, header=False)
s_aureus_results.to_csv('S_aureus.txt', index=False, header=False)

print("Files 'peptides.txt', 'E_coli.txt', and 'S_aureus.txt' have been created.")
