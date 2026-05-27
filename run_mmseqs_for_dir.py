import os
import argparse
from mean_mmseqs_score import compute_mean_normalized_bitscore
import pandas as pd
import re

def main():
    parser = argparse.ArgumentParser(
        description="Compute mean normalized MMseqs scores for a directory of FASTA files"
    )

    parser.add_argument(
        "-q",
        "--query_dir",
        required=True,
        help="Directory containing query FASTA files",
    )
    parser.add_argument(
        "-d", "--database", required=True, help="Reference/database FASTA file"
    )
    parser.add_argument(
        "-r", "--results_dir", required=True, help="Directory to store result CSV files"
    )
    parser.add_argument(
        "-n", "--name", required=True, help="Reference name used in output filenames"
    )

    args = parser.parse_args()

    base_path = args.query_dir
    ref_path = args.database
    results_path = args.results_dir
    ref_name = args.name

    os.makedirs(results_path, exist_ok=True)

    for fname in os.listdir(base_path):
        file_path = os.path.join(base_path, fname)
        
        if not fname.endswith(".fasta"):
             continue
            
        # elif fname.endswith(".csv"):
        #     # Load CSV without assuming the first row is a header
        #     # then check if the first row actually looks like a header or a sequence
        #     df_test = pd.read_csv(file_path, nrows=1)
            
        #     # If the first 'column name' is actually a sequence (contains A, R, N, D...), 
        #     # reload with header=None
        #     first_col_name = str(df_test.columns[0])
        #     if re.search(r'[ARNDCEQGHILKMFPSTWYV]{5,}', first_col_name.upper()):
        #         df = pd.read_csv(file_path, header=None)
        #     else:
        #         df = pd.read_csv(file_path)
    
        #     fasta_fname = fname.replace(".csv", ".fasta")
        #     fasta_path = os.path.join(base_path, fasta_fname)
            
        #     with open(fasta_path, "w") as f:
        #         for i, row in df.iterrows():
        #             # 1. Identify which column is the sequence
        #             # We look for the first column that contains amino acid-like strings
        #             seq = ""
        #             seq_col_idx = 0
        #             for col_idx, val in enumerate(row):
        #                 val_str = str(val).strip()
        #                 if re.search(r'[ARNDCEQGHILKMFPSTWYV]{3,}', val_str.upper()):
        #                     seq = val_str
        #                     seq_col_idx = col_idx
        #                     break
                    
        #             if not seq or seq.lower() == 'nan':
        #                 continue
    
        #             # 2. Identify Metadata (anything that isn't the sequence)
        #             # For your example, this will grab the numbers (0, 1, 2...)
        #             meta_data = [str(row[c]).strip() for c in range(len(row)) if c != seq_col_idx]
        #             meta_data = [m for m in meta_data if m.lower() != 'nan' and m != ""]
                    
        #             # 3. Build the Header
        #             if meta_data:
        #                 # e.g., >0 or >1
        #                 header_text = "_".join(meta_data)
        #             else:
        #                 # fallback if there truly is only one column
        #                 header_text = f"seq_{i}"
                    
        #             f.write(f">{header_text}\n{seq}\n")
            
        #     print(f"Successfully converted: {fname}")
        #     fname = fasta_fname
        
        # else:
        #     continue
            
        query = os.path.join(base_path, fname)
        base_name = os.path.splitext(fname)[0]

        mean_score, coverage = compute_mean_normalized_bitscore(
            query,
            ref_path,
            os.path.join(results_path, f"{base_name}_vs_{ref_name}.csv"),
        )

        print(
            f"Query: {fname}, Mean MMseqs Score: {mean_score:.4f}, Coverage: {coverage:.4f}"
        )


if __name__ == "__main__":
    main()
