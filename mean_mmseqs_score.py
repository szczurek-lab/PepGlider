import subprocess
import pandas as pd
import os
import argparse
from pathlib import Path
from Bio import SeqIO
import shutil

MMSEQ_BIN = "/home/gwiazale/AR-VAE/mmseq2/mmseqs/bin/mmseqs"
PROFILE = "easy-search"


def get_num_sequences(fasta_path):
    return sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))


def safe_load_tsv(path):
    if os.path.getsize(path) == 0:
        return pd.DataFrame(columns=[0, 11])  # empty DF with required cols
    return pd.read_csv(path, sep="\t", header=None)


def run_mmseq(query_fasta, database, i, add_self_matches=False):
    cmd = [
        MMSEQ_BIN,
        PROFILE,
        query_fasta,
        database,
        f"result_{i}.tsv",
        "tmp",
        "-v",
        "0",
        "-e",
        "1000",
    ]

    if add_self_matches:
        cmd += ["--add-self-matches", "--max-seqs", "1"]

    subprocess.run(cmd, check=True)


def compute_mean_normalized_bitscore(
    query_fasta: str,
    database_fasta: str,
    output_results_csv: str | None = None,
) -> tuple[float, float]:
    i = 0

    # ------------------------------------------------------------------
    # Step 1: Load sequences
    # ------------------------------------------------------------------
    query_records = list(SeqIO.parse(query_fasta, "fasta"))
    db_records = list(SeqIO.parse(database_fasta, "fasta"))
    db_seqs = {str(r.seq) for r in db_records}

    total_queries = len(query_records)
    query_ids = [r.id for r in query_records]

    # ------------------------------------------------------------------
    # Step 2: Split shared vs non-shared queries
    # ------------------------------------------------------------------
    shared_records = [r for r in query_records if str(r.seq) in db_seqs]
    filtered_records = [r for r in query_records if str(r.seq) not in db_seqs]

    shared_ids = {r.id for r in shared_records}
    filtered_ids = {r.id for r in filtered_records}

    # ------------------------------------------------------------------
    # Prepare result table with defaults (no-hit = 0)
    # ------------------------------------------------------------------
    results = pd.DataFrame(
        {
            "queryId": query_ids,
            "targetId": None,
            "bit_score": None,
            "self_targetId": None,
            "self_bit_score": None,
            "normalized": 0.0,
        }
    ).set_index("queryId")

    # Shared queries score 1 by definition
    results.loc[list(shared_ids), "normalized"] = 1.0
    # # results.loc[list(shared_ids), "targetId"] = list(shared_ids)
    # # Create a temporary list of IDs to match the length of the selected rows
    ids_to_assign = list(shared_ids)
    
    # # Check if the lengths match before assigning
    # if len(results.loc[ids_to_assign]) == len(ids_to_assign):
    #     results.loc[ids_to_assign, "targetId"] = ids_to_assign
    # else:
    #     # If there are duplicates or mismatches, this mapping approach is safer
    #     for identifier in ids_to_assign:
    #         results.loc[identifier, "targetId"] = identifier
    # results.loc[list(shared_ids), "self_targetId"] = list(shared_ids)
    
    for identifier in ids_to_assign:
        # results.loc[identifier] targets ALL rows with that ID
        results.loc[identifier, "targetId"] = identifier
        results.loc[identifier, "self_targetId"] = identifier

    # ------------------------------------------------------------------
    # If all queries are shared → done
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # If all queries are shared → done
    # ------------------------------------------------------------------
    if not filtered_records:
        mean_norm = 1.0
        coverage = 1.0

        if output_results_csv:
            results.reset_index().to_csv(output_results_csv, index=False)

        return mean_norm, coverage

    # ------------------------------------------------------------------
    # Write filtered queries to FASTA
    # ------------------------------------------------------------------
    filtered_fasta = "fasta_filtered.fasta"

    for idx, r in enumerate(filtered_records):
        if not r.id or r.id.strip() == "":
            r.id = f"query_{idx}"
            r.description = r.id # Some tools check description too
        
    SeqIO.write(filtered_records, filtered_fasta, "fasta")
    print("Filtered queries written to:", filtered_fasta)

    # ------------------------------------------------------------------
    # Step 3: Query vs database
    # ------------------------------------------------------------------
    run_mmseq(filtered_fasta, database_fasta, i, add_self_matches=False)
    df_db = safe_load_tsv(f"result_{i}.tsv")

    if df_db.empty:
        # No hits at all: filtered queries remain 0
        mean_norm = results["normalized"].mean()
        coverage = len(shared_ids) / total_queries

        if output_results_csv:
            results.reset_index().to_csv(output_results_csv, index=False)

        return mean_norm, coverage

    idx = df_db.groupby(0)[df_db.columns[-1]].idxmax()
    best_db_hits = df_db.loc[idx, [0, 1, df_db.columns[-1]]]
    best_db_hits.columns = ["queryId", "targetId", "bit_score"]
    i += 1

    # ------------------------------------------------------------------
    # Step 4: Self-alignment
    # ------------------------------------------------------------------
    run_mmseq(filtered_fasta, filtered_fasta, i, add_self_matches=True)
    df_self = safe_load_tsv(f"result_{i}.tsv")

    if df_self.empty:
        mean_norm = results["normalized"].mean()
        coverage = len(shared_ids) / total_queries

        if output_results_csv:
            results.reset_index().to_csv(output_results_csv, index=False)

        return mean_norm, coverage

    idx = df_self.groupby(0)[df_self.columns[-1]].idxmax()
    best_self_hits = df_self.loc[idx, [0, 1, df_self.columns[-1]]]
    best_self_hits.columns = ["queryId", "self_targetId", "self_bit_score"]

    # ------------------------------------------------------------------
    # Step 5: Normalize
    # ------------------------------------------------------------------
    merged = best_db_hits.merge(best_self_hits, on="queryId", how="inner")
    merged["normalized"] = merged["bit_score"] / merged["self_bit_score"]

    if (merged["normalized"] > 1).any():
        print("Warning: Some normalized bitscores > 1")
        print(merged[merged["normalized"] > 1])

    # Update results table
    results.update(merged.set_index("queryId"))

    # ------------------------------------------------------------------
    # Final stats
    # ------------------------------------------------------------------
    mean_norm = results["normalized"].mean()
    coverage = (results["normalized"] > 0).mean()

    assert len(results) == total_queries

    if output_results_csv:
        results.reset_index().to_csv(output_results_csv, index=False)

    return mean_norm, coverage


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute mean normalized bitscore using MMSEQ2."
    )
    parser.add_argument("--query", required=True, help="Query FASTA file")
    parser.add_argument("--database", required=True, help="Database FASTA file")

    args = parser.parse_args()

    mean_score = compute_mean_normalized_bitscore(args.query, args.database)
    print("Mean normalized bitscore:", mean_score)
