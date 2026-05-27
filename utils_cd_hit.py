from Bio import SeqIO
import subprocess
from itertools import groupby
import shlex
from utils import read_fasta_file


def run_cdhit(cd_hit_path, input_path, output_path, threshold, vocab_size=5, verbose=True):
    """Run CD-HIT with the given parameters."""
    cmd = f"{shlex.quote(cd_hit_path)} -i {shlex.quote(input_path)} -o {shlex.quote(output_path)} -c {threshold} -n {vocab_size}"
    
    if verbose:
        # Show CD-HIT output in console
        subprocess.run(cmd, shell=True, check=True)
    else:
        # Suppress CD-HIT output
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        
def get_clusters(output_path, filter_small_clusters=True):
    """Parse CD-HIT cluster output and return clusters."""
    output_cluster_path = output_path + '.clstr'
    
    clusters = []
    with open(output_cluster_path, 'r') as f:
        for _, group in groupby(f, lambda line: line.startswith('>')):
            cluster = list(group)
            if not cluster[0].startswith('>'):  # Ignore headers
                clusters.append(cluster)

    # Filter clusters smaller than 2 sequences
    return [c for c in clusters if len(c) > 1] if filter_small_clusters else clusters


def get_cluster_representatives(clusters):
    """Extract representative sequences from clusters."""
    return [
        seq.split('>')[1].split('...')[0]
        for cluster in clusters
        for seq in cluster if '*' in seq
    ]


def get_representative_sequences(representatives, output_path):
    """Retrieve representative sequences from the FASTA file."""
    bio_sequences = {record.id: str(record.seq) for record in SeqIO.parse(output_path, 'fasta')}

    # Ensure all representatives exist in the sequence dictionary
    return [bio_sequences.get(rep, f"Missing: {rep}") for rep in representatives]


def get_cd_coverage(
    cd_hit_path,
    sequences_path,
    output_path,
    threshold,
    vocab_size=5,
    filter_small_clusters=False,
    verbose=False,
):
    run_cdhit(
        cd_hit_path, sequences_path, output_path, threshold, vocab_size, verbose
    )

    clusters = get_clusters(output_path, filter_small_clusters)
    representatives = get_cluster_representatives(clusters)
    sequences = get_representative_sequences(representatives, output_path)

    original_number_of_sequences = len(read_fasta_file(sequences_path))
    after_clustering_number_of_sequences = len(sequences)
    coverage = after_clustering_number_of_sequences / original_number_of_sequences

    if verbose:
        print("Coverage:", coverage)

    return coverage
    