"""
Clustered train/test split for peptide sequences.

Random splits leak information in AMP datasets because many sequences are
near-duplicate point-mutant variants of each other. This clusters sequences
by k-mer similarity (Jaccard over 3-mers) using agglomerative clustering,
then assigns whole clusters to train or test -- so test sequences are never
near-duplicates of train sequences.

This is a lightweight stand-in for CD-HIT (CD-HIT must be used in linux env).
For rigorous work, prefer CD-HIT if you have access to it locally.
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split


def kmer_set(seq, k=3):
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def jaccard_distance_matrix(sequences, k=3):
    kmers = [kmer_set(s, k) for s in sequences]
    n = len(sequences)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = kmers[i], kmers[j]
            union = len(a | b)
            inter = len(a & b)
            d = 1.0 - (inter / union if union > 0 else 0.0)
            dist[i, j] = dist[j, i] = d
    return dist


"""
Cluster unique sequences by k-mer Jaccard similarity.
distance_threshold: lower = stricter (fewer sequences per cluster,
i.e. only near-identical sequences grouped together).
Returns dict: sequence -> cluster_id
"""
def cluster_sequences(unique_sequences, distance_threshold=0.7, k=3):
    dist = jaccard_distance_matrix(unique_sequences, k=k)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist)
    return dict(zip(unique_sequences, labels))


"""
Split a dataframe into train/test by sequence cluster, so near-duplicate
or highly similar sequences never appear in both sets.
Returns (train_df, test_df)
"""
def clustered_train_test_split(df, sequence_col="sequence", test_size=0.2,
                                 distance_threshold=0.7, random_state=42):
    unique_seqs = df[sequence_col].unique().tolist()
    seq_to_cluster = cluster_sequences(unique_seqs, distance_threshold=distance_threshold)

    cluster_ids = sorted(set(seq_to_cluster.values()))
    train_clusters, test_clusters = train_test_split(
        cluster_ids, test_size=test_size, random_state=random_state
    )
    train_clusters = set(train_clusters)

    df = df.copy()
    df["_cluster"] = df[sequence_col].map(seq_to_cluster)
    train_df = df[df["_cluster"].isin(train_clusters)].drop(columns="_cluster")
    test_df = df[~df["_cluster"].isin(train_clusters)].drop(columns="_cluster")

    return train_df, test_df