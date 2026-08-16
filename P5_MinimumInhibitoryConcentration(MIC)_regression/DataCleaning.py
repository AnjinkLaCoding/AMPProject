"""
Full cleaning pipeline for the multi-organism AMP MIC dataset.

Starting from the raw GRAMPA export (grampa.csv), this applies, in order:
  1. Remove chemically modified peptides / non-standard amino acids
  2. Filter to a reasonable peptide length range
  3. Consolidate organism-name typos via fuzzy matching (e.g. 'S. aureuss' -> 'S. aureus')
  4. Aggregate repeat measurements per (sequence, organism), computing mean + std
  5. Drop sequence-organism pairs with high measurement disagreement (unreliable labels)
  6. Filter to well-represented organisms only
"""
import pandas as pd
from difflib import SequenceMatcher

RAW_FILE = "grampa.csv"
#OUTPUT_FILE = "multi_organism_mic_filtered.csv"

MIN_LEN = 5
MAX_LEN = 50
DISAGREEMENT_STD_THRESHOLD = 1.0   # drop pairs where repeat measurements disagree by more than this (log10 units)
TYPO_MATCH_THRESHOLD = 0.85        # similarity ratio required to merge a name into a canonical one
CANONICAL_MIN_COUNT = 100          # only names with at least this many records are eligible to BE a canonical name
FINAL_MIN_RECORDS = 250            # organisms kept in the final filtered file need at least this many rows


def consolidate_organism_typos(df, threshold=TYPO_MATCH_THRESHOLD, canonical_min_count=CANONICAL_MIN_COUNT):
    """Fuzzy-match rare organism labels onto a well-represented canonical name
    of the same genus, e.g. 'S. aureuss' -> 'S. aureus'. Distinct species are
    not merged, since matching is restricted to the same genus prefix."""
    counts = df.bacterium.value_counts()
    canonical_pool = counts[counts >= canonical_min_count].index.tolist()

    def best_match(name):
        if name in canonical_pool:
            return name
        genus = name.split(".")[0]
        candidates = [c for c in canonical_pool if c.split(".")[0] == genus]
        best, best_score = name, 0
        for c in candidates:
            score = SequenceMatcher(None, name.lower(), c.lower()).ratio()
            if score > best_score:
                best, best_score = c, score
        return best if best_score >= threshold else name

    mapping = {name: best_match(name) for name in counts.index}
    n_merged = sum(1 for k, v in mapping.items() if k != v)
    print(f"  Merged {n_merged} typo/near-duplicate organism labels into canonical names")

    df = df.copy()
    df["bacterium"] = df["bacterium"].map(mapping)
    return df


def CleanFile(df):
    df = df[df.is_modified == False]
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    df = df[df.sequence.apply(lambda s: set(str(s)).issubset(valid_aa))]
    print(f"After removing modified/non-standard-AA peptides: {len(df)} records")

    # ----Length filter ----
    seqlen = df.sequence.str.len()
    before = len(df)
    df = df[(seqlen >= MIN_LEN) & (seqlen <= MAX_LEN)]
    print(f"After length filter ({MIN_LEN}-{MAX_LEN} aa): {len(df)} records (dropped {before - len(df)})")

    # ----Organism typo consolidation ----
    print("Consolidating organism name typos...")
    df = consolidate_organism_typos(df)

    # ----Aggregate repeat measurements, keeping std for disagreement filtering ----
    agg = df.groupby(["sequence", "bacterium"], as_index=False).agg(
        pMIC=("value", "mean"),
        pMIC_std=("value", "std"),
        n_measurements=("value", "count"),
    )
    agg["pMIC_std"] = agg["pMIC_std"].fillna(0.0)  # single-measurement pairs have no std
    print(f"After aggregating repeat measurements: {len(agg)} unique (sequence, organism) pairs")

    # ----Drop high-disagreement pairs ----
    before = len(agg)
    agg = agg[agg.pMIC_std <= DISAGREEMENT_STD_THRESHOLD]
    print(f"After dropping high-disagreement pairs (std > {DISAGREEMENT_STD_THRESHOLD}): "
          f"{len(agg)} records (dropped {before - len(agg)})")

    # ----Filter to well-represented organisms ----
    counts = agg.bacterium.value_counts()
    keep_organisms = counts[counts >= FINAL_MIN_RECORDS].index
    final = agg[agg.bacterium.isin(keep_organisms)].copy()

    print(f"\nKept {len(keep_organisms)} organisms with >= {FINAL_MIN_RECORDS} records:")
    print(final.bacterium.value_counts())
    print(f"\nFinal filtered dataset: {len(final)} rows")

    #final = final.drop(columns=["pMIC_std"])  # not needed downstream, was only for filtering
    #final.to_csv(OUTPUT_FILE, index=False)
    #print(f"\nSaved to {OUTPUT_FILE}")

    return final