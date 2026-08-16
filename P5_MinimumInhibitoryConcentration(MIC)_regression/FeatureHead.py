import numpy as np
import pandas as pd
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
 
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
 

"""Amino acid composition: fraction of each of the 20 standard AAs."""
def aac_features(sequences):
    rows = []
    for seq in sequences:
        L = len(seq)
        rows.append([seq.count(a) / L for a in AA_ORDER])
    cols = [f"AAC_{a}" for a in AA_ORDER]
    return pd.DataFrame(rows, columns=cols)
 

"""modlAMP global physicochemical descriptors."""
def global_desc_features(sequences):
    gd = GlobalDescriptor(list(sequences))
    gd.calculate_all()
    return pd.DataFrame(gd.descriptor, columns=gd.featurenames)
 

"""
Eisenberg hydrophobic moment (amphipathicity) + global hydrophobicity

Notice these describe different properties:
HydrophMoment = distribution of hydrophobic residues
HydrophGlobal = overall amount of hydrophobicity

These two features capture membrane-binding behavior better than amino acid composition alone.

Feature                         What it measures                            Why it matters for AMPs

HydrophGlobal                  Overall hydrophobicity                   Helps peptides insert into 
                                                                            lipid membranes

HydrophMoment                  Amphipathic segregation                  Enables one face to interact 
                                                                        with lipids while the other 
                                                                        faces water, a hallmark of many 
                                                                        α-helical AMPs
"""
def hydrophobic_moment_features(sequences):
    moment = PeptideDescriptor(list(sequences), "eisenberg")
    moment.calculate_moment()
    hmom = moment.descriptor.flatten()
 
    hydro = PeptideDescriptor(list(sequences), "eisenberg")
    hydro.calculate_global()
    hglob = hydro.descriptor.flatten()
 
    return pd.DataFrame({"HydrophMoment": hmom, "HydrophGlobal": hglob})
 

"""
Build the full feature matrix for a dataframe with a sequence column.
If organism_col is given, one-hot encodes it and appends to the features.
"""
def featurize(df, sequence_col="sequence", organism_col=None):
    seqs = df[sequence_col].tolist()
 
    gd_feats = global_desc_features(seqs)
    hm_feats = hydrophobic_moment_features(seqs)
    aac_feats = aac_features(seqs)
 
    X = pd.concat(
        [gd_feats.reset_index(drop=True),
         hm_feats.reset_index(drop=True),
         aac_feats.reset_index(drop=True)],
        axis=1
    )
 
    if organism_col is not None:
        organism_dummies = pd.get_dummies(df[organism_col], prefix="org")
        X = pd.concat([X.reset_index(drop=True), organism_dummies.reset_index(drop=True)], axis=1)

    print(X.head())
    print(X.shape)
    return X

#The final dataframe
#   Length       MW  Charge  ChargeDensity  ...  org_P. aeruginosa  org_S. aureus  org_S. epidermidis  org_S. typhimurium
#0    33.0  3176.71   4.030       0.001269  ...              False          False               False               False
#1    33.0  3176.71   4.030       0.001269  ...               True          False               False               False
#2    33.0  3176.71   4.030       0.001269  ...              False           True               False               False
#3    16.0  1423.67   1.989       0.001397  ...              False          False               False               False
#4    17.0  1941.29   3.679       0.001895  ...              False          False               False               False

#[5 rows x 43 columns]
#The shape: (9877, 43)
#Feature matrix: 43 features