import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from IPython.display import display
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from Bio import SeqIO
import xgboost as xgb
import glob
import os
from sklearn.ensemble import RandomForestClassifier


df_AntiBac = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/AntiBacterial.csv')
df_AntiVir = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antiviral.csv')
df_AntiDiab = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antidiabetic.csv')
df_AntiOxi = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antioxidant.csv')
df_AntiParas = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antiparasitic.csv')
df_AntiMic = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antimicrobial.csv')
df_AntiInfla = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antiinflamatory.csv')
df_AntiFun = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antifungal.csv')
df_AntiCan = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/AntiCancer.csv')
df_Neurotoxin = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Neurotoxin.csv')

#Put everything into a dataframe
dfs = []
for filepath in glob.glob('*.csv'):  # adjust folder path if needed
    temp = pd.read_csv(filepath)
    activity_name = os.path.splitext(os.path.basename(filepath))[0]  # e.g. "Antiviral"
    temp['activity'] = activity_name
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

#print(df.shape)
#print(df['activity'].value_counts())
#print(df.head())

# Check for duplicate sequences
#print(df['sequence'].duplicated().sum())

# See how many unique activities (should be 10)
#print(df['activity'].nunique())
#print(df['activity'].unique())

# ============================================================
# FEATURE ENGINEERING
# ============================================================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

HYDROPHOBICITY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

POSITIVE_RESIDUES = set("KR")
NEGATIVE_RESIDUES = set("DE")

def sequence_to_features(seq):

    seq = "".join(aa for aa in seq if aa in AMINO_ACIDS)
    length = len(seq)
    if length == 0:
        return [0.0] * 24
    
    # 20 amino acid composition
    composition = [seq.count(aa) / length for aa in AMINO_ACIDS]

    # Net charge
    net_charge = (sum(seq.count(aa) for aa in POSITIVE_RESIDUES) - sum(seq.count(aa) for aa in NEGATIVE_RESIDUES))

    # Mean hydrophobicity
    mean_hydrophobicity = (sum(HYDROPHOBICITY[aa] for aa in seq) / length)

    # Positive residue fraction
    positive_fraction = (sum(seq.count(aa) for aa in POSITIVE_RESIDUES) / length)

    return composition + [
        length,
        net_charge,
        mean_hydrophobicity,
        positive_fraction
    ]

FEATURE_NAMES = [f"frac_{aa}" for aa in AMINO_ACIDS] + ["length", "net_charge", "mean_hydrophobicity", "positive_fraction"]

def build_feature_matrix(sequences):
    return pd.DataFrame([sequence_to_features(seq) for seq in sequences], columns=FEATURE_NAMES)


# ============================================================
# TRAIN + GRID SEARCH
# ============================================================

def train_and_compare(df):
    results = []
    for activity in sorted(df["activity"].unique()):
        print("\n" + "=" * 60)
        print(f"Activity: {activity}")
        print("=" * 60)

        # ----------------------------------------------------
        # Get only this activity
        # ----------------------------------------------------

        sub = df[df["activity"] == activity]
        X = build_feature_matrix(sub["sequence"])
        y = sub["label"]

        # ----------------------------------------------------
        # 80/20 stratified split
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # ----------------------------------------------------
        # 5-fold Stratified CV
        # ----------------------------------------------------

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        # ====================================================
        # RANDOM FOREST
        # ====================================================

        rf = RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        rf_grid = {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }

        rf_search = GridSearchCV(
            rf,
            rf_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )

        rf_search.fit(X_train, y_train)
        rf_best = rf_search.best_estimator_
        rf_pred = rf_best.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)

        # ====================================================
        # XGBOOST
        # ====================================================

        xgb = XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

        xgb_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1]
        }

        xgb_search = GridSearchCV(
            xgb,
            xgb_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )

        xgb_search.fit(X_train, y_train)
        xgb_best = xgb_search.best_estimator_
        xgb_pred = xgb_best.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)

        # ====================================================
        # PRINT RESULT
        # ====================================================

        print(f"Random Forest accuracy : {rf_accuracy:.4f}")
        print(f"XGBoost accuracy       : {xgb_accuracy:.4f}")
        print(f"Best RF parameters     : {rf_search.best_params_}")
        print(f"Best XGBoost parameters: {xgb_search.best_params_}")

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({
            "activity": activity,
            "RF_accuracy": rf_accuracy,
            "XGBoost_accuracy": xgb_accuracy
        })

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(results)
    print("\n\n")
    print("=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    print(
        results_df.to_string(index=False)
    )

    # Average accuracy
    print("\nAverage accuracy:")
    print(f"Random Forest : {results_df['RF_accuracy'].mean():.4f}")
    print(f"XGBoost       : {results_df['XGBoost_accuracy'].mean():.4f}")

    return results_df


print("\nTraining models...")

results = train_and_compare(df)

# Save comparison
#results.to_csv("RF_vs_XGBoost_accuracy.csv",index=False)

#print("\nResults saved to ""RF_vs_XGBoost_accuracy.csv")

#The result table:
#============================================================
#FINAL MODEL COMPARISON
#============================================================
       #activity  RF_accuracy  XGBoost_accuracy
  #AntiBacterial     0.882497          0.869732
     #AntiCancer     0.874740          0.867249
   #Antidiabetic     0.673267          0.678218
     #Antifungal     0.871218          0.862684
#Antiinflamatory     0.673190          0.652316
  #Antimicrobial     0.872320          0.856360
    #Antioxidant     0.609865          0.616592
  #Antiparasitic     0.868246          0.852702
      #Antiviral     0.820167          0.793834
     #Neurotoxin     0.768987          0.724684

#Average accuracy:
#Random Forest : 0.7914
#XGBoost       : 0.7774

#Random Forest achieved a higher average prediction accuracy (79.14%) than XGBoost (77.74%) across the ten peptide activities evaluated.
#Random Forest outperformed XGBoost in eight of the ten activities, indicating that it provided more consistent predictive performance 
#for the current sequence-derived feature set. The highest prediction accuracies were observed for antibacterial, anticancer, antifungal,
#antimicrobial, and antiparasitic activities, with accuracies exceeding 86% using Random Forest.
#In contrast, antioxidant, antidiabetic, and anti-inflammatory activities showed comparatively lower accuracies, 
#suggesting that these activities may be more difficult to distinguish based on the current features. Overall,
#the results indicate that Random Forest is the more suitable model for the current dataset, although additional feature engineering
#and evaluation metrics would be useful to further assess model performance.