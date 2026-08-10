import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
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

hydrophobicity = {
    "I": 4.5, "V": 4.2, "L": 3.8,
    "F": 2.8, "C": 2.5, "M": 1.9,
    "A": 1.8, "G": -0.4, "T": -0.7,
    "S": -0.8, "W": -0.9, "Y": -1.3,
    "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5,
    "K": -3.9, "R": -4.5
}

amino_acids = [
    "A", "R", "N", "D", "C",
    "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V"
]

RandomState = 42

def GetCharge(seq):
    positive = seq.count("K") + seq.count("R")
    negative = seq.count("D") + seq.count("E")
    return positive - negative

def GetHydropho(seq):
    Temp = [hydrophobicity[aa] for aa in seq]
    return sum(Temp)/len(Temp)

def GetLen(seq):
    return len(seq)

def Get_aa_composition(seq):
    length = len(seq)
    return {f"{aa}_fraction": seq.count(aa) / length for aa in amino_acids}

#Take the sequences and label from FASTA file, form a dataframe from the data
fasta_file = "C:/Users/Matthew/Downloads/AMPProj1/AMPBenchmark_public.fasta"

data = []

for record in SeqIO.parse(fasta_file, "fasta"):
    sequence = str(record.seq)

    label = 1 if "AMP=1" in record.id else 0

    data.append({
        "id": record.id,
        "sequence": sequence,
        "label": label
    })

df = pd.DataFrame(data)
#Add features into the dataframe, seq length, hydrophobicity, Charge
df["seq_length"] = df["sequence"].apply(GetLen)
df["charge"] = df["sequence"].apply(GetCharge)
df["hydrophobicity"] = df["sequence"].apply(GetHydropho)
aa_composition = df["sequence"].apply(Get_aa_composition)
# Convert dictionaries into columns
aa_composition_df = pd.DataFrame(aa_composition.tolist(), index=df.index)
# Add to original dataframe
df = pd.concat([df, aa_composition_df], axis=1)

#print(df.head())
#print(df.shape)

#Check for duplicate seq
#print(df['sequence'].duplicated().sum())
#print(df['label'].value_counts())

#We got 16067 duplicates, kinda lot
df = df.drop_duplicates(subset='sequence').reset_index(drop=True)
#print("After erasing the duplicates")
#print(df['sequence'].duplicated().sum())
#print(df['label'].value_counts())

#We split the data into x and y
X = df.drop(columns=["id", "sequence", "label"])
Y = df["label"]

#The data is significantly consist of Non-amp labeled data, as i cant find any dataset that has balanced composition.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=RandomState
)

print(X_train.head())
print(y_train.head())

def build_model(**overrides):

    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        verbosity=0,
        random_state=RandomState
    )

    params.update(overrides)

    return xgb.XGBClassifier(**params)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RandomState)
param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}
#param_grid = {
    #"n_estimators": [200, 300],
    #"max_depth": [3, 5, 7],
    #"learning_rate": [0.05, 0.1],
    #"subsample": [0.8, 1.0],
    #"colsample_bytree": [0.8, 1.0]
#}
search = GridSearchCV(
    build_model(),
    param_grid,
    cv=skf,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
    refit=True
)
search.fit(X_train, y_train)
print("Best parameters:")
print(search.best_params_)
print("\nBest CV ROC-AUC:")
print(search.best_score_)
best_model = search.best_estimator_

#Best parameter is:
#{'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 500, 'subsample': 0.8}
#Best CV ROC-AUC: 0.9515416749398099
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

#Below is the table
              #precision    recall  f1-score   support

           #0       0.99      1.00      1.00     22275
           #1       0.88      0.37      0.52       208

    #accuracy                           0.99     22483
   #macro avg       0.93      0.68      0.76     22483
#weighted avg       0.99      0.99      0.99     22483

#The recall score for AMP is low only 0.37
#the model is only catching 37% of actual AMPs. It's missing 63% of the peptides you actually care about identifying.