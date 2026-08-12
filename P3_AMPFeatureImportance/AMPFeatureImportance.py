import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
import glob
import os

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

# ============================================================
# FEATURES Engineering
# ============================================================
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
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

def calculate_features(sequence):
    length = len(sequence)

    # --------------------------------
    # Amino acid composition
    # --------------------------------

    features = {}
    for aa in amino_acids:
        features[f"{aa}_fraction"] = sequence.count(aa) / length

    # --------------------------------
    # Sequence length
    # --------------------------------

    features["seq_length"] = length

    # --------------------------------
    # Charge
    # --------------------------------

    positive = sequence.count("K") + sequence.count("R")
    negative = sequence.count("D") + sequence.count("E")
    features["charge"] = positive - negative

    # --------------------------------
    # Hydrophobicity
    # --------------------------------

    mean_hydrophobicity = (sum(HYDROPHOBICITY[aa] for aa in sequence) / length)
    features["hydrophobicity"] = mean_hydrophobicity

    return features

feature_df = df["sequence"].apply(calculate_features).apply(pd.Series)
df = pd.concat([df[["sequence", "label", "activity"]],feature_df], axis=1)
print(df.head())

# ============================================================
# PREPARE FEATURES
# ============================================================

drop_columns = [
    "sequence",
    "label",
    "activity"
]
# X = numerical sequence features
X = df.drop(columns=drop_columns).copy()
# y -> activity we want to predict
y = df["activity"].copy()
print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# CHECK FOR MISSING VALUES
# ============================================================

print("\nMissing values:")
missing = X.isnull().sum()
print(missing[missing > 0])
# If there are no missing values, this does nothing
X = X.fillna(0)


# ============================================================
# ENCODE ACTIVITY LABELS
# ============================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nActivity encoding:")
for number, activity in enumerate(label_encoder.classes_):
    print(number, "=", activity)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# DEFINE MODELS
# ============================================================

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


xgb = XGBClassifier(
    n_estimators=300,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1
)


# ============================================================
# STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)


# -----------------------------
# Random Forest CV
# -----------------------------

rf_cv_scores = cross_val_score(
    rf,
    X,
    y_encoded,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("\nRandom Forest CV scores:")
print(rf_cv_scores)
print("Random Forest mean accuracy:", rf_cv_scores.mean())
print("Random Forest std:", rf_cv_scores.std())


# -----------------------------
# XGBoost CV
# -----------------------------

xgb_cv_scores = cross_val_score(
    xgb,
    X,
    y_encoded,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("\nXGBoost CV scores:")
print(xgb_cv_scores)
print("XGBoost mean accuracy:", xgb_cv_scores.mean())
print("XGBoost std:", xgb_cv_scores.std())


# ============================================================
# COMPARE MODELS
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"Random Forest : {rf_cv_scores.mean():.4f}")
print(f"XGBoost       : {xgb_cv_scores.mean():.4f}")

if rf_cv_scores.mean() > xgb_cv_scores.mean():
    print("\nRandom Forest performs better.")
    best_model = rf
    best_model_name = "Random Forest"
else:
    print("\nXGBoost performs better.")
    best_model = xgb
    best_model_name = "XGBoost"


# ============================================================
# TRAIN BOTH MODELS ON TRAINING DATA
# ============================================================

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)


# ============================================================
# TEST SET PERFORMANCE
# ============================================================

rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
xgb_accuracy = accuracy_score(y_test, xgb_pred)


print("\n" + "=" * 60)
print("TEST SET ACCURACY")
print("=" * 60)
print(f"Random Forest : {rf_accuracy:.4f}")
print(f"XGBoost       : {xgb_accuracy:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test,rf_pred,target_names=label_encoder.classes_))


print("\n" + "=" * 60)
print("XGBOOST CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test,xgb_pred,target_names=label_encoder.classes_))


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_importance = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})

rf_importance = rf_importance.sort_values("importance", ascending=False)

print("\n" + "=" * 60)
print("RANDOM FOREST TOP FEATURES")
print("=" * 60)
print(rf_importance.head(20).to_string(index=False))


# ============================================================
# XGBOOST FEATURE IMPORTANCE
# ============================================================

xgb_importance = pd.DataFrame({"feature": X.columns, "importance": xgb.feature_importances_})

xgb_importance = xgb_importance.sort_values("importance", ascending=False)

print("\n" + "=" * 60)
print("XGBOOST TOP FEATURES")
print("=" * 60)
print(xgb_importance.head(20).to_string(index=False))


# ============================================================
# COMPARE FEATURE IMPORTANCE
# ============================================================

importance_comparison = pd.merge(
    rf_importance.rename(columns={"importance": "RF_importance"}),
    xgb_importance.rename(columns={"importance": "XGBoost_importance"}),
    on="feature"
)

importance_comparison["average_importance"] = (importance_comparison["RF_importance"] + importance_comparison["XGBoost_importance"]) / 2


importance_comparison = importance_comparison.sort_values("average_importance", ascending=False)
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE COMPARISON")
print("=" * 60)
print(importance_comparison.head(20).to_string(index=False))


# ============================================================
# PLOT RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

top_rf = rf_importance.head(15)

plt.figure(figsize=(9, 6))

plt.barh(top_rf["feature"], top_rf["importance"])

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest - Top 15 Features")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# ============================================================
# PLOT XGBOOST FEATURE IMPORTANCE
# ============================================================

top_xgb = xgb_importance.head(15)

plt.figure(figsize=(9, 6))

plt.barh(top_xgb["feature"], top_xgb["importance"])

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost - Top 15 Features")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# ============================================================
# TRAIN BEST MODEL FOR INTERPRETABILITY
# ============================================================

if best_model_name == "Random Forest":
    best_model.fit(X_train, y_train)
else:
    best_model.fit(X_train, y_train)

print("\nModel selected for interpretation:", best_model_name)


# ============================================================
# SHAP INTERPRETABILITY
# ============================================================

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
class_names = label_encoder.classes_
print("Classes:")
print(class_names)


# ============================================================
# SHAP plot for each activity
# ============================================================

for i, activity in enumerate(class_names):
    shap.summary_plot(
        shap_values[:, :, i],
        X_test,
        feature_names=X.columns,
        show=False
    )
    plt.title(f"SHAP Feature Importance - {activity}")
    plt.tight_layout()
    plt.show()


# ============================================================
# SAVE FEATURE IMPORTANCE RESULTS
# ============================================================

#importance_comparison.to_csv("RF_vs_XGBoost_feature_importance.csv", index=False)

#rf_importance.to_csv("RF_feature_importance.csv", index=False)

#xgb_importance.to_csv("XGBoost_feature_importance.csv", index=False)

#print("\nResults saved:")
#print("RF_vs_XGBoost_feature_importance.csv")
#print("RF_feature_importance.csv")
#print("XGBoost_feature_importance.csv")

#Result comparison RF and XgBoost
#============================================================
#FEATURE IMPORTANCE COMPARISON
#============================================================
#       feature  RF_importance  XGBoost_importance  average_importance
#    seq_length       0.071194            0.182816            0.127005
#        charge       0.053101            0.141481            0.097291
#hydrophobicity       0.097709            0.040096            0.068902
#    L_fraction       0.055120            0.031189            0.043154
#    R_fraction       0.047227            0.034312            0.040770
#    K_fraction       0.049629            0.031850            0.040740
#    A_fraction       0.047271            0.029803            0.038537
#    V_fraction       0.046270            0.030429            0.038349
#    G_fraction       0.045086            0.031083            0.038085
#    I_fraction       0.045450            0.029594            0.037522
#    S_fraction       0.044168            0.028963            0.036565
#    M_fraction       0.033686            0.039021            0.036354
#    C_fraction       0.030081            0.042501            0.036291
#    P_fraction       0.039774            0.031782            0.035778
#    F_fraction       0.042185            0.027648            0.034916
#    T_fraction       0.038588            0.029712            0.034150
#    N_fraction       0.033607            0.031038            0.032322
#    W_fraction       0.027913            0.035135            0.031524
#    Q_fraction       0.032450            0.029781            0.031116
#    E_fraction       0.031068            0.030999            0.031034