"""
Train and evaluate a baseline MIC regression model on the multi-organism
AMP dataset (multi_organism_mic_filtered.csv), using a leakage-safe
clustered train/test split and physicochemical + AAC + organism features.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor
from FeatureHead import featurize
from ClusterSplit import clustered_train_test_split
from DataCleaning import CleanFile

RANDOM_STATE = 42

# ----Load data ----
df = pd.read_csv("C:/Users/Matthew/Downloads/AMPProj1/MICRegress/grampa.csv")
df = CleanFile(df)
print(df.head())
print(df.shape)

# ----Clustered train/test split (leakage-safe) ----
train_df, test_df = clustered_train_test_split(
    df, sequence_col="sequence", test_size=0.2,
    distance_threshold=0.7, random_state=RANDOM_STATE
)
print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
assert len(set(train_df.sequence) & set(test_df.sequence)) == 0, "Leakage detected!"

# ----Featurize ----
X_train = featurize(train_df, sequence_col="sequence", organism_col="bacterium")
X_test = featurize(test_df, sequence_col="sequence", organism_col="bacterium")
y_train = train_df["pMIC"].values
y_test = test_df["pMIC"].values
print(f"Feature matrix: {X_train.shape[1]} features")

# ----Train baseline model ----
model = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ----Evaluate: overall ----
preds = model.predict(X_test)
rmse = root_mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
rho, _ = spearmanr(y_test, preds)
print("\n=== Overall test performance (pMIC = log10 MIC in uM) ===")
print(f"RMSE:            {rmse:.3f}")
print(f"R^2:              {r2:.3f}")
print(f"Spearman rho:    {rho:.3f}")

# ----Evaluate: per organism ----
print("\n=== Per-organism test performance ===")
test_eval = test_df.copy()
test_eval["pred"] = preds
rows = []
for org, g in test_eval.groupby("bacterium"):
    if len(g) < 10:
        continue  # too few test points for a meaningful metric
    org_rmse = root_mean_squared_error(g["pMIC"], g["pred"])
    org_r2 = r2_score(g["pMIC"], g["pred"])
    org_rho, _ = spearmanr(g["pMIC"], g["pred"])
    rows.append({"organism": org, "n_test": len(g),
                      "RMSE": org_rmse, "R2": org_r2, "Spearman": org_rho})
per_org = pd.DataFrame(rows).sort_values("n_test", ascending=False)
print(per_org.to_string(index=False))

# ----Feature importance ----
importances = pd.Series(model.feature_importances_, index=X_train.columns)
print("\n=== Top 15 features by importance ===")
print(importances.sort_values(ascending=False).head(15).to_string())

# ----Save artifacts ----
#test_eval[["sequence", "bacterium", "pMIC", "pred"]].to_csv(
    #"test_predictions.csv", index=False
#)
#per_org.to_csv("per_organism_metrics.csv", index=False)
#model.save_model("mic_xgb_model.json")
#print("\nSaved: test_predictions.csv, per_organism_metrics.csv, mic_xgb_model.json")


#The results:

#=== Overall test performance (pMIC = log10 MIC in uM) ===
#RMSE:            0.691
#R^2:              0.152
#Spearman rho:    0.420

#=== Per-organism test performance ===
#      organism  n_test     RMSE        R2  Spearman
#       E. coli     627 0.694362  0.184183  0.463232
#     S. aureus     555 0.692479  0.175639  0.419340
# P. aeruginosa     253 0.624069  0.139259  0.402553
#   C. albicans     242 0.709013  0.004660  0.283869
#   B. subtilis     188 0.694834  0.135687  0.364605
#S. typhimurium     106 0.621201  0.055374  0.334711
#     M. luteus     104 0.877841 -0.117334  0.066641
#S. epidermidis      96 0.691244  0.058288  0.300258
#     B. cereus      87 0.655808  0.187533  0.536343
# K. pneumoniae      73 0.669165  0.136295  0.454839
#   E. faecalis      63 0.664849  0.173404  0.483207

#=== Top 15 features by importance ===
#MW                    0.078829
#AAC_E                 0.072869
#Charge                0.052133
#AAC_W                 0.034945
#AAC_D                 0.034222
#AAC_Q                 0.032518
#AAC_M                 0.027128
#AAC_S                 0.026463
#AAC_C                 0.026307
#AAC_Y                 0.024927
#AAC_V                 0.023945
#org_S. epidermidis    0.023906
#AAC_N                 0.023635
#ChargeDensity         0.022760
#HydrophRatio          0.022491


#obviously, need more improvement lol