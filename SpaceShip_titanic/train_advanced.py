#!/usr/bin/env python3
"""train_advanced.py

Advanced training pipeline for the Spaceship Titanic competition.
Goal: improve validation CV by using:
  * StratifiedGroupKFold (group = GroupId) to avoid family leakage
  * Domain‑logic imputation (CryoSleep → spending = 0)
  * KNN imputer for numeric spending columns
  * Target encoding (HomePlanet, Destination, Deck, Side) with CV
  * Frequency encoding for high‑cardinality Name
  * LightGBM, XGBoost and CatBoost models
  * Simple stacking (Ridge meta‑learner) on out‑of‑fold predictions

The script saves `submission_advanced.csv` in the project root.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss
# Removed category_encoders import; will use OrdinalEncoder from sklearn

# CatBoost, LightGBM, XGBoost imports
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
import category_encoders as ce

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
train_path = os.path.join(os.path.dirname(__file__), "train.csv")
test_path = os.path.join(os.path.dirname(__file__), "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

y = train_df["Transported"].astype(int)
train_df = train_df.drop(columns=["Transported"])

# ---------------------------------------------------------------------------
# 2. Basic feature engineering (same as previous scripts)
# ---------------------------------------------------------------------------

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Extract group information from PassengerId (e.g., "0013_01" -> group "0013")
    df["GroupId"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
    # Family size based on Name (simple heuristic: count commas)
    df["FamilySize"] = df["Name"].fillna("").apply(lambda x: x.count(",") + 1)
    # IsAlone flag
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    # Deck, Side, CabinNum from Cabin (if present)
    def parse_cabin(c):
        if pd.isna(c):
            return pd.Series([np.nan, np.nan, np.nan])
        parts = c.split('/')
        deck = parts[0] if len(parts) > 0 else np.nan
        side = parts[1] if len(parts) > 1 else np.nan
        num = parts[2] if len(parts) > 2 else np.nan
        return pd.Series([deck, side, num])
    df[["Deck", "Side", "CabinNum"]] = df["Cabin"].apply(parse_cabin)
    return df

train_df = add_basic_features(train_df)
test_df = add_basic_features(test_df)

# ---------------------------------------------------------------------------
# 3. Domain‑logic imputation
# ---------------------------------------------------------------------------
spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
# If CryoSleep == True, all spending must be 0
train_df.loc[train_df["CryoSleep"] == True, spending_cols] = 0
test_df.loc[test_df["CryoSleep"] == True, spending_cols] = 0

# ---------------------------------------------------------------------------
# 4. KNN imputation for numeric spending columns (after domain logic)
# ---------------------------------------------------------------------------
numeric_cols = ["Age"] + spending_cols + ["CabinNum"]
# Convert CabinNum to numeric (it may be string like "1")
train_df["CabinNum"] = pd.to_numeric(train_df["CabinNum"], errors="coerce")
test_df["CabinNum"] = pd.to_numeric(test_df["CabinNum"], errors="coerce")
# Simple median imputation for numeric columns (Age and spending features)
for col in numeric_cols:
    median = train_df[col].median()
    train_df[col] = train_df[col].fillna(median)
    test_df[col] = test_df[col].fillna(median)


# Fill remaining categorical missing values with mode
cat_cols = ["HomePlanet", "CryoSleep", "Destination", "Deck", "Side", "VIP"]
for col in cat_cols:
    mode = train_df[col].mode()[0]
    train_df[col].fillna(mode, inplace=True)
    test_df[col].fillna(mode, inplace=True)

# ---------------------------------------------------------------------------
# 5. Target Encoding (CV to avoid leakage)
# ---------------------------------------------------------------------------
# Encode categorical columns using OrdinalEncoder (handles unseen categories)
# Target encoding with smoothing (CV‑safe)
te_cols = ["HomePlanet", "Destination", "Deck", "Side"]
te = ce.TargetEncoder(cols=te_cols, smoothing=1.0)
te.fit(train_df, train_df["Transported"])  # use target column for encoding
train_df = te.transform(train_df)
test_df = te.transform(test_df)

# Encode remaining binary/categorical columns with OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder
cat_cols = ["CryoSleep", "VIP"]
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_df[cat_cols] = ord_enc.fit_transform(train_df[cat_cols])
test_df[cat_cols] = ord_enc.transform(test_df[cat_cols])

# ---------------------------------------------------------------------------
# 6. Frequency encoding for Name (high cardinality)
# ---------------------------------------------------------------------------
name_freq = train_df["Name"].value_counts()
train_df["NameFreq"] = train_df["Name"].map(name_freq)
test_df["NameFreq"] = test_df["Name"].map(name_freq).fillna(0)
# Drop raw Name (too noisy)
train_df = train_df.drop(columns=["Name"])
test_df = test_df.drop(columns=["Name"])

# ---------------------------------------------------
# Pseudo‑labeling: add high‑confidence test predictions to training data
# ---------------------------------------------------
# Train a quick LightGBM model on current data
lgb_params_ps = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
}
ps_lgb = lgb.train(lgb_params_ps, lgb.Dataset(X, label=y), num_boost_round=500)
test_pred_ps = ps_lgb.predict(X_test)
# Select high‑confidence predictions (prob > 0.9 or < 0.1)
mask = (test_pred_ps > 0.9) | (test_pred_ps < 0.1)
pseudo_labels = (test_pred_ps[mask] > 0.5).astype(int)
X_pseudo = X_test[mask]
y_pseudo = pd.Series(pseudo_labels, index=X_pseudo.index)
# Augment training set
X = pd.concat([X, X_pseudo], axis=0)
y = pd.concat([y, y_pseudo], axis=0)

# ---------------------------------------------------------------------------
# 7. Prepare data for models
# ---------------------------------------------------------------------------
# Identify categorical columns for CatBoost (bools are treated as categorical)
cat_features = ["HomePlanet", "CryoSleep", "Destination", "Deck", "Side", "VIP"]

X = train_df.copy()
X_test = test_df.copy()

# ---------------------------------------------------------------------------
# 8. StratifiedGroupKFold (group = GroupId)
# ---------------------------------------------------------------------------
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Containers for out‑of‑fold predictions
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

# Store test predictions for averaging
preds_lgb = np.zeros(len(X_test))
preds_xgb = np.zeros(len(X_test))
preds_cat = np.zeros(len(X_test))

# LightGBM parameters (tuned from earlier runs)
lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.048,
    "num_leaves": 23,
    "max_depth": 9,
    "feature_fraction": 0.95,
    "bagging_fraction": 0.68,
    "verbosity": -1,
    "seed": 42,
}

# XGBoost parameters (tuned)
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.045,
    "max_depth": 4,
    "subsample": 0.77,
    "colsample_bytree": 0.82,
    "reg_alpha": 0.02,
    "reg_lambda": 0.40,
    "seed": 42,
    "verbosity": 0,
}

# CatBoost parameters (tuned)
cat_params = {
    "iterations": 1000,
    "learning_rate": 0.041,
    "depth": 6,
    "l2_leaf_reg": 2.15,
    "border_count": 212,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "random_seed": 42,
    "early_stopping_rounds": 100,
    "verbose": False,
}

for fold, (train_idx, val_idx) in enumerate(
    sgkf.split(X, y, groups=X["GroupId"])):
    print(f"--- Fold {fold+1}/5 ---")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # LightGBM
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=5000,
    valid_sets=[lgb_val],
    verbose_eval=False,
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
)
    oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    preds_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / sgkf.n_splits

    # XGBoost
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=5000,
        evals=[(dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    oof_xgb[val_idx] = xgb_model.predict(dval, ntree_limit=xgb_model.best_ntree_limit)
    preds_xgb += xgb_model.predict(dtest, ntree_limit=xgb_model.best_ntree_limit) / sgkf.n_splits

    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True,
        verbose=False,
    )
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    preds_cat += cat_model.predict_proba(X_test)[:, 1] / sgkf.n_splits

# ---------------------------------------------------------------------------
# 9. Stacking meta‑learner (Ridge)
# ---------------------------------------------------------------------------
stack_train = np.vstack([oof_lgb, oof_xgb, oof_cat]).T
stack_test = np.vstack([preds_lgb, preds_xgb, preds_cat]).T

meta = Ridge(alpha=1.0, random_state=42)
meta.fit(stack_train, y)
final_pred = meta.predict(stack_test)
# Clip to [0,1]
final_pred = np.clip(final_pred, 0, 1)

# ---------------------------------------------------------------------------
# 10. Save submission
# ---------------------------------------------------------------------------
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Transported": final_pred})
submission_path = os.path.join(os.path.dirname(__file__), "submission_advanced.csv")
submission.to_csv(submission_path, index=False)
print(f"✅ Saved submission to {submission_path}")

# Optional: print CV log‑loss for reference
print("\nCV LogLoss (average over folds):")
print("LGB", log_loss(y, oof_lgb))
print("XGB", log_loss(y, oof_xgb))
print("CAT", log_loss(y, oof_cat))
print("Stacked", log_loss(y, meta.predict(stack_train)))
