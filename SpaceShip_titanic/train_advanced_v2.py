#!/usr/bin/env python3
"""train_advanced_v2.py

Enhanced training pipeline for the Spaceship Titanic competition.
Improvements:
- MICE imputation for numeric columns
- Target encoding with smoothing (category_encoders)
- Pseudo‑labeling using high‑confidence test predictions
- Stacking of LightGBM, XGBoost, CatBoost with Ridge meta‑learner
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import category_encoders as ce
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(__file__)
train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# ---------------------------------------------------------------------------
# 2. Basic feature engineering (same as previous scripts)
# ---------------------------------------------------------------------------
def add_basic_features(df):
    # GroupId from PassengerId
    df["GroupId"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
    # Family size based on Name (count commas + 1)
    df["FamilySize"] = df["Name"].fillna("").apply(lambda x: x.count(",") + 1)
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    # Parse Cabin into Deck, Side, CabinNum
    def parse_cabin(c):
        if pd.isna(c):
            return pd.Series([np.nan, np.nan, np.nan])
        parts = c.split('/')
        deck = parts[0] if len(parts) > 0 else np.nan
        side = parts[1] if len(parts) > 1 else np.nan
        num = parts[2] if len(parts) > 2 else np.nan
        return pd.Series([deck, side, num])
    df[["Deck", "Side", "CabinNum"]] = df["Cabin"].apply(parse_cabin)
    df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")
    # Spending features
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpending"] = df[spend_cols].sum(axis=1)
    df["HasSpending"] = (df["TotalSpending"] > 0).astype(int)
    df["SpendingPerAge"] = df["TotalSpending"] / df["Age"].replace(0, np.nan)
    return df

train_df = add_basic_features(train_df)
test_df = add_basic_features(test_df)

# ---------------------------------------------------------------------------
# 3. Domain‑logic imputation for CryoSleep
# ---------------------------------------------------------------------------
spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
train_df.loc[train_df["CryoSleep"] == True, spend_cols] = 0
test_df.loc[test_df["CryoSleep"] == True, spend_cols] = 0

# ---------------------------------------------------------------------------
# 4. Imputation
# ---------------------------------------------------------------------------
numeric_cols = ["Age", "CabinNum", "TotalSpending", "SpendingPerAge"]
# Filter out columns that are all NaN to prevent MICE from failing
numeric_cols = [c for c in numeric_cols if train_df[c].notna().sum() > 0]

# MICE for numeric columns
mice = IterativeImputer(max_iter=5, random_state=42)
train_df[numeric_cols] = mice.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = mice.transform(test_df[numeric_cols])

# Mode for categoricals
cat_cols = ["HomePlanet", "CryoSleep", "Destination", "Deck", "Side", "VIP"]
for col in cat_cols:
    mode = train_df[col].mode()[0]
    train_df[col] = train_df[col].fillna(mode)
    test_df[col] = test_df[col].fillna(mode)

# ---------------------------------------------------------------------------
# 5. Target Encoding with Smoothing (Manual CV implementation)
# ---------------------------------------------------------------------------
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

class TargetEncoderCV(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, n_splits=5, smooth=10):
        self.cols = cols
        self.n_splits = n_splits
        self.smooth = smooth
        self.map_ = {}
        self.global_mean_ = {}

    def fit(self, X, y):
        for col in self.cols:
            self.global_mean_[col] = y.mean()
            # Smoothed mean: (mean * count + global * smooth) / (count + smooth)
            agg = y.groupby(X[col]).agg(['mean', 'count'])
            smooth_mean = (agg['mean'] * agg['count'] + self.global_mean_[col] * self.smooth) / (agg['count'] + self.smooth)
            self.map_[col] = smooth_mean
        return self

    def transform(self, X):
        # For test set, use the fitted map
        X_out = X.copy()
        for col in self.cols:
            X_out[col] = X_out[col].map(self.map_[col]).fillna(self.global_mean_[col])
        return X_out

    def fit_transform(self, X, y):
        # For training set, use CV to avoid leakage
        self.fit(X, y) # Fit global map for later use on test
        X_out = X.copy()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for col in self.cols:
            # Initialize with global mean as fallback
            X_out[f"{col}_enc"] = np.nan
            for train_idx, val_idx in kf.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr = y.iloc[train_idx]
                
                # Compute smoothed mean on training fold
                global_mean = y_tr.mean()
                agg = y_tr.groupby(X_tr[col]).agg(['mean', 'count'])
                smooth_mean = (agg['mean'] * agg['count'] + global_mean * self.smooth) / (agg['count'] + self.smooth)
                
                # Map to validation fold
                X_out.loc[X_out.index[val_idx], f"{col}_enc"] = X_val[col].map(smooth_mean)
            
            # Fill NaNs (unseen categories in validation folds) with global mean
            X_out[f"{col}_enc"] = X_out[f"{col}_enc"].fillna(self.global_mean_[col])
            # Replace original column
            X_out[col] = X_out[f"{col}_enc"]
            X_out.drop(columns=[f"{col}_enc"], inplace=True)
            
        return X_out

te = TargetEncoderCV(cols=["HomePlanet", "Destination", "Deck", "Side"], smooth=10)
train_df = te.fit_transform(train_df, train_df["Transported"])
test_df = te.transform(test_df)

# Frequency encoding for Name (high‑cardinality)
name_counts = train_df["Name"].value_counts()
train_df["NameFreq"] = train_df["Name"].map(name_counts)
test_df["NameFreq"] = test_df["Name"].map(name_counts).fillna(0)
train_df = train_df.drop(columns=["Name"])
test_df = test_df.drop(columns=["Name"])

# Cast binary columns to int
for col in ["CryoSleep", "VIP"]:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)

# ---------------------------------------------------------------------------
# 6. Prepare matrices
# ---------------------------------------------------------------------------
y = train_df["Transported"].astype(int)

# Extract groups for CV
train_groups = train_df["GroupId"]
test_groups = test_df["GroupId"]

# Drop non-feature columns
drop_cols = ["PassengerId", "Transported", "Cabin", "GroupId"]
X = train_df.drop(columns=drop_cols)
X_test = test_df.drop(columns=[c for c in drop_cols if c != "Transported"])

# Standardize numeric columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ---------------------------------------------------------------------------
# 7. Pseudo‑labeling (high‑confidence predictions)
# ---------------------------------------------------------------------------
# Quick LightGBM model on full data
lgb_params_ps = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 42}
ps_model = lgb.train(lgb_params_ps, lgb.Dataset(X, label=y), num_boost_round=500)
test_pred_ps = ps_model.predict(X_test)
mask = (test_pred_ps > 0.9) | (test_pred_ps < 0.1)
pseudo_labels = (test_pred_ps[mask] > 0.5).astype(int)
X_pseudo = X_test[mask]
y_pseudo = pd.Series(pseudo_labels, index=X_pseudo.index)
# Augment training data
# Augment training data
X_aug = pd.concat([X, X_pseudo], axis=0)
y_aug = pd.concat([y, y_pseudo], axis=0)
# Augment groups
groups_pseudo = test_groups[mask]
groups_aug = pd.concat([train_groups, groups_pseudo], axis=0)

# ---------------------------------------------------------------------------
# 8. Stacking models
# ---------------------------------------------------------------------------
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
# Containers for OOF predictions
oof_lgb = np.zeros(len(X_aug))
oof_xgb = np.zeros(len(X_aug))
oof_cat = np.zeros(len(X_aug))
# Test predictions
preds_test = np.zeros((X_test.shape[0], 3))

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_aug, y_aug, groups=groups_aug)):
    X_tr, X_val = X_aug.iloc[train_idx], X_aug.iloc[val_idx]
    y_tr, y_val = y_aug.iloc[train_idx], y_aug.iloc[val_idx]
    # LightGBM
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    lgb_model = lgb.train(
        lgb_params_ps,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    preds_test[:, 0] += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / sgkf.n_splits
    # XGBoost
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "verbosity": 0,
    }
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=2000, evals=[(dval, "val")], early_stopping_rounds=100, verbose_eval=False)
    oof_xgb[val_idx] = xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration + 1))
    preds_test[:, 1] += xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration + 1)) / sgkf.n_splits
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=False,
    )
    # Target encoded columns are now numeric, so we don't pass them as cat_features
    # CryoSleep and VIP are binary ints, treated as numeric
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    preds_test[:, 2] += cat_model.predict_proba(X_test)[:, 1] / sgkf.n_splits

# ---------------------------------------------------------------------------
# 9. Meta‑learner (Ridge)
# ---------------------------------------------------------------------------
stack_train = np.vstack([oof_lgb, oof_xgb, oof_cat]).T
stack_test = preds_test
meta = Ridge(alpha=1.0, random_state=42)
meta.fit(stack_train, y_aug)
final_pred = meta.predict(stack_test)
final_pred = np.clip(final_pred, 0, 1)

# ---------------------------------------------------------------------------
# 10. Save submission
# ---------------------------------------------------------------------------
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Transported": final_pred})
submission_path = os.path.join(base_dir, "submission_advanced.csv")
submission.to_csv(submission_path, index=False)
print(f"✅ Saved submission to {submission_path}")
