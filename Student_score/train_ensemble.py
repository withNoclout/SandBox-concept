import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Load Data
print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# --- Preprocessing for XGB/LGBM (Need Numerical Inputs) ---
# CatBoost handles categories, but XGB/LGBM need encoding.
# We will use Label Encoding for simplicity and tree-compatibility.

cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

# Combine for consistent encoding
all_data = pd.concat([train, test], axis=0).reset_index(drop=True)

for col in cat_cols:
    le = LabelEncoder()
    # Fill NaNs with 'Missing' before encoding
    all_data[col] = all_data[col].fillna('Missing').astype(str)
    all_data[col] = le.fit_transform(all_data[col])

train_enc = all_data.iloc[:len(train)]
test_enc = all_data.iloc[len(train):]

X = train_enc.drop(['id', 'exam_score'], axis=1)
y = train_enc['exam_score']
X_test = test_enc.drop(['id', 'exam_score'], axis=1)

# --- Model Definitions ---

# 1. CatBoost (Strong Manual Params)
cb_params = {
    'iterations': 1500,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'verbose': 0,
    'random_seed': 42,
    'allow_writing_files': False
}

# 2. XGBoost
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'random_state': 42,
    'enable_categorical': True # XGBoost now supports this, but we encoded anyway
}

# 3. LightGBM
lgbm_params = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}

# --- Training & Ensembling ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_cb = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))
oof_preds_lgbm = np.zeros(len(X))

test_preds_cb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_lgbm = np.zeros(len(X_test))

print(f"Starting {kf.get_n_splits()}-Fold Ensemble Training...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # --- CatBoost ---
    cb = CatBoostRegressor(**cb_params)
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
    oof_preds_cb[val_idx] = cb.predict(X_val)
    test_preds_cb += cb.predict(X_test) / kf.get_n_splits()
    
    # --- XGBoost ---
    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_preds_xgb[val_idx] = xgb.predict(X_val)
    test_preds_xgb += xgb.predict(X_test) / kf.get_n_splits()
    
    # --- LightGBM ---
    lgbm = LGBMRegressor(**lgbm_params)
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[]) # callbacks empty to suppress verbose
    oof_preds_lgbm[val_idx] = lgbm.predict(X_val)
    test_preds_lgbm += lgbm.predict(X_test) / kf.get_n_splits()
    
    print(f"Fold {fold+1} Done.")

# --- Evaluation ---
rmse_cb = np.sqrt(mean_squared_error(y, oof_preds_cb))
rmse_xgb = np.sqrt(mean_squared_error(y, oof_preds_xgb))
rmse_lgbm = np.sqrt(mean_squared_error(y, oof_preds_lgbm))

print(f"\nCatBoost RMSE: {rmse_cb:.4f}")
print(f"XGBoost RMSE:  {rmse_xgb:.4f}")
print(f"LightGBM RMSE: {rmse_lgbm:.4f}")

# --- Blending ---
# Simple average for now, or weighted if one is clearly better
# Let's try equal weights first
blend_preds = (oof_preds_cb + oof_preds_xgb + oof_preds_lgbm) / 3
rmse_blend = np.sqrt(mean_squared_error(y, blend_preds))

print(f"\nEnsemble (Blend) RMSE: {rmse_blend:.4f}")
print(f"Baseline RMSE: 8.7888")
print(f"Improvement: {8.7888 - rmse_blend:.4f}")

# --- Submission ---
final_test_preds = (test_preds_cb + test_preds_xgb + test_preds_lgbm) / 3
submission['exam_score'] = final_test_preds
submission.to_csv('submission_ensemble.csv', index=False)
print("Submission file 'submission_ensemble.csv' created successfully.")
