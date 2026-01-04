"""
Spaceship Titanic - 4-Model Stacking Ensemble
==============================================
LightGBM + XGBoost + CatBoost + HistGradientBoosting
Meta-Learner: Logistic Regression

Kaggle Progress: 0.801 -> 0.804 -> Target: 0.81+
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
SEED = 42
N_FOLDS = 5
TARGET = 'Transported'

# --- Load Data ---
print("📊 Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test_ids = test['PassengerId'].copy()

train['is_train'] = 1
test['is_train'] = 0
test[TARGET] = np.nan
df = pd.concat([train, test], axis=0, ignore_index=True)

print(f"Combined shape: {df.shape}")

# --- Feature Engineering ---
print("🛠️ Feature Engineering...")

# 1. PassengerId features
df['GroupId'] = df['PassengerId'].str.split('_').str[0].astype(int)
df['PersonInGroup'] = df['PassengerId'].str.split('_').str[1].astype(int)
df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')
df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

# 2. Cabin features
df['Deck'] = df['Cabin'].str.split('/').str[0]
df['CabinNum'] = df['Cabin'].str.split('/').str[1].astype(float)
df['Side'] = df['Cabin'].str.split('/').str[2]

# 3. Spending features
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalSpending'] = df[spending_cols].sum(axis=1)
df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
df['SpendingPerAge'] = df['TotalSpending'] / (df['Age'] + 1)

# Log transform for spending
for col in spending_cols:
    df[f'{col}_log'] = np.log1p(df[col])
df['TotalSpending_log'] = np.log1p(df['TotalSpending'])

# 4. CryoSleep logic
for col in spending_cols:
    df.loc[df['CryoSleep'] == True, col] = 0
    df.loc[df['CryoSleep'] == True, f'{col}_log'] = 0
df.loc[df['CryoSleep'] == True, 'TotalSpending'] = 0
df.loc[df['CryoSleep'] == True, 'TotalSpending_log'] = 0

# 5. Age features
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])
df['AgeGroup'] = df['AgeGroup'].astype(float)
df['IsChild'] = (df['Age'] < 18).astype(int)

# 6. Name -> Family
df['LastName'] = df['Name'].str.split().str[-1]
df['FamilySize'] = df.groupby('LastName')['LastName'].transform('count')

# --- Handle Missing Values ---
print("🩹 Handling Missing Values...")

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 
            'TotalSpending', 'CabinNum', 'SpendingPerAge', 'AgeGroup',
            'RoomService_log', 'FoodCourt_log', 'ShoppingMall_log', 'Spa_log', 'VRDeck_log', 'TotalSpending_log']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# --- Encode Categorical Features ---
print("🔤 Encoding Categorical Features...")

label_encoders = {}
encode_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
for col in encode_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# --- Prepare Features ---
features = [
    'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'RoomService_log', 'FoodCourt_log', 'ShoppingMall_log', 'Spa_log', 'VRDeck_log',
    'TotalSpending', 'TotalSpending_log', 'HasSpending', 'SpendingPerAge',
    'GroupId', 'PersonInGroup', 'GroupSize', 'IsAlone',
    'Deck', 'CabinNum', 'Side', 'FamilySize', 'AgeGroup', 'IsChild'
]

train_df = df[df['is_train'] == 1].copy()
test_df = df[df['is_train'] == 0].copy()

X = train_df[features].values
y = train_df[TARGET].astype(int).values
X_test = test_df[features].values

print(f"Training features: {X.shape}")
print(f"Test features: {X_test.shape}")

# ============================================================
# 4-MODEL STACKING ENSEMBLE
# ============================================================
print("\n🚀 Training 4-Model Stacking Ensemble...")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# OOF predictions for each model
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_hgb = np.zeros(len(X))

# Test predictions for each model
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_hgb = np.zeros(len(X_test))

# LightGBM params
lgb_params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 6,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'verbose': -1, 'seed': SEED
}

# XGBoost params
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'learning_rate': 0.03, 'max_depth': 6, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'seed': SEED, 'verbosity': 0
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # --- LightGBM ---
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    lgb_model = lgb.train(
        lgb_params, train_data, num_boost_round=1500,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_lgb += lgb_model.predict(X_test) / N_FOLDS
    
    # --- XGBoost ---
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    xgb_model = xgb.train(
        xgb_params, dtrain, num_boost_round=1500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=100, verbose_eval=False
    )
    oof_xgb[val_idx] = xgb_model.predict(dval)
    test_xgb += xgb_model.predict(dtest) / N_FOLDS
    
    # --- CatBoost ---
    cat_model = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6,
        random_seed=SEED, verbose=0, early_stopping_rounds=100
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_cat += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    # --- HistGradientBoosting (NEW) ---
    hgb_model = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=6,
        random_state=SEED, early_stopping=True, validation_fraction=0.1
    )
    hgb_model.fit(X_train, y_train)
    oof_hgb[val_idx] = hgb_model.predict_proba(X_val)[:, 1]
    test_hgb += hgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    # Fold scores
    lgb_acc = ((oof_lgb[val_idx] > 0.5) == y_val).mean()
    xgb_acc = ((oof_xgb[val_idx] > 0.5) == y_val).mean()
    cat_acc = ((oof_cat[val_idx] > 0.5) == y_val).mean()
    hgb_acc = ((oof_hgb[val_idx] > 0.5) == y_val).mean()
    print(f"LGB: {lgb_acc:.4f} | XGB: {xgb_acc:.4f} | CAT: {cat_acc:.4f} | HGB: {hgb_acc:.4f}")

# ============================================================
# META-LEARNER
# ============================================================
print("\n🧠 Training Meta-Learner (Logistic Regression)...")

# Stack OOF predictions (now 4 columns)
meta_train = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_hgb])
meta_test = np.column_stack([test_lgb, test_xgb, test_cat, test_hgb])

# Train meta-learner
meta_model = LogisticRegression(random_state=SEED, max_iter=1000)
meta_model.fit(meta_train, y)

# Final predictions
final_preds = meta_model.predict_proba(meta_test)[:, 1]
final_labels = (final_preds > 0.5)

# ============================================================
# RESULTS
# ============================================================
print("\n🏆 Final Results:")

lgb_cv = ((oof_lgb > 0.5) == y).mean()
xgb_cv = ((oof_xgb > 0.5) == y).mean()
cat_cv = ((oof_cat > 0.5) == y).mean()
hgb_cv = ((oof_hgb > 0.5) == y).mean()

# Simple average baseline
avg_oof = (oof_lgb + oof_xgb + oof_cat + oof_hgb) / 4
avg_cv = ((avg_oof > 0.5) == y).mean()

# Meta-learner CV
meta_cv = ((meta_model.predict_proba(meta_train)[:, 1] > 0.5) == y).mean()

print(f"LightGBM CV:          {lgb_cv:.4f}")
print(f"XGBoost CV:           {xgb_cv:.4f}")
print(f"CatBoost CV:          {cat_cv:.4f}")
print(f"HistGradientBoost CV: {hgb_cv:.4f}")
print(f"Simple Avg (4 models): {avg_cv:.4f}")
print(f"Meta-Learner:          {meta_cv:.4f}")

# --- Create Submission ---
print("\n💾 Creating submission...")
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': final_labels
})
submission.to_csv('submission_4model.csv', index=False)
print(f"Saved submission_4model.csv with {len(submission)} predictions")
print(submission.head())
