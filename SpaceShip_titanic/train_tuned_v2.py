"""
Spaceship Titanic - Tuned v2 (Back to Basics + Weight Optimization)
===================================================================
Based on train_tuned.py (0.807 LB)
Improvements:
1. Added Deck_Side interaction feature
2. Replaced Meta-Learner with Optimized Weighted Average (Soft Voting)
3. Fixed CabinNum handling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
import warnings
import os

warnings.filterwarnings('ignore')

# --- Configuration ---
SEED = 42
N_FOLDS = 5
TARGET = 'Transported'

# --- Load Data ---
print("📊 Loading data...")
base_dir = os.path.dirname(__file__)
train = pd.read_csv(os.path.join(base_dir, 'train.csv'))
test = pd.read_csv(os.path.join(base_dir, 'test.csv'))

test_ids = test['PassengerId'].copy()

train['is_train'] = 1
test['is_train'] = 0
test[TARGET] = np.nan
df = pd.concat([train, test], axis=0, ignore_index=True)

# --- Feature Engineering ---
print("🛠️ Feature Engineering...")

# 1. PassengerId features
df['GroupId'] = df['PassengerId'].str.split('_').str[0].astype(int)
df['PersonInGroup'] = df['PassengerId'].str.split('_').str[1].astype(int)
df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')
df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

# 2. Cabin features
def split_cabin(x):
    if pd.isna(x):
        return pd.Series([np.nan, np.nan, np.nan])
    return pd.Series(x.split('/'))

df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].apply(split_cabin)
df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')

# NEW: Deck_Side interaction
df['Deck_Side'] = df['Deck'] + '_' + df['Side']

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

cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Deck_Side']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# --- Encode Categorical Features ---
print("🔤 Encoding Categorical Features...")

label_encoders = {}
encode_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Deck_Side']
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
    'Deck', 'CabinNum', 'Side', 'Deck_Side', 'FamilySize', 'AgeGroup', 'IsChild'
]

train_df = df[df['is_train'] == 1].copy()
test_df = df[df['is_train'] == 0].copy()

X = train_df[features].values
y = train_df[TARGET].astype(int).values
X_test = test_df[features].values

print(f"Training features: {X.shape}")

# ============================================================
# TUNED PARAMETERS (Same as train_tuned.py)
# ============================================================
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 700,
    'learning_rate': 0.01,
    'max_depth': 7,
    'num_leaves': 40,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': SEED
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.02,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'seed': SEED,
    'verbosity': 0
}

cat_params = {
    'learning_rate': 0.018,
    'depth': 6,
    'l2_leaf_reg': 7.8,
    'border_count': 182,
    'iterations': 1500,
    'random_seed': SEED,
    'verbose': 0,
    'early_stopping_rounds': 100
}

# ============================================================
# TRAINING LOOP
# ============================================================
print("\n🚀 Training Models...")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# OOF predictions
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_hgb = np.zeros(len(X))

# Test predictions
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_hgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold + 1}/{N_FOLDS} ---")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    lgb_model = lgb.train(
        lgb_params, train_data, num_boost_round=1500,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_lgb += lgb_model.predict(X_test) / N_FOLDS
    
    # XGBoost
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
    
    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_cat += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    # HistGradientBoosting
    hgb_model = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=6,
        random_state=SEED, early_stopping=True, validation_fraction=0.1
    )
    hgb_model.fit(X_train, y_train)
    oof_hgb[val_idx] = hgb_model.predict_proba(X_val)[:, 1]
    test_hgb += hgb_model.predict_proba(X_test)[:, 1] / N_FOLDS

# ============================================================
# OPTIMIZE WEIGHTS
# ============================================================
print("\n⚖️ Optimizing Ensemble Weights...")

def loss_func(weights):
    final_prediction = (weights[0] * oof_lgb + weights[1] * oof_xgb + 
                        weights[2] * oof_cat + weights[3] * oof_hgb)
    # Clip to avoid log(0)
    final_prediction = np.clip(final_prediction, 1e-15, 1-1e-15)
    # Calculate Log Loss (negative log likelihood)
    ll = -np.mean(y * np.log(final_prediction) + (1 - y) * np.log(1 - final_prediction))
    return ll

# Initial weights (equal)
init_weights = [0.25, 0.25, 0.25, 0.25]
# Constraints: sum of weights = 1
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
# Bounds: weights between 0 and 1
bounds = [(0, 1)] * 4

res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = res.x

print(f"Optimal Weights: LGB={best_weights[0]:.4f}, XGB={best_weights[1]:.4f}, CAT={best_weights[2]:.4f}, HGB={best_weights[3]:.4f}")

# ============================================================
# FINAL PREDICTION
# ============================================================
final_oof = (best_weights[0] * oof_lgb + best_weights[1] * oof_xgb + 
             best_weights[2] * oof_cat + best_weights[3] * oof_hgb)
final_acc = ((final_oof > 0.5) == y).mean()

print(f"🏆 Final Weighted Ensemble CV Accuracy: {final_acc:.5f}")

final_test_pred = (best_weights[0] * test_lgb + best_weights[1] * test_xgb + 
                   best_weights[2] * test_cat + best_weights[3] * test_hgb)
final_labels = (final_test_pred > 0.5)

# --- Save Submission ---
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': final_labels
})
submission_path = os.path.join(base_dir, 'submission_tuned_v2.csv')
submission.to_csv(submission_path, index=False)
print(f"✅ Saved submission to {submission_path}")
