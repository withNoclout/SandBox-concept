"""
Spaceship Titanic - Multi-Seed Averaging
=========================================
Train each model with 3 different seeds, average all probabilities
for maximum variance reduction.

Total models: 4 models × 5 folds × 3 seeds = 60 models!

Kaggle Progress: 0.801 -> 0.807 -> Target: 0.81+
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
SEEDS = [42, 123, 456]  # 3 different seeds
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

# Feature Engineering (same as before)
df['GroupId'] = df['PassengerId'].str.split('_').str[0].astype(int)
df['PersonInGroup'] = df['PassengerId'].str.split('_').str[1].astype(int)
df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')
df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

df['Deck'] = df['Cabin'].str.split('/').str[0]
df['CabinNum'] = df['Cabin'].str.split('/').str[1].astype(float)
df['Side'] = df['Cabin'].str.split('/').str[2]

spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalSpending'] = df[spending_cols].sum(axis=1)
df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
df['SpendingPerAge'] = df['TotalSpending'] / (df['Age'] + 1)

for col in spending_cols:
    df[f'{col}_log'] = np.log1p(df[col])
df['TotalSpending_log'] = np.log1p(df['TotalSpending'])

for col in spending_cols:
    df.loc[df['CryoSleep'] == True, col] = 0
    df.loc[df['CryoSleep'] == True, f'{col}_log'] = 0
df.loc[df['CryoSleep'] == True, 'TotalSpending'] = 0
df.loc[df['CryoSleep'] == True, 'TotalSpending_log'] = 0

df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])
df['AgeGroup'] = df['AgeGroup'].astype(float)
df['IsChild'] = (df['Age'] < 18).astype(int)

df['LastName'] = df['Name'].str.split().str[-1]
df['FamilySize'] = df.groupby('LastName')['LastName'].transform('count')

# Handle Missing Values
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 
            'TotalSpending', 'CabinNum', 'SpendingPerAge', 'AgeGroup',
            'RoomService_log', 'FoodCourt_log', 'ShoppingMall_log', 'Spa_log', 'VRDeck_log', 'TotalSpending_log']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

encode_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
for col in encode_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

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

print(f"Training: {X.shape}, Test: {X_test.shape}")
print(f"Seeds: {SEEDS}")
print(f"Total model instances: {4} models × {N_FOLDS} folds × {len(SEEDS)} seeds = {4 * N_FOLDS * len(SEEDS)} models")

# ============================================================
# MULTI-SEED TRAINING
# ============================================================
print("\n🚀 Multi-Seed Averaging Training...")

# Accumulate predictions across ALL seeds
test_lgb_all = np.zeros(len(X_test))
test_xgb_all = np.zeros(len(X_test))
test_cat_all = np.zeros(len(X_test))
test_hgb_all = np.zeros(len(X_test))

oof_lgb_all = np.zeros(len(X))
oof_xgb_all = np.zeros(len(X))
oof_cat_all = np.zeros(len(X))
oof_hgb_all = np.zeros(len(X))

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*60}")
    print(f"🌱 SEED {seed} ({seed_idx + 1}/{len(SEEDS)})")
    print(f"{'='*60}")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    oof_hgb = np.zeros(len(X))
    
    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))
    test_hgb = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # LightGBM
        lgb_params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
            'learning_rate': 0.048, 'num_leaves': 23, 'max_depth': 9,
            'reg_alpha': 0.19, 'reg_lambda': 0.13, 'feature_fraction': 0.95,
            'bagging_fraction': 0.68, 'bagging_freq': 5, 'verbose': -1, 'seed': seed
        }
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        lgb_model = lgb.train(lgb_params, train_data, num_boost_round=1500,
                             valid_sets=[val_data], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        test_lgb += lgb_model.predict(X_test) / N_FOLDS
        
        # XGBoost
        xgb_params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'learning_rate': 0.045, 'max_depth': 4, 'subsample': 0.77,
            'colsample_bytree': 0.82, 'reg_alpha': 0.02, 'reg_lambda': 0.4,
            'seed': seed, 'verbosity': 0
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=1500,
                             evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=False)
        oof_xgb[val_idx] = xgb_model.predict(dval)
        test_xgb += xgb_model.predict(dtest) / N_FOLDS
        
        # CatBoost
        cat_model = CatBoostClassifier(
            learning_rate=0.041, depth=6, l2_leaf_reg=2.15, border_count=212,
            iterations=1500, random_seed=seed, verbose=0, early_stopping_rounds=100
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        test_cat += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS
        
        # HistGradientBoosting
        hgb_model = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=6,
            random_state=seed, early_stopping=True, validation_fraction=0.1
        )
        hgb_model.fit(X_train, y_train)
        oof_hgb[val_idx] = hgb_model.predict_proba(X_val)[:, 1]
        test_hgb += hgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    # Seed-level CV scores
    lgb_cv = ((oof_lgb > 0.5) == y).mean()
    xgb_cv = ((oof_xgb > 0.5) == y).mean()
    cat_cv = ((oof_cat > 0.5) == y).mean()
    hgb_cv = ((oof_hgb > 0.5) == y).mean()
    print(f"Seed {seed} CV: LGB={lgb_cv:.4f} XGB={xgb_cv:.4f} CAT={cat_cv:.4f} HGB={hgb_cv:.4f}")
    
    # Accumulate across seeds
    test_lgb_all += test_lgb / len(SEEDS)
    test_xgb_all += test_xgb / len(SEEDS)
    test_cat_all += test_cat / len(SEEDS)
    test_hgb_all += test_hgb / len(SEEDS)
    
    oof_lgb_all += oof_lgb / len(SEEDS)
    oof_xgb_all += oof_xgb / len(SEEDS)
    oof_cat_all += oof_cat / len(SEEDS)
    oof_hgb_all += oof_hgb / len(SEEDS)

# ============================================================
# FINAL ENSEMBLE
# ============================================================
print("\n🧠 Final Multi-Seed Ensemble...")

# Average ALL probabilities
final_probs = (test_lgb_all + test_xgb_all + test_cat_all + test_hgb_all) / 4
final_labels = (final_probs > 0.5)

# OOF CV
oof_avg = (oof_lgb_all + oof_xgb_all + oof_cat_all + oof_hgb_all) / 4
avg_cv = ((oof_avg > 0.5) == y).mean()

print(f"\n🏆 Final Results (Multi-Seed Averaged):")
print(f"LightGBM CV:  {((oof_lgb_all > 0.5) == y).mean():.4f}")
print(f"XGBoost CV:   {((oof_xgb_all > 0.5) == y).mean():.4f}")
print(f"CatBoost CV:  {((oof_cat_all > 0.5) == y).mean():.4f}")
print(f"HistGrad CV:  {((oof_hgb_all > 0.5) == y).mean():.4f}")
print(f"**Avg All CV: {avg_cv:.4f}**")

# Submission
submission = pd.DataFrame({'PassengerId': test_ids, 'Transported': final_labels})
submission.to_csv('submission_multiseed.csv', index=False)
print(f"\n💾 Saved submission_multiseed.csv (60 models averaged!)")
print(submission.head())
