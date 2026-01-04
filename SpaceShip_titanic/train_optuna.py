"""
Spaceship Titanic - Optuna Auto-Tuning
======================================
Systematically optimize hyperparameters for all 4 models

Kaggle Progress: 0.801 -> 0.804 -> 0.806 -> 0.807 -> Target: 0.815
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
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Configuration ---
SEED = 42
N_FOLDS = 5
N_TRIALS = 30  # Number of optimization trials per model
TARGET = 'Transported'

# --- Load and Prepare Data (Same as before) ---
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

# Encode
encode_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
for col in encode_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Features
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

# ============================================================
# OPTUNA OPTIMIZATION
# ============================================================
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)  # Use 3 folds for speed

def objective_lgb(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': 5,
        'verbose': -1,
        'seed': SEED
    }
    
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        train_data = lgb.Dataset(X[train_idx], label=y[train_idx])
        val_data = lgb.Dataset(X[val_idx], label=y[val_idx], reference=train_data)
        model = lgb.train(params, train_data, num_boost_round=1000,
                         valid_sets=[val_data], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        preds = model.predict(X[val_idx])
        acc = ((preds > 0.5) == y[val_idx]).mean()
        scores.append(acc)
    return np.mean(scores)

def objective_xgb(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'seed': SEED,
        'verbosity': 0
    }
    
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
        dval = xgb.DMatrix(X[val_idx], label=y[val_idx])
        model = xgb.train(params, dtrain, num_boost_round=1000,
                         evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        preds = model.predict(dval)
        acc = ((preds > 0.5) == y[val_idx]).mean()
        scores.append(acc)
    return np.mean(scores)

def objective_cat(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'iterations': 1000,
        'random_seed': SEED,
        'verbose': 0,
        'early_stopping_rounds': 50
    }
    
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = CatBoostClassifier(**params)
        model.fit(X[train_idx], y[train_idx], eval_set=(X[val_idx], y[val_idx]), verbose=0)
        preds = model.predict_proba(X[val_idx])[:, 1]
        acc = ((preds > 0.5) == y[val_idx]).mean()
        scores.append(acc)
    return np.mean(scores)

# --- Run Optimization ---
print(f"\n🔍 Optuna Optimization ({N_TRIALS} trials per model)...")

print("\n📊 Optimizing LightGBM...")
study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
study_lgb.optimize(objective_lgb, n_trials=N_TRIALS, show_progress_bar=True)
best_lgb = study_lgb.best_params
print(f"Best LGB: {study_lgb.best_value:.4f}")

print("\n📊 Optimizing XGBoost...")
study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=True)
best_xgb = study_xgb.best_params
print(f"Best XGB: {study_xgb.best_value:.4f}")

print("\n📊 Optimizing CatBoost...")
study_cat = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
study_cat.optimize(objective_cat, n_trials=N_TRIALS, show_progress_bar=True)
best_cat = study_cat.best_params
print(f"Best CAT: {study_cat.best_value:.4f}")

# --- Print Best Params ---
print("\n" + "="*60)
print("🏆 BEST PARAMETERS FOUND:")
print("="*60)
print(f"\nLightGBM: {best_lgb}")
print(f"\nXGBoost: {best_xgb}")
print(f"\nCatBoost: {best_cat}")

# ============================================================
# TRAIN WITH OPTIMIZED PARAMS
# ============================================================
print("\n🚀 Training with OPTIMIZED parameters (5-Fold)...")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_hgb = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_hgb = np.zeros(len(X_test))

# Merge best params with defaults
lgb_params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'bagging_freq': 5, 'verbose': -1, 'seed': SEED, **best_lgb
}
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'seed': SEED, 'verbosity': 0, **best_xgb
}
cat_params = {
    'iterations': 1500, 'random_seed': SEED, 'verbose': 0,
    'early_stopping_rounds': 100, **best_cat
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    lgb_model = lgb.train(lgb_params, train_data, num_boost_round=1500,
                         valid_sets=[val_data], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_lgb += lgb_model.predict(X_test) / N_FOLDS
    
    # XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=1500,
                         evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=False)
    oof_xgb[val_idx] = xgb_model.predict(dval)
    test_xgb += xgb_model.predict(dtest) / N_FOLDS
    
    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_cat += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    # HistGradientBoosting
    hgb_model = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=6,
                                               random_state=SEED, early_stopping=True, validation_fraction=0.1)
    hgb_model.fit(X_train, y_train)
    oof_hgb[val_idx] = hgb_model.predict_proba(X_val)[:, 1]
    test_hgb += hgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    lgb_acc = ((oof_lgb[val_idx] > 0.5) == y_val).mean()
    xgb_acc = ((oof_xgb[val_idx] > 0.5) == y_val).mean()
    cat_acc = ((oof_cat[val_idx] > 0.5) == y_val).mean()
    hgb_acc = ((oof_hgb[val_idx] > 0.5) == y_val).mean()
    print(f"LGB: {lgb_acc:.4f} | XGB: {xgb_acc:.4f} | CAT: {cat_acc:.4f} | HGB: {hgb_acc:.4f}")

# Meta-learner
print("\n🧠 Training Meta-Learner...")
meta_train = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_hgb])
meta_test = np.column_stack([test_lgb, test_xgb, test_cat, test_hgb])

meta_model = LogisticRegression(random_state=SEED, max_iter=1000)
meta_model.fit(meta_train, y)

final_preds = meta_model.predict_proba(meta_test)[:, 1]
final_labels = (final_preds > 0.5)

# Results
print("\n🏆 Final Results (OPTUNA OPTIMIZED):")
print(f"LightGBM CV:     {((oof_lgb > 0.5) == y).mean():.4f}")
print(f"XGBoost CV:      {((oof_xgb > 0.5) == y).mean():.4f}")
print(f"CatBoost CV:     {((oof_cat > 0.5) == y).mean():.4f}")
print(f"HistGradBoost:   {((oof_hgb > 0.5) == y).mean():.4f}")
avg_oof = (oof_lgb + oof_xgb + oof_cat + oof_hgb) / 4
print(f"Simple Avg (4):  {((avg_oof > 0.5) == y).mean():.4f}")
print(f"Meta-Learner:    {((meta_model.predict_proba(meta_train)[:, 1] > 0.5) == y).mean():.4f}")

# Submission
submission = pd.DataFrame({'PassengerId': test_ids, 'Transported': final_labels})
submission.to_csv('submission_optuna.csv', index=False)
print(f"\n💾 Saved submission_optuna.csv")
print(submission.head())
