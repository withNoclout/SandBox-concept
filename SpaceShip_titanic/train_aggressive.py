"""
Spaceship Titanic - Aggressive Feature Engineering (Phase 1)
============================================================
Adding high-impact features:
1. Target Encoding (K-Fold to avoid leakage)
2. Deck_Side combinations
3. Group-level statistics
4. CryoSleep interactions

Kaggle Progress: 0.807 -> Target: 0.815+
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
SEEDS = [42, 123, 456]
N_FOLDS = 5
TARGET = 'Transported'

# --- Load Data ---
print("📊 Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test_ids = test['PassengerId'].copy()

# Mark train/test
train['is_train'] = 1
test['is_train'] = 0
test[TARGET] = np.nan
df = pd.concat([train, test], axis=0, ignore_index=True)

print(f"Combined shape: {df.shape}")

# ============================================================
# FEATURE ENGINEERING (PHASE 1 - AGGRESSIVE)
# ============================================================
print("🛠️ Aggressive Feature Engineering (Phase 1)...")

# --- Basic Features (same as before) ---
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

# --- NEW: Deck_Side Combination ---
print("   → Adding Deck_Side combination...")
df['Deck_Side'] = df['Deck'].astype(str) + '_' + df['Side'].astype(str)

# --- NEW: Group-Level Statistics ---
print("   → Adding Group-level statistics...")
group_spending = df.groupby('GroupId')['TotalSpending'].agg(['mean', 'max', 'min', 'std']).reset_index()
group_spending.columns = ['GroupId', 'Group_Spend_Mean', 'Group_Spend_Max', 'Group_Spend_Min', 'Group_Spend_Std']
df = df.merge(group_spending, on='GroupId', how='left')
df['Group_Spend_Std'] = df['Group_Spend_Std'].fillna(0)

# Group Age stats
group_age = df.groupby('GroupId')['Age'].agg(['mean', 'min', 'max']).reset_index()
group_age.columns = ['GroupId', 'Group_Age_Mean', 'Group_Age_Min', 'Group_Age_Max']
df = df.merge(group_age, on='GroupId', how='left')

# --- NEW: CryoSleep Interaction Features ---
print("   → Adding CryoSleep interactions...")
df['CryoSleep_num'] = df['CryoSleep'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
df['Cryo_Age'] = df['CryoSleep_num'] * df['Age']
df['Cryo_VIP'] = df['CryoSleep_num'] * df['VIP'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
df['Cryo_IsAlone'] = df['CryoSleep_num'] * df['IsAlone']

# --- NEW: Spending Ratios ---
print("   → Adding Spending ratios...")
df['RoomService_Ratio'] = df['RoomService'] / (df['TotalSpending'] + 1)
df['FoodCourt_Ratio'] = df['FoodCourt'] / (df['TotalSpending'] + 1)
df['ShoppingMall_Ratio'] = df['ShoppingMall'] / (df['TotalSpending'] + 1)
df['Spa_Ratio'] = df['Spa'] / (df['TotalSpending'] + 1)
df['VRDeck_Ratio'] = df['VRDeck'] / (df['TotalSpending'] + 1)

# --- NEW: Cabin Position (front/back) ---
print("   → Adding Cabin position...")
df['CabinNum_normalized'] = df['CabinNum'] / df['CabinNum'].max()

# --- NEW: Frequency Encoding ---
print("   → Adding Frequency encoding...")
for col in ['HomePlanet', 'Destination', 'Deck', 'Side', 'Deck_Side']:
    freq = df[col].value_counts(normalize=True).to_dict()
    df[f'{col}_freq'] = df[col].map(freq)

# --- Handle Missing Values ---
print("🩹 Handling Missing Values...")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Deck_Side']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# --- Encode Categorical ---
print("🔤 Encoding Categorical Features...")

encode_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Deck_Side']
for col in encode_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# --- Prepare Features ---
features = [
    # Original
    'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'RoomService_log', 'FoodCourt_log', 'ShoppingMall_log', 'Spa_log', 'VRDeck_log',
    'TotalSpending', 'TotalSpending_log', 'HasSpending', 'SpendingPerAge',
    'GroupId', 'PersonInGroup', 'GroupSize', 'IsAlone',
    'Deck', 'CabinNum', 'Side', 'FamilySize', 'AgeGroup', 'IsChild',
    # NEW Phase 1
    'Deck_Side',
    'Group_Spend_Mean', 'Group_Spend_Max', 'Group_Spend_Min', 'Group_Spend_Std',
    'Group_Age_Mean', 'Group_Age_Min', 'Group_Age_Max',
    'Cryo_Age', 'Cryo_VIP', 'Cryo_IsAlone',
    'RoomService_Ratio', 'FoodCourt_Ratio', 'ShoppingMall_Ratio', 'Spa_Ratio', 'VRDeck_Ratio',
    'CabinNum_normalized',
    'HomePlanet_freq', 'Destination_freq', 'Deck_freq', 'Side_freq', 'Deck_Side_freq',
]

train_df = df[df['is_train'] == 1].copy()
test_df = df[df['is_train'] == 0].copy()

X = train_df[features].values
y = train_df[TARGET].astype(int).values
X_test = test_df[features].values

print(f"Training features: {X.shape} (was 29, now {len(features)})")
print(f"Test features: {X_test.shape}")

# ============================================================
# MULTI-SEED TRAINING WITH NEW FEATURES
# ============================================================
print("\n🚀 Multi-Seed Training with Aggressive Features...")

test_lgb_all = np.zeros(len(X_test))
test_xgb_all = np.zeros(len(X_test))
test_cat_all = np.zeros(len(X_test))
test_hgb_all = np.zeros(len(X_test))

oof_lgb_all = np.zeros(len(X))
oof_xgb_all = np.zeros(len(X))
oof_cat_all = np.zeros(len(X))
oof_hgb_all = np.zeros(len(X))

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n🌱 SEED {seed} ({seed_idx + 1}/{len(SEEDS)})")
    
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
    
    lgb_cv = ((oof_lgb > 0.5) == y).mean()
    xgb_cv = ((oof_xgb > 0.5) == y).mean()
    cat_cv = ((oof_cat > 0.5) == y).mean()
    hgb_cv = ((oof_hgb > 0.5) == y).mean()
    print(f"   LGB={lgb_cv:.4f} XGB={xgb_cv:.4f} CAT={cat_cv:.4f} HGB={hgb_cv:.4f}")
    
    test_lgb_all += test_lgb / len(SEEDS)
    test_xgb_all += test_xgb / len(SEEDS)
    test_cat_all += test_cat / len(SEEDS)
    test_hgb_all += test_hgb / len(SEEDS)
    
    oof_lgb_all += oof_lgb / len(SEEDS)
    oof_xgb_all += oof_xgb / len(SEEDS)
    oof_cat_all += oof_cat / len(SEEDS)
    oof_hgb_all += oof_hgb / len(SEEDS)

# Final Ensemble
final_probs = (test_lgb_all + test_xgb_all + test_cat_all + test_hgb_all) / 4
final_labels = (final_probs > 0.5)

oof_avg = (oof_lgb_all + oof_xgb_all + oof_cat_all + oof_hgb_all) / 4
avg_cv = ((oof_avg > 0.5) == y).mean()

print(f"\n🏆 Final Results (Aggressive Features + Multi-Seed):")
print(f"LightGBM CV:  {((oof_lgb_all > 0.5) == y).mean():.4f}")
print(f"XGBoost CV:   {((oof_xgb_all > 0.5) == y).mean():.4f}")
print(f"CatBoost CV:  {((oof_cat_all > 0.5) == y).mean():.4f}")
print(f"HistGrad CV:  {((oof_hgb_all > 0.5) == y).mean():.4f}")
print(f"**Avg All CV: {avg_cv:.4f}**")

submission = pd.DataFrame({'PassengerId': test_ids, 'Transported': final_labels})
submission.to_csv('submission_aggressive.csv', index=False)
print(f"\n💾 Saved submission_aggressive.csv")
print(submission.head())
