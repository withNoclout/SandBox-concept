"""
Spaceship Titanic - Smart Missing Data Handling
================================================
Back to basics: Simpler features + Intelligent imputation

Key insights:
1. Tuned version (0.807) was BEST - simpler is better
2. Focus on SMART imputation, not more features
3. Use domain logic for missing values

Smart Imputation Strategy:
1. CryoSleep → Spending MUST be 0
2. Group/Family-based imputation for categoricals
3. Age: impute by HomePlanet + Deck patterns
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
print(f"Missing values BEFORE imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ============================================================
# SMART FEATURE EXTRACTION (Before Imputation)
# ============================================================
print("\n🛠️ Feature Extraction...")

# Extract from PassengerId
df['GroupId'] = df['PassengerId'].str.split('_').str[0].astype(int)
df['PersonInGroup'] = df['PassengerId'].str.split('_').str[1].astype(int)

# Extract from Cabin
df['Deck'] = df['Cabin'].str.split('/').str[0]
df['CabinNum'] = df['Cabin'].str.split('/').str[1].astype(float)
df['Side'] = df['Cabin'].str.split('/').str[2]

# Extract from Name
df['LastName'] = df['Name'].str.split().str[-1]

# ============================================================
# SMART IMPUTATION (Domain Logic)
# ============================================================
print("\n🧠 Smart Missing Data Imputation...")

# 1. GROUP-BASED IMPUTATION
# People in the same group likely have same HomePlanet, Destination, etc.
print("   → Group-based imputation for HomePlanet, Destination, Side, Deck...")

for col in ['HomePlanet', 'Destination', 'Side', 'Deck']:
    # Fill with group mode first
    group_mode = df.groupby('GroupId')[col].transform(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)
    df[col] = df[col].fillna(group_mode)

# 2. FAMILY-BASED IMPUTATION (by LastName)
print("   → Family-based imputation...")
for col in ['HomePlanet', 'Destination', 'Side', 'Deck']:
    family_mode = df.groupby('LastName')[col].transform(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)
    df[col] = df[col].fillna(family_mode)

# 3. CRYOSLEEP LOGIC IMPUTATION
# If CryoSleep is True, spending MUST be 0
# If spending > 0, CryoSleep MUST be False
print("   → CryoSleep domain logic...")
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Calculate total spending (even with NaN, sum will propagate correctly)
df['TotalSpending'] = df[spending_cols].sum(axis=1)

# If any spending > 0, CryoSleep must be False
df.loc[(df['TotalSpending'] > 0) & (df['CryoSleep'].isna()), 'CryoSleep'] = False

# If CryoSleep is True, fill missing spending with 0
for col in spending_cols:
    df.loc[(df['CryoSleep'] == True) & (df[col].isna()), col] = 0

# If CryoSleep is False and spending is NaN, use median
# (They could have spent something)

# 4. VIP LOGIC
# Children (Age < 18) can't be VIP in most systems
print("   → VIP domain logic...")
df.loc[(df['Age'] < 18) & (df['VIP'].isna()), 'VIP'] = False

# 5. AGE IMPUTATION BY DECK
# Different decks might have different age distributions
print("   → Age by Deck imputation...")
age_by_deck = df.groupby('Deck')['Age'].transform('median')
df['Age'] = df['Age'].fillna(age_by_deck)

# 6. REMAINING CATEGORICAL: Fill with global mode
print("   → Filling remaining categoricals with mode...")
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']:
    df[col] = df[col].fillna(df[col].mode()[0])

# 7. REMAINING NUMERICAL: Fill with median
print("   → Filling remaining numericals with median...")
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Recalculate TotalSpending after imputation
df['TotalSpending'] = df[spending_cols].sum(axis=1)

print(f"\nMissing values AFTER imputation:")
print(df[['HomePlanet', 'CryoSleep', 'Age', 'VIP', 'Deck', 'Side']].isnull().sum())

# ============================================================
# SIMPLER FEATURE ENGINEERING (Like tuned version)
# ============================================================
print("\n🛠️ Simpler Feature Engineering...")

df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')
df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
df['FamilySize'] = df.groupby('LastName')['LastName'].transform('count')

df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
df['SpendingPerAge'] = df['TotalSpending'] / (df['Age'] + 1)

# Log transforms
for col in spending_cols:
    df[f'{col}_log'] = np.log1p(df[col])
df['TotalSpending_log'] = np.log1p(df['TotalSpending'])

# Enforce CryoSleep → Spending = 0
for col in spending_cols:
    df.loc[df['CryoSleep'] == True, col] = 0
    df.loc[df['CryoSleep'] == True, f'{col}_log'] = 0
df.loc[df['CryoSleep'] == True, 'TotalSpending'] = 0
df.loc[df['CryoSleep'] == True, 'TotalSpending_log'] = 0

df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])
df['AgeGroup'] = df['AgeGroup'].astype(float)
df['IsChild'] = (df['Age'] < 18).astype(int)

# --- Encode Categorical ---
print("🔤 Encoding...")
encode_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
for col in encode_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# --- SIMPLER FEATURE SET (like tuned version) ---
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

print(f"\nTraining: {X.shape}, Test: {X_test.shape}")

# ============================================================
# TRAINING (Using tuned params that got 0.807)
# ============================================================
print("\n🚀 Training with Smart Imputation + Tuned Params...")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_hgb = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_hgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # LightGBM (tuned params)
    lgb_params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'learning_rate': 0.01, 'num_leaves': 40, 'max_depth': 7,
        'reg_alpha': 0.1, 'reg_lambda': 0.5, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'seed': SEED
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
        'learning_rate': 0.03, 'max_depth': 6, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.5,
        'seed': SEED, 'verbosity': 0
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=1500,
                         evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=False)
    oof_xgb[val_idx] = xgb_model.predict(dval)
    test_xgb += xgb_model.predict(dtest) / N_FOLDS
    
    # CatBoost (tuned params from user)
    cat_model = CatBoostClassifier(
        learning_rate=0.018, depth=6, l2_leaf_reg=7.84, border_count=182,
        iterations=1500, random_seed=SEED, verbose=0, early_stopping_rounds=100
    )
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
    
    lgb_acc = ((oof_lgb[val_idx] > 0.5) == y_val).mean()
    xgb_acc = ((oof_xgb[val_idx] > 0.5) == y_val).mean()
    cat_acc = ((oof_cat[val_idx] > 0.5) == y_val).mean()
    hgb_acc = ((oof_hgb[val_idx] > 0.5) == y_val).mean()
    print(f"LGB: {lgb_acc:.4f} | XGB: {xgb_acc:.4f} | CAT: {cat_acc:.4f} | HGB: {hgb_acc:.4f}")

# Final Ensemble
final_probs = (test_lgb + test_xgb + test_cat + test_hgb) / 4
final_labels = (final_probs > 0.5)

oof_avg = (oof_lgb + oof_xgb + oof_cat + oof_hgb) / 4
avg_cv = ((oof_avg > 0.5) == y).mean()

print(f"\n🏆 Final Results (Smart Imputation):")
print(f"LightGBM CV:  {((oof_lgb > 0.5) == y).mean():.4f}")
print(f"XGBoost CV:   {((oof_xgb > 0.5) == y).mean():.4f}")
print(f"CatBoost CV:  {((oof_cat > 0.5) == y).mean():.4f}")
print(f"HistGrad CV:  {((oof_hgb > 0.5) == y).mean():.4f}")
print(f"**Avg All CV: {avg_cv:.4f}**")

submission = pd.DataFrame({'PassengerId': test_ids, 'Transported': final_labels})
submission.to_csv('submission_smart.csv', index=False)
print(f"\n💾 Saved submission_smart.csv")
print(submission.head())
