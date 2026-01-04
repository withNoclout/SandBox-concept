"""
Spaceship Titanic - Baseline Model
===================================
This script implements a LightGBM classifier with feature engineering.

Features:
- Cabin split (Deck, CabinNum, Side)
- PassengerId split (GroupId, PersonInGroup)
- TotalSpending
- CryoSleep spending fix
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
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

# Save test IDs for submission
test_ids = test['PassengerId'].copy()

# Combine for consistent preprocessing
train['is_train'] = 1
test['is_train'] = 0
test[TARGET] = np.nan
df = pd.concat([train, test], axis=0, ignore_index=True)

print(f"Combined shape: {df.shape}")

# --- Feature Engineering ---
print("🛠️ Feature Engineering...")

# 1. Extract from PassengerId: GroupId, PersonInGroup
df['GroupId'] = df['PassengerId'].str.split('_').str[0].astype(int)
df['PersonInGroup'] = df['PassengerId'].str.split('_').str[1].astype(int)
df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')

# 2. Extract from Cabin: Deck, CabinNum, Side
df['Deck'] = df['Cabin'].str.split('/').str[0]
df['CabinNum'] = df['Cabin'].str.split('/').str[1].astype(float)
df['Side'] = df['Cabin'].str.split('/').str[2]

# 3. Spending Features
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalSpending'] = df[spending_cols].sum(axis=1)
df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)

# 4. CryoSleep Logic: If CryoSleep=True, spending should be 0
for col in spending_cols:
    df.loc[df['CryoSleep'] == True, col] = 0
df.loc[df['CryoSleep'] == True, 'TotalSpending'] = 0

# 5. Name -> LastName
df['LastName'] = df['Name'].str.split().str[-1]
df['FamilySize'] = df.groupby('LastName')['LastName'].transform('count')

# --- Handle Missing Values ---
print("🩹 Handling Missing Values...")

# Numerical: Fill with median
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpending', 'CabinNum']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical: Fill with mode
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

# --- Prepare Final Features ---
features = [
    'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'TotalSpending', 'HasSpending',
    'GroupId', 'PersonInGroup', 'GroupSize',
    'Deck', 'CabinNum', 'Side', 'FamilySize'
]

# Split back to train/test
train_df = df[df['is_train'] == 1].copy()
test_df = df[df['is_train'] == 0].copy()

X = train_df[features]
y = train_df[TARGET].astype(int)
X_test = test_df[features]

print(f"Training features: {X.shape}")
print(f"Test features: {X_test.shape}")

# --- Train with K-Fold ---
print(f"\n🚀 Training LightGBM with {N_FOLDS}-Fold CV...")

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_scores = []

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': SEED
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / N_FOLDS
    
    fold_acc = ((oof_preds[val_idx] > 0.5) == y_val).mean()
    fold_scores.append(fold_acc)
    print(f"Fold {fold + 1} Accuracy: {fold_acc:.4f}")

# --- Final Evaluation ---
print(f"\n🏆 CV Results:")
print(f"Fold Accuracies: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean CV Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

# --- Create Submission ---
print("\n💾 Creating submission...")
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': (test_preds > 0.5)
})
submission.to_csv('submission.csv', index=False)
print(f"Saved submission.csv with {len(submission)} predictions")
print(submission.head())
