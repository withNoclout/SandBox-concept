import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.optimize import differential_evolution

# 1. Load Data
print("Loading Data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['PassengerId']

train['is_train'] = 1
test['is_train'] = 0
test['Survived'] = np.nan
all_data = pd.concat([train, test], sort=False).reset_index(drop=True)

# 2. Feature Engineering (Combined Robust + Neural Friendly)
print("Feature Engineering...")
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['Title'] = all_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')

# Encode Title (Ordinal for Trees, One-Hot for NN - we'll use One-Hot for both for simplicity in this blend script)
all_data = pd.get_dummies(all_data, columns=['Title'], prefix='Title')

all_data['Sex'] = all_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())
all_data['Embarked'] = all_data['Embarked'].fillna('S')
all_data = pd.get_dummies(all_data, columns=['Embarked'], prefix='Emb')

all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)

# Normalize Numerical Features (Crucial for MLP, fine for XGB)
scaler = StandardScaler()
num_cols = ['Age', 'Fare', 'FamilySize', 'Pclass']
all_data[num_cols] = scaler.fit_transform(all_data[num_cols])

features = [c for c in all_data.columns if c not in ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'is_train', 'SibSp', 'Parch']]

X = all_data[all_data['is_train'] == 1][features].values
y = all_data[all_data['is_train'] == 1]['Survived'].values
X_test = all_data[all_data['is_train'] == 0][features].values

# 3. Generate OOF Predictions
print("Generating OOF Predictions (This may take a minute)...")

# Model 1: XGBoost (V14 Style)
print("  Training XGBoost (Tree)...")
xgb_model = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    use_label_encoder=False, eval_metric='logloss', n_jobs=1
)
# Get OOF Probs
xgb_oof = cross_val_predict(xgb_model, X, y, cv=5, method='predict_proba')[:, 1]
# Fit on full data for Test
xgb_model.fit(X, y)
xgb_test = xgb_model.predict_proba(X_test)[:, 1]

# Model 2: MLP (V19 Style)
print("  Training MLP (Brain)...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
    alpha=0.0001, batch_size=32, learning_rate_init=0.001, max_iter=500,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42
)
# Get OOF Probs
mlp_oof = cross_val_predict(mlp_model, X, y, cv=5, method='predict_proba')[:, 1]
# Fit on full data for Test
mlp_model.fit(X, y)
mlp_test = mlp_model.predict_proba(X_test)[:, 1]

# 4. Genetic Optimization (Differential Evolution)
print("Optimizing Weights via Genetic Algorithm...")

def objective(weights):
    # Normalize weights
    w = np.abs(weights)
    w = w / w.sum()
    
    # Blend
    blend_prob = w[0] * xgb_oof + w[1] * mlp_oof
    blend_pred = (blend_prob > 0.5).astype(int)
    
    # Minimize Error (1 - Accuracy)
    accuracy = np.mean(blend_pred == y)
    return 1 - accuracy

# Bounds for weights [0, 1]
bounds = [(0, 1), (0, 1)]

# Run Evolution
result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=100, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=42)

best_w = np.abs(result.x)
best_w = best_w / best_w.sum()

print(f"  Best Weights Found: XGB={best_w[0]:.4f}, MLP={best_w[1]:.4f}")
print(f"  Best OOF Accuracy: {1 - result.fun:.4f}")

# 5. Final Blend
final_probs = best_w[0] * xgb_test + best_w[1] * mlp_test
final_preds = (final_probs > 0.5).astype(int)

# 6. Apply WCG Corrections (The Detective)
print("Applying WCG Corrections...")
# Re-load for Logic (Need original values)
train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')
test_raw['Survived'] = final_preds
all_raw = pd.concat([train_raw, test_raw], sort=False).reset_index(drop=True)

all_raw['Surname'] = all_raw['Name'].apply(lambda x: x.split(',')[0])
all_raw['Title'] = all_raw['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
all_raw['IsWC'] = ((all_raw['Sex'] == 'female') | (all_raw['Title'] == 'Master')).astype(int)

# Update test_raw with IsWC from all_raw
test_raw = all_raw[all_raw['PassengerId'].isin(test_ids)].copy()

# Logic on Train
train_wcg = all_raw[all_raw['Survived'].notna() & (all_raw['PassengerId'].isin(train_raw['PassengerId']))]
wc_stats = train_wcg[train_wcg['IsWC'] == 1].groupby('Ticket')['Survived'].agg(['count', 'mean'])
dead_tickets = wc_stats[wc_stats['mean'] == 0.0].index

male_stats = train_wcg[train_wcg['Sex'] == 'male'].groupby('Ticket')['Survived'].agg(['count', 'mean'])
alive_tickets = male_stats[male_stats['mean'] == 1.0].index

# Apply to Test
mask_die = (test_raw['Ticket'].isin(dead_tickets)) & (test_raw['IsWC'] == 1)
test_raw.loc[mask_die, 'Survived'] = 0

mask_live = (test_raw['Ticket'].isin(alive_tickets)) & (test_raw['Sex'] == 'male')
test_raw.loc[mask_live, 'Survived'] = 1

print(f"  Corrected {mask_die.sum()} to Die, {mask_live.sum()} to Live.")

# 7. Submission
print("Saving Submission...")
submission = pd.DataFrame({
    'PassengerId': test_ids.values,
    'Survived': test_raw['Survived'].astype(int).values
})
submission.to_csv('submission_v20.csv', index=False)
print("Saved 'submission_v20.csv'")
