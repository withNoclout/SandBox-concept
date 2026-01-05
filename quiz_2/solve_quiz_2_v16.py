import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# 1. Load Data
print("Loading Data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['PassengerId']

# Feature Engineering (Standard V14-like)
def preprocess(df):
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).fillna(0)
    
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S').map({'S':0, 'C':1, 'Q':2})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df

train = preprocess(train)
test = preprocess(test)

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
X = train[features].values
y = train['Survived'].values
X_test = test[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 2. Identify Noisy Samples (Cross-Validation)
print("Identifying Noisy Samples...")
clf1 = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
clf2 = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss')
eclf = VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2)], voting='soft')

# Get OOF Predictions
oof_probs = cross_val_predict(eclf, X_scaled, y, cv=5, method='predict_proba')[:, 1]
oof_preds = (oof_probs > 0.5).astype(int)

# Find "Confident Errors"
# Error: Prediction != Truth
# Confident: Prob > 0.8 or Prob < 0.2
error_mask = (oof_preds != y)
conf_mask = (oof_probs > 0.8) | (oof_probs < 0.2)
noise_mask = error_mask & conf_mask

noisy_indices = np.where(noise_mask)[0]
print(f"  Found {len(noisy_indices)} Noisy Samples (Confident Errors).")
print(f"  Example Noise: {train.iloc[noisy_indices[:5]][['Name', 'Survived', 'Sex', 'Pclass']].values}")

# 3. Remove Noise
X_clean = np.delete(X_scaled, noisy_indices, axis=0)
y_clean = np.delete(y, noisy_indices, axis=0)
print(f"  Cleaned Dataset Size: {len(X_clean)} (Original: {len(X)})")

# 4. Retrain on Clean Data
print("Retraining on Clean Data...")
eclf.fit(X_clean, y_clean)
final_preds = eclf.predict(X_test_scaled)

# 5. Apply WCG Corrections (The Detective)
print("Applying WCG Corrections...")
test['Survived'] = final_preds
all_data = pd.concat([train, test], sort=False).reset_index(drop=True)
all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0])
all_data['IsWC'] = ((all_data['Sex'] == 1) | (all_data['Title'] == 4)).astype(int)

# Logic
train_df_wcg = all_data[all_data['Survived'].notna()] 
# Actually, we should use the CLEAN train data for logic? 
# No, for WCG, we want the TRUTH about families, even if they are outliers.
# So we use the original 'train' dataframe for WCG stats.

# Re-calculate IsWC on original train for stats
train['Surname'] = train['Name'].apply(lambda x: x.split(',')[0])
train['IsWC'] = ((train['Sex'] == 1) | (train['Title'] == 4)).astype(int)

wc_stats = train[train['IsWC'] == 1].groupby('Ticket')['Survived'].agg(['count', 'mean'])
dead_tickets = wc_stats[wc_stats['mean'] == 0.0].index

male_stats = train[train['Sex'] == 0].groupby('Ticket')['Survived'].agg(['count', 'mean'])
alive_tickets = male_stats[male_stats['mean'] == 1.0].index

# Calculate IsWC for Test
test['Surname'] = test['Name'].apply(lambda x: x.split(',')[0])
test['IsWC'] = ((test['Sex'] == 1) | (test['Title'] == 4)).astype(int)

# Apply to Test
mask_die = (test['Ticket'].isin(dead_tickets)) & (test['IsWC'] == 1)
test.loc[mask_die, 'Survived'] = 0

mask_live = (test['Ticket'].isin(alive_tickets)) & (test['Sex'] == 0)
test.loc[mask_live, 'Survived'] = 1

print(f"  Corrected {mask_die.sum()} to Die, {mask_live.sum()} to Live.")

# 6. Submission
print("Saving Submission...")
submission = pd.DataFrame({
    'PassengerId': test_ids.values,
    'Survived': test['Survived'].astype(int).values
})
submission.to_csv('submission_v16.csv', index=False)
print("Saved 'submission_v16.csv'")
