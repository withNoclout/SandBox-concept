import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Load Data
print("Loading Data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['PassengerId']

# Combine for Preprocessing
train['is_train'] = 1
test['is_train'] = 0
test['Survived'] = np.nan
all_data = pd.concat([train, test], sort=False)

# 2. Feature Engineering
print("Feature Engineering...")
# Title
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')
all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
all_data['Title'] = all_data['Title'].map(title_mapping).fillna(0)

# Sex
all_data['Sex'] = all_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Age (Smart Imputation)
all_data['Age'] = all_data['Age'].fillna(all_data.groupby('Title')['Age'].transform('median'))

# Embarked
all_data['Embarked'] = all_data['Embarked'].fillna('S')
all_data['Embarked'] = all_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Fare
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())

# Family
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone'] = 0
all_data.loc[all_data['FamilySize'] == 1, 'IsAlone'] = 1

# Features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
X = all_data[all_data['is_train'] == 1][features]
y = all_data[all_data['is_train'] == 1]['Survived']
X_test = all_data[all_data['is_train'] == 0][features]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 3. Teacher Model (Ensemble)
print("Training Teacher Ensemble...")
clf1 = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=1)
clf2 = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=1, use_label_encoder=False, eval_metric='logloss')
clf3 = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=1)

eclf = VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2), ('gb', clf3)], voting='soft')
eclf.fit(X_scaled, y)

# 4. Pseudo-Labeling
print("Generating Pseudo-Labels...")
probs = eclf.predict_proba(X_test_scaled)
preds = eclf.predict(X_test_scaled)

# Select High Confidence
# > 0.95 or < 0.05
high_conf_mask = (probs[:, 1] > 0.95) | (probs[:, 1] < 0.05)
X_pseudo = X_test_scaled[high_conf_mask]
y_pseudo = preds[high_conf_mask]

print(f"  Found {len(X_pseudo)} high-confidence samples out of {len(X_test)}.")

# Augment Training Data
X_aug = np.vstack([X_scaled, X_pseudo])
y_aug = np.hstack([y, y_pseudo])

# 5. Student Model (Retrain)
print("Retraining Student on Augmented Data...")
student = VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2), ('gb', clf3)], voting='soft')
student.fit(X_aug, y_aug)

# Final Predictions
final_preds = student.predict(X_test_scaled)

# 6. Apply WCG Corrections (The Detective)
print("Applying WCG Corrections...")
# Ticket Grouping Logic (Simplified from V12)
all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0])
all_data['IsWC'] = ((all_data['Sex'] == 1) | (all_data['Title'] == 4)).astype(int) # 4 is Master

# We need to reconstruct the dataframe for WCG logic
test_df = all_data[all_data['is_train'] == 0].copy()
test_df['Survived'] = final_preds

train_df_wcg = all_data[all_data['is_train'] == 1]
wc_stats = train_df_wcg[train_df_wcg['IsWC'] == 1].groupby('Ticket')['Survived'].agg(['count', 'mean'])
dead_tickets = wc_stats[wc_stats['mean'] == 0.0].index

male_stats = train_df_wcg[train_df_wcg['Sex'] == 0].groupby('Ticket')['Survived'].agg(['count', 'mean'])
alive_tickets = male_stats[male_stats['mean'] == 1.0].index

# Apply
mask_die = (test_df['Ticket'].isin(dead_tickets)) & (test_df['IsWC'] == 1)
test_df.loc[mask_die, 'Survived'] = 0

mask_live = (test_df['Ticket'].isin(alive_tickets)) & (test_df['Sex'] == 0)
test_df.loc[mask_live, 'Survived'] = 1

print(f"  Corrected {mask_die.sum()} to Die, {mask_live.sum()} to Live.")

# 7. Submission
print("Saving Submission...")
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': test_df['Survived'].astype(int)
})
submission.to_csv('submission_v13.csv', index=False)
print("Saved 'submission_v13.csv'")
