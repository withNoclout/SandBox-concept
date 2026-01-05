import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# 1. Load Data
print("Loading Data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['PassengerId']

train['is_train'] = 1
test['is_train'] = 0
test['Survived'] = np.nan
all_data = pd.concat([train, test], sort=False).reset_index(drop=True)

# 2. Feature Engineering (V14 Style - Robust)
print("Feature Engineering...")
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['Title'] = all_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
all_data['Title'] = all_data['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).fillna(0)

all_data['Sex'] = all_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())
all_data['Embarked'] = all_data['Embarked'].fillna('S').map({'S':0, 'C':1, 'Q':2})
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize']
X_all = all_data[features].values
y_all = all_data['is_train'].values # Target is "Is_Train" (reverse of Is_Test)

# 3. Adversarial Validation
print("Running Adversarial Validation...")
# We want to predict "Is_Test" (0 for Train, 1 for Test)
# So let's flip y_all: 0 -> 1 (Test), 1 -> 0 (Train)
y_adv = 1 - y_all

# Train Classifier
adv_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
# We only need predictions for the TRAINING set to weight them.
# But to train the adv_model, we need both Train and Test data.

# Cross-Validation to check AUC
cv_preds = cross_val_predict(adv_model, X_all, y_adv, cv=5, method='predict_proba')[:, 1]
auc = roc_auc_score(y_adv, cv_preds)
print(f"  Adversarial AUC: {auc:.4f}")
if auc > 0.6:
    print("  WARNING: Significant Drift Detected! (AUC > 0.6)")
else:
    print("  Drift is minimal. Reweighting might have small effect.")

# 4. Calculate Sample Weights
# w = p / (1 - p) where p = P(Is_Test)
# We only need weights for the TRAINING samples (where y_adv == 0)
train_indices = (y_adv == 0)
p_train = cv_preds[train_indices]

# Clip p to avoid division by zero or explosion
p_train = np.clip(p_train, 0.01, 0.99)
weights = p_train / (1 - p_train)

# Normalize weights
weights = weights / weights.mean()

print(f"  Weights Stats: Min={weights.min():.2f}, Max={weights.max():.2f}, Mean={weights.mean():.2f}")

# 5. Retrain Main Model with Weights
print("Retraining Survival Model with Adversarial Weights...")
X_train = all_data[all_data['is_train'] == 1][features].values
y_train = all_data[all_data['is_train'] == 1]['Survived'].values
X_test = all_data[all_data['is_train'] == 0][features].values

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Bagging Ensemble (50 Models)
n_models = 50
test_preds_sum = np.zeros(len(X_test))

for i in range(n_models):
    if i % 10 == 0: print(f"  Model {i}/{n_models}...")
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=i,
        use_label_encoder=False, eval_metric='logloss', n_jobs=1
    )
    # Pass Weights!
    model.fit(X_train_sc, y_train, sample_weight=weights)
    test_preds_sum += model.predict_proba(X_test_sc)[:, 1]

avg_preds = test_preds_sum / n_models
final_preds = (avg_preds > 0.5).astype(int)

# 6. Apply WCG Corrections (The Detective)
print("Applying WCG Corrections...")
# Logic
all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0])
all_data['IsWC'] = ((all_data['Sex'] == 1) | (all_data['Title'] == 'Master')).astype(int)

test_df = all_data[all_data['is_train'] == 0].copy()
test_df['Survived'] = final_preds

train_df_wcg = all_data[all_data['is_train'] == 1]
wc_stats = train_df_wcg[train_df_wcg['IsWC'] == 1].groupby('Ticket')['Survived'].agg(['count', 'mean'])
dead_tickets = wc_stats[wc_stats['mean'] == 0.0].index

male_stats = train_df_wcg[train_df_wcg['Sex'] == 0].groupby('Ticket')['Survived'].agg(['count', 'mean'])
alive_tickets = male_stats[male_stats['mean'] == 1.0].index

mask_die = (test_df['Ticket'].isin(dead_tickets)) & (test_df['IsWC'] == 1)
test_df.loc[mask_die, 'Survived'] = 0

mask_live = (test_df['Ticket'].isin(alive_tickets)) & (test_df['Sex'] == 0)
test_df.loc[mask_live, 'Survived'] = 1

print(f"  Corrected {mask_die.sum()} to Die, {mask_live.sum()} to Live.")

# 7. Submission
print("Saving Submission...")
submission = pd.DataFrame({
    'PassengerId': test_ids.values,
    'Survived': test_df['Survived'].astype(int).values
})
submission.to_csv('submission_v18.csv', index=False)
print("Saved 'submission_v18.csv'")
