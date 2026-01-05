import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold

# 1. Load Data
print("Loading Data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['PassengerId']

train['is_train'] = 1
test['is_train'] = 0
test['Survived'] = np.nan
all_data = pd.concat([train, test], sort=False).reset_index(drop=True)

# 2. Feature Engineering (Neural Friendly)
print("Feature Engineering...")
# Neural Networks like normalized, dense features.
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['Title'] = all_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
# One-Hot Encode Title for NN
all_data = pd.get_dummies(all_data, columns=['Title'], prefix='Title')

all_data['Sex'] = all_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())
all_data['Embarked'] = all_data['Embarked'].fillna('S')
all_data = pd.get_dummies(all_data, columns=['Embarked'], prefix='Emb')

all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)

# Normalize Numerical Features
scaler = StandardScaler()
num_cols = ['Age', 'Fare', 'FamilySize', 'Pclass']
all_data[num_cols] = scaler.fit_transform(all_data[num_cols])

# Select Features (All numeric/encoded)
features = [c for c in all_data.columns if c not in ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'is_train', 'SibSp', 'Parch']]
print(f"  Using {len(features)} Features: {features}")

X = all_data[all_data['is_train'] == 1][features].values
y = all_data[all_data['is_train'] == 1]['Survived'].values
X_test = all_data[all_data['is_train'] == 0][features].values

# 3. Neural Bagging (10 Deep MLPs)
print("Training Neural Bagging Ensemble (10 MLPs)...")
n_models = 10
test_preds_sum = np.zeros(len(X_test))

for i in range(n_models):
    print(f"  Model {i+1}/{n_models}...")
    # Deep Architecture: 128 -> 64 -> 32
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001, # L2 Regularization
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=i
    )
    model.fit(X, y)
    test_preds_sum += model.predict_proba(X_test)[:, 1]

avg_preds = test_preds_sum / n_models
final_preds = (avg_preds > 0.5).astype(int)

# 4. Apply WCG Corrections (The Detective)
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

# 5. Submission
print("Saving Submission...")
submission = pd.DataFrame({
    'PassengerId': test_ids.values,
    'Survived': test_raw['Survived'].astype(int).values
})
submission.to_csv('submission_v19.csv', index=False)
print("Saved 'submission_v19.csv'")
