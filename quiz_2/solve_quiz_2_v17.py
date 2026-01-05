import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
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

# 2. Basic Feature Engineering
print("Basic Feature Engineering...")
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['Title'] = all_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')

all_data['Sex'] = all_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
all_data['Embarked'] = all_data['Embarked'].fillna('S')
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1

# 3. MICE Imputation (Smarter Age)
print("Applying MICE Imputation...")
# We impute Age using Pclass, Sex, SibSp, Parch, Fare
impute_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age']
mice_data = all_data[impute_cols].copy()

imputer = IterativeImputer(random_state=42)
mice_filled = imputer.fit_transform(mice_data)
all_data['Age'] = mice_filled[:, 5] # Update Age

# 4. Target Encoding (Title, Embarked)
print("Applying Target Encoding...")
# We must only compute means on TRAIN data to avoid leakage
# Smoothing: (n * mean + m * global_mean) / (n + m)
def target_encode(df, col, target, m=10):
    global_mean = df[target].mean()
    agg = df.groupby(col)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth = (counts * means + m * global_mean) / (counts + m)
    return df[col].map(smooth)

# Apply to Train
train_df = all_data[all_data['is_train'] == 1].copy()
# Map for Test (using Train stats)
te_title = target_encode(train_df, 'Title', 'Survived')
te_embarked = target_encode(train_df, 'Embarked', 'Survived')

# Create mapping dictionaries
title_map = train_df.groupby('Title')['Survived'].mean().to_dict() # Simplified for mapping
embarked_map = train_df.groupby('Embarked')['Survived'].mean().to_dict()

all_data['Title_TE'] = all_data['Title'].map(title_map)
all_data['Embarked_TE'] = all_data['Embarked'].map(embarked_map)
# Fill NaNs in Test (if new categories) with global mean
all_data['Title_TE'] = all_data['Title_TE'].fillna(train_df['Survived'].mean())
all_data['Embarked_TE'] = all_data['Embarked_TE'].fillna(train_df['Survived'].mean())

# 5. Clustering Features (Unsupervised)
print("Generating Cluster Features...")
# Cluster based on: Pclass, Age, Fare, FamilySize, Sex
cluster_cols = ['Pclass', 'Age', 'Fare', 'FamilySize', 'Sex']
scaler = StandardScaler()
X_cluster = scaler.fit_transform(all_data[cluster_cols])

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
all_data['Cluster'] = kmeans.fit_predict(X_cluster)
# One-Hot Encode Cluster
all_data = pd.get_dummies(all_data, columns=['Cluster'], prefix='Cluster')

# 6. Prepare Final Dataset
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Title_TE', 'Embarked_TE'] + \
           [c for c in all_data.columns if c.startswith('Cluster_')]

X = all_data[all_data['is_train'] == 1][features].values
y = all_data[all_data['is_train'] == 1]['Survived'].values
X_test = all_data[all_data['is_train'] == 0][features].values

X_scaled = scaler.fit_transform(X) # Re-scale everything
X_test_scaled = scaler.transform(X_test)

# 7. Train Bagging Ensemble (V14 Style)
print("Training Bagging Ensemble (50 XGBoost Models)...")
n_models = 50
test_preds_sum = np.zeros(len(X_test))

for i in range(n_models):
    if i % 10 == 0: print(f"  Model {i}/{n_models}...")
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=i,
        use_label_encoder=False, eval_metric='logloss', n_jobs=1
    )
    model.fit(X_scaled, y)
    test_preds_sum += model.predict_proba(X_test_scaled)[:, 1]

avg_preds = test_preds_sum / n_models
final_preds = (avg_preds > 0.5).astype(int)

# 8. Apply WCG Corrections (The Detective)
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

# 9. Submission
print("Saving Submission...")
submission = pd.DataFrame({
    'PassengerId': test_ids.values,
    'Survived': test_df['Survived'].astype(int).values
})
submission.to_csv('submission_v17.csv', index=False)
print("Saved 'submission_v17.csv'")
