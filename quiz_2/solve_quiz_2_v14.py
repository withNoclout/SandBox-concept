import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
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

# 2. Advanced Feature Engineering (TF-DF Style)
print("Feature Engineering (Tokenization)...")

# A. Name Tokens (TF-IDF)
# TF-DF handles text by tokenizing. We simulate this.
# We use char_wb ngrams to capture sub-parts of names/titles
tfidf_name = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=50)
name_features = tfidf_name.fit_transform(all_data['Name'].fillna("")).toarray()
name_cols = [f'name_tfidf_{i}' for i in range(name_features.shape[1])]
df_name = pd.DataFrame(name_features, columns=name_cols)

# B. Ticket Tokens
# Clean ticket: remove punctuation
all_data['Ticket_Clean'] = all_data['Ticket'].str.replace('[^a-zA-Z0-9]', ' ', regex=True)
tfidf_ticket = TfidfVectorizer(analyzer='word', token_pattern=r'\w+', max_features=20)
ticket_features = tfidf_ticket.fit_transform(all_data['Ticket_Clean'].fillna("")).toarray()
ticket_cols = [f'ticket_tfidf_{i}' for i in range(ticket_features.shape[1])]
df_ticket = pd.DataFrame(ticket_features, columns=ticket_cols)

# C. Standard Features
all_data['Sex'] = all_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())
all_data['Embarked'] = all_data['Embarked'].fillna('S').map({'S':0, 'C':1, 'Q':2})
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1

# Concatenate All
df_final = pd.concat([
    all_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']].reset_index(drop=True),
    df_name,
    df_ticket
], axis=1)

X = df_final[all_data['is_train'] == 1].values
y = all_data[all_data['is_train'] == 1]['Survived'].values
X_test = df_final[all_data['is_train'] == 0].values

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 3. Bagging Ensemble (Many GBTs)
print("Training Bagging Ensemble (50 XGBoost Models)...")
n_models = 50
test_preds_sum = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))

# We use different seeds for diversity
for i in range(n_models):
    if i % 10 == 0: print(f"  Model {i}/{n_models}...")
    
    # Randomize parameters slightly for diversity
    depth = np.random.choice([3, 4, 5, 6])
    subsample = np.random.uniform(0.6, 0.9)
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=depth,
        learning_rate=0.05,
        subsample=subsample,
        colsample_bytree=0.8,
        random_state=i, # Different seed
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=1
    )
    
    model.fit(X_scaled, y)
    test_preds_sum += model.predict_proba(X_test_scaled)[:, 1]

avg_preds = test_preds_sum / n_models
final_preds = (avg_preds > 0.5).astype(int)

# 4. Apply WCG Corrections (The Detective)
print("Applying WCG Corrections...")
# Logic
all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0])
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
all_data['IsWC'] = ((all_data['Sex'] == 1) | (all_data['Title'] == 'Master')).astype(int)

# Reconstruct DF for Logic
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

# 5. Submission
print("Saving Submission...")
submission = pd.DataFrame({
    'PassengerId': test_ids.values,
    'Survived': test_df['Survived'].astype(int).values
})
submission.to_csv('submission_v14.csv', index=False)
print("Saved 'submission_v14.csv'")
