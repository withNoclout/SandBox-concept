import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
print("Loading Titanic data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
passenger_ids = test_df['PassengerId']

# 2. Feature Engineering (Same as V6 - The Imitator)
def process_data(df):
    # Title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping).fillna(0)
    
    # Sex
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
    # Age
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
    
    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Embarked
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Fare
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Deck
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'M')
    deck_mapping = {"A": 1, "B": 1, "C": 1, "D": 2, "E": 2, "F": 3, "G": 3, "T": 3, "M": 4}
    df['Deck'] = df['Deck'].map(deck_mapping).fillna(4).astype(int)
    
    # --- OBLIQUE FEATURES ---
    df['Age_Class'] = df['Age'] * df['Pclass']
    df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']
    df['Age_Sex'] = df['Age'] * df['Sex']
    df['Class_Sex'] = df['Pclass'] * df['Sex']
    df['Fare_Age'] = df['Fare'] * df['Age']
    
    df = df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
    return df

train_df = process_data(train_df)
test_df = process_data(test_df)

X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]
X_test = test_df.copy()

# 3. The Swarm (100 Models)
print("\nUnleashing The Swarm (Training 100 Models)...")

# Best Params from V6 (Optuna)
# Note: We use the params found in the previous step
best_params = {
    'n_estimators': 875, 
    'max_depth': 7, 
    'learning_rate': 0.0124, 
    'subsample': 0.985, 
    'colsample_bytree': 0.739, 
    'gamma': 0.283, 
    'reg_alpha': 4.044, 
    'reg_lambda': 3.648, 
    'min_child_weight': 8,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'n_jobs': 4
}

predictions = np.zeros(len(X_test))
num_models = 100

for i in range(num_models):
    if i % 10 == 0:
        print(f"Training model {i}/{num_models}...")
    
    # Set unique seed for each model
    model = XGBClassifier(**best_params, random_state=i)
    model.fit(X, y)
    
    # Add probabilities
    predictions += model.predict_proba(X_test)[:, 1]

# Average
predictions /= num_models

# Threshold
final_preds = (predictions >= 0.5).astype(int)

# 4. Save
output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': final_preds})
output.to_csv('submission_v7.csv', index=False)
print("Saved 'submission_v7.csv'")

with open('prediction_summary_v7.txt', 'w') as f:
    f.write("=== Titanic V7 (The Swarm) ===\n")
    f.write(f"Models Trained: {num_models}\n")
    f.write("Strategy: Bagging (Averaging 100 Seeded XGBoost Models)\n")
    f.write("Base Model: V6 Optimized XGBoost\n")
    f.write("Goal: Maximum Stability / Variance Reduction\n")

print("Saved 'prediction_summary_v7.txt'")
