import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
print("Loading Titanic data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
passenger_ids = test_df['PassengerId']

# 2. Feature Engineering (The "Oblique" Strategy)
def process_data(df):
    # --- Basic Features ---
    # Title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping).fillna(0)
    
    # Sex
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
    # Age (Impute)
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
    
    # --- OBLIQUE FEATURES (Interaction Terms) ---
    # Mimicking "Age + Fare" splits by creating them explicitly
    
    # 1. Age * Class (Young rich vs Old poor)
    df['Age_Class'] = df['Age'] * df['Pclass']
    
    # 2. Fare / FamilySize (Cost per person)
    df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']
    
    # 3. Age * Sex (Young male vs Young female)
    df['Age_Sex'] = df['Age'] * df['Sex']
    
    # 4. Class * Sex (Rich female vs Poor male)
    df['Class_Sex'] = df['Pclass'] * df['Sex']
    
    # 5. Fare * Age (Wealth accumulation?)
    df['Fare_Age'] = df['Fare'] * df['Age']
    
    # Drop unused
    df = df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
    
    return df

train_df = process_data(train_df)
test_df = process_data(test_df)

X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]
X_test = test_df.copy()

print(f"Features: {X.columns.tolist()}")

# 3. Optuna Optimization
print("\nRunning Optuna Optimization (100 Trials)...")

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 1,
        'n_jobs': 4
    }
    
    model = XGBClassifier(**param)
    
    # Stratified K-Fold for better stability
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"\nBest Trial:")
print(f"  Value: {study.best_trial.value:.5f}")
print(f"  Params: {study.best_trial.params}")

# 4. Final Training & Prediction
print("\nTraining Final Model with Best Params...")
best_params = study.best_trial.params
best_params['use_label_encoder'] = False
best_params['eval_metric'] = 'logloss'
best_params['random_state'] = 1
best_params['n_jobs'] = 4

final_model = XGBClassifier(**best_params)
final_model.fit(X, y)
predictions = final_model.predict(X_test)

output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
output.to_csv('submission_v6.csv', index=False)
print("Saved 'submission_v6.csv'")

with open('prediction_summary_v6.txt', 'w') as f:
    f.write("=== Titanic V6 (The Imitator: Oblique XGBoost) ===\n")
    f.write(f"Best CV Accuracy: {study.best_trial.value:.5f}\n")
    f.write(f"Best Params: {study.best_trial.params}\n")
    f.write("Strategy: Manual Interaction Features + Optuna Tuning\n")
    f.write("New Features: Age*Class, Fare/Person, Age*Sex, etc.\n")

print("Saved 'prediction_summary_v6.txt'")
