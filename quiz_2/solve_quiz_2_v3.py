import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
print("Loading Titanic data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
passenger_ids = test_df['PassengerId']

# 2. Feature Engineering Function
def process_data(df):
    # Title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping).fillna(0)
    
    # Sex
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
    # Age (Impute with Title Median)
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
    # Bin Age
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age'] = 4
    df['Age'] = df['Age'].astype(int)
    
    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Embarked
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Fare
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    # Bin Fare (Quantiles)
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    
    # --- NEW FEATURE: DECK ---
    # Cabin often looks like 'C85', 'B42'. The first letter is the Deck.
    # Missing cabins are usually lower decks or unknown, mapped to 'M' (Missing)
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'M')
    
    # Group Decks: A,B,C (Top), D,E (Middle), F,G (Bottom), M (Unknown)
    # Map to numbers
    deck_mapping = {"A": 1, "B": 1, "C": 1, "D": 2, "E": 2, "F": 3, "G": 3, "T": 3, "M": 4}
    df['Deck'] = df['Deck'].map(deck_mapping)
    df['Deck'] = df['Deck'].fillna(4).astype(int) # Fill any weird ones with 4
    
    # Drop unused
    df = df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
    
    return df

# Apply Processing
train_df = process_data(train_df)
test_df = process_data(test_df)

X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]
X_test  = test_df.copy()

print(f"Features: {X_train.columns.tolist()}")

# 3. GridSearchCV (Automated Tuning)
print("\nRunning GridSearchCV (This might take a minute)...")

rf = RandomForestClassifier(random_state=1)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.5f}")

# 4. Final Prediction
best_rf = grid_search.best_estimator_
predictions = best_rf.predict(X_test)

# 5. Save
output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
output.to_csv('submission_v3.csv', index=False)
print("Saved 'submission_v3.csv'")

with open('prediction_summary_v3.txt', 'w') as f:
    f.write("=== Titanic V3 (Tuned Random Forest) ===\n")
    f.write(f"Best CV Score: {grid_search.best_score_:.5f}\n")
    f.write(f"Best Params: {grid_search.best_params_}\n")
    f.write("New Features: Deck (from Cabin)\n")
    f.write("Strategy: GridSearchCV Optimization\n")

print("Saved 'prediction_summary_v3.txt'")
