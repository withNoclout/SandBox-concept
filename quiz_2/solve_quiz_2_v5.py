import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# 1. Load Data
print("Loading Titanic data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
passenger_ids = test_df['PassengerId']

# 2. Feature Engineering (From V3 - The Best So Far)
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
    # Bin Fare
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    
    # Deck
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'M')
    deck_mapping = {"A": 1, "B": 1, "C": 1, "D": 2, "E": 2, "F": 3, "G": 3, "T": 3, "M": 4}
    df['Deck'] = df['Deck'].map(deck_mapping).fillna(4).astype(int)
    
    # Drop unused
    df = df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
    
    return df

train_df = process_data(train_df)
test_df = process_data(test_df)

X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]
X_test  = test_df.copy()

print(f"Features: {X_train.columns.tolist()}")

# 3. Models

# Random Forest (Tuned from V3)
# Best Params: {'max_depth': 7, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}
rf = RandomForestClassifier(max_depth=7, max_features='sqrt', min_samples_leaf=4, 
                            min_samples_split=2, n_estimators=200, random_state=1)

# KNN (Needs Scaling)
# We use a Pipeline to scale data automatically before feeding to KNN
knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=10))
])

# 4. Ensemble (Soft Voting)
# We give slightly more weight to RF because it's proven
ensemble = VotingClassifier(estimators=[
    ('rf', rf), 
    ('knn', knn)
], voting='soft', weights=[0.6, 0.4])

# 5. Validation
print("\nTraining Hybrid Ensemble...")
scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {scores.mean():.5f} (+/- {scores.std():.5f})")

# 6. Prediction
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)

output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
output.to_csv('submission_v5.csv', index=False)
print("Saved 'submission_v5.csv'")

with open('prediction_summary_v5.txt', 'w') as f:
    f.write("=== Titanic V5 (The Hybrid: RF + KNN) ===\n")
    f.write(f"CV Accuracy: {scores.mean():.5f}\n")
    f.write("Models: Random Forest (Tuned) + KNN (n=10)\n")
    f.write("Strategy: Diversity (Tree Logic + Distance Logic)\n")

print("Saved 'prediction_summary_v5.txt'")
