import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# 1. Load Data
print("Loading Titanic data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
passenger_ids = test_df['PassengerId']

# 2. Feature Engineering Function
def process_data(df):
    # Extract Title from Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Map Titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    
    # Sex
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
    # Age (Impute based on Title median - smarter than global median)
    # We'll just use simple median for now to keep it robust, or Pclass/Sex grouping
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
    
    # Bin Age
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age'] = 4
    
    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # IsAlone
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
    
    # Drop unused
    df = df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
    
    return df

# Apply Processing
train_df = process_data(train_df)
test_df = process_data(test_df)

X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]
X_test  = test_df.copy()

print(f"Features: {X_train.columns.tolist()}")

# 3. Models

# Random Forest (Robust Baseline)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Gradient Boosting (The "Boost" user asked for)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)

# XGBoost (The "Extreme Boost")
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, use_label_encoder=False, eval_metric='logloss', random_state=1)

# Voting Ensemble (Soft Voting)
voting_clf = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('xgb', xgb)], voting='soft')

# 4. Training & Validation
print("\nTraining Ensemble...")
scores = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {scores.mean():.5f} (+/- {scores.std():.5f})")

voting_clf.fit(X_train, y_train)
predictions = voting_clf.predict(X_test)

# 5. Save
output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
output.to_csv('submission_v2.csv', index=False)
print("Saved 'submission_v2.csv'")

with open('prediction_summary_v2.txt', 'w') as f:
    f.write("=== Titanic V2 (Feature Engineering + Boosting) ===\n")
    f.write(f"CV Accuracy: {scores.mean():.5f}\n")
    f.write("New Features: Title, FamilySize, IsAlone, AgeBin, FareBin\n")
    f.write("Models: Random Forest + GradientBoosting + XGBoost\n")

print("Saved 'prediction_summary_v2.txt'")
