import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

# 1. Load Data
print("Loading Titanic data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
passenger_ids = test_df['PassengerId']

# 2. Smart Preprocessing (The "Polished" Part)
# We use Title for Age Imputation, but NOT as a feature for the model.
# This keeps the model "Concise" (few features) but the data "High Quality".

def fill_age(df):
    # Extract Title temporarily
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping).fillna(0)
    
    # Impute Age based on Title
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
    
    # Drop Title (We don't want the model to overfit on it, just use it for Age)
    df = df.drop('Title', axis=1)
    return df

train_df = fill_age(train_df)
test_df = fill_age(test_df)

# Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = train_df[features].copy()
y = train_df['Survived']
X_test = test_df[features].copy()

# 3. Pipeline Construction
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Fallback
    ('scaler', StandardScaler())
])

categorical_features = ['Pclass', 'Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Tuning Logistic Regression (Giving it "More Time" and "Better Params")
print("\nTuning Logistic Regression...")

lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(random_state=1, solver='liblinear', max_iter=1000))])

param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(lr_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

best_lr = grid_search.best_estimator_
print(f"Best LR Params: {grid_search.best_params_}")
print(f"Best LR CV Score: {grid_search.best_score_:.5f}")

# 5. Robust Random Forest (Giving it "More Time")
# n_estimators=1000 (More trees = More stable)
rf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=1)
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', rf)])

# 6. Ensemble
# We combine the Tuned LR and the Robust RF
ensemble = VotingClassifier(estimators=[
    ('lr', best_lr), 
    ('rf', rf_pipeline)
], voting='soft')

# 7. Final Training & Prediction
print("\nTraining Final Polished Ensemble...")
ensemble.fit(X, y)
predictions = ensemble.predict(X_test)

output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
output.to_csv('submission_v9.csv', index=False)
print("Saved 'submission_v9.csv'")

with open('prediction_summary_v9.txt', 'w') as f:
    f.write("=== Titanic V9 (The Polished Minimalist) ===\n")
    f.write(f"Best LR CV Score: {grid_search.best_score_:.5f}\n")
    f.write("Strategy: Smart Age Imputation + Tuned LR + Robust RF (1000 Trees)\n")
    f.write("Philosophy: High Quality Data + Simple Models\n")

print("Saved 'prediction_summary_v9.txt'")
