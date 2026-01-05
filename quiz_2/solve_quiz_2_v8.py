import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# 1. Load Data
print("Loading Titanic data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
passenger_ids = test_df['PassengerId']

# 2. Minimal Preprocessing (No "Smart" Features)
# We stick to the raw data: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = train_df[features].copy()
y = train_df['Survived']
X_test = test_df[features].copy()

# 3. Pipeline Construction
# Numerical: Age, SibSp, Parch, Fare
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) # LR needs scaling
])

# Categorical: Pclass, Sex, Embarked
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

# 4. Models

# Logistic Regression (The "Simple Truth")
lr = LogisticRegression(random_state=1, solver='liblinear')

# Random Forest (The "Baseline Champion")
# Using the parameters from our successful V1/Baseline
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Ensemble (Soft Voting)
ensemble = VotingClassifier(estimators=[
    ('lr', lr), 
    ('rf', rf)
], voting='soft')

# Bundle into Pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('ensemble', ensemble)])

# 5. Validation
print("\nTraining Minimalist Ensemble...")
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {scores.mean():.5f} (+/- {scores.std():.5f})")

# 6. Prediction
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
output.to_csv('submission_v8.csv', index=False)
print("Saved 'submission_v8.csv'")

with open('prediction_summary_v8.txt', 'w') as f:
    f.write("=== Titanic V8 (The Minimalist) ===\n")
    f.write(f"CV Accuracy: {scores.mean():.5f}\n")
    f.write("Features: Raw Data Only (No Titles/Decks)\n")
    f.write("Models: Logistic Regression + Random Forest (Baseline)\n")
    f.write("Strategy: Simplicity & Robustness\n")

print("Saved 'prediction_summary_v8.txt'")
