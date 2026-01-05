import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# 1. Load Data
print("Loading Titanic data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
passenger_ids = test_df['PassengerId']

# 2. Feature Engineering (Keep the good stuff from V3)
def process_data(df):
    # Title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Deck
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'M')
    
    return df

train_df = process_data(train_df)
test_df = process_data(test_df)

# 3. Define Features
# We will let the Pipeline handle encoding/imputing
numerical_cols = ['Age', 'Fare', 'FamilySize']
categorical_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'Deck']

X = train_df[numerical_cols + categorical_cols].copy()
y = train_df['Survived']
X_test = test_df[numerical_cols + categorical_cols].copy()

# 4. Build Pipeline (The "Intermediate ML" Way)

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define Model (XGBoost)
xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, n_jobs=4, random_state=1, use_label_encoder=False, eval_metric='logloss')

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', xgb)
                             ])

# 5. GridSearchCV (Tuning the Pipeline)
print("\nRunning GridSearchCV on Pipeline...")

param_grid = {
    'model__n_estimators': [100, 200, 500],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(my_pipeline, param_grid, cv=5, verbose=1)
grid_search.fit(X, y)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.5f}")

# 6. Final Prediction
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
output.to_csv('submission_v4.csv', index=False)
print("Saved 'submission_v4.csv'")

with open('prediction_summary_v4.txt', 'w') as f:
    f.write("=== Titanic V4 (Intermediate ML: Pipelines + Tuned XGB) ===\n")
    f.write(f"Best CV Score: {grid_search.best_score_:.5f}\n")
    f.write(f"Best Params: {grid_search.best_params_}\n")
    f.write("Technique: sklearn Pipeline + ColumnTransformer + GridSearchCV\n")

print("Saved 'prediction_summary_v4.txt'")
