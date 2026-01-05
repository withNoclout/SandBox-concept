import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
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

# 2. Preprocessing (Robust & Scaled)
# We use the "Simple" feature set from V8 because Stacking handles complexity well enough.
# We MUST scale data for the Neural Network (MLP).

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = train_df[features].copy()
y = train_df['Survived']
X_test = test_df[features].copy()

numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) # Critical for MLP
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

# 3. Define Level 1 Models (The Experts)

# Expert 1: Random Forest (The Consistent One)
rf = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=1)

# Expert 2: XGBoost (The Booster)
xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, 
                    use_label_encoder=False, eval_metric='logloss', random_state=1)

# Expert 3: Neural Network (The Brain)
# MLP = Multi-Layer Perceptron. 
# Hidden Layers: (128, 64) neurons. 
# Activation: ReLU. 
# Solver: Adam.
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=1)

# 4. Define Stacking Classifier (The Boss)
# The "Final Estimator" (Logistic Regression) takes the predictions of RF, XGB, and MLP
# and learns the best way to combine them.

estimators = [
    ('rf', rf),
    ('xgb', xgb),
    ('mlp', mlp)
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5 # Cross-validation for generating Level 1 predictions
)

# Bundle into Pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('stack', stack)])

# 5. Validation
print("\nTraining Neural Stack...")
# Note: Stacking is computationally expensive because it trains models multiple times (CV).
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {scores.mean():.5f} (+/- {scores.std():.5f})")

# 6. Prediction
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
output.to_csv('submission_v11.csv', index=False)
print("Saved 'submission_v11.csv'")

with open('prediction_summary_v11.txt', 'w') as f:
    f.write("=== Titanic V11 (The Neural Stack) ===\n")
    f.write(f"CV Accuracy: {scores.mean():.5f}\n")
    f.write("Level 1 Models: Random Forest, XGBoost, MLP (Neural Net)\n")
    f.write("Level 2 Model: Logistic Regression\n")
    f.write("Technique: Stacking (Learning the Ensemble)\n")

print("Saved 'prediction_summary_v11.txt'")
