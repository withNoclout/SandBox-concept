import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# 1. Load Data
print("Loading Titanic data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")

# 2. Basic EDA
print("\nMissing Values (Train):")
print(train_df.isnull().sum())

# 3. Preprocessing
# Combine for consistent processing
y = train_df["Survived"]
train_df.drop("Survived", axis=1, inplace=True)
all_data = pd.concat([train_df, test_df]).reset_index(drop=True)

# Feature Selection (Basic)
features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Age"]
X = all_data[features].copy()

# Handling Missing Values
# Age: Median
X["Age"] = X["Age"].fillna(X["Age"].median())
# Fare: Median
X["Fare"] = X["Fare"].fillna(X["Fare"].median())
# Embarked: Mode
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

# Encoding
X = pd.get_dummies(X, columns=["Sex", "Embarked"], drop_first=True)

# Split back
X_train = X.iloc[:len(y), :]
X_test = X.iloc[len(y):, :]

# 4. Model (Random Forest Baseline)
print("\nTraining Random Forest Baseline...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y)

# Validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=1)
model.fit(X_tr, y_tr)
val_preds = model.predict(X_val)
acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {acc:.5f}")

# 5. Prediction
model.fit(X_train, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Saved 'submission.csv'")

with open('prediction_summary.txt', 'w') as f:
    f.write("=== Titanic Baseline (Random Forest) ===\n")
    f.write(f"Validation Accuracy: {acc:.5f}\n")
    f.write("Features: Pclass, Sex, SibSp, Parch, Fare, Embarked, Age\n")
    f.write("Strategy: Basic Imputation + One-Hot Encoding + Random Forest\n")

print("Saved 'prediction_summary.txt'")
