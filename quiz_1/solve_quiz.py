import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Preprocessing
# Identify target and features
y = train_df['SalePrice']
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select features (simplified for this example, but can be expanded)
# We will use a mix of numerical and categorical features for a better model
# Let's select all numerical columns and a few key categorical ones
numerical_cols = [cname for cname in train_df.columns if train_df[cname].dtype in ['int64', 'float64'] and cname != 'SalePrice' and cname != 'Id']
categorical_cols = [cname for cname in train_df.columns if train_df[cname].dtype == "object" and train_df[cname].nunique() < 10]

my_cols = numerical_cols + categorical_cols
X = train_df[my_cols].copy()
X_test = test_df[my_cols].copy()

# Handle Missing Values (Imputation)
# Numerical: Fill with median
num_imputer = SimpleImputer(strategy='median')
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

# Categorical: Fill with 'Missing' and then Encode
cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

# Ordinal Encoding for Categoricals
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_cols] = ordinal_encoder.fit_transform(X[categorical_cols])
X_test[categorical_cols] = ordinal_encoder.transform(X_test[categorical_cols])

# 3. Validation Split (to prove it works)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

# 4. Model Training
print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(X_train, y_train)

# 5. Validation
val_predictions = model.predict(X_val)
mae = mean_absolute_error(y_val, val_predictions)
print(f"\nValidation MAE: ${mae:,.2f}")
print("This means our predictions are, on average, off by this amount.")

# 6. Final Prediction on Test Data
print("\nGenerating predictions for test.csv...")
# Retrain on all data for better results
model.fit(X, y)
test_predictions = model.predict(X_test)

# 7. Create Submission File
output = pd.DataFrame({'Id': test_df.Id, 'SalePrice': test_predictions})
output.to_csv('submission.csv', index=False)
print("Saved 'submission.csv'")

# 8. Generate "Proof of Work" (Feature Importance)
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

print("\n=== Top 10 Factors Driving Price ===")
print(feature_importance_df)

# Save a summary file
with open('prediction_summary.txt', 'w') as f:
    f.write("=== House Price Prediction Summary ===\n")
    f.write(f"Model: Random Forest Regressor\n")
    f.write(f"Validation Mean Absolute Error: ${mae:,.2f}\n\n")
    f.write("Top 10 Most Important Features:\n")
    f.write(feature_importance_df.to_string(index=False))
    f.write("\n\nSample Predictions (First 5):\n")
    f.write(output.head().to_string(index=False))

print("\nSaved 'prediction_summary.txt'")
