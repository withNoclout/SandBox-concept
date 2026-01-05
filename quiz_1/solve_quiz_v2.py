import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Feature Engineering (The "Super Features")
def create_super_features(df):
    # Total Square Footage (Basement + 1st + 2nd) - The #1 Predictor
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # Total Bathrooms (Full + Half)
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    
    # Quality * Age (New Luxury vs Old Luxury)
    df['Qual_Age'] = df['OverallQual'] * df['YearBuilt']
    
    return df

print("Creating Super Features...")
train_df = create_super_features(train_df)
test_df = create_super_features(test_df)

# 3. Preprocessing
y = train_df['SalePrice']
# Add our new features to the list
features = ['TotalSF', 'TotalBath', 'Qual_Age', 'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt']

# Select features
numerical_cols = [cname for cname in train_df.columns if train_df[cname].dtype in ['int64', 'float64'] and cname != 'SalePrice' and cname != 'Id']
categorical_cols = [cname for cname in train_df.columns if train_df[cname].dtype == "object" and train_df[cname].nunique() < 10]

my_cols = numerical_cols + categorical_cols
X = train_df[my_cols].copy()
X_test = test_df[my_cols].copy()

# Imputation
num_imputer = SimpleImputer(strategy='median')
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

# Encoding
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_cols] = ordinal_encoder.fit_transform(X[categorical_cols])
X_test[categorical_cols] = ordinal_encoder.transform(X_test[categorical_cols])

# 4. Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

# 5. Model Training (XGBoost)
print("Training XGBoost model...")
# n_estimators=1000, learning_rate=0.05 are standard starting points for "winning" models
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, random_state=1, early_stopping_rounds=5)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 6. Validation
val_predictions = model.predict(X_val)
mae = mean_absolute_error(y_val, val_predictions)
print(f"\nValidation MAE (V2): ${mae:,.2f}")

# 7. Final Prediction
print("\nGenerating V2 predictions for test.csv...")
# Re-initialize without early stopping for the final fit on all data
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, random_state=1)
model.fit(X, y, verbose=False)
test_predictions = model.predict(X_test)

output = pd.DataFrame({'Id': test_df.Id, 'SalePrice': test_predictions})
output.to_csv('submission_v2.csv', index=False)
print("Saved 'submission_v2.csv'")

# 8. Summary
with open('prediction_summary_v2.txt', 'w') as f:
    f.write("=== House Price Prediction V2 (XGBoost + Feature Engineering) ===\n")
    f.write(f"Validation MAE: ${mae:,.2f}\n")
    f.write("(Lower is better. Compare with V1 MAE: $16,508)\n\n")
    f.write("Key Improvements:\n")
    f.write("1. Added 'TotalSF' (Total Square Footage)\n")
    f.write("2. Added 'TotalBath' (Total Bathrooms)\n")
    f.write("3. Used XGBoost (Gradient Boosting) instead of Random Forest\n")

print("Saved 'prediction_summary_v2.txt'")
