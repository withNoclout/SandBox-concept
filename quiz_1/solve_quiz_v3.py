import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import skew

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Preprocessing & Feature Engineering
# Combine for consistent preprocessing
all_data = pd.concat([train_df.drop(['SalePrice'], axis=1), test_df]).reset_index(drop=True)

# A. Log-Transform Target (The "Secret Weapon")
# We predict log(Price) instead of Price. This minimizes RMSLE (the competition metric).
y = np.log1p(train_df["SalePrice"])

# B. Feature Engineering
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath'])
all_data['Qual_Age'] = all_data['OverallQual'] * all_data['YearBuilt']

# C. Fix Skewness
numerical_features = all_data.select_dtypes(include=['int64', 'float64']).columns
skewed_feats = all_data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 0.75]
print(f"Fixing skewness for {skewness.shape[0]} features...")

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

# D. Handling Categorical Data (One-Hot Encoding)
all_data = pd.get_dummies(all_data)

# Split back to train/test
X = all_data.iloc[:len(y), :]
X_test = all_data.iloc[len(y):, :]

# Handle any remaining NaNs (created by dummies or existing)
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

# Scale data (Important for Lasso/Ridge)
scaler = RobustScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# 3. Model Definition (The Ensemble)

# Model 1: XGBoost (Tree-based)
xgb = XGBRegressor(n_estimators=3000, learning_rate=0.01, max_depth=4, 
                   min_child_weight=0, gamma=0, subsample=0.7, 
                   colsample_bytree=0.7, objective='reg:squarederror', 
                   n_jobs=4, random_state=1)

# Model 2: Lasso (Linear, good for feature selection)
lasso = Lasso(alpha=0.0005, random_state=1)

# Model 3: Ridge (Linear, good for preventing overfitting)
ridge = Ridge(alpha=13, random_state=1)

# 4. Training & Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

print("Training XGBoost...")
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_val)

print("Training Lasso...")
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_val)

print("Training Ridge...")
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_val)

# Weighted Average Ensemble
# We give more weight to XGBoost, but let linear models correct it
final_pred_val = (0.5 * xgb_pred) + (0.3 * lasso_pred) + (0.2 * ridge_pred)

# Calculate RMSLE (Root Mean Squared Log Error) - The actual competition metric
rmsle = np.sqrt(mean_squared_error(y_val, final_pred_val))
print(f"\nValidation RMSLE (V3): {rmsle:.5f}")
print("(Target for Top 1% is < 0.11. Lower is better.)")

# 5. Final Prediction on Test Data
print("\nGenerating V3 predictions...")
# Retrain on full data
xgb.fit(X, y)
lasso.fit(X, y)
ridge.fit(X, y)

final_xgb = xgb.predict(X_test)
final_lasso = lasso.predict(X_test)
final_ridge = ridge.predict(X_test)

# Combine
log_predictions = (0.5 * final_xgb) + (0.3 * final_lasso) + (0.2 * final_ridge)

# Inverse Log Transformation (Convert back to real prices)
final_predictions = np.expm1(log_predictions)

output = pd.DataFrame({'Id': test_df.Id, 'SalePrice': final_predictions})
output.to_csv('submission_v3.csv', index=False)
print("Saved 'submission_v3.csv'")

# 6. Summary
with open('prediction_summary_v3.txt', 'w') as f:
    f.write("=== House Price Prediction V3 (Elite Ensemble) ===\n")
    f.write(f"Validation RMSLE: {rmsle:.5f}\n")
    f.write("Techniques Used:\n")
    f.write("1. Log-Transformation of Target (Minimizes error)\n")
    f.write("2. Box-Cox Transformation of Skewed Features\n")
    f.write("3. Ensemble: XGBoost (50%) + Lasso (30%) + Ridge (20%)\n")

print("Saved 'prediction_summary_v3.txt'")
