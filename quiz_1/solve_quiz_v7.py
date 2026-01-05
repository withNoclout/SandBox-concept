import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import skew
from scipy.special import boxcox1p

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Outlier Removal
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)

# Base Data
y = np.log1p(train_df["SalePrice"])
all_data = pd.concat([train_df.drop(['SalePrice'], axis=1), test_df]).reset_index(drop=True)

# Feature Engineering (Common)
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath'])
all_data['Qual_Age'] = all_data['OverallQual'] * all_data['YearBuilt']

# Skewness Correction (Common)
numerical_features = all_data.select_dtypes(include=['int64', 'float64']).columns
skewed_feats = all_data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 0.75]
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

# --- PIPELINE A: Linear Models (One-Hot Encoding) ---
print("Preparing Pipeline A (One-Hot for Linear)...")
data_linear = pd.get_dummies(all_data)
X_linear = data_linear.iloc[:len(y), :]
X_test_linear = data_linear.iloc[len(y):, :]

# Impute & Scale Linear
imputer_lin = SimpleImputer(strategy='median')
X_linear = imputer_lin.fit_transform(X_linear)
X_test_linear = imputer_lin.transform(X_test_linear)

scaler_lin = RobustScaler()
X_linear = scaler_lin.fit_transform(X_linear)
X_test_linear = scaler_lin.transform(X_test_linear)

# --- PIPELINE B: Tree Models (Label Encoding) ---
print("Preparing Pipeline B (Label Encoding for Trees)...")
data_tree = all_data.copy()
categorical_cols = data_tree.select_dtypes(include=['object']).columns

for col in categorical_cols:
    lbl = LabelEncoder() 
    # Handle new categories in test set by converting to string
    data_tree[col] = data_tree[col].astype(str)
    lbl.fit(list(data_tree[col].values)) 
    data_tree[col] = lbl.transform(list(data_tree[col].values))

X_tree = data_tree.iloc[:len(y), :]
X_test_tree = data_tree.iloc[len(y):, :]

# Impute Tree (No Scaling needed usually, but doesn't hurt)
imputer_tree = SimpleImputer(strategy='median')
X_tree = imputer_tree.fit_transform(X_tree)
X_test_tree = imputer_tree.transform(X_test_tree)

# --- MODEL TRAINING ---

# Linear Models (On Pipeline A)
print("Training Lasso (Linear)...")
lasso = Lasso(alpha=0.0005, random_state=1)
lasso.fit(X_linear, y)

print("Training Ridge (Linear)...")
ridge = Ridge(alpha=0.6, random_state=1)
ridge.fit(X_linear, y)

# Tree Models (On Pipeline B)
print("Training GradientBoosting (Tree)...")
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, 
                                loss='huber', random_state=42)
gbr.fit(X_tree, y)

print("Training XGBoost (Tree)...")
xgb = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                   max_depth=3, min_child_weight=0,
                   gamma=0, subsample=0.7,
                   colsample_bytree=0.7,
                   objective='reg:squarederror', nthread=-1,
                   scale_pos_weight=1, seed=27,
                   reg_alpha=0.00006, random_state=42)
xgb.fit(X_tree, y)

# --- ENSEMBLE ---
print("Generating V7 predictions...")

pred_lasso = lasso.predict(X_test_linear)
pred_ridge = ridge.predict(X_test_linear)
pred_gbr = gbr.predict(X_test_tree)
pred_xgb = xgb.predict(X_test_tree)

# Weighted Average (Balanced)
# Linear: 50% (25+25)
# Tree: 50% (25+25)
log_pred = (0.25 * pred_lasso) + (0.25 * pred_ridge) + (0.25 * pred_gbr) + (0.25 * pred_xgb)
final_pred = np.expm1(log_pred)

output = pd.DataFrame({'Id': test_df.Id, 'SalePrice': final_pred})
output.to_csv('submission_v7.csv', index=False)
print("Saved 'submission_v7.csv'")

with open('prediction_summary_v7.txt', 'w') as f:
    f.write("=== House Price Prediction V7 (The Specialist) ===\n")
    f.write("Strategy: Dual Pipeline (One-Hot for Linear, Label for Trees).\n")
    f.write("Weights:\n")
    f.write("- Lasso (One-Hot): 25%\n")
    f.write("- Ridge (One-Hot): 25%\n")
    f.write("- GBR (Label): 25%\n")
    f.write("- XGB (Label): 25%\n")

print("Saved 'prediction_summary_v7.txt'")
