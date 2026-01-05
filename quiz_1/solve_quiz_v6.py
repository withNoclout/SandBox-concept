import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew
from scipy.special import boxcox1p

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Outlier Removal (Keep this, it's good)
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)

# 3. Preprocessing
all_data = pd.concat([train_df.drop(['SalePrice'], axis=1), test_df]).reset_index(drop=True)
y = np.log1p(train_df["SalePrice"])

# Feature Engineering
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath'])
all_data['Qual_Age'] = all_data['OverallQual'] * all_data['YearBuilt']
all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['Has2ndFloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Skewness
numerical_features = all_data.select_dtypes(include=['int64', 'float64']).columns
skewed_feats = all_data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 0.75]
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

all_data = pd.get_dummies(all_data)
X = all_data.iloc[:len(y), :]
X_test = all_data.iloc[len(y):, :]

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

scaler = RobustScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# 4. Define Models (Linear Heavy)

# Lasso (Robust Linear)
lasso = Lasso(alpha=0.0005, random_state=1)

# Ridge (Regularized Linear)
ridge = Ridge(alpha=0.6, random_state=1)

# ElasticNet (Mix of Lasso/Ridge)
elastic = ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3)

# Gradient Boosting (Robust Tree)
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, 
                                loss='huber', random_state=42)

# XGBoost (Powerful Tree)
xgb = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                   max_depth=3, min_child_weight=0,
                   gamma=0, subsample=0.7,
                   colsample_bytree=0.7,
                   objective='reg:squarederror', nthread=-1,
                   scale_pos_weight=1, seed=27,
                   reg_alpha=0.00006, random_state=42)

# 5. Training
print("Training Lasso...")
lasso.fit(X, y)

print("Training Ridge...")
ridge.fit(X, y)

print("Training ElasticNet...")
elastic.fit(X, y)

print("Training GradientBoosting...")
gbr.fit(X, y)

print("Training XGBoost...")
xgb.fit(X, y)

# 6. Ensemble Prediction (Linear Heavy Weights)
print("Generating V6 predictions...")

pred_lasso = lasso.predict(X_test)
pred_ridge = ridge.predict(X_test)
pred_elastic = elastic.predict(X_test)
pred_gbr = gbr.predict(X_test)
pred_xgb = xgb.predict(X_test)

# WEIGHTS: 60% Linear, 40% Tree
# Lasso: 25%
# Ridge: 25%
# ElasticNet: 10%
# GBR: 20%
# XGB: 20%
log_pred = (0.25 * pred_lasso) + (0.25 * pred_ridge) + (0.10 * pred_elastic) + (0.20 * pred_gbr) + (0.20 * pred_xgb)

final_pred = np.expm1(log_pred)

output = pd.DataFrame({'Id': test_df.Id, 'SalePrice': final_pred})
output.to_csv('submission_v6.csv', index=False)
print("Saved 'submission_v6.csv'")

with open('prediction_summary_v6.txt', 'w') as f:
    f.write("=== House Price Prediction V6 (Linear Heavy) ===\n")
    f.write("Strategy: Prioritize Linear Models to prevent Overfitting.\n")
    f.write("Weights:\n")
    f.write("- Lasso: 25%\n")
    f.write("- Ridge: 25%\n")
    f.write("- ElasticNet: 10%\n")
    f.write("- GradientBoosting: 20%\n")
    f.write("- XGBoost: 20%\n")
    f.write("Total: 60% Linear / 40% Tree\n")

print("Saved 'prediction_summary_v6.txt'")
