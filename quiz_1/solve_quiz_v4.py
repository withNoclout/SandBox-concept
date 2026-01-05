import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew
from scipy.special import boxcox1p

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Outlier Removal (The "Silver Bullet")
# Removing houses > 4000 sqft that sold for cheap (outliers)
print(f"Original Train Shape: {train_df.shape}")
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)
print(f"New Train Shape (Outliers Removed): {train_df.shape}")

# 3. Preprocessing & Feature Engineering
all_data = pd.concat([train_df.drop(['SalePrice'], axis=1), test_df]).reset_index(drop=True)

# Log-Transform Target
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

# Fix Skewness
numerical_features = all_data.select_dtypes(include=['int64', 'float64']).columns
skewed_feats = all_data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 0.75]
print(f"Fixing skewness for {skewness.shape[0]} features...")

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

# One-Hot Encoding
all_data = pd.get_dummies(all_data)

# Split back
X = all_data.iloc[:len(y), :]
X_test = all_data.iloc[len(y):, :]

# Impute & Scale
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

scaler = RobustScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# 4. Define Models (The "Holy Trinity" + Linear Friends)

# LightGBM
lightgbm = LGBMRegressor(objective='regression', 
                         num_leaves=4,
                         learning_rate=0.01, 
                         n_estimators=5000,
                         max_bin=200, 
                         bagging_fraction=0.75,
                         bagging_freq=5, 
                         bagging_seed=7,
                         feature_fraction=0.2,
                         feature_fraction_seed=7,
                         verbose=-1,
                         random_state=42)

# XGBoost
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=3460,
                       max_depth=3,
                       min_child_weight=0,
                       gamma=0,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)

# CatBoost
catboost = CatBoostRegressor(iterations=3000,
                             learning_rate=0.01,
                             depth=4,
                             l2_leaf_reg=3,
                             loss_function='RMSE',
                             eval_metric='RMSE',
                             random_seed=42,
                             verbose=False)

# Gradient Boosting
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, 
                                loss='huber', random_state=42)

# Linear Models (for Stacking)
ridge = Ridge(alpha=0.6, random_state=42)
lasso = Lasso(alpha=0.0005, random_state=42)
elasticnet = ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42)

# 5. Stacking Ensemble
# We use a StackingRegressor where the "Final Estimator" learns how to combine the base models
print("Training Stacking Ensemble (this may take a minute)...")
stack_gen = StackingRegressor(
    estimators=[
        ('xgboost', xgboost),
        ('lightgbm', lightgbm),
        ('catboost', catboost),
        ('gbr', gbr),
        ('ridge', ridge),
        ('lasso', lasso),
        ('elasticnet', elasticnet)
    ],
    final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
    n_jobs=4
)

# Cross-Validation Score
kf = KFold(n_splits=5, shuffle=True, random_state=42)
score = cross_val_score(stack_gen, X, y, cv=kf, scoring='neg_root_mean_squared_error')
rmsle_score = -score.mean()
print(f"\nCross-Validation RMSLE (V4): {rmsle_score:.5f}")
print("(Target for Top 1% is < 0.11. Lower is better.)")

# 6. Final Training & Prediction
print("Retraining on full data...")
stack_gen.fit(X, y)

print("Generating V4 predictions...")
log_pred = stack_gen.predict(X_test)
final_pred = np.expm1(log_pred)

output = pd.DataFrame({'Id': test_df.Id, 'SalePrice': final_pred})
output.to_csv('submission_v4.csv', index=False)
print("Saved 'submission_v4.csv'")

# 7. Summary
with open('prediction_summary_v4.txt', 'w') as f:
    f.write("=== House Price Prediction V4 (Grandmaster Stack) ===\n")
    f.write(f"CV RMSLE: {rmsle_score:.5f}\n")
    f.write("Techniques Used:\n")
    f.write("1. Outlier Removal (GrLivArea > 4000)\n")
    f.write("2. Stacking Ensemble (XGB + LGBM + CatBoost + GBR + Linear)\n")
    f.write("3. Meta-Model: XGBoost\n")

print("Saved 'prediction_summary_v4.txt'")
