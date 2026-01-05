import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.stats import skew
from scipy.special import boxcox1p

# 1. Load Data
print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. Preprocessing (The "Surgeon" Part)

# Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

# Drop 'Id'
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# Remove Outliers (We are keeping the V3/V8 logic of NOT removing too many, 
# but let's remove the 2 most extreme ones that are universally agreed upon in kernels)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# Log-Transform Target
train["SalePrice"] = np.log1p(train["SalePrice"])
y_train = train.SalePrice.values

# Concatenate
ntrain = train.shape[0]
ntest = test.shape[0]
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

print("Smart Imputation...")
# LotFrontage: Group by Neighborhood and fill with median
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# Categorical "None"
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'):
    all_data[col] = all_data[col].fillna('None')

# Numerical 0
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    all_data[col] = all_data[col].fillna(0)

# Mode Imputation (for very few missing)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['Functional'] = all_data['Functional'].fillna("Typ")

# Fix Typos
# Some houses have GarageYrBlt > 2020 (impossible in this dataset). Fix to 2007 (last year of data)
all_data.loc[all_data['GarageYrBlt'] > 2010, 'GarageYrBlt'] = 2007

# Feature Engineering
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['Total_Bath'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                          all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))
all_data['YrBltAndRemod'] = all_data['YearBuilt'] + all_data['YearRemodAdd']
all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['Has2ndFloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Fix Skewness
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 0.75]

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

# One-Hot Encoding
all_data = pd.get_dummies(all_data)
print(f"Total Features: {all_data.shape[1]}")

train = all_data[:ntrain]
test = all_data[ntrain:]

# 3. Modeling

# Validation function
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# Lasso
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

# ElasticNet
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

# Kernel Ridge (The Secret Weapon)
KRR = make_pipeline(RobustScaler(), KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))

# Gradient Boosting
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

# XGBoost
model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# LightGBM
model_lgb = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11,
                              verbose=-1)

print("Training models...")
lasso.fit(train.values, y_train)
ENet.fit(train.values, y_train)
KRR.fit(train.values, y_train)
GBoost.fit(train.values, y_train)
model_xgb.fit(train.values, y_train)
model_lgb.fit(train.values, y_train)

print("Generating V9 predictions...")
pred_lasso = lasso.predict(test.values)
pred_ENet = ENet.predict(test.values)
pred_KRR = KRR.predict(test.values)
pred_GBoost = GBoost.predict(test.values)
pred_xgb = model_xgb.predict(test.values)
pred_lgb = model_lgb.predict(test.values)

# Weighted Average
# We trust the robust linear models (Lasso/ENet/KRR) quite a bit
final_pred = (0.20 * pred_lasso) + \
             (0.20 * pred_ENet) + \
             (0.20 * pred_KRR) + \
             (0.15 * pred_GBoost) + \
             (0.15 * pred_xgb) + \
             (0.10 * pred_lgb)

final_pred = np.expm1(final_pred)

output = pd.DataFrame({'Id': test_ID, 'SalePrice': final_pred})
output.to_csv('submission_v9.csv', index=False)
print("Saved 'submission_v9.csv'")

with open('prediction_summary_v9.txt', 'w') as f:
    f.write("=== House Price Prediction V9 (The Surgeon) ===\n")
    f.write("Strategy: Smart Imputation + KernelRidge + Robust Ensemble.\n")
    f.write("Weights:\n")
    f.write("- Lasso: 20%\n")
    f.write("- ElasticNet: 20%\n")
    f.write("- KernelRidge: 20%\n")
    f.write("- GBoost: 15%\n")
    f.write("- XGB: 15%\n")
    f.write("- LGBM: 10%\n")

print("Saved 'prediction_summary_v9.txt'")
