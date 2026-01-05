import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew
from scipy.special import boxcox1p
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Outlier Removal
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

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. Optuna Optimization

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.00001, 0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.00001, 0.1),
        'n_jobs': 4,
        'random_state': 42
    }
    model = xgb.XGBRegressor(**params)
    score = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error').mean()
    return -score

def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
        'num_leaves': trial.suggest_int('num_leaves', 4, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'verbose': -1,
        'random_state': 42
    }
    model = lgb.LGBMRegressor(**params)
    score = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error').mean()
    return -score

print("Tuning XGBoost (50 trials)...")
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=50)
print(f"Best XGB Params: {study_xgb.best_params}")

print("Tuning LightGBM (50 trials)...")
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=50)
print(f"Best LGB Params: {study_lgb.best_params}")

# 5. Final Training with Best Params

print("Training Final Models...")
best_xgb = xgb.XGBRegressor(**study_xgb.best_params, n_jobs=4, random_state=42)
best_lgb = lgb.LGBMRegressor(**study_lgb.best_params, verbose=-1, random_state=42)
catboost = CatBoostRegressor(iterations=3000, learning_rate=0.01, depth=4, l2_leaf_reg=3, loss_function='RMSE', verbose=False, random_seed=42)
lasso = Lasso(alpha=0.0005, random_state=42)
ridge = Ridge(alpha=0.6, random_state=42)

best_xgb.fit(X, y)
best_lgb.fit(X, y)
catboost.fit(X, y)
lasso.fit(X, y)
ridge.fit(X, y)

# 6. Ensemble Prediction (Weighted Average)
# Weights based on typical performance: XGB/LGB/Cat get more, Linear gets less but stabilizes
print("Generating V5 predictions...")
pred_xgb = best_xgb.predict(X_test)
pred_lgb = best_lgb.predict(X_test)
pred_cat = catboost.predict(X_test)
pred_lasso = lasso.predict(X_test)
pred_ridge = ridge.predict(X_test)

# Weighted Average
log_pred = (0.3 * pred_xgb) + (0.3 * pred_lgb) + (0.2 * pred_cat) + (0.1 * pred_lasso) + (0.1 * pred_ridge)
final_pred = np.expm1(log_pred)

output = pd.DataFrame({'Id': test_df.Id, 'SalePrice': final_pred})
output.to_csv('submission_v5.csv', index=False)
print("Saved 'submission_v5.csv'")

with open('prediction_summary_v5.txt', 'w') as f:
    f.write("=== House Price Prediction V5 (Optuna Tuned) ===\n")
    f.write(f"Best XGB Score (CV): {study_xgb.best_value:.5f}\n")
    f.write(f"Best LGB Score (CV): {study_lgb.best_value:.5f}\n")
    f.write("Strategy: Optuna Tuning + Weighted Ensemble (XGB+LGB+Cat+Lasso+Ridge)\n")

print("Saved 'prediction_summary_v5.txt'")
