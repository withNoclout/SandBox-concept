import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Load Data
print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# Use original features since v2 didn't improve
categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

X = train.drop(['id', 'exam_score'], axis=1)
y = train['exam_score']
X_test = test.drop(['id'], axis=1)

def objective(trial):
    params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'verbose': 0,
        'random_seed': 42,
        'cat_features': categorical_features
    }
    
    # Use a smaller K-Fold for speed during tuning (e.g., 3 folds)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)
        
    return np.mean(rmse_scores)

print("Starting Optuna Optimization...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20) # 20 trials for speed, increase for better results

print("\nBest trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Train Final Model with Best Params
print("\nTraining Final Model with Best Params...")
best_params = trial.params
best_params['iterations'] = 1500 # Increase iterations for final training
best_params['cat_features'] = categorical_features
best_params['loss_function'] = 'RMSE'
best_params['eval_metric'] = 'RMSE'
best_params['random_seed'] = 42
best_params['verbose'] = 0

kf = KFold(n_splits=5, shuffle=True, random_state=42)
final_preds = np.zeros(len(X_test))
rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = CatBoostRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, use_best_model=True)

    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    rmse_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")
    
    final_preds += model.predict(X_test) / kf.get_n_splits()

print(f"\nFinal Average RMSE: {np.mean(rmse_scores):.4f}")

submission['exam_score'] = final_preds
submission.to_csv('submission_v3_optuna.csv', index=False)
print("Submission file 'submission_v3_optuna.csv' created successfully.")
