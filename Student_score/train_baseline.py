import pandas as pd
import numpy as np
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

# Feature Engineering (Basic)
# CatBoost handles categorical features automatically, but we need to specify them.
categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

# Fill missing values if any (CatBoost can handle NaNs, but explicit filling is often safer)
# For simplicity in baseline, we'll let CatBoost handle NaNs or fill with mode/mean if needed.
# Checking for NaNs
print("Missing values in Train:\n", train.isnull().sum()[train.isnull().sum() > 0])

# Define Target and Features
X = train.drop(['id', 'exam_score'], axis=1)
y = train['exam_score']
X_test = test.drop(['id'], axis=1)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
test_preds = np.zeros(len(X_test))

print(f"Starting {kf.get_n_splits()}-Fold Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Initialize CatBoost
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=categorical_features,
        verbose=0,
        early_stopping_rounds=50,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    rmse_scores.append(rmse)
    
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    # Predict on Test Set
    test_preds += model.predict(X_test) / kf.get_n_splits()

mean_rmse = np.mean(rmse_scores)
print(f"\nAverage RMSE: {mean_rmse:.4f}")

# Create Submission
submission['exam_score'] = test_preds
submission.to_csv('submission_baseline.csv', index=False)
print("Submission file 'submission_baseline.csv' created successfully.")
