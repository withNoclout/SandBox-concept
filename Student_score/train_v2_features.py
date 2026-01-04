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

# --- Feature Engineering ---
def engineer_features(df):
    df = df.copy()
    
    # 1. Interaction: Study Efficiency
    # Avoid division by zero by adding a small epsilon or clipping
    df['study_efficiency'] = df['study_hours'] / (df['sleep_hours'] + 1e-5)
    
    # 2. Interaction: Dedication (Time * Attendance)
    df['dedication'] = df['study_hours'] * (df['class_attendance'] / 100)
    
    # 3. Polynomial: Diminishing returns of study hours
    df['study_hours_sq'] = df['study_hours'] ** 2
    
    # 4. Interaction: Sleep Quality Impact
    # Map sleep quality to numeric to interact with sleep hours
    sleep_map = {'poor': 1, 'average': 2, 'good': 3}
    # Handle potential missing or unknown values safely
    df['sleep_quality_num'] = df['sleep_quality'].map(sleep_map).fillna(2) 
    df['rest_effectiveness'] = df['sleep_hours'] * df['sleep_quality_num']
    
    # 5. Risk Factor: Low Attendance AND Low Study
    # Flag students who are likely to fail
    df['high_risk'] = ((df['class_attendance'] < 60) & (df['study_hours'] < 2)).astype(int)
    
    # 6. Exam Difficulty Adjustment
    # If exam is hard, scores might be lower generally.
    difficulty_map = {'low': 1, 'medium': 2, 'high': 3, 'moderate': 2} # 'moderate' appears in data
    df['difficulty_num'] = df['exam_difficulty'].map(difficulty_map).fillna(2)
    
    return df

print("Engineering features...")
train_eng = engineer_features(train)
test_eng = engineer_features(test)

# Define Categorical Features (Original + New if any)
# Note: 'sleep_quality_num' and 'difficulty_num' are numerical representations, 
# but we keep original categorical columns for CatBoost to use as well.
categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

# Define Target and Features
X = train_eng.drop(['id', 'exam_score'], axis=1)
y = train_eng['exam_score']
X_test = test_eng.drop(['id'], axis=1)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
test_preds = np.zeros(len(X_test))

print(f"Starting {kf.get_n_splits()}-Fold Cross-Validation with New Features...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Initialize CatBoost (Same params as baseline for fair comparison)
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
print(f"Baseline RMSE: 8.7888")
print(f"Improvement: {8.7888 - mean_rmse:.4f}")

# Create Submission
submission['exam_score'] = test_preds
submission.to_csv('submission_v2_features.csv', index=False)
print("Submission file 'submission_v2_features.csv' created successfully.")
