import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Load Data
print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# --- Feature Engineering (Ported from Top 20 Snippet) ---
def preprocess(df):
    df_temp = df.copy()
    
    # Numeric Transformations
    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['study_hours_cubed'] = df_temp['study_hours'] ** 3
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['age_squared'] = df_temp['age'] ** 2

    df_temp['log_study_hours'] = np.log1p(df_temp['study_hours'])
    df_temp['log_class_attendance'] = np.log1p(df_temp['class_attendance'])
    df_temp['log_sleep_hours'] = np.log1p(df_temp['sleep_hours'])
    df_temp['sqrt_study_hours'] = np.sqrt(df_temp['study_hours'])
    df_temp['sqrt_class_attendance'] = np.sqrt(df_temp['class_attendance'])

    # Interaction features
    df_temp['study_hours_times_attendance'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_hours_times_sleep'] = df_temp['study_hours'] * df_temp['sleep_hours']
    df_temp['attendance_times_sleep'] = df_temp['class_attendance'] * df_temp['sleep_hours']

    # Ratio features (add small epsilon to avoid division by zero)
    eps = 1e-5
    df_temp['study_hours_over_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_sleep'] = df_temp['class_attendance'] / (df_temp['sleep_hours'] + eps)

    # Encode categorical variables to numeric ordinal values
    # Note: Using map/fillna/astype(int) as in snippet
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}

    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map(sleep_quality_map).fillna(1).astype(int)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map(facility_rating_map).fillna(1).astype(int)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map(exam_difficulty_map).fillna(1).astype(int)

    # Interaction between encoded categoricals and key numeric features
    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']
    df_temp['age_times_study_hours'] = df_temp['age'] * df_temp['study_hours']
    df_temp['age_times_attendance'] = df_temp['age'] * df_temp['class_attendance']

    # Composite feature: learning efficiency
    df_temp['efficiency'] = (df_temp['study_hours'] * df_temp['class_attendance']) / (df_temp['sleep_hours'] + 1)
    
    return df_temp

print("Applying Top 20 Feature Engineering...")
# Combine train and test for consistent Label Encoding of other categoricals
train_len = len(train)
all_data = pd.concat([train.drop('exam_score', axis=1), test], axis=0).reset_index(drop=True)

# Apply the numeric engineering
all_data_eng = preprocess(all_data)

# Label Encode remaining categoricals (gender, course, internet_access, study_method)
# The snippet didn't explicitly show this part but 'base_features' implies they are used.
cat_cols = ['gender', 'course', 'internet_access', 'study_method', 'sleep_quality', 'facility_rating', 'exam_difficulty']
# Note: sleep_quality, facility_rating, exam_difficulty were manually mapped to numerics, 
# but we might want to keep the label encoded versions too or drop them. 
# The snippet kept 'base_features' + 'numeric_features'. 
# Let's assume base_features included the original categoricals.
# We will Label Encode ALL categoricals to be safe for XGBoost.

for col in cat_cols:
    le = LabelEncoder()
    all_data_eng[col] = all_data_eng[col].fillna('Missing').astype(str)
    all_data_eng[col] = le.fit_transform(all_data_eng[col])

# Split back
X = all_data_eng.iloc[:train_len].drop(['id'], axis=1)
y = train['exam_score']
X_test = all_data_eng.iloc[train_len:].drop(['id'], axis=1)

# --- XGBoost Training (Fast Approximation of Top 20) ---
xgb_params = {
    'n_estimators': 3000, # Reduced from 15000 for speed
    'learning_rate': 0.02, # Increased from 0.005
    'max_depth': 9,
    'subsample': 0.75,
    'reg_lambda': 5,
    'reg_alpha': 0.1,
    'colsample_bytree': 0.5,
    'colsample_bynode': 0.6,
    'min_child_weight': 5,
    'tree_method': 'hist',
    'random_state': 42,
    'eval_metric': 'rmse',
    'enable_categorical': True,
    'n_jobs': -1
}

kf = KFold(n_splits=5, shuffle=True, random_state=1003) # Reduced from 10 folds

rmse_scores = []
test_preds = np.zeros(len(X_test))

print(f"Starting {kf.get_n_splits()}-Fold Training with Top 20 Strategy...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = XGBRegressor(**xgb_params)
    
    # Manual early stopping implementation since we removed it from params
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=False
    )
    
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    rmse_scores.append(rmse)
    
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")
    
    test_preds += model.predict(X_test) / kf.get_n_splits()

mean_rmse = np.mean(rmse_scores)
print(f"\nAverage RMSE: {mean_rmse:.4f}")

# Create Submission
submission['exam_score'] = test_preds
submission.to_csv('submission_top20.csv', index=False)
print("Submission file 'submission_top20.csv' created successfully.")
