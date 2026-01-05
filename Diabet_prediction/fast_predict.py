import pandas as pd
from autogluon.tabular import TabularPredictor
import os

def engineer_features(df):
    # Must match the training script exactly!
    df = df.copy()
    df['tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-5)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['family_bmi'] = df['family_history_diabetes'] * df['bmi']
    df['family_age'] = df['family_history_diabetes'] * df['age']
    df['family_glucose'] = df['family_history_diabetes'] * df['diet_score']
    df['bmi_age'] = df['bmi'] * df['age']
    df['bmi_waist'] = df['bmi'] * df['waist_to_hip_ratio']
    return df

print("Loading Test Data...")
test_df = pd.read_csv("test.csv")
submission_df = pd.read_csv("sample_submission.csv")

if 'id' in test_df.columns:
    test_df = test_df.drop(columns=['id'])

print("Engineering features...")
test_df = engineer_features(test_df)

print("Loading Saved Model...")
predictor = TabularPredictor.load("autogluon_diabetes_model")

# Use LightGBM_BAG_L1 (Score: ~0.7266) for SPEED
# The full ensemble (Score: ~0.7270) is too slow on CPU
model_to_use = 'LightGBM_BAG_L1' 

print(f"Predicting with {model_to_use} (Fast & Accurate)...")
preds_proba = predictor.predict_proba(test_df, model=model_to_use)
positive_class_probs = preds_proba[1]

submission_df['diagnosed_diabetes'] = positive_class_probs
submission_df.to_csv('submission_diabetes_fast.csv', index=False)

print("✅ Done! Saved submission_diabetes_fast.csv")
print(submission_df.head())
