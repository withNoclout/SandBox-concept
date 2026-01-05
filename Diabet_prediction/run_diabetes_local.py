import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import os

# Configuration
pd.set_option("display.max_columns", 50)

def engineer_features(df):
    df = df.copy()
    
    # --- Based on User's EDA Insights: Genetics + Vitals are key ---
    
    # 1. Metabolic Markers (Triglycerides & Cholesterol)
    # Triglyceride/HDL Ratio is a major marker for metabolic syndrome/insulin resistance
    # Add small epsilon to avoid division by zero
    df['tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-5)
    
    # LDL/HDL Ratio (Cardiovascular risk)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    
    # Non-HDL Cholesterol (Total - HDL) - often better predictor than LDL alone
    df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    # 2. Blood Pressure
    # EDA showed Systolic is strong, Diastolic is weak.
    # Pulse Pressure (Systolic - Diastolic) captures arterial stiffness better.
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    # 3. Genetics & Vitals Interactions
    # Family history is the strongest predictor. Let's interact it with other risk factors.
    # "Bad genetics + Bad habits/state" = High Risk
    df['family_bmi'] = df['family_history_diabetes'] * df['bmi']
    df['family_age'] = df['family_history_diabetes'] * df['age']
    df['family_glucose'] = df['family_history_diabetes'] * df['diet_score'] # Proxy for sugar intake? Or just general interaction
    
    # 4. BMI Interactions
    # BMI impact worsens with Age
    df['bmi_age'] = df['bmi'] * df['age']
    # Waist-to-Hip is another obesity measure, interact with BMI
    df['bmi_waist'] = df['bmi'] * df['waist_to_hip_ratio']
    
    # 5. Drop Weak Predictors?
    # AutoGluon handles weak features well, but we can try dropping noise if needed.
    # For now, we keep them but rely on the strong engineered features.
    
    return df

def main():
    print("Loading data...")
    if not os.path.exists('train.csv'):
        print("Error: train.csv not found!")
        return

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    submission_df = pd.read_csv("sample_submission.csv")
    
    # Drop ID
    if 'id' in train_df.columns:
        train_df = train_df.drop(columns=['id'])
    if 'id' in test_df.columns:
        test_df = test_df.drop(columns=['id'])
        
    print("Engineering features (Focusing on Vitals + Genetics)...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    print(f"Training with {train_df.shape[1]} features...")
    print("Running AutoGluon locally (CPU)...")
    
    predictor = TabularPredictor(
        label='diagnosed_diabetes',
        eval_metric='roc_auc',  # Optimize for AUC
        problem_type='binary',
        path='autogluon_diabetes_model'
    ).fit(
        train_df,
        presets='best_quality',
        time_limit=3600,        # 1 Hour (Adjustable)
        ag_args_fit={'num_gpus': 0} # CPU only
    )
    
    print("Training complete. Predicting on test set...")
    # Get probability of class 1
    preds_proba = predictor.predict_proba(test_df)
    positive_class_probs = preds_proba[1]
    
    submission_df['diagnosed_diabetes'] = positive_class_probs
    submission_df.to_csv('submission_diabetes_local_enhanced.csv', index=False)
    
    print("-" * 30)
    print("Saved submission_diabetes_local_enhanced.csv")
    print("-" * 30)
    
    # Show leaderboard
    print(predictor.leaderboard(silent=True))

if __name__ == "__main__":
    main()
