import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    return text

def run_linearsvc_oof():
    print("--- Loading Data ---")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    
    train_df['Original_Message'] = train_df['Original_Message'].fillna("")
    test_df['Original_Message'] = test_df['Original_Message'].fillna("")
    
    print("--- Cleaning Data ---")
    train_df['clean_text'] = train_df['Original_Message'].apply(clean_text)
    test_df['clean_text'] = test_df['Original_Message'].apply(clean_text)
    
    # TF-IDF Vectorization
    print("--- Feature Extraction (TF-IDF) ---")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(train_df['clean_text'])
    y = train_df['Extremism_Label']
    X_test = vectorizer.transform(test_df['clean_text'])
    
    # OOF Implementation
    print("--- Training LinearSVC (5-Fold OOF) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df), dtype=object)
    test_preds_list = []
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # LinearSVC does not output probabilities by default, so we use CalibratedClassifierCV if we needed probs
        # But for simple accuracy/label prediction, plain LinearSVC is fine.
        # However, to be safe and robust, let's just predict classes directly.
        
        model = LinearSVC(C=1.0, random_state=42, max_iter=2000)
        model.fit(X_train_fold, y_train_fold)
        
        val_pred = model.predict(X_val_fold)
        oof_preds[val_idx] = val_pred
        
        score = accuracy_score(y_val_fold, val_pred)
        fold_scores.append(score)
        print(f"Fold {fold+1} Accuracy: {score:.4f}")
        
        # Predict on test set
        test_pred = model.predict(X_test)
        test_preds_list.append(test_pred)
        
    print("\n--- OOF Results ---")
    overall_acc = accuracy_score(y, oof_preds)
    print(f"Overall OOF Accuracy: {overall_acc:.4f}")
    print(f"Average Fold Accuracy: {np.mean(fold_scores):.4f}")
    print("\nClassification Report:\n", classification_report(y, oof_preds))
    
    # Majority Vote for Test Predictions
    print("--- Generating Submission (Majority Vote) ---")
    # Transpose to get (n_samples, n_folds)
    test_preds_matrix = np.array(test_preds_list).T 
    
    final_test_preds = []
    for row in test_preds_matrix:
        # Find the most common prediction
        unique, counts = np.unique(row, return_counts=True)
        final_test_preds.append(unique[np.argmax(counts)])
        
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Extremism_Label': final_test_preds
    })
    
    submission.to_csv("submission_linearsvc_oof.csv", index=False)
    print("Saved submission_linearsvc_oof.csv")

if __name__ == "__main__":
    run_linearsvc_oof()
