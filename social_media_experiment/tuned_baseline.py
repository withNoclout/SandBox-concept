import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    return text

def run_tuning():
    print("--- Loading Data ---")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    
    train_df['Original_Message'] = train_df['Original_Message'].fillna("")
    test_df['Original_Message'] = test_df['Original_Message'].fillna("")
    
    print("--- Cleaning Data ---")
    train_df['clean_text'] = train_df['Original_Message'].apply(clean_text)
    test_df['clean_text'] = test_df['Original_Message'].apply(clean_text)
    
    # Define Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    # Define Hyperparameters to tune
    # ngram_range: (1,1) = unigrams only, (1,2) = unigrams + bigrams (pairs of words)
    # C: Inverse of regularization strength (smaller values = stronger regularization)
    parameters = {
        'tfidf__max_features': [5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.1, 1.0, 10.0]
    }
    
    print("--- Starting Grid Search (Hyperparameter Tuning) ---")
    print("This might take a minute...")
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search.fit(train_df['clean_text'], train_df['Extremism_Label'])
    
    print("\n--- Best Parameters ---")
    print(grid_search.best_params_)
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    
    print("\n--- Generating Submission with Best Model ---")
    best_model = grid_search.best_estimator_
    test_preds = best_model.predict(test_df['clean_text'])
    
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Extremism_Label': test_preds
    })
    
    submission.to_csv("submission_tuned.csv", index=False)
    print("Saved submission_tuned.csv")

if __name__ == "__main__":
    run_tuning()
