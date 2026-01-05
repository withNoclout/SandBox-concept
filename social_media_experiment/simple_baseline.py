import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove special characters and lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    return text

def run_baseline():
    print("--- Loading Data ---")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    
    # Handle missing values
    train_df['Original_Message'] = train_df['Original_Message'].fillna("")
    test_df['Original_Message'] = test_df['Original_Message'].fillna("")
    
    print("--- Cleaning Data ---")
    train_df['clean_text'] = train_df['Original_Message'].apply(clean_text)
    test_df['clean_text'] = test_df['Original_Message'].apply(clean_text)
    
    print("--- Feature Extraction (TF-IDF) ---")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(train_df['clean_text'])
    y = train_df['Extremism_Label'] # Assuming this is the target column name from EDA
    
    X_test_final = vectorizer.transform(test_df['clean_text'])
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("--- Training Model (Logistic Regression) ---")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    print("--- Evaluation ---")
    val_preds = model.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, val_preds))
    print("\nClassification Report:\n", classification_report(y_val, val_preds))
    
    print("--- Generating Submission ---")
    test_preds = model.predict(X_test_final)
    
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Extremism_Label': test_preds
    })
    
    submission.to_csv("submission_baseline.csv", index=False)
    print("Saved submission_baseline.csv")

if __name__ == "__main__":
    run_baseline()
