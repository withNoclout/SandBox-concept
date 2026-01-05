import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 50)

print("Loading data...")
# Load data
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    submission_df = pd.read_csv('sample_submission.csv')

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    print("\nFirst 5 rows of Train:")
    print(train_df.head())

    print("\nTrain Info:")
    print(train_df.info())

    print("\nMissing values in Train:")
    missing = train_df.isnull().sum()
    print(missing[missing > 0])

    print("\nTarget Distribution (diagnosed_diabetes):")
    if 'diagnosed_diabetes' in train_df.columns:
        print(train_df['diagnosed_diabetes'].value_counts(normalize=True))
    else:
        print("Column 'diagnosed_diabetes' not found in train.csv")

    print("\nNumerical Summary:")
    print(train_df.describe())

except FileNotFoundError:
    print("Error: Data files not found. Make sure train.csv is in the current directory.")
