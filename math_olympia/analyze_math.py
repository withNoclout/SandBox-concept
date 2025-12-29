import pandas as pd
import os

pd.set_option('display.max_colwidth', None)

print("--- Reference Data (Training?) ---")
if os.path.exists('reference.csv'):
    ref_df = pd.read_csv('reference.csv')
    print(f"Shape: {ref_df.shape}")
    print(ref_df.head())
    print("\nColumns:", ref_df.columns.tolist())
else:
    print("reference.csv not found")

print("\n--- Test Data ---")
if os.path.exists('test.csv'):
    test_df = pd.read_csv('test.csv')
    print(f"Shape: {test_df.shape}")
    print(test_df.head())
else:
    print("test.csv not found")

print("\n--- Sample Submission ---")
if os.path.exists('sample_submission.csv'):
    sub_df = pd.read_csv('sample_submission.csv')
    print(sub_df.head())
