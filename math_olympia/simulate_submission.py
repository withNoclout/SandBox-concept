import polars as pl
import os
from submission_deepseek import predict

# Load test data
test_df = pl.read_csv("test.csv")
sample_submission = pl.read_csv("sample_submission.csv")

print(f"Loaded {len(test_df)} test problems.")

# Create a list to store results
results = []

# Iterate through each problem
for i in range(len(test_df)):
    # Slice to get a single row DataFrame, mimicking the API
    test_row = test_df.slice(i, 1)
    # Create a dummy sample submission for this row
    sample_row = sample_submission.slice(0, 1).with_columns(pl.lit(test_row['id'][0]).alias('id'))
    
    print(f"\n--- Processing Problem {i+1}/{len(test_df)} ---")
    try:
        # Call the predict function
        result_df = predict(test_row, sample_row)
        
        # Extract answer
        answer = result_df['answer'][0]
        print(f"Final Answer: {answer}")
        
        results.append({'id': test_row['id'][0], 'answer': answer})
    except Exception as e:
        print(f"Error processing row {i}: {e}")
        results.append({'id': test_row['id'][0], 'answer': 0})

# Create final submission DataFrame
final_df = pl.DataFrame(results)
final_df.write_csv("submission.csv")
print("\n✅ Simulation Complete. Saved 'submission.csv'.")
print(final_df)
