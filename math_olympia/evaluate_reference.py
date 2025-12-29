import polars as pl
import os
from submission_deepseek import predict

# Load reference data
if not os.path.exists("reference.csv"):
    print("reference.csv not found!")
    exit()

ref_df = pl.read_csv("reference.csv")
# Create a dummy sample submission
sample_submission = pl.DataFrame({'id': ref_df['id'], 'answer': [0]*len(ref_df)})

print(f"Loaded {len(ref_df)} reference problems.")

correct_count = 0
results = []

for i in range(len(ref_df)):
    test_row = ref_df.slice(i, 1)
    sample_row = sample_submission.slice(0, 1).with_columns(pl.lit(test_row['id'][0]).alias('id'))
    
    true_answer = test_row['answer'][0]
    
    print(f"\n--- Problem {i+1}: ID {test_row['id'][0]} ---")
    # print(f"Question: {test_row['problem'][0]}")
    
    try:
        result_df = predict(test_row, sample_row)
        pred_answer = result_df['answer'][0]
        
        print(f"Predicted: {pred_answer} | True: {true_answer}")
        
        if pred_answer == true_answer:
            print("✅ CORRECT")
            correct_count += 1
        else:
            print("❌ WRONG")
            
        results.append({'id': test_row['id'][0], 'pred': pred_answer, 'true': true_answer, 'correct': pred_answer == true_answer})
    except Exception as e:
        print(f"Error: {e}")

print(f"\n=== Summary ===")
print(f"Accuracy: {correct_count}/{len(ref_df)} ({correct_count/len(ref_df)*100:.1f}%)")
