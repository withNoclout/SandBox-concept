import pandas as pd
import os

# 1. Load Submissions
print("Loading submissions...")

# Check if files exist
if not os.path.exists('submission_v3.csv'):
    print("Error: submission_v3.csv not found! Please run solve_quiz_v3.py first.")
    exit()

if not os.path.exists('submission_v8.csv'):
    print("Error: submission_v8.csv not found! Please run solve_quiz_v8.py first.")
    exit()

v3 = pd.read_csv('submission_v3.csv')
v8 = pd.read_csv('submission_v8.csv')

print(f"V3 Shape: {v3.shape}")
print(f"V8 Shape: {v8.shape}")

# 2. Blend
# V3 (0.11898) is our best single model. It gets higher weight.
# V8 (0.11992) is our best "complex" model. It gets lower weight to provide diversity.
w_v3 = 0.6
w_v8 = 0.4

print(f"Blending: {w_v3} * V3 + {w_v8} * V8")

final_pred = (w_v3 * v3['SalePrice']) + (w_v8 * v8['SalePrice'])

# 3. Save
output = pd.DataFrame({'Id': v3.Id, 'SalePrice': final_pred})
output.to_csv('submission_v10.csv', index=False)
print("Saved 'submission_v10.csv'")

# 4. Summary
with open('prediction_summary_v10.txt', 'w') as f:
    f.write("=== House Price Prediction V10 (The Final Blend) ===\n")
    f.write("Strategy: Submission Blending (Ensemble of Ensembles).\n")
    f.write(f"Formula: ({w_v3} * V3_Simple) + ({w_v8} * V8_Complex)\n")
    f.write("Rationale: Combining the low-variance V3 with the high-bias V8 to minimize error.\n")

print("Saved 'prediction_summary_v10.txt'")
