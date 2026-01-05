import pandas as pd
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv('train.csv')

# Check correlation between ID and Target
# If ID is just a random index, correlation should be ~0.
# If there is drift, it might be non-zero, or the mean target changes over ID.
corr = train_df['id'].corr(train_df['diagnosed_diabetes'])
print(f"Correlation between ID and Target: {corr}")

# Check target mean in chunks of IDs
train_df['id_group'] = pd.qcut(train_df['id'], 10, labels=False)
mean_target_by_group = train_df.groupby('id_group')['diagnosed_diabetes'].mean()
print("\nTarget Mean by ID Decile (0=Start, 9=End):")
print(mean_target_by_group)

# Check if the last 22k rows have a different distribution
last_22k = train_df.iloc[-22000:]
rest = train_df.iloc[:-22000]

print(f"\nOverall Target Mean: {train_df['diagnosed_diabetes'].mean():.4f}")
print(f"Rest Target Mean:    {rest['diagnosed_diabetes'].mean():.4f}")
print(f"Last 22k Target Mean: {last_22k['diagnosed_diabetes'].mean():.4f}")
