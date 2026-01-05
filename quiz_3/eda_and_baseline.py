import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# social_graph = pd.read_csv('social_graph.csv') # Ignore for baseline

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")

# 2. Analyze Target
# The target is 'is_cheating'.
# However, the training set also has 'high_conf_clean'.
# Rows with 'is_cheating' NaN are unlabeled? Or are they the 'high_conf_clean' ones?
print("\nTarget Distribution:")
print(train_df['is_cheating'].value_counts(dropna=False))
print("\nHigh Conf Clean Distribution:")
print(train_df['high_conf_clean'].value_counts(dropna=False))

# Strategy:
# We only have labels where 'is_cheating' is not NaN.
# But 'high_conf_clean' == True implies 'is_cheating' == 0 (presumably).
# Let's verify if there is overlap.

# Create a unified target for training
# If is_cheating is 0 or 1, use it.
# If is_cheating is NaN and high_conf_clean is True, assume 0.
train_df['target'] = train_df['is_cheating']
train_df.loc[(train_df['is_cheating'].isna()) & (train_df['high_conf_clean'] == True), 'target'] = 0

# Filter out rows that are still NaN (if any)
labeled_df = train_df[train_df['target'].notna()].copy()
print(f"\nLabeled Data Shape: {labeled_df.shape}")
print(labeled_df['target'].value_counts())

# 3. Cost Function
def calculate_cost(y_true, y_prob, t_pass, t_block):
    # Regions:
    # Auto-pass: prob < t_pass
    # Manual Review: t_pass <= prob < t_block
    # Auto-block: prob >= t_block
    
    cost = 0
    
    # Vectorized calculation
    # 1. False Negatives (Cheating passes through)
    # y_true=1 AND prob < t_pass
    fn_mask = (y_true == 1) & (y_prob < t_pass)
    cost += fn_mask.sum() * 600
    
    # 2. False Positive in Auto-block
    # y_true=0 AND prob >= t_block
    fp_block_mask = (y_true == 0) & (y_prob >= t_block)
    cost += fp_block_mask.sum() * 300
    
    # 3. False Positive in Manual Review
    # y_true=0 AND t_pass <= prob < t_block
    fp_manual_mask = (y_true == 0) & (y_prob >= t_pass) & (y_prob < t_block)
    cost += fp_manual_mask.sum() * 150
    
    # 4. True Positive requiring Manual Review
    # y_true=1 AND t_pass <= prob < t_block
    tp_manual_mask = (y_true == 1) & (y_prob >= t_pass) & (y_prob < t_block)
    cost += tp_manual_mask.sum() * 5
    
    # 5. Correct Auto-pass (TN) -> $0
    # 6. Correct Auto-block (TP) -> $0
    
    return cost

# 4. Baseline Model (XGBoost)
features = [c for c in train_df.columns if c.startswith('feature_')]
X = labeled_df[features]
y = labeled_df['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining Baseline XGBoost...")
model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

val_probs = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_probs)
print(f"Validation AUC: {auc:.5f}")

# 5. Find Optimal Thresholds
print("\nOptimizing Thresholds...")
best_cost = float('inf')
best_t_pass = 0
best_t_block = 0

# Grid Search for thresholds
# t_pass must be < t_block
thresholds = np.linspace(0, 1, 101)

for t_p in thresholds:
    for t_b in thresholds:
        if t_p >= t_b:
            continue
            
        cost = calculate_cost(y_val, val_probs, t_p, t_b)
        if cost < best_cost:
            best_cost = cost
            best_t_pass = t_p
            best_t_block = t_b

print(f"Best Cost: ${best_cost}")
print(f"Best Thresholds: Pass < {best_t_pass:.2f}, Block >= {best_t_block:.2f}")
print(f"Normalized Cost (per candidate): ${best_cost / len(y_val):.2f}")
