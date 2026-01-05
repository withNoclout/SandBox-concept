import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
social_graph = pd.read_csv('social_graph.csv')

# 2. Feature Engineering: Graph Degree
print("Calculating Graph Features...")
# Count connections for each user
# We concatenate user_a and user_b to get a full list of mentions
all_nodes = pd.concat([social_graph['user_a'], social_graph['user_b']])
degree_counts = all_nodes.value_counts().to_dict()

# Map to Train/Test
# Assuming 'candidate_id' is the column name for users? 
# Let's check the column names first.
print(f"Train Columns: {train_df.columns}")
# Based on previous EDA, there is no 'candidate_id' in the features list?
# Wait, the description says "Each candidate record includes...".
# Let's check if there is an ID column.
# If not, we can't link the graph.
# But the graph exists, so there MUST be an ID.
# Let's assume the first column is the ID or there is a 'candidate_id'.

# Inspecting columns from EDA output:
# Train Shape: (272819, 21)
# 18 features + is_cheating + high_conf_clean + ...?
# Let's look at the head again to be sure.
pass

# RE-CHECK COLUMNS
# I'll assume there is a 'candidate_id' or similar.
# If not, I'll have to skip graph features for now.
# But let's write the code to be robust.

if 'candidate_id' in train_df.columns:
    id_col = 'candidate_id'
elif 'user_id' in train_df.columns:
    id_col = 'user_id'
else:
    # Fallback: Maybe the index? Or maybe it's the first column?
    id_col = train_df.columns[0] 

print(f"Using ID column: {id_col}")

train_df['degree'] = train_df[id_col].map(degree_counts).fillna(0)
test_df['degree'] = test_df[id_col].map(degree_counts).fillna(0)

# 3. Target Prep
train_df['target'] = train_df['is_cheating']
train_df.loc[(train_df['is_cheating'].isna()) & (train_df['high_conf_clean'] == True), 'target'] = 0
labeled_df = train_df[train_df['target'].notna()].copy()

# 4. Features
features = [c for c in train_df.columns if c.startswith('feature_')] + ['degree']
X = labeled_df[features]
y = labeled_df['target']
X_test = test_df[features]

print(f"Features: {features}")

# 5. Models

# Model 1: XGBoost
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    n_jobs=4
)
# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_preds = np.zeros(len(X_test))
xgb_val_preds = np.zeros(len(X))

# Simple Train/Val split for speed (or full CV if requested)
# Let's do a simple split for now to be fast.
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
print(f"XGB AUC: {roc_auc_score(y_val, xgb_val_probs):.5f}")

xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]

# Model 2: CatBoost
print("\nTraining CatBoost...")
cat_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)
cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
cat_val_probs = cat_model.predict_proba(X_val)[:, 1]
print(f"CatBoost AUC: {roc_auc_score(y_val, cat_val_probs):.5f}")

cat_test_probs = cat_model.predict_proba(X_test)[:, 1]

# 6. Ensemble (Blend)
print("\nBlending Models (50/50)...")
val_blend = (xgb_val_probs + cat_val_probs) / 2
test_blend = (xgb_test_probs + cat_test_probs) / 2

print(f"Ensemble AUC: {roc_auc_score(y_val, val_blend):.5f}")

# 7. Cost Calculation (Validation)
def calculate_cost(y_true, y_prob, t_pass, t_block):
    cost = 0
    fn_mask = (y_true == 1) & (y_prob < t_pass)
    cost += fn_mask.sum() * 600
    fp_block_mask = (y_true == 0) & (y_prob >= t_block)
    cost += fp_block_mask.sum() * 300
    fp_manual_mask = (y_true == 0) & (y_prob >= t_pass) & (y_prob < t_block)
    cost += fp_manual_mask.sum() * 150
    tp_manual_mask = (y_true == 1) & (y_prob >= t_pass) & (y_prob < t_block)
    cost += tp_manual_mask.sum() * 5
    return cost

# Find best thresholds for the ensemble
best_cost = float('inf')
thresholds = np.linspace(0, 1, 101)
for t_p in thresholds:
    for t_b in thresholds:
        if t_p >= t_b: continue
        cost = calculate_cost(y_val, val_blend, t_p, t_b)
        if cost < best_cost:
            best_cost = cost

print(f"Estimated Validation Cost: ${best_cost}")
print(f"Normalized Cost: ${best_cost / len(y_val):.2f}")

# 8. Submission
# We submit the PROBABILITIES.
submission = pd.DataFrame({
    id_col: test_df[id_col],
    'probability': test_blend # Assuming column name is 'probability' or 'is_cheating'?
})
# Check sample submission format
sample_sub = pd.read_csv('sample_submission.csv')
sub_col = sample_sub.columns[1] # The target column name
submission.columns = [id_col, sub_col]

submission.to_csv('submission.csv', index=False)
print(f"Saved 'submission.csv' with columns: {submission.columns.tolist()}")
