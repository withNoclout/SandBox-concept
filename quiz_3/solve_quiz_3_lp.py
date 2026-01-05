import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
social_graph = pd.read_csv('social_graph.csv')

# 2. Label Propagation
print("Building Graph & Propagating Labels...")

# --- VECTORIZED IMPLEMENTATION (SCIPY SPARSE) ---
from scipy import sparse

print("Vectorizing Graph...")

# Define Known Labels
cheaters = train_df[train_df['is_cheating'] == 1]['user_hash']
clean = train_df[(train_df['is_cheating'] == 0) | (train_df['high_conf_clean'] == 1)]['user_hash']
iterations = 5

# 1. Map all users to 0..N indices
all_users = set(train_df['user_hash']).union(set(test_df['user_hash']))
all_users_list = list(all_users.union(set(social_graph['user_a'])).union(set(social_graph['user_b'])))
user_to_idx = {u: i for i, u in enumerate(all_users_list)}
idx_to_user = {i: u for i, u in enumerate(all_users_list)}
N = len(all_users_list)
print(f"Total Nodes: {N}")

# 2. Build Adjacency Matrix (Sparse)
# We need integer indices for u and v
# Map the edges
u_indices = social_graph['user_a'].map(user_to_idx).values
v_indices = social_graph['user_b'].map(user_to_idx).values

# Create symmetric edges (u->v and v->u)
rows = np.concatenate([u_indices, v_indices])
cols = np.concatenate([v_indices, u_indices])
data = np.ones(len(rows))

# COO Matrix
adj_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(N, N))

# 3. Normalize (Row-stochastic Transition Matrix)
# D = Degree Matrix
degrees = np.array(adj_matrix.sum(axis=1)).flatten()
# Avoid divide by zero
degrees[degrees == 0] = 1
D_inv = sparse.diags(1.0 / degrees)
T = D_inv @ adj_matrix # Transition Matrix

# 4. Initialize Label Vector Y
# 0.5 for unknown
Y = np.full(N, 0.5)

# Set Knowns
cheater_indices = [user_to_idx[u] for u in cheaters if u in user_to_idx]
clean_indices = [user_to_idx[u] for u in clean if u in user_to_idx]

Y[cheater_indices] = 1.0
Y[clean_indices] = 0.0

# 5. Propagate (Matrix Multiplication)
print(f"Propagating for {iterations} iterations (Vectorized)...")
for i in range(iterations):
    Y = T @ Y
    # Reset Knowns (Clamp)
    Y[cheater_indices] = 1.0
    Y[clean_indices] = 0.0
    
    # Check convergence (optional, but fast)
    # print(f"  Iter {i+1} done.")

print("Propagation Complete.")

# 6. Map back
node_scores = {idx_to_user[i]: s for i, s in enumerate(Y)}
node_counts = {idx_to_user[i]: d for i, d in enumerate(degrees)}

# 3. Map back to DataFrames
train_df['lp_score'] = train_df['user_hash'].map(node_scores).fillna(0.5)
test_df['lp_score'] = test_df['user_hash'].map(node_scores).fillna(0.5)

# Add Degree as well (re-using logic)
train_df['degree'] = train_df['user_hash'].map(node_counts).fillna(0)
test_df['degree'] = test_df['user_hash'].map(node_counts).fillna(0)

# 4. Train Ensemble (Boost + Cat + LP)
print("\nTraining Ensemble with LP Features...")

# Target Prep
train_df['target'] = train_df['is_cheating']
train_df.loc[(train_df['is_cheating'].isna()) & (train_df['high_conf_clean'] == True), 'target'] = 0
labeled_df = train_df[train_df['target'].notna()].copy()

features = [c for c in train_df.columns if c.startswith('feature_')] + ['degree', 'lp_score']
X = labeled_df[features]
y = labeled_df['target']
X_test = test_df[features]

print(f"Features: {features}")

# Split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42, n_jobs=4)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]
print(f"XGB AUC: {roc_auc_score(y_val, xgb_val_probs):.5f}")

# CatBoost
print("Training CatBoost...")
cat_model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, loss_function='Logloss', eval_metric='AUC', random_seed=42, verbose=100)
cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
cat_val_probs = cat_model.predict_proba(X_val)[:, 1]
cat_test_probs = cat_model.predict_proba(X_test)[:, 1]
print(f"CatBoost AUC: {roc_auc_score(y_val, cat_val_probs):.5f}")

# Blend
val_blend = (xgb_val_probs + cat_val_probs) / 2
test_blend = (xgb_test_probs + cat_test_probs) / 2
print(f"Ensemble AUC: {roc_auc_score(y_val, val_blend):.5f}")

# Cost Calc
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

best_cost = float('inf')
thresholds = np.linspace(0, 1, 101)
for t_p in thresholds:
    for t_b in thresholds:
        if t_p >= t_b: continue
        cost = calculate_cost(y_val, val_blend, t_p, t_b)
        if cost < best_cost: best_cost = cost

print(f"Estimated Validation Cost: ${best_cost}")
print(f"Normalized Cost: ${best_cost / len(y_val):.2f}")

# Submission
submission = pd.DataFrame({'user_hash': test_df['user_hash'], 'prediction': test_blend})
submission.to_csv('submission_lp.csv', index=False)
print("Saved 'submission_lp.csv'")
