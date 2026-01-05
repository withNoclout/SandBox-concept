import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from tqdm import tqdm

# Config
SOCIAL_PATH = 'social_graph.csv'
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
INPUT_SUB_PATH = 'submission_v3.csv' # The Blend
ALPHA = 0.5 # Balance between Neighbor (Graph) and Self (Model)
MAX_ITER = 10

def run_corrected_LP():
    print("1. Loading Data...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    df_sub = pd.read_csv(INPUT_SUB_PATH)
    df_social = pd.read_csv(SOCIAL_PATH)
    
    print(f"Train: {len(df_train)}, Test: {len(df_test)}, Edges: {len(df_social)}")
    
    # 2. Prepare Nodes
    # We need a unified index for ALL users (Train + Test + Graph)
    # Note: Graph might contain users not in Train/Test?
    # Let's collect all unique users.
    all_users = set(df_train['user_hash']) | set(df_test['user_hash']) | set(df_social['user_a']) | set(df_social['user_b'])
    all_users = list(all_users)
    user_to_idx = {u: i for i, u in enumerate(all_users)}
    idx_to_user = {i: u for i, u in enumerate(all_users)}
    N = len(all_users)
    
    print(f"Total Unique Nodes: {N}")
    
    # 3. Build Sparse Adjacency Matrix
    print("Building Adjacency Matrix...")
    u_idx = df_social['user_a'].map(user_to_idx).values
    v_idx = df_social['user_b'].map(user_to_idx).values
    
    # Symmetric
    rows = np.concatenate([u_idx, v_idx])
    cols = np.concatenate([v_idx, u_idx])
    data = np.ones(len(rows))
    
    adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    # Normalize (Row Stochastic)
    # D^-1 * A
    # Calculate degrees
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1 # Avoid div/0
    D_inv = sp.diags(1.0 / degrees)
    P = D_inv @ adj # Propagation Matrix
    
    # 4. Initialize Labels (Y)
    print("Initializing Labels...")
    Y = np.zeros(N)
    
    # A. Train Nodes (Ground Truth) -> Fixed
    # Handle NaNs in Train (Unlabeled High Conf Clean?)
    # Same logic as before: is_cheating=1 -> 1, is_cheating=0 -> 0, high_conf_clean=1 -> 0
    # Map train users to indices
    
    # Create a map for fast lookup
    train_map = {}
    # 1. Cheaters
    cheaters = df_train[df_train['is_cheating'] == 1]['user_hash']
    for u in cheaters: train_map[u] = 1.0
    
    # 2. Clean
    clean = df_train[(df_train['is_cheating'] == 0) | (df_train['high_conf_clean'] == 1)]['user_hash']
    for u in clean: train_map[u] = 0.0
    
    # B. Test Nodes (Model Predictions) -> Initial State
    # Map sub users
    sub_map = dict(zip(df_sub['user_hash'], df_sub['prediction']))
    
    # Fill Y
    # We also need masks to know which nodes are Train (Fixed)
    is_fixed = np.zeros(N, dtype=bool)
    
    for i, u in enumerate(all_users):
        if u in train_map:
            Y[i] = train_map[u]
            is_fixed[i] = True
        elif u in sub_map:
            Y[i] = sub_map[u]
            # Not fixed, can change
        else:
            # Graph-only nodes? Initialize to 0.5 or 0?
            # Let's assume 0.0 (Clean) or average?
            # Or 0.5 (Unknown).
            Y[i] = 0.5
            
    Y_init = Y.copy()
    
    # 5. Propagate
    print(f"Propagating for {MAX_ITER} iterations (Alpha={ALPHA})...")
    # Formula: Y_new = Alpha * (P @ Y_curr) + (1 - Alpha) * Y_init
    # BUT for Fixed nodes, we force them back to Ground Truth every step.
    
    for i in range(MAX_ITER):
        neighbor_avg = P @ Y
        
        # Update
        Y = ALPHA * neighbor_avg + (1 - ALPHA) * Y_init
        
        # Clamp Fixed Nodes
        Y[is_fixed] = Y_init[is_fixed]
        
        # Optional: Clamp Test nodes to be somewhat close to their model prediction?
        # The formula above does that via (1-Alpha)*Y_init.
        
    # 6. Extract Results
    print("Saving Submission...")
    # We only care about Test users
    test_users = df_test['user_hash'].values
    test_indices = [user_to_idx[u] for u in test_users]
    test_preds = Y[test_indices]
    
    submission = pd.DataFrame({'user_hash': test_users, 'prediction': test_preds})
    submission.to_csv('submission_v4.csv', index=False)
    print("Saved 'submission_v4.csv'")

if __name__ == "__main__":
    run_corrected_LP()
