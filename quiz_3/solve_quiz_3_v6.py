import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
import scipy.sparse as sp
import gc
import warnings

warnings.filterwarnings('ignore')

# Config
BASE_PATH = './'
# Dynamic Thresholds for Rounds: [High, Low]
# Round 1: Very Strict (Only the obvious)
# Round 2: Strict
# Round 3: Slightly looser to expand reach
PL_THRESHOLDS = [
    (0.98, 0.01), # Round 1
    (0.97, 0.015), # Round 2
    (0.95, 0.02)  # Round 3
]

# Model Params (Same as V5)
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist',
    'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 1000, 'subsample': 0.8,
    'colsample_bytree': 0.7, 'n_jobs': 4, 'verbosity': 0, 'random_state': 42
}
lgb_params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'num_leaves': 63,
    'learning_rate': 0.03, 'n_estimators': 1000, 'feature_fraction': 0.7,
    'bagging_fraction': 0.8, 'bagging_freq': 3, 'n_jobs': 4, 'verbose': -1, 'random_state': 42
}
cb_params = {
    'loss_function': 'Logloss', 'iterations': 1000, 'learning_rate': 0.04,
    'depth': 6, 'verbose': 0, 'allow_writing_files': False, 'random_seed': 42
}
mlp_params = {
    'hidden_layer_sizes': (64, 32), 'activation': 'relu', 'solver': 'adam',
    'max_iter': 300, 'early_stopping': True, 'random_state': 42
}

# --- GRAPH FUNCTIONS (Reused from V5) ---
def build_adjacency_matrix(social_graph, all_users):
    user_to_idx = {u: i for i, u in enumerate(all_users)}
    N = len(all_users)
    u_idx = social_graph['user_a'].map(user_to_idx).values
    v_idx = social_graph['user_b'].map(user_to_idx).values
    rows = np.concatenate([u_idx, v_idx])
    cols = np.concatenate([v_idx, u_idx])
    data = np.ones(len(rows))
    adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    return adj, user_to_idx

def calculate_pagerank(adj, alpha=0.85, max_iter=30, tol=1e-4, personalization=None):
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    D_inv = sp.diags(1.0 / degrees)
    P = D_inv @ adj
    M = P.T
    N = adj.shape[0]
    if personalization is None:
        v = np.ones(N) / N
        p = np.ones(N) / N
    else:
        v = personalization.copy()
        p = personalization.copy()
    for i in range(max_iter):
        v_new = alpha * (M @ v) + (1 - alpha) * p
        if np.linalg.norm(v_new - v, 1) < tol: break
        v = v_new
    return v

def get_graph_features_cpu(social_graph, all_users):
    adj, user_to_idx = build_adjacency_matrix(social_graph, all_users)
    degrees = np.array(adj.sum(axis=1)).flatten()
    pr = calculate_pagerank(adj)
    df = pd.DataFrame({'user_hash': all_users, 'degree': degrees, 'pagerank': pr})
    return df.set_index('user_hash'), adj, user_to_idx

def run_risk_propagation_cpu(adj, seeds_series, user_to_idx, N):
    p = np.zeros(N)
    seed_users = seeds_series[seeds_series == 1].index
    seed_indices = [user_to_idx[u] for u in seed_users if u in user_to_idx]
    if not seed_indices: return np.zeros(N)
    p[seed_indices] = 1.0 / len(seed_indices)
    return calculate_pagerank(adj, personalization=p)

# --- TRAINING ---
def train_ensemble_cv(X, y, X_test, sample_weight=None):
    """Trains 4 models and returns OOF and Test preds."""
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros((len(X), 4))
    test_p = np.zeros((len(X_test), 4))
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_test_sc = scaler.transform(X_test)
    
    # Handle sample_weight for CV? 
    # It's tricky because we need to split weights too.
    # If sample_weight is None, create ones.
    if sample_weight is None: sample_weight = np.ones(len(X))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        w_tr = sample_weight[tr_idx]
        
        # XGB
        m = xgb.XGBClassifier(**xgb_params)
        m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof[val_idx, 0] = m.predict_proba(X_val)[:, 1]
        test_p[:, 0] += m.predict_proba(X_test)[:, 1] / 3
        
        # LGB
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[val_idx, 1] = m.predict_proba(X_val)[:, 1]
        test_p[:, 1] += m.predict_proba(X_test)[:, 1] / 3
        
        # Cat
        m = CatBoostClassifier(**cb_params)
        m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        oof[val_idx, 2] = m.predict_proba(X_val)[:, 1]
        test_p[:, 2] += m.predict_proba(X_test)[:, 1] / 3
        
        # MLP (No weights for MLP usually, or handled differently)
        m = MLPClassifier(**mlp_params)
        m.fit(X_sc[tr_idx], y_tr)
        oof[val_idx, 3] = m.predict_proba(X_sc[val_idx])[:, 1]
        test_p[:, 3] += m.predict_proba(X_test_sc)[:, 1] / 3
        
    return oof, test_p

def optimize_weights(oof, y):
    """Find best weights for ensemble."""
    def loss_func(weights):
        w = np.exp(weights) / np.sum(np.exp(weights))
        p = np.average(oof, axis=1, weights=w)
        # LogLoss
        epsilon = 1e-15
        p = np.clip(p, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    res = minimize(loss_func, [1, 1, 1, 1], method='Nelder-Mead')
    return np.exp(res.x) / np.sum(np.exp(res.x))

# --- MAIN ---
def main():
    print("="*50 + "\nQuiz 3 V6: Refined Self-Trainer\n" + "="*50)
    
    # 1. Load & Graph
    print("1. Loading & Graph Features...")
    train = pd.read_csv(BASE_PATH + 'train.csv')
    test = pd.read_csv(BASE_PATH + 'test.csv')
    social = pd.read_csv(BASE_PATH + 'social_graph.csv')
    all_users = pd.concat([social['user_a'], social['user_b'], train['user_hash'], test['user_hash']]).unique()
    
    graph_stats, adj, user_to_idx = get_graph_features_cpu(social, all_users)
    
    for df in [train, test]:
        df['pagerank'] = df['user_hash'].map(graph_stats['pagerank']).fillna(0)
        df['degree'] = df['user_hash'].map(graph_stats['degree']).fillna(0)
        
    # 2. Risk Prop
    print("2. Risk Propagation...")
    labeled = train[train['is_cheating'].notna()]
    y = labeled['is_cheating'].values
    N = len(all_users)
    
    # OOF Risk
    oof_risk = pd.Series(index=labeled['user_hash'], dtype=float)
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(labeled, y):
        tr_fold = labeled.iloc[tr_idx]
        risk = run_risk_propagation_cpu(adj, tr_fold.set_index('user_hash')['is_cheating'], user_to_idx, N)
        val_users = labeled.iloc[val_idx]['user_hash']
        val_indices = [user_to_idx[u] for u in val_users if u in user_to_idx]
        oof_risk.loc[val_users] = risk[val_indices]
    train['risk_score'] = train['user_hash'].map(oof_risk).fillna(0)
    
    # Test Risk
    all_seeds = labeled.set_index('user_hash')['is_cheating']
    test_risk = run_risk_propagation_cpu(adj, all_seeds, user_to_idx, N)
    test_indices = [user_to_idx[u] for u in test['user_hash'] if u in user_to_idx]
    test['risk_score'] = test_risk[test_indices]
    
    # 3. Features
    print("3. Feature Engineering...")
    for df in [train, test]:
        df['risk_log'] = np.log1p(df['risk_score'] * 1e5)
        df['risk_degree'] = df['risk_log'] * np.log1p(df['degree'])
        df['risk_pagerank'] = df['risk_score'] * df['pagerank']
        if 'feature_012' in df.columns:
            df['risk_f12'] = df['risk_score'] * df['feature_012'].fillna(-1)
        else:
            df['risk_f12'] = 0
            
    feats = [c for c in train.columns if c.startswith('feature_')] + ['degree', 'pagerank', 'risk_score', 'risk_degree', 'risk_pagerank', 'risk_log', 'risk_f12']
    train[feats] = train[feats].fillna(-1)
    test[feats] = test[feats].fillna(-1)
    
    X = train.loc[labeled.index, feats].values
    y = labeled['is_cheating'].values
    X_test_orig = test[feats].values
    
    # 4. Multi-Round Loop
    current_X = X
    current_y = y
    current_weights = np.ones(len(X))
    
    for round_num in range(3):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Train
        oof, test_preds_stack = train_ensemble_cv(current_X, current_y, X_test_orig, current_weights)
        
        # Optimize Weights (on Original Labeled Data only!)
        # We need to extract the OOF part corresponding to original data
        # But current_X grows...
        # Actually, OOF is only valid for the data we trained on.
        # But we care about performance on the REAL validation set (Original).
        # So we should track indices.
        # Simplified: Just optimize on the current OOF (which includes pseudo labels in CV).
        # Wait, optimizing on Pseudo Labels might be biased.
        # Better: Optimize on the OOF of the *Original* samples.
        # Since we append pseudo labels at the end, the first len(X) are original.
        
        orig_len = len(X)
        oof_orig = oof[:orig_len]
        y_orig = current_y[:orig_len]
        
        best_w = optimize_weights(oof_orig, y_orig)
        print(f"  Best Weights: XGB={best_w[0]:.2f}, LGB={best_w[1]:.2f}, Cat={best_w[2]:.2f}, MLP={best_w[3]:.2f}")
        
        final_test_preds = np.average(test_preds_stack, axis=1, weights=best_w)
        
        # Generate Pseudo Labels
        high, low = PL_THRESHOLDS[round_num]
        idx_cheat = np.where(final_test_preds > high)[0]
        idx_clean = np.where(final_test_preds < low)[0]
        
        print(f"  Pseudo Labels: {len(idx_cheat)} Cheaters, {len(idx_clean)} Clean")
        
        # Prepare Next Round Data
        # Weights
        w_cheat = np.clip(final_test_preds[idx_cheat], 0.5, 0.9)
        w_clean = np.clip(1 - final_test_preds[idx_clean], 0.05, 0.3)
        
        new_weights = np.hstack([np.ones(len(X)), w_cheat, w_clean])
        
        X_pseudo_cheat = X_test_orig[idx_cheat]
        y_pseudo_cheat = np.ones(len(idx_cheat))
        
        X_pseudo_clean = X_test_orig[idx_clean]
        y_pseudo_clean = np.zeros(len(idx_clean))
        
        current_X = np.vstack([X, X_pseudo_cheat, X_pseudo_clean])
        current_y = np.hstack([y, y_pseudo_cheat, y_pseudo_clean])
        current_weights = new_weights
        
        print(f"  New Training Size: {len(current_X)}")

    # 5. Submission
    print("\nSaving Submission...")
    sub = pd.DataFrame({'user_hash': test['user_hash'], 'prediction': final_test_preds})
    sub.to_csv('submission_v6.csv', index=False)
    print("Saved 'submission_v6.csv'")

if __name__ == "__main__":
    main()
