import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import scipy.sparse as sp
import gc
import warnings

warnings.filterwarnings('ignore')

# Config
BASE_PATH = './'
PSEUDO_LABEL_THRESHOLD_HIGH = 0.96  
PSEUDO_LABEL_THRESHOLD_LOW = 0.02   

# Model Params (V5/V6)
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

# --- GRAPH FUNCTIONS ---
def build_adjacency_matrix(social_graph, all_users):
    print("  Building Sparse Adjacency Matrix...")
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

def compute_neighbor_features(adj, feature_matrix):
    """
    Computes Mean of neighbors' features.
    New_F = D^-1 * A * F
    """
    print("  Computing Neighbor Mean Features (Graph Convolution)...")
    # Normalize Adjacency
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    D_inv = sp.diags(1.0 / degrees)
    P = D_inv @ adj # Row Stochastic
    
    # Propagate Features
    # F_new = P @ F
    # We do this for each feature column
    neighbor_means = P @ feature_matrix
    return neighbor_means

# --- TRAINING ---
def train_round_1_cv(X, y, X_test):
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros((len(X), 4))
    test_p = np.zeros((len(X_test), 4))
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_test_sc = scaler.transform(X_test)
    
    for fold, (tr, val) in enumerate(kf.split(X, y)):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr], y[val]
        
        # XGB
        m = xgb.XGBClassifier(**xgb_params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof[val, 0] = m.predict_proba(X_val)[:, 1]
        test_p[:, 0] += m.predict_proba(X_test)[:, 1] / 3
        
        # LGB
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[val, 1] = m.predict_proba(X_val)[:, 1]
        test_p[:, 1] += m.predict_proba(X_test)[:, 1] / 3
        
        # Cat
        m = CatBoostClassifier(**cb_params)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        oof[val, 2] = m.predict_proba(X_val)[:, 1]
        test_p[:, 2] += m.predict_proba(X_test)[:, 1] / 3
        
        # MLP
        m = MLPClassifier(**mlp_params)
        m.fit(X_sc[tr], y_tr)
        oof[val, 3] = m.predict_proba(X_sc[val])[:, 1]
        test_p[:, 3] += m.predict_proba(X_test_sc)[:, 1] / 3
        
    return oof, test_p

def train_round_2_full(X, y, w, X_test):
    test_p = np.zeros((len(X_test), 4))
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_test_sc = scaler.transform(X_test)
    
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(X, y, sample_weight=w, verbose=False)
    test_p[:, 0] = m.predict_proba(X_test)[:, 1]
    
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X, y, sample_weight=w)
    test_p[:, 1] = m.predict_proba(X_test)[:, 1]
    
    m = CatBoostClassifier(**cb_params)
    m.fit(X, y, sample_weight=w, verbose=False)
    test_p[:, 2] = m.predict_proba(X_test)[:, 1]
    
    m = MLPClassifier(**mlp_params)
    m.fit(X_sc, y)
    test_p[:, 3] = m.predict_proba(X_test_sc)[:, 1]
    
    return test_p

# --- MAIN ---
def main():
    print("="*50 + "\nQuiz 3 V7: The Graph Convolution\n" + "="*50)
    
    print("1. Loading Data...")
    train = pd.read_csv(BASE_PATH + 'train.csv')
    test = pd.read_csv(BASE_PATH + 'test.csv')
    social = pd.read_csv(BASE_PATH + 'social_graph.csv')
    all_users = pd.concat([social['user_a'], social['user_b'], train['user_hash'], test['user_hash']]).unique()
    
    # 2. Graph Features (Basic)
    adj, user_to_idx = build_adjacency_matrix(social, all_users)
    degrees = np.array(adj.sum(axis=1)).flatten()
    pr = calculate_pagerank(adj)
    
    # 3. Graph Convolution (Neighbor Mean Features)
    print("3. Generating Neighbor Features...")
    # We need a matrix of features for ALL users (Train + Test + Graph)
    # Graph-only users have missing features -> Fill with Mean? Or 0?
    # Let's fill with global mean.
    
    feature_cols = [c for c in train.columns if c.startswith('feature_')]
    N = len(all_users)
    F = np.zeros((N, len(feature_cols)))
    
    # Fill F with known values
    # Map user to row index
    # This is slow if done row by row.
    # Faster: Create a DF indexed by user_hash, then reindex to all_users
    
    df_all_features = pd.concat([train.set_index('user_hash')[feature_cols], test.set_index('user_hash')[feature_cols]])
    # Reindex to all_users (sorted by user_to_idx)
    # user_to_idx maps user -> 0..N
    # We need a list of users in order 0..N
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    ordered_users = [idx_to_user[i] for i in range(N)]
    
    df_F = df_all_features.reindex(ordered_users)
    # Fill NaNs (Graph-only nodes) with Mean
    df_F = df_F.fillna(df_F.mean())
    F = df_F.values
    
    # Compute Neighbor Means
    F_neighbor_mean = compute_neighbor_features(adj, F)
    
    # Create DF for new features
    neighbor_cols = [f'neighbor_mean_{c}' for c in feature_cols]
    df_neighbor = pd.DataFrame(F_neighbor_mean, columns=neighbor_cols)
    df_neighbor['user_hash'] = ordered_users
    
    # Merge back
    train = train.merge(df_neighbor, on='user_hash', how='left')
    test = test.merge(df_neighbor, on='user_hash', how='left')
    
    # 4. Risk Propagation (Standard V5)
    print("4. Risk Propagation...")
    labeled = train[train['is_cheating'].notna()]
    y = labeled['is_cheating'].values
    
    # OOF Risk
    oof_risk = pd.Series(index=labeled['user_hash'], dtype=float)
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for tr, val in kf.split(labeled, y):
        tr_fold = labeled.iloc[tr]
        # Seeds
        p = np.zeros(N)
        seed_users = tr_fold[tr_fold['is_cheating']==1]['user_hash']
        seed_idx = [user_to_idx[u] for u in seed_users if u in user_to_idx]
        if seed_idx:
            p[seed_idx] = 1.0/len(seed_idx)
            risk = calculate_pagerank(adj, personalization=p)
            val_users = labeled.iloc[val]['user_hash']
            val_idx_list = [user_to_idx[u] for u in val_users if u in user_to_idx]
            oof_risk.loc[val_users] = risk[val_idx_list]
            
    train['risk_score'] = train['user_hash'].map(oof_risk).fillna(0)
    
    # Test Risk
    p = np.zeros(N)
    seed_users = labeled[labeled['is_cheating']==1]['user_hash']
    seed_idx = [user_to_idx[u] for u in seed_users if u in user_to_idx]
    if seed_idx:
        p[seed_idx] = 1.0/len(seed_idx)
        test_risk = calculate_pagerank(adj, personalization=p)
        test_idx_list = [user_to_idx[u] for u in test['user_hash'] if u in user_to_idx]
        test['risk_score'] = test_risk[test_idx_list]
    else:
        test['risk_score'] = 0
        
    # 5. Final Feature Assembly
    print("5. Feature Assembly...")
    # Add Basic Graph Features
    graph_df = pd.DataFrame({'user_hash': ordered_users, 'degree': degrees, 'pagerank': pr})
    train = train.merge(graph_df, on='user_hash', how='left')
    test = test.merge(graph_df, on='user_hash', how='left')
    
    for df in [train, test]:
        df['risk_log'] = np.log1p(df['risk_score'] * 1e5)
        df['risk_degree'] = df['risk_log'] * np.log1p(df['degree'])
        df['risk_pagerank'] = df['risk_score'] * df['pagerank']
        
    # Columns
    feats = [c for c in train.columns if c.startswith('feature_') or c.startswith('neighbor_mean_')] + \
            ['degree', 'pagerank', 'risk_score', 'risk_degree', 'risk_pagerank', 'risk_log']
            
    train[feats] = train[feats].fillna(0) # Fill NaNs
    test[feats] = test[feats].fillna(0)
    
    X = train.loc[labeled.index, feats].values
    y = labeled['is_cheating'].values
    X_test_orig = test[feats].values
    
    print(f"  Feature Count: {len(feats)}")
    
    # 6. Round 1 Training
    print("6. Round 1 Training...")
    oof, test_stack = train_round_1_cv(X, y, X_test_orig)
    # Weights from V6 (approx)
    w = [0.25, 0.50, 0.20, 0.05]
    preds = np.average(test_stack, axis=1, weights=w)
    
    # 7. Pseudo Labeling (Single Round for speed, but high quality)
    print("7. Pseudo Labeling & Retraining...")
    high, low = 0.98, 0.01
    idx_cheat = np.where(preds > high)[0]
    idx_clean = np.where(preds < low)[0]
    
    w_cheat = np.clip(preds[idx_cheat], 0.5, 0.9)
    w_clean = np.clip(1 - preds[idx_clean], 0.05, 0.3)
    
    X_comb = np.vstack([X, X_test_orig[idx_cheat], X_test_orig[idx_clean]])
    y_comb = np.hstack([y, np.ones(len(idx_cheat)), np.zeros(len(idx_clean))])
    w_comb = np.hstack([np.ones(len(X)), w_cheat, w_clean])
    
    print(f"  Added {len(idx_cheat)} Cheaters, {len(idx_clean)} Clean")
    
    test_stack_2 = train_round_2_full(X_comb, y_comb, w_comb, X_test_orig)
    final_preds = np.average(test_stack_2, axis=1, weights=w)
    
    # 8. Submission
    print("Saving Submission...")
    sub = pd.DataFrame({'user_hash': test['user_hash'], 'prediction': final_preds})
    sub.to_csv('submission_v7.csv', index=False)
    print("Saved 'submission_v7.csv'")

if __name__ == "__main__":
    main()
