import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
import scipy.sparse as sp
import gc
import warnings

warnings.filterwarnings('ignore')

# Config
BASE_PATH = './'
PSEUDO_LABEL_THRESHOLD_HIGH = 0.96  
PSEUDO_LABEL_THRESHOLD_LOW = 0.02   

# Model Params (CPU Optimized)
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

# --- GRAPH FUNCTIONS (CPU) ---

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
    """
    Power Iteration for PageRank (CPU Sparse).
    v = alpha * M * v + (1-alpha) * personalization
    """
    N = adj.shape[0]
    
    # Create Transition Matrix M
    # M = A * D^-1 (Column Stochastic)
    # Actually, standard PR uses Row Stochastic P = D^-1 * A, then v = P.T * v
    # Let's stick to: v_new = alpha * P.T @ v_old + (1-alpha) * e
    
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    D_inv = sp.diags(1.0 / degrees)
    P = D_inv @ adj
    M = P.T # Transpose for power iteration on column vector
    
    # Init v
    if personalization is None:
        v = np.ones(N) / N
        p = np.ones(N) / N
    else:
        v = personalization.copy()
        p = personalization.copy()
        
    for i in range(max_iter):
        v_new = alpha * (M @ v) + (1 - alpha) * p
        err = np.linalg.norm(v_new - v, 1)
        v = v_new
        if err < tol:
            break
            
    return v

def get_graph_features_cpu(social_graph, all_users):
    print("  Computing Graph Features (CPU)...")
    adj, user_to_idx = build_adjacency_matrix(social_graph, all_users)
    
    # 1. Degree
    degrees = np.array(adj.sum(axis=1)).flatten()
    
    # 2. PageRank (Global)
    pr = calculate_pagerank(adj)
    
    # 3. Clustering Coeff (Too slow on CPU for 1.8M nodes without optimized C++ lib)
    # We will skip Triangle Count/Clustering for CPU version to save time.
    # Or use a very rough approximation? Better to skip.
    clustering = np.zeros(len(all_users))
    
    # Map back to DataFrame
    df = pd.DataFrame({
        'user_hash': all_users,
        'degree': degrees,
        'pagerank': pr,
        'clustering': clustering
    })
    return df.set_index('user_hash'), adj, user_to_idx

def run_risk_propagation_cpu(adj, seeds_series, user_to_idx, N):
    """Personalized PageRank with Seeds."""
    # Create personalization vector
    p = np.zeros(N)
    
    # Map seeds to indices
    seed_users = seeds_series[seeds_series == 1].index
    seed_indices = [user_to_idx[u] for u in seed_users if u in user_to_idx]
    
    if len(seed_indices) == 0:
        return np.zeros(N)
        
    p[seed_indices] = 1.0 / len(seed_indices)
    
    # Run PPR
    risk_scores = calculate_pagerank(adj, personalization=p)
    return risk_scores

# --- TRAINING FUNCTIONS ---

def train_round_1_cv(X, y, X_test, feature_cols):
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_stack = np.zeros((len(X), 4))
    test_stack = np.zeros((len(X_test), 4))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    print("  Starting 3-Fold CV...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # XGB
        model_xgb = xgb.XGBClassifier(**xgb_params)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_stack[val_idx, 0] = model_xgb.predict_proba(X_val)[:, 1]
        test_stack[:, 0] += model_xgb.predict_proba(X_test)[:, 1] / 3
        
        # LGB
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_stack[val_idx, 1] = model_lgb.predict_proba(X_val)[:, 1]
        test_stack[:, 1] += model_lgb.predict_proba(X_test)[:, 1] / 3
        
        # Cat
        model_cb = CatBoostClassifier(**cb_params)
        model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        oof_stack[val_idx, 2] = model_cb.predict_proba(X_val)[:, 1]
        test_stack[:, 2] += model_cb.predict_proba(X_test)[:, 1] / 3
        
        # MLP
        model_mlp = MLPClassifier(**mlp_params)
        model_mlp.fit(X_scaled[train_idx], y_tr)
        oof_stack[val_idx, 3] = model_mlp.predict_proba(X_scaled[val_idx])[:, 1]
        test_stack[:, 3] += model_mlp.predict_proba(X_test_scaled)[:, 1] / 3
        
    return oof_stack, test_stack

def train_round_2_full(X_combined, y_combined, weights, X_test, feature_cols):
    print("  Training full models with sample weights...")
    test_preds = np.zeros((len(X_test), 4))
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_combined)
    X_test_sc = scaler.transform(X_test)

    # XGB
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_combined, y_combined, sample_weight=weights, verbose=False)
    test_preds[:, 0] = model_xgb.predict_proba(X_test)[:, 1]

    # LGB
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_combined, y_combined, sample_weight=weights)
    test_preds[:, 1] = model_lgb.predict_proba(X_test)[:, 1]

    # Cat
    model_cb = CatBoostClassifier(**cb_params)
    model_cb.fit(X_combined, y_combined, sample_weight=weights, verbose=False)
    test_preds[:, 2] = model_cb.predict_proba(X_test)[:, 1]

    # MLP
    model_mlp = MLPClassifier(**mlp_params)
    model_mlp.fit(X_sc, y_combined)
    test_preds[:, 3] = model_mlp.predict_proba(X_test_sc)[:, 1]
    
    return test_preds

# --- MAIN ---

def main():
    print("=" * 50)
    print("Mercor Cheating Detection - Weighted Pseudo Labeling (CPU)")
    print("=" * 50)
    
    # 1. Load Data
    print("\n1. Loading data...")
    train = pd.read_csv(BASE_PATH + 'train.csv')
    test = pd.read_csv(BASE_PATH + 'test.csv')
    social_graph = pd.read_csv(BASE_PATH + 'social_graph.csv')
    
    all_users = pd.concat([
        social_graph['user_a'], social_graph['user_b'], 
        train['user_hash'], test['user_hash']
    ]).unique()
    
    # 2. Graph Features
    print("\n2. Computing Graph Features...")
    graph_stats, adj, user_to_idx = get_graph_features_cpu(social_graph, all_users)
    
    # Map features
    def map_feature(df, col):
        return df['user_hash'].map(graph_stats[col]).fillna(0)
        
    for df in [train, test]:
        df['pagerank'] = map_feature(df, 'pagerank')
        df['degree'] = map_feature(df, 'degree')
        # df['clustering'] = map_feature(df, 'clustering') # Skipped
        
    # 3. Risk Propagation
    print("\n3. Risk Propagation (PPR)...")
    labeled_train = train[train['is_cheating'].notna()]
    y = labeled_train['is_cheating'].values
    
    # OOF Risk Scores
    oof_risk = pd.Series(index=labeled_train['user_hash'], dtype=float)
    kf_feat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    N = len(all_users)
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    
    for fold, (train_idx, val_idx) in enumerate(kf_feat.split(labeled_train, y)):
        train_fold = labeled_train.iloc[train_idx]
        train_seeds = train_fold.set_index('user_hash')['is_cheating']
        
        risk_vec = run_risk_propagation_cpu(adj, train_seeds, user_to_idx, N)
        
        # Map back
        val_users = labeled_train.iloc[val_idx]['user_hash']
        val_indices = [user_to_idx[u] for u in val_users if u in user_to_idx]
        oof_risk.loc[val_users] = risk_vec[val_indices]
        
    train['risk_score'] = train['user_hash'].map(oof_risk).fillna(0)
    
    # Test Risk Scores (Full Train Seeds)
    all_seeds = labeled_train.set_index('user_hash')['is_cheating']
    test_risk_vec = run_risk_propagation_cpu(adj, all_seeds, user_to_idx, N)
    
    # Map back efficiently
    test_indices = [user_to_idx[u] for u in test['user_hash'] if u in user_to_idx]
    test['risk_score'] = test_risk_vec[test_indices]
    
    # 4. Feature Engineering
    print("\n4. Feature Engineering...")
    for df in [train, test]:
        df['risk_log'] = np.log1p(df['risk_score'] * 1e5)
        df['risk_degree'] = df['risk_log'] * np.log1p(df['degree'])
        df['risk_pagerank'] = df['risk_score'] * df['pagerank']
        # df['risk_f12'] = df['risk_score'] * df['feature_012'].fillna(-1) # feature_012 might not exist?
        # Let's check columns first. Assuming feature_012 exists.
        if 'feature_012' in df.columns:
            df['risk_f12'] = df['risk_score'] * df['feature_012'].fillna(-1)
        else:
            df['risk_f12'] = 0

    feature_cols = [c for c in train.columns if c.startswith('feature_')] + \
                   ['degree', 'pagerank', 'risk_score', 
                    'risk_degree', 'risk_pagerank', 'risk_log', 'risk_f12']
    
    train[feature_cols] = train[feature_cols].fillna(-1)
    test[feature_cols] = test[feature_cols].fillna(-1)
    
    X = train.loc[labeled_train.index, feature_cols].values
    y = labeled_train['is_cheating'].values
    X_test_orig = test[feature_cols].values
    
    # 5. Round 1
    print("\n5. Round 1: Initial Ensemble...")
    oof_stack_1, test_stack_1 = train_round_1_cv(X, y, X_test_orig, feature_cols)
    
    ensemble_weights = np.array([0.35, 0.30, 0.25, 0.10])
    test_preds_1 = np.average(test_stack_1, axis=1, weights=ensemble_weights)
    
    # 6. Pseudo Labeling
    print("\n6. Generating Weighted Pseudo Labels...")
    high_conf_idx_cheat = np.where(test_preds_1 > PSEUDO_LABEL_THRESHOLD_HIGH)[0]
    high_conf_idx_clean = np.where(test_preds_1 < PSEUDO_LABEL_THRESHOLD_LOW)[0]
    
    pseudo_cheat_weights = np.clip(test_preds_1[high_conf_idx_cheat], 0.5, 0.9)
    pseudo_clean_weights = np.clip(1 - test_preds_1[high_conf_idx_clean], 0.05, 0.3)
    
    orig_weights = np.ones(len(X))
    sample_weights = np.hstack([orig_weights, pseudo_cheat_weights, pseudo_clean_weights])
    
    X_pseudo_cheat = X_test_orig[high_conf_idx_cheat]
    y_pseudo_cheat = np.ones(len(high_conf_idx_cheat))
    
    X_pseudo_clean = X_test_orig[high_conf_idx_clean]
    y_pseudo_clean = np.zeros(len(high_conf_idx_clean))
    
    X_combined = np.vstack([X, X_pseudo_cheat, X_pseudo_clean])
    y_combined = np.hstack([y, y_pseudo_cheat, y_pseudo_clean])
    
    print(f"  Orig: {len(X)} | Pseudo Cheat: {len(X_pseudo_cheat)} | Pseudo Clean: {len(X_pseudo_clean)}")
    
    # 7. Round 2
    print("\n7. Round 2: Retraining with Pseudo Labels...")
    test_stack_2 = train_round_2_full(X_combined, y_combined, sample_weights, X_test_orig, feature_cols)
    test_preds_final = np.average(test_stack_2, axis=1, weights=ensemble_weights)
    
    # 8. Submission
    print("\n8. Saving Submission...")
    submission = pd.DataFrame({
        'user_hash': test['user_hash'],
        'prediction': test_preds_final
    })
    submission.to_csv('submission_v5.csv', index=False)
    print("Saved 'submission_v5.csv'")

if __name__ == "__main__":
    main()
