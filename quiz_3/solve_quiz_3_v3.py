import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
social_graph = pd.read_csv('social_graph.csv')

# 2. Feature Engineering: Graph Degree (Safe)
print("Calculating Graph Degree...")
all_nodes = pd.concat([social_graph['user_a'], social_graph['user_b']])
degree_counts = all_nodes.value_counts().to_dict()

train_df['degree'] = train_df['user_hash'].map(degree_counts).fillna(0)
test_df['degree'] = test_df['user_hash'].map(degree_counts).fillna(0)

# 3. Target Prep
train_df['target'] = train_df['is_cheating']
train_df.loc[(train_df['is_cheating'].isna()) & (train_df['high_conf_clean'] == True), 'target'] = 0
labeled_df = train_df[train_df['target'].notna()].copy()

features = [c for c in train_df.columns if c.startswith('feature_')] + ['degree']
X = labeled_df[features]
y = labeled_df['target']
X_test = test_df[features]

print(f"Features: {features}")

# Split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train Models

# Model 1: XGBoost
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42, n_jobs=4)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
p_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_xgb_test = xgb_model.predict_proba(X_test)[:, 1]

# Model 2: CatBoost
print("\nTraining CatBoost...")
cat_model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, loss_function='Logloss', eval_metric='AUC', random_seed=42, verbose=100)
cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
p_cat = cat_model.predict_proba(X_val)[:, 1]
p_cat_test = cat_model.predict_proba(X_test)[:, 1]

# Model 3: LightGBM
print("\nTraining LightGBM...")
lgb_model = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4)
lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc') # Removed verbose=100 for LGBM compatibility check
p_lgb = lgb_model.predict_proba(X_val)[:, 1]
p_lgb_test = lgb_model.predict_proba(X_test)[:, 1]

# 5. Optimize Weights (The Simple Blend)
print("\nOptimizing Blend Weights...")

def calculate_cost(y_true, y_prob, t_pass=0.2, t_block=0.9):
    # Fixed thresholds for weight optimization to save time, 
    # or we can optimize thresholds inside? 
    # Let's use the thresholds found in EDA (0.2, 0.95) as a baseline anchor.
    # Actually, the evaluator optimizes thresholds, so we should optimize AUC or LogLoss.
    # But the user wants to minimize Cost.
    # Let's optimize for AUC/LogLoss as a proxy, OR optimize Cost with fixed thresholds.
    # Let's optimize LogLoss (Cross-Entropy) as it produces well-calibrated probabilities.
    
    # LogLoss proxy
    epsilon = 1e-15
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

def objective(weights):
    # Softmax to ensure sum=1 and positive
    w = np.exp(weights) / np.sum(np.exp(weights))
    blend = w[0]*p_xgb + w[1]*p_cat + w[2]*p_lgb
    return calculate_cost(y_val, blend)

# Initial weights (equal)
init_weights = [1, 1, 1]
res = minimize(objective, init_weights, method='Nelder-Mead')
best_w = np.exp(res.x) / np.sum(np.exp(res.x))

print(f"Best Weights: XGB={best_w[0]:.3f}, Cat={best_w[1]:.3f}, LGB={best_w[2]:.3f}")

# 6. Final Blend
val_blend = best_w[0]*p_xgb + best_w[1]*p_cat + best_w[2]*p_lgb
test_blend = best_w[0]*p_xgb_test + best_w[1]*p_cat_test + best_w[2]*p_lgb_test

print(f"Ensemble AUC: {roc_auc_score(y_val, val_blend):.5f}")

# 7. Cost Check
def exact_cost(y_true, y_prob, t_pass, t_block):
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

# Find best thresholds for the blend
best_cost = float('inf')
thresholds = np.linspace(0, 1, 101)
for t_p in thresholds:
    for t_b in thresholds:
        if t_p >= t_b: continue
        cost = exact_cost(y_val, val_blend, t_p, t_b)
        if cost < best_cost: best_cost = cost

print(f"Estimated Validation Cost: ${best_cost}")
print(f"Normalized Cost: ${best_cost / len(y_val):.2f}")

# 8. Submission
submission = pd.DataFrame({'user_hash': test_df['user_hash'], 'prediction': test_blend})
submission.to_csv('submission_v3.csv', index=False)
print("Saved 'submission_v3.csv'")
