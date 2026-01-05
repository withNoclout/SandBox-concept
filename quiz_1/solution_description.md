# 📚 Comprehensive Solution History

This document tracks the evolution of our solutions across all quizzes. Use this to identify methods we have already tried and find gaps for future improvements.

---

## 🏠 Quiz 1: House Prices (Regression)
**Goal**: Predict the final sale price of homes.

### 🟢 V1: The Baseline (Random Forest)
- **Method**: Basic Random Forest Regressor.
- **Why**: Handles non-linear relationships better than simple Linear Regression.
- **Result**: Good baseline, but lacks precision.

### 🟢 V2: The Booster (XGBoost + Feature Eng)
- **Method**: XGBoost with "Super Features" (TotalSF, Quality-Adjusted Age).
- **Why**: Gradient boosting corrects errors iteratively; new features capture domain knowledge.

### 🟢 V3: The Elite (Ensemble + Log Transform)
- **Method**: Weighted Average of XGBoost (50%), Lasso (30%), Ridge (20%).
- **Technique**: Log-Transformation of Target (to match RMSLE metric) and Box-Cox for features.
- **Why**: Combines boosting power with linear stability.

### 🟢 V4: The Grandmaster (Stacking + Outlier Removal)
- **Method**: Stacking Regressor (Meta-Learner) combining XGB, LGBM, CatBoost, ElasticNet.
- **Technique**: Removed outliers (Large houses with low prices).
- **Why**: Stacking learns how to combine models better than simple averaging.

### 🟢 V5: The Sniper (Optuna Tuning)
- **Method**: Hyperparameter optimization using Optuna (100 trials).
- **Why**: Mathematically optimal parameters beat manual guessing.

### 🟢 V6: The Recovery (Linear Heavy)
- **Method**: Weighted Ensemble favoring Linear Models (60% Lasso/Ridge/ElasticNet).
- **Why**: Overfitting was an issue; linear models generalize better on small datasets.

### 🟢 V7: The Specialist (Dual Pipeline)
- **Method**: Two separate preprocessing pipelines.
    - **Linear Models**: One-Hot Encoding (for categorical).
    - **Tree Models**: Label Encoding (for ordinal).
- **Why**: Different model types need different data representations.

### 🟢 V8: The Final Truth (Full Data)
- **Method**: Restored "Outliers" but modeled them.
- **Why**: Removing data reduces information. Better to learn from it.

### 🟢 V9: The Surgeon (Smart Imputation)
- **Method**: Imputed `LotFrontage` based on `Neighborhood` median (not global median).
- **Why**: Local context matters (houses in same neighborhood look alike).

### 🟢 V10: The Final Blend (Ensemble of Ensembles)
- **Method**: Blended V3 (Simple) and V8 (Complex) predictions.
- **Why**: Diversity is key. Blending a simple and complex model reduces variance.

---

## 🚢 Quiz 2: Titanic (Classification)
**Goal**: Predict survival (0/1).

### 🟢 V1-V10: The Evolution
- **V1**: Baseline.
- **V2**: Feature Engineering (Title, FamilySize).
- **V3-V5**: Tuning & Hybrid Models (KNN + RF).
- **V6**: Interaction Features (Age * Class).
- **V7**: Bagging (100 XGBoost models).
- **V8-V10**: Minimalist approaches & Blending.

### 🟢 V11: The Neural Stack
- **Method**: Stacking Classifier including an **MLP (Neural Network)**.
- **Why**: Neural Networks capture different patterns than Trees.

### 🟢 V12: The Detective (Ticket Grouping / WCG)
- **Method**: **Post-Processing Correction**.
    - Identified groups (Same Ticket).
    - **Woman-Child-Group (WCG)**: If all women/children in a group died, predict the others died too.
    - If all men in a group lived, predict the others lived too.
- **Why**: Captures "Family Fate" dependency (families lived/died together).

### 🟢 V13: The Self-Taught (Pseudo-Labeling)
- **Method**: **Semi-Supervised Learning**.
    - Trained Teacher on Train.
    - Predicted Test (High Confidence > 95%).
    - Added these "Pseudo-Labels" to Train.
    - Retrained Student.
- **Why**: Increases training size by learning from the Test distribution.

### 🟢 V14: The TF-DF Replica (NLP Features + Bagging)
- **Method**: **Text Analysis**.
    - TF-IDF Vectorization on **Names** and **Tickets**.
    - **Bagging**: 50 XGBoost models with different seeds.
- **Why**: Captures hidden patterns in names ("Mr", "William") and ticket prefixes.

### 🟢 V15: The Tutorial Baseline
- **Method**: Simple Random Forest (4 Features).
- **Why**: User request for a simple baseline to understand the concept.

### 🟢 V16: The Purist (Noise Filtering)
- **Method**: **Data Cleaning**.
    - Identified "Confident Errors" in Training (Model sure, but wrong).
    - Removed these 56 "Noisy" samples.
    - Retrained on Clean data.
- **Why**: Removing confusing outliers helps the model learn general rules.

---

## 🎓 Quiz 3: Mercor Cheating Detection (Binary Classification + Cost)
**Goal**: Minimize Cost (FN=$600, FP=$300/$150).

### 🟢 V1: Baseline (XGB + Graph Degree)
- **Method**: XGBoost/CatBoost with simple `degree` feature.

### 🟢 V2: Label Propagation (Leakage)
- **Method**: Iterative Label Propagation on Social Graph.
- **Issue**: Target Leakage (Perfect Score).

### 🟢 V3: The Simple Blend
- **Method**: Weighted Average of XGB, LGBM, CatBoost. Optimized weights with `scipy.optimize`.

### 🟢 V4: Graph Smoothing
- **Method**: Corrected Label Propagation.
- **Technique**: Used LP to "smooth" the predictions of the V3 ensemble based on neighbors.

### 🟢 V5: Weighted Pseudo-Labeling
- **Method**: **Risk Propagation** (Personalized PageRank) + **2-Round Self-Training**.
- **Why**: Propagates "Risk" from known cheaters to their friends.

### 🟢 V6: Refined Self-Trainer
- **Method**: **3-Round Self-Training** + Scipy Weight Optimization.
- **Why**: Iteratively uncovers more hidden cheaters.

### 🟢 V7: The Graph Convolution (GNN-Style)
- **Method**: **Feature Propagation**.
    - Calculated Mean/Std of **Neighbors' Features** (e.g., `neighbor_mean_feature_1`).
    - Added 18+ new features.
- **Why**: Captures "Homophily in Behavior" (Cheaters act like their friends).

### � V17: The Modernist (MICE + Clustering + Target Encoding)
- **Method**: **Advanced Feature Engineering**.
    - **MICE**: Smart Age imputation.
    - **Clustering**: KMeans (10 clusters) features.
    - **Target Encoding**: Encoded Title/Embarked with survival rates.
- **Result**: Score dropped. Likely overfitting due to complex features on small data.

### 🟢 V18: The Adversarial Validator (Drift Check)
- **Method**: **Adversarial Validation**.
    - Trained classifier to distinguish Train vs Test.
    - **Result**: AUC ~0.53 (No Drift).
    - **Action**: Reweighted training samples anyway.

### 🟢 V19: The Neural Architect (Deep Learning)
- **Method**: **Deep MLP Ensemble**.
    - 3-Layer Neural Network (128-64-32).
    - **Neural Bagging**: 10 Models averaged.
- **Why**: Captures non-linear patterns.
- **Result**: Good (0.791), but slightly behind Trees (0.794).

### 🟢 V20: The Geneticist (Evolutionary Blending)
- **Method**: **Genetic Algorithms**.
    - Used `Differential Evolution` to find optimal weights for V14 (XGB) and V19 (MLP).
    - **Result**: Weights [0.96, 0.04].
    - **Conclusion**: Trees are vastly superior to NNs for this specific dataset.

---

## 🔍 Potential Missing Methods (What we haven't tried yet)

1.  **Symbolic Regression**: Finding a mathematical formula (e.g., `gplearn`).
2.  **AutoML**: H2O / AutoGluon.
3.  **External Data**: Using the extended Titanic dataset (if allowed).

