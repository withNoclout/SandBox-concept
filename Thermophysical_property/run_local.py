import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import optuna
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, GraphDescriptors, Fragments, rdFingerprintGenerator
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
import os

# Configuration
pd.set_option("display.max_columns", 50)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
            
        # 1. Basic Descriptors
        features = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'RingCount': Descriptors.RingCount(mol),
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
            'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
            'BertzCT': GraphDescriptors.BertzCT(mol),
            'HallKierAlpha': GraphDescriptors.HallKierAlpha(mol),
        }
        
        # 2. Fragment Counts (Functional Groups - Critical for Melting Point)
        # Adding a few key ones, there are many more in Fragments
        features['fr_Al_COO'] = Fragments.fr_Al_COO(mol)
        features['fr_Al_OH'] = Fragments.fr_Al_OH(mol)
        features['fr_Ar_COO'] = Fragments.fr_Ar_COO(mol)
        features['fr_Ar_N'] = Fragments.fr_Ar_N(mol)
        features['fr_Ar_OH'] = Fragments.fr_Ar_OH(mol)
        features['fr_COO'] = Fragments.fr_COO(mol)
        features['fr_C_O'] = Fragments.fr_C_O(mol)
        features['fr_C_O_noCOO'] = Fragments.fr_C_O_noCOO(mol)
        features['fr_NH0'] = Fragments.fr_NH0(mol)
        features['fr_NH1'] = Fragments.fr_NH1(mol)
        features['fr_NH2'] = Fragments.fr_NH2(mol)
        features['fr_benzene'] = Fragments.fr_benzene(mol)
        features['fr_bicyclic'] = Fragments.fr_bicyclic(mol)
        features['fr_halogen'] = Fragments.fr_halogen(mol)
        
        # 3. Morgan Fingerprints (Radius 2, 1024 bits)
        mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fp = mfgen.GetFingerprint(mol)
        fp_bits = list(fp)
        for i, bit in enumerate(fp_bits):
            features[f'fp_{i}'] = bit
            
        return features
    except:
        return None

def objective(trial, X, y):
    # Split for validation within the trial
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=42)

    param = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 1),
        'loss_function': 'MAE',
        'verbose': 0,
        'allow_writing_files': False,
        'random_seed': 42
    }

    model = CatBoostRegressor(**param)
    model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], early_stopping_rounds=100, verbose=0)

    preds = model.predict(valid_x)
    mae = mean_absolute_error(valid_y, preds)
    return mae

def main():
    print("Loading data...")
    if not os.path.exists('train.csv'):
        print("Error: train.csv not found!")
        return

    df_train = pd.read_csv("train.csv")[['SMILES', 'Tm']]
    
    print("Extracting features (Descriptors + Fragments + Morgan)...")
    train_features_df = df_train['SMILES'].apply(get_mol_features).apply(pd.Series)
    train_full = pd.concat([df_train, train_features_df], axis=1)
    
    features = [c for c in train_full.columns if c not in ['id', 'SMILES', 'Tm']]
    X = train_full[features].fillna(0)
    y = train_full['Tm']
    
    print(f"Training with {len(features)} features...")
    
    # --- Optuna Tuning ---
    print("Running Optuna to find best CatBoost hyperparameters (20 trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=20)
    
    print("-" * 30)
    print("Best Hyperparameters:")
    print(study.best_params)
    print(f"Best Validation MAE (Trial): {study.best_value:.4f}")
    print("-" * 30)
    
    # --- Final Training with Best Params ---
    print("Training final model with best parameters...")
    best_params = study.best_params
    best_params['loss_function'] = 'MAE'
    best_params['verbose'] = 200
    best_params['allow_writing_files'] = False
    best_params['random_seed'] = 42
    
    # Train/Test Split for final verification
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    
    model = CatBoostRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    
    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    
    print("-" * 30)
    print(f"Final Test MAE (Optimized CatBoost): {mae:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
