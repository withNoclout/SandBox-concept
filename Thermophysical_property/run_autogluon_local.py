import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, GraphDescriptors, Fragments, rdFingerprintGenerator
from autogluon.tabular import TabularPredictor
import os

# Configuration
pd.set_option("display.max_columns", 50)

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
        
        # 2. Fragment Counts (Functional Groups)
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

def main():
    print("Loading data...")
    if not os.path.exists('train.csv'):
        print("Error: train.csv not found!")
        return

    df_train = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    submission_df = pd.read_csv("sample_submission.csv")
    
    print("Extracting features (Descriptors + Fragments + Morgan)...")
    # Train
    train_features = df_train['SMILES'].apply(get_mol_features).apply(pd.Series)
    train_data = pd.concat([df_train, train_features], axis=1)
    
    # Test
    test_features = test_df['SMILES'].apply(get_mol_features).apply(pd.Series)
    test_data = pd.concat([test_df, test_features], axis=1)
    
    # Drop ID and SMILES
    train_data = train_data.drop(columns=['id', 'SMILES'])
    test_data = test_data.drop(columns=['id', 'SMILES'])
    
    print(f"Training AutoGluon with {train_data.shape[1]} features...")
    print("This will take some time (set to 30 mins limit)...")
    
    predictor = TabularPredictor(
        label='Tm',
        eval_metric='mean_absolute_error',
        problem_type='regression',
        path='autogluon_model'
    ).fit(
        train_data,
        presets='best_quality', # High accuracy
        time_limit=7200,        # 2 hours (Increased for maximum performance)
        ag_args_fit={'num_gpus': 0} # CPU only for Mac stability
    )
    
    print("Training complete. Predicting on test set...")
    preds = predictor.predict(test_data)
    
    submission_df['Tm'] = preds
    submission_df.to_csv('submission_autogluon_local.csv', index=False)
    
    print("-" * 30)
    print("Saved submission_autogluon_local.csv")
    print("-" * 30)
    
    # Show leaderboard
    print(predictor.leaderboard(silent=True))

if __name__ == "__main__":
    main()
