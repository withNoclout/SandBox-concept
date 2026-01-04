import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from dataset import LLMPreferenceDataset
from model import LLMPreferenceModel

# --- Hyperparameters ---
MODEL_NAME = "microsoft/deberta-v3-xsmall"
MAX_LENGTH = 512
BATCH_SIZE = 8
MODEL_PATH = "saved_model/best_model.pt"

def inference():
    print("--- Starting Inference ---")
    
    # 1. Load Data
    print("Loading test data...")
    df = pd.read_csv("test.csv")
    print(f"Loaded {len(df)} rows.")
    
    # 2. Prepare Tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 3. Create Dataset
    # Note: is_train=False so it doesn't look for labels
    dataset = LLMPreferenceDataset(df, tokenizer, max_length=MAX_LENGTH, is_train=False)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Initialize Model
    print("Initializing model...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model = LLMPreferenceModel(MODEL_NAME)
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Warning: No saved model found. Using random weights (for testing only).")
        
    model.to(device)
    model.eval()
    
    # 5. Inference Loop
    all_probs = []
    
    print("Predicting...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            
    all_probs = np.concatenate(all_probs, axis=0)
    
    # 6. Create Submission File
    submission = pd.DataFrame({
        'id': df['id'],
        'winner_model_a': all_probs[:, 0],
        'winner_model_b': all_probs[:, 1],
        'winner_tie': all_probs[:, 2]
    })
    
    submission.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")
    print(submission.head())

if __name__ == "__main__":
    inference()
