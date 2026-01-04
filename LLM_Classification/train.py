import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from dataset import LLMPreferenceDataset
from model import LLMPreferenceModel

# --- Hyperparameters ---
MODEL_NAME = "microsoft/deberta-v3-xsmall" # Use xsmall for fast testing, base/large for real training
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 2e-5
SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train():
    print("--- Starting Training Pipeline ---")
    set_seed(SEED)
    
    # 1. Load Data
    print("Loading data...")
    # Load subset for speed
    df = pd.read_csv("train.csv", nrows=2000) 
    print(f"Loaded {len(df)} rows.")
    
    # 2. Prepare Tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 3. Create Dataset
    full_dataset = LLMPreferenceDataset(df, tokenizer, max_length=MAX_LENGTH)
    
    # Split into Train/Val
    train_size = int(0.9 * len(full_dataset)) # 90/10 split
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
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
    model.to(device)
    
    # 5. Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # 6. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        # --- Training ---
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        
        # --- Validation ---
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        print("Validating...")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0)
                
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("New best model! Saving...")
            if not os.path.exists("saved_model"):
                os.makedirs("saved_model")
            torch.save(model.state_dict(), "saved_model/best_model.pt")
            
    print("\nTraining Complete!")

if __name__ == "__main__":
    train()
