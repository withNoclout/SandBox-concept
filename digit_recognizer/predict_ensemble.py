import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torchvision.transforms as transforms

# Configuration
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
NUM_FOLDS = 5

# Dataset Class
class TestDataset(Dataset):
    def __init__(self, data):
        self.images = data.values.reshape(-1, 28, 28).astype(np.uint8)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(image)
        return image

# Wider CNN Architecture (Must match train_ensemble.py)
class WideCNN(nn.Module):
    def __init__(self):
        super(WideCNN, self).__init__()
        
        # Block 1: 64 Filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        # Block 2: 128 Filters
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

def predict_ensemble():
    print(f"🚀 Predicting with Stratified K-Fold Ensemble (K={NUM_FOLDS}) on {DEVICE}")
    
    # Load Data
    print("📊 Loading test data...")
    test_df = pd.read_csv("test.csv")
    test_dataset = TestDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize aggregated probabilities
    aggregated_probs = torch.zeros(len(test_dataset), 10).to(DEVICE)
    
    # Loop through each fold model
    for fold in range(NUM_FOLDS):
        model_path = f"model_fold_{fold}.pth"
        if not os.path.exists(model_path):
            print(f"❌ Error: {model_path} not found! Skipping...")
            continue
            
        print(f"🤖 Loading Model Fold {fold+1}/{NUM_FOLDS}...")
        model = WideCNN().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        # Inference
        probs = []
        with torch.no_grad():
            for images in test_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                batch_probs = torch.softmax(outputs, dim=1)
                probs.append(batch_probs)
        
        full_probs = torch.cat(probs, dim=0)
        aggregated_probs += full_probs
        
    # Final Prediction
    print("🔮 Calculating final votes...")
    _, predictions = torch.max(aggregated_probs, 1)
    predictions = predictions.cpu().numpy()
            
    # Create Submission
    submission = pd.DataFrame({
        "ImageId": range(1, len(predictions) + 1),
        "Label": predictions
    })
    
    submission.to_csv("submission_kfold.csv", index=False)
    print(f"💾 Saved submission_kfold.csv with {len(submission)} predictions")
    print(submission.head())

if __name__ == "__main__":
    predict_ensemble()
