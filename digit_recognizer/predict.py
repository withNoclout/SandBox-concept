import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Configuration
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Dataset Class (Same as train.py but for test data)
class TestDataset(Dataset):
    def __init__(self, data):
        self.images = data.values.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        return image

# 2. CNN Model (Must match train.py)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def predict():
    print(f"🚀 Predicting on {DEVICE}")
    
    # Load Data
    print("📊 Loading test data...")
    test_df = pd.read_csv("test.csv")
    test_dataset = TestDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load Model
    if not os.path.exists("best_model.pth"):
        print("❌ Error: best_model.pth not found! Train the model first.")
        return

    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.eval()
    print("✅ Model loaded")
    
    # Prediction
    predictions = []
    print("🔮 Generating predictions...")
    
    with torch.no_grad():
        for images in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            
    # Create Submission
    submission = pd.DataFrame({
        "ImageId": range(1, len(predictions) + 1),
        "Label": predictions
    })
    
    submission.to_csv("submission.csv", index=False)
    print(f"💾 Saved submission.csv with {len(submission)} predictions")
    print(submission.head())

if __name__ == "__main__":
    predict()
