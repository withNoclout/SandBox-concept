import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Configuration
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 1. Dataset Class (Same as train.py but for test data)
class TestDataset(Dataset):
    def __init__(self, data):
        self.images = data.values.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        return image

# ============================================================================
# ResNet9 Architecture (Must match train_resnet9.py)
# ============================================================================

def conv_block(in_channels, out_channels, pool=False):
    """Basic conv block: Conv -> BatchNorm -> ReLU (-> MaxPool)"""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def predict():
    print(f"🚀 Predicting with ResNet9 on {DEVICE}")
    
    # Load Data
    print("📊 Loading test data...")
    test_df = pd.read_csv("test.csv")
    test_dataset = TestDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load Model
    if not os.path.exists("resnet9_best.pth"):
        print("❌ Error: resnet9_best.pth not found! Train the model first.")
        return

    model = ResNet9(in_channels=1, num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load("resnet9_best.pth", map_location=DEVICE))
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
    
    submission.to_csv("submission_resnet9.csv", index=False)
    print(f"💾 Saved submission_resnet9.csv with {len(submission)} predictions")
    print(submission.head())

if __name__ == "__main__":
    predict()
