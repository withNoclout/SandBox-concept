import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# Configuration
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
NUM_FOLDS = 5

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# Dataset Class
class DigitDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.images = data.reshape(-1, 28, 28).astype(np.uint8)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
            
        if self.labels is not None:
            return image, torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            return image

# Data Augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# Wider CNN Architecture (64 -> 128 filters)
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

def train_ensemble():
    print(f"🚀 Training Stratified K-Fold Ensemble (K={NUM_FOLDS}) on {DEVICE}")
    
    # Load Data
    print("📊 Loading data...")
    train_df = pd.read_csv("train.csv")
    X = train_df.iloc[:, 1:].values
    y = train_df.iloc[:, 0].values
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    history = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🤖 Training Fold {fold+1}/{NUM_FOLDS}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_dataset = DigitDataset(X_train, y_train, transform=train_transform)
        val_dataset = DigitDataset(X_val, y_val, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Initialize Model
        set_seed(42 + fold) # Different seed per fold (though data is already different)
        model = WideCNN().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        
        best_acc = 0.0
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = 100 * correct / total
            scheduler.step(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"model_fold_{fold}.pth")
                
            print(f"   Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
            
        print(f"✅ Fold {fold+1} Finished. Best Acc: {best_acc:.2f}%")
        history.append(best_acc)
        
    print(f"\n🏆 K-Fold Training Complete!")
    print(f"Fold Accuracies: {history}")
    print(f"Average Accuracy: {sum(history)/len(history):.2f}%")

if __name__ == "__main__":
    train_ensemble()
