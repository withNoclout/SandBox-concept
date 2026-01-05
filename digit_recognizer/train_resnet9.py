"""
ResNet9 Model for Digit Recognizer
==================================
ResNet9 uses residual connections (skip connections) which allow gradients 
to flow more easily during backpropagation, enabling faster and better training.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.003
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(SEED)

# Dataset Class
class DigitDataset(Dataset):
    def __init__(self, data, is_test=False):
        if is_test:
            self.labels = None
            self.images = data.values.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        else:
            self.labels = data.iloc[:, 0].values
            self.images = data.iloc[:, 1:].values.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        return image

# ============================================================================
# ResNet9 Architecture
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
    """
    ResNet9 Architecture:
    - 9 layers total with residual connections
    - Designed for small images (28x28 for MNIST)
    """
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        
        # Initial conv
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)  # 28 -> 14
        
        # Residual block 1
        self.res1 = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 128)
        )
        
        self.conv3 = conv_block(128, 256, pool=True)  # 14 -> 7
        self.conv4 = conv_block(256, 512, pool=True)  # 7 -> 3
        
        # Residual block 2
        self.res2 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out  # Residual connection 1
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out  # Residual connection 2
        out = self.classifier(out)
        return out


# ============================================================================
# Training Loop with OneCycleLR (faster convergence)
# ============================================================================

def train_model():
    print(f"🚀 Training ResNet9 on {DEVICE}")
    
    # Load Data
    print("📊 Loading data...")
    train_df = pd.read_csv("train.csv")
    
    # Split Data
    train_data, val_data = train_test_split(
        train_df, test_size=0.1, random_state=SEED, stratify=train_df['label']
    )
    
    train_dataset = DigitDataset(train_data)
    val_dataset = DigitDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize Model
    model = ResNet9(in_channels=1, num_classes=10).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📐 Total parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # OneCycleLR for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE * 10,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader)
    )
    
    # Training
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    
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
            scheduler.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
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
        history['val_acc'].append(val_acc)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "resnet9_best.pth")
            print("💾 Model saved!")
            
    print(f"\n🏆 Best Validation Accuracy: {best_acc:.2f}%")
    
    # Plot history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='orange')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.axhline(y=99, color='r', linestyle='--', label='99% Target')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('resnet9_training_history.png')
    print("🖼️ Saved resnet9_training_history.png")

if __name__ == "__main__":
    train_model()
