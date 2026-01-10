import os
import glob
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# --- Config ---
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 3
IMG_SIZE = 32  # ResNet standard-ish for CIFAR

print(f"Using device: {DEVICE}")

# --- Data Preparation ---
class StreetViewDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        # Ensure Resize is part of transform or added here
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = imread(img_path)
        
        # Color -> Grayscale
        if len(img.shape) == 3:
            img = img.mean(axis=2)
        
        img = img.astype(np.uint8) # PIL needs uint8
        
        # Apply base transform (Resize + ToTensor)
        img_tensor = self.base_transform(img)

        # Apply augmentation if any
        if self.transform:
            # transform expects tensor or PIL? 
            # My train_transform uses RandomAffine, which works on Tensor or PIL.
            # But RandomAffine on Tensor might be slower than PIL?
            # Actually torchvision transforms usually expect PIL or Tensor. 
            # Let's apply transform AFTER base if it expects tensor, or merge.
            # train_transform in main uses RandomAffine. 
            # Let's just apply it.
            img_tensor = self.transform(img_tensor)

        if self.labels is not None:
            label = self.labels[idx]
            return img_tensor, label
        return img_tensor

class FastCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastCNN, self).__init__()
        # Input: 1 x 32 x 32
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_sota():
    base_path = r"d:\SandBox-concept\Street_veiw"
    train_dir = os.path.join(base_path, "train")
    test_dir = os.path.join(base_path, "test")
    labels_path = os.path.join(base_path, "trainLabels.csv")

    # 1. Prepare Data
    labels_df = pd.read_csv(labels_path)
    labels_df['filepath'] = labels_df['ID'].apply(lambda x: os.path.join(train_dir, f"{x}.Bmp"))
    
    le = LabelEncoder()
    labels_df['encoded_label'] = le.fit_transform(labels_df['Class'])
    num_classes = len(le.classes_)
    
    X = labels_df['filepath'].values
    y = labels_df['encoded_label'].values
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    ])
    
    # 2. K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    fold_models = []
    
    print(f"Starting {N_FOLDS}-Fold Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/{N_FOLDS} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_ds = StreetViewDataset(X_train, y_train, transform=train_transform)
        val_ds = StreetViewDataset(X_val, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = FastCNN(num_classes).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=EPOCHS)
        
        best_acc = 0.0
        best_model_state = None
        
        for epoch in range(EPOCHS):
            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(imgs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = correct / total
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = model.state_dict()
                
            print(f"Epoch {epoch+1}: Val Acc {val_acc:.4f} (Best: {best_acc:.4f})")
            
        print(f"Fold {fold+1} Finished. Best Acc: {best_acc:.4f}")
        
        # Save best model of this fold
        torch.save(best_model_state, f"resnet_fold{fold}.pth")
        fold_models.append(f"resnet_fold{fold}.pth")
    
    # 3. Ensemble Inference
    print("--- Generating Ensemble Submission ---")
    test_files = glob.glob(os.path.join(test_dir, "*.Bmp"))
    test_ds = StreetViewDataset(test_files)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize probabilities
    avg_probs = None
    
    for i, model_path in enumerate(fold_models):
        print(f"Inference with model {i+1}...")
        model = FastCNN(num_classes).to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for imgs in test_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                fold_probs.append(probs.cpu().numpy())
                
        fold_probs = np.concatenate(fold_probs, axis=0)
        
        if avg_probs is None:
            avg_probs = fold_probs
        else:
            avg_probs += fold_probs
            
    avg_probs /= N_FOLDS
    final_preds = np.argmax(avg_probs, axis=1)
    
    pred_chars = le.inverse_transform(final_preds)
    
    ids = []
    for f in test_files:
        ids.append(int(os.path.splitext(os.path.basename(f))[0]))
        
    sub_df = pd.DataFrame({'ID': ids, 'Class': pred_chars})
    sub_df = sub_df.sort_values(by='ID')
    sub_df.to_csv(os.path.join(base_path, "submission_ensemble.csv"), index=False)
    print("Ensemble submission saved.")

if __name__ == "__main__":
    train_sota()
