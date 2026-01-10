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
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# --- Config ---
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 3
IMG_SIZE = 20 # Keep native size

print(f"Using device: {DEVICE}")

# --- Data Preparation ---
class StreetViewDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = imread(img_path)
        if len(img.shape) == 3:
            img = img.mean(axis=2)
        
        # Native 20x20 resize
        img = resize(img, (20, 20), anti_aliasing=True).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img_tensor = torch.from_numpy(img)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.labels is not None:
            label = self.labels[idx]
            return img_tensor, label
        return img_tensor

# --- Model (The "Winner" Architecture) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 10x10
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 5x5
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 2x2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def run_ensemble():
    base_path = r"d:\SandBox-concept\Street_veiw"
    train_dir = os.path.join(base_path, "train")
    test_dir = os.path.join(base_path, "test")
    labels_path = os.path.join(base_path, "trainLabels.csv")

    # Load Data
    labels_df = pd.read_csv(labels_path)
    labels_df['filepath'] = labels_df['ID'].apply(lambda x: os.path.join(train_dir, f"{x}.Bmp"))
    le = LabelEncoder()
    labels_df['encoded_label'] = le.fit_transform(labels_df['Class'])
    
    X = labels_df['filepath'].values
    y = labels_df['encoded_label'].values
    num_classes = len(le.classes_)
    
    # Augmentation
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])
    
    # K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_paths = []
    
    print(f"Starting {N_FOLDS}-Fold Ensemble Training...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_ds = StreetViewDataset(X_train, y_train, transform=train_transform)
        val_ds = StreetViewDataset(X_val, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = SimpleCNN(num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
        
        best_acc = 0.0
        
        for epoch in range(EPOCHS):
            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
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
            scheduler.step(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(base_path, f"model_fold{fold}.pth"))
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}: Val Acc {val_acc:.4f} (Best: {best_acc:.4f})")
                
        print(f"Fold {fold+1} Finished. Best Acc: {best_acc:.4f}")
        fold_paths.append(os.path.join(base_path, f"model_fold{fold}.pth"))

    # --- Ensemble Inference ---
    print("\nStarting Ensemble Inference...")
    test_files = glob.glob(os.path.join(test_dir, "*.Bmp"))
    
    # TTA Transforms
    tta_transforms = [
        None,
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.RandomAffine(degrees=0, scale=(0.95, 1.05))
    ]
    
    final_probs = np.zeros((len(test_files), num_classes))
    total_votes = 0
    
    for fold_idx, model_path in enumerate(fold_paths):
        print(f"Predicting with Model Fold {fold_idx+1}...")
        model = SimpleCNN(num_classes).to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        for t_idx, tta in enumerate(tta_transforms):
            # print(f"  TTA {t_idx+1}...")
            ds = StreetViewDataset(test_files, transform=tta)
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
            
            fold_preds = []
            with torch.no_grad():
                for imgs in dl:
                    imgs = imgs.to(DEVICE)
                    out = model(imgs)
                    prob = torch.softmax(out, dim=1)
                    fold_preds.append(prob.cpu().numpy())
            
            final_probs += np.concatenate(fold_preds, axis=0)
            total_votes += 1
            
    # Average
    # final_probs /= total_votes # Not strictly necessary for argmax but good for debug
    
    final_preds = np.argmax(final_probs, axis=1)
    pred_chars = le.inverse_transform(final_preds)
    
    ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in test_files]
    df = pd.DataFrame({'ID': ids, 'Class': pred_chars}).sort_values('ID')
    df.to_csv(os.path.join(base_path, "submission_ensemble_final.csv"), index=False)
    print("Saved submission_ensemble_final.csv")

if __name__ == "__main__":
    run_ensemble()
