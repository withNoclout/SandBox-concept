import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import optuna

# --- Config ---
BATCH_SIZE = 64
EPOCHS = 15 # Shorter search epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 20

# --- Dataset ---
class StreetViewDataset(Dataset):
    def __init__(self, image_paths, labels=None):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = imread(img_path)
        if len(img.shape) == 3: img = img.mean(axis=2)
        img = resize(img, (20, 20), anti_aliasing=True).astype(np.float32)
        img = np.expand_dims(img, axis=0) # (1, 20, 20)
        return torch.from_numpy(img), self.labels[idx]

# --- Dynamic Model ---
class DynamicCNN(nn.Module):
    def __init__(self, num_classes, filter_mult, dropout_rate):
        super(DynamicCNN, self).__init__()
        base = filter_mult
        self.features = nn.Sequential(
            nn.Conv2d(1, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(),
            nn.MaxPool2d(2), # 10x10
            
            nn.Conv2d(base, base*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(),
            nn.MaxPool2d(2), # 5x5
            
            nn.Conv2d(base*2, base*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 2x2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base*4 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def objective(trial):
    # Hyperparams
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.6)
    filter_mult = trial.suggest_categorical('filter_mult', [16, 32, 48])
    
    # Data Setup (Once)
    base_path = r"d:\SandBox-concept\Street_veiw"
    train_dir = os.path.join(base_path, "train")
    labels_path = os.path.join(base_path, "trainLabels.csv")
    
    labels_df = pd.read_csv(labels_path)
    labels_df['filepath'] = labels_df['ID'].apply(lambda x: os.path.join(train_dir, f"{x}.Bmp"))
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels_df['Class'])
    X = labels_df['filepath'].values
    y = encoded_labels
    
    # Use just 1 fold for search speed
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(X, y))
    
    train_ds = StreetViewDataset(X[train_idx], y[train_idx]) # No aug for speed/apples-to-apples in search
    val_ds = StreetViewDataset(X[val_idx], y[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = DynamicCNN(len(le.classes_), filter_mult, dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                _, pred = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        
        val_acc = correct / total
        
        # Pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_acc

if __name__ == "__main__":
    print("Starting Optuna Search...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    print("Best params:")
    print(study.best_params)
    
    # Save best params to text file
    with open("d:\\SandBox-concept\\Street_veiw\\best_params.txt", "w") as f:
        f.write(str(study.best_params))
