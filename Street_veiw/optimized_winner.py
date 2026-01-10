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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Config ---
BATCH_SIZE = 64
EPOCHS = 40  # Increased from 25
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        # KEY: Using 20x20 again (Native resolution seems best)
        img = resize(img, (20, 20), anti_aliasing=True).astype(np.float32)
        img = np.expand_dims(img, axis=0) # (1, 20, 20)
        img_tensor = torch.from_numpy(img)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.labels is not None:
            label = self.labels[idx]
            return img_tensor, label
        return img_tensor

def get_data_loaders(base_path):
    train_dir = os.path.join(base_path, "train")
    test_dir = os.path.join(base_path, "test")
    labels_path = os.path.join(base_path, "trainLabels.csv")

    labels_df = pd.read_csv(labels_path)
    labels_df['filepath'] = labels_df['ID'].apply(lambda x: os.path.join(train_dir, f"{x}.Bmp"))
    
    le = LabelEncoder()
    labels_df['encoded_label'] = le.fit_transform(labels_df['Class'])
    
    train_df, val_df = train_test_split(labels_df, test_size=0.1, random_state=42, stratify=labels_df['encoded_label'])
    
    # Same Augmentation as before (it worked)
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])
    
    train_dataset = StreetViewDataset(train_df['filepath'].values, train_df['encoded_label'].values, transform=train_transform)
    val_dataset = StreetViewDataset(val_df['filepath'].values, val_df['encoded_label'].values)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    test_files = glob.glob(os.path.join(test_dir, "*.Bmp")) 
    
    return train_loader, val_loader, test_files, le

# --- Model (SimpleCNN) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Input: 1 x 20 x 20
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 10x10
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 5x5
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> 2x2
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularization
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_winner():
    base_path = r"d:\SandBox-concept\Street_veiw"
    train_loader, val_loader, test_files, le = get_data_loaders(base_path)
    num_classes = len(le.classes_)
    
    model = SimpleCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    best_acc = 0.0
    
    print(f"Starting optimized training ({EPOCHS} Epochs)...")
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
        print(f"Epoch {epoch+1}/{EPOCHS}: Val Acc {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(base_path, "winner_cnn.pth"))
            
    print(f"Best Acc: {best_acc:.4f}")

    # --- TTA Inference ---
    print("Generating TTA Submission...")
    model.load_state_dict(torch.load(os.path.join(base_path, "winner_cnn.pth")))
    model.eval()
    
    # Define TTA transforms (Affine variation)
    tta_transforms = [
        None, # Original
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
    ]
    
    final_probs = np.zeros((len(test_files), num_classes))
    
    # We need to map file path -> index in final_probs carefully
    # We will just predict for all test_files in order
    
    for t_idx, tta_t in enumerate(tta_transforms):
        print(f"TTA Pass {t_idx+1}...")
        ds = StreetViewDataset(test_files, transform=tta_t)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for imgs in dl:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                preds.append(probs.cpu().numpy())
        
        final_probs += np.concatenate(preds, axis=0)
        
    final_preds = np.argmax(final_probs, axis=1)
    pred_chars = le.inverse_transform(final_preds)
    
    ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in test_files]
    df = pd.DataFrame({'ID': ids, 'Class': pred_chars}).sort_values('ID')
    df.to_csv(os.path.join(base_path, "submission_winner.csv"), index=False)
    print("Saved submission_winner.csv")

if __name__ == "__main__":
    train_winner()
