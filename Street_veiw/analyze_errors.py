import os
import glob
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# --- Config ---
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"d:\SandBox-concept\Street_veiw\winner_cnn.pth"

# --- Recreate Model & Dataset ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
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
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img), self.labels[idx]

def analyze():
    base_path = r"d:\SandBox-concept\Street_veiw"
    train_dir = os.path.join(base_path, "train")
    labels_path = os.path.join(base_path, "trainLabels.csv")

    # 1. Recreate Validation Split
    labels_df = pd.read_csv(labels_path)
    labels_df['filepath'] = labels_df['ID'].apply(lambda x: os.path.join(train_dir, f"{x}.Bmp"))
    le = LabelEncoder()
    labels_df['encoded_label'] = le.fit_transform(labels_df['Class'])
    
    _, val_df = train_test_split(labels_df, test_size=0.1, random_state=42, stratify=labels_df['encoded_label'])
    
    val_ds = StreetViewDataset(val_df['filepath'].values, val_df['encoded_label'].values)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    num_classes = len(le.classes_)
    model = SimpleCNN(num_classes).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # 3. Predict
    all_preds = []
    all_labels = []
    
    print("Evaluating validation set...")
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(lbls.numpy())
            
    # 4. Analyze Errors
    cm = confusion_matrix(all_labels, all_preds)
    
    # Get misclassified items
    misclassified = []
    for true_idx, pred_idx in zip(all_labels, all_preds):
        if true_idx != pred_idx:
            misclassified.append((le.classes_[true_idx], le.classes_[pred_idx]))
            
    # Count confusion pairs
    from collections import Counter
    pair_counts = Counter(misclassified)
    
    print("\n" + "="*40)
    print(f"Validation Accuracy: {1 - len(misclassified)/len(all_labels):.4f}")
    print(f"Total Errors: {len(misclassified)}")
    print("="*40)
    print("Top 15 Confused Pairs (True -> Predicted):")
    for (true, pred), count in pair_counts.most_common(15):
        print(f"  {true} -> {pred}: {count} times")
    print("="*40)

if __name__ == "__main__":
    analyze()
