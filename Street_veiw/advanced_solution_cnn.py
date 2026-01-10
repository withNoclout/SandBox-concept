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
EPOCHS = 25
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
        # Read image
        img = imread(img_path)
        
        # Convert to grayscale if color
        if len(img.shape) == 3:
            img = img.mean(axis=2)
            
        # Resize to 20x20
        img = resize(img, (20, 20), anti_aliasing=True)
        # Scale to 0-1 (already done by resize usually, but ensure float32)
        img = img.astype(np.float32)
        
        # Add channel dim: (H, W) -> (1, H, W) for PyTorch
        img = np.expand_dims(img, axis=0)
        
        # Convert to tensor
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

    # 1. Load Labels and Paths
    labels_df = pd.read_csv(labels_path)
    # Ensure string sorting or numerical sorting matches file system?
    # Actually, we should construct paths based on ID to be safe
    # But IDs in csv are integers. Files are "1.Bmp", "10.Bmp".
    
    labels_df['filepath'] = labels_df['ID'].apply(lambda x: os.path.join(train_dir, f"{x}.Bmp"))
    
    # Check if files exist (some might be .bmp lowercase, but glob in previous step showed .Bmp)
    # Let's assume standard naming based on previous check.
    
    # Label Encoding (Chars to 0-N)
    le = LabelEncoder()
    labels_df['encoded_label'] = le.fit_transform(labels_df['Class'])
    
    # Split Train/Val
    train_df, val_df = train_test_split(labels_df, test_size=0.1, random_state=42, stratify=labels_df['encoded_label'])
    
    # Transforms (Augmentation)
    # Since we manually resize/grayscale in dataset, we use torchvision for affine
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])
    # No transform for validation
    
    train_dataset = StreetViewDataset(train_df['filepath'].values, train_df['encoded_label'].values, transform=train_transform)
    val_dataset = StreetViewDataset(val_df['filepath'].values, val_df['encoded_label'].values)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Test Data
    test_files = glob.glob(os.path.join(test_dir, "*.Bmp")) 
    # Sort test files by ID to maintain order for submission or just track IDs
    # We will extract ID from filename during prediction to stay robust
    
    test_dataset = StreetViewDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, le, test_files

# --- Model ---
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
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Training ---
def train_model():
    base_path = r"d:\SandBox-concept\Street_veiw"
    train_loader, val_loader, test_loader, le, test_files = get_data_loaders(base_path)
    
    num_classes = len(le.classes_)
    print(f"Classes: {num_classes}")
    
    model = SimpleCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    
    best_acc = 0.0
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
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
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(base_path, "best_cnn.pth"))
            
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    
    # --- Inference ---
    print("Generating submission...")
    model.load_state_dict(torch.load(os.path.join(base_path, "best_cnn.pth")))
    model.eval()
    
    predictions = []
    ids = []
    
    # Map test files to IDs and keep order
    # test_files list corresponds to test_loader iteration order if shuffle=False
    
    # Wait, we need to carefully match IDs. 
    # The DataLoader iterates in the order of `test_files` passed to Dataset.
    # So we iterate loader and `test_files` together or just trust the index.
    
    all_preds = []
    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            
    # Decode labels
    pred_chars = le.inverse_transform(all_preds)
    
    # Create rows
    for i, file_path in enumerate(test_files):
        file_name = os.path.basename(file_path)
        file_id = int(os.path.splitext(file_name)[0])
        ids.append(file_id)
        
    submission_df = pd.DataFrame({
        'ID': ids,
        'Class': pred_chars
    })
    
    submission_df = submission_df.sort_values(by='ID')
    submission_df.to_csv(os.path.join(base_path, "submission_cnn.csv"), index=False)
    print("Saved submission_cnn.csv")

if __name__ == "__main__":
    train_model()
