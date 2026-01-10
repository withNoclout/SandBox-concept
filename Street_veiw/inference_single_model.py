import os
import glob
import numpy as np
import pandas as pd
from skimage.io import imread
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

# --- Config ---
BATCH_SIZE = 64
IMG_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"d:\SandBox-concept\Street_veiw\resnet_fold0.pth"

# --- Model Definition (Must match Saved Model) ---
class FastCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastCNN, self).__init__()
        # Input: 1 x 32 x 32
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 8x8
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 4x4
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> 2x2
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Dataset ---
class StreetViewDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
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
        if len(img.shape) == 3:
            img = img.mean(axis=2)
        img = img.astype(np.uint8)
        img_tensor = self.base_transform(img)
        return img_tensor

def main():
    base_path = r"d:\SandBox-concept\Street_veiw"
    test_dir = os.path.join(base_path, "test")
    labels_path = os.path.join(base_path, "trainLabels.csv")

    # Re-fit LabelEncoder to get classes
    labels_df = pd.read_csv(labels_path)
    le = LabelEncoder()
    le.fit(labels_df['Class'])
    num_classes = len(le.classes_)
    
    # Load Model
    model = FastCNN(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print(f"Loaded model from {MODEL_PATH}")

    # Test Data
    print("Preparing Test Data...")
    test_files = glob.glob(os.path.join(test_dir, "*.Bmp"))
    test_ds = StreetViewDataset(test_files)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    preds = []
    with torch.no_grad():
        for i, imgs in enumerate(test_loader):
            if i % 10 == 0: print(f"Batch {i}")
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.cpu().numpy())
            
    pred_chars = le.inverse_transform(preds)
    
    ids = []
    for f in test_files:
        ids.append(int(os.path.splitext(os.path.basename(f))[0]))
        
    sub_df = pd.DataFrame({'ID': ids, 'Class': pred_chars})
    sub_df = sub_df.sort_values(by='ID')
    sub_path = os.path.join(base_path, "submission_fold0.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved {sub_path}")

if __name__ == "__main__":
    main()
