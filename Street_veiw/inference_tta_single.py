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
MODEL_PATH = r"d:\SandBox-concept\Street_veiw\custom_cnn_fold0.pth"

# --- Model (FastCNN) ---
class FastCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastCNN, self).__init__()
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
            nn.MaxPool2d(2),
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

class StreetViewDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
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

        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor

def main():
    base_path = r"d:\SandBox-concept\Street_veiw"
    test_dir = os.path.join(base_path, "test")
    labels_path = os.path.join(base_path, "trainLabels.csv")

    # Labels
    labels_df = pd.read_csv(labels_path)
    le = LabelEncoder()
    le.fit(labels_df['Class'])
    num_classes = len(le.classes_)
    
    # Model
    model = FastCNN(num_classes).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}. Exiting.")
        return
    model.eval()

    # Data
    test_files = glob.glob(os.path.join(test_dir, "*.Bmp"))
    print(f"Found {len(test_files)} test images.")
    
    # TTA Transforms
    tta_transforms = [
        None,
        transforms.RandomAffine(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    ]
    
    final_probs = np.zeros((len(test_files), num_classes))
    
    print("Starting TTA Inference...")
    for t_idx, tta_t in enumerate(tta_transforms):
        print(f"  TTA Pass {t_idx+1}/{len(tta_transforms)}...")
        ds = StreetViewDataset(test_files, transform=tta_t)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        
        fold_probs = []
        with torch.no_grad():
            for imgs in dl:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                fold_probs.append(probs.cpu().numpy())
        
        final_probs += np.concatenate(fold_probs, axis=0)
        
    final_preds = np.argmax(final_probs, axis=1)
    pred_chars = le.inverse_transform(final_preds)
    
    ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in test_files]
    sub_df = pd.DataFrame({'ID': ids, 'Class': pred_chars}).sort_values('ID')
    save_path = os.path.join(base_path, "submission_final_boost.csv")
    sub_df.to_csv(save_path, index=False)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
