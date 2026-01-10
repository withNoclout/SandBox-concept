import os
import glob
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

# --- Config ---
BATCH_SIZE = 64
IMG_SIZE = 20 # Winner size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"d:\SandBox-concept\Street_veiw\winner_cnn.pth"

# --- Model (SimpleCNN - Must match optimized_winner.py) ---
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

class StreetViewDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = imread(img_path)
        if len(img.shape) == 3:
            img = img.mean(axis=2)
        
        img = resize(img, (20, 20), anti_aliasing=True).astype(np.float32)
        img = np.expand_dims(img, axis=0) # (1, 20, 20)
        img_tensor = torch.from_numpy(img)

        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor

def main():
    base_path = r"d:\SandBox-concept\Street_veiw"
    test_dir = os.path.join(base_path, "test")
    labels_path = os.path.join(base_path, "trainLabels.csv")

    labels_df = pd.read_csv(labels_path)
    le = LabelEncoder()
    le.fit(labels_df['Class'])
    num_classes = len(le.classes_)
    
    print(f"Loading model from {MODEL_PATH}")
    model = SimpleCNN(num_classes).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("Model file not found!")
        return
    model.eval()

    test_files = glob.glob(os.path.join(test_dir, "*.Bmp"))
    print(f"Found {len(test_files)} test images.")
    
    # TTA Transforms
    tta_transforms = [
        None, # Original
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
    ]
    
    final_probs = np.zeros((len(test_files), num_classes))
    
    print("Starting TTA Inference...")
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
    main()
