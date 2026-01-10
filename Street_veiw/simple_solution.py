import os
import glob
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def load_data(data_dir, labels_file=None):
    print(f"Loading data from {data_dir}...")
    image_files = glob.glob(os.path.join(data_dir, "*.Bmp"))
    if not image_files:
        # Try lowercase extension just in case
        image_files = glob.glob(os.path.join(data_dir, "*.bmp"))
    
    print(f"Found {len(image_files)} images.")
    
    X = []
    ids = []
    
    for i, file_path in enumerate(image_files):
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} images...")
            
        # Read image
        img = imread(file_path)
        
        # Convert to grayscale if it is a color image
        if len(img.shape) == 3:
             # Just take average or use rgb2gray (need import).
             # Simple way: mean across last axis (channels)
             img = img.mean(axis=2)
             
        # Resize image
        img_resized = resize(img, (20, 20), anti_aliasing=True)
        
        # Flatten image to 1D array
        # 20x20 = 400 features
        X.append(img_resized.flatten())
        
        # Extract ID from filename (e.g., "d:\path\123.Bmp" -> 123)
        file_name = os.path.basename(file_path)
        file_id = int(os.path.splitext(file_name)[0])
        ids.append(file_id)
        
    return np.array(X), np.array(ids)

def main():
    base_path = r"d:\SandBox-concept\Street_veiw"
    train_dir = os.path.join(base_path, "train")
    test_dir = os.path.join(base_path, "test")
    labels_path = os.path.join(base_path, "trainLabels.csv")
    output_path = os.path.join(base_path, "submission.csv")

    # 1. Load Training Data
    print("--- Loading Training Data ---")
    X_train, train_ids = load_data(train_dir)
    
    # 2. Load Labels
    print("--- Loading Labels ---")
    labels_df = pd.read_csv(labels_path)
    # Map labels to the loaded images using ID
    # Verify that we have labels for all loaded images
    # Create a dictionary for fast lookup
    label_map = dict(zip(labels_df['ID'], labels_df['Class']))
    
    y_train = []
    # Filter X_train to ensure it matches order or reorder it. 
    # Simpler: Re-construct X_train/y_train based on IDs to be safe
    
    # Let's align them.
    # Create a dataframe for the loaded images
    train_data_df = pd.DataFrame({'ID': train_ids})
    train_data_df['features'] = list(X_train)
    
    # Merge with labels
    merged_train = pd.merge(train_data_df, labels_df, on='ID')
    
    X_train_ordered = np.stack(merged_train['features'].values)
    y_train_ordered = merged_train['Class'].values
    
    print(f"Training data shape: {X_train_ordered.shape}")
    print(f"Labels shape: {y_train_ordered.shape}")

    # 3. Train Model
    print("--- Training Random Forest Model ---")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train_ordered, y_train_ordered)
    print("Model trained.")

    # 4. Load Test Data
    print("--- Loading Test Data ---")
    X_test, test_ids = load_data(test_dir)
    print(f"Test data shape: {X_test.shape}")

    # 5. Predict
    print("--- Predicting ---")
    predictions = clf.predict(X_test)

    # 6. Save Submission
    print("--- Saving Submission ---")
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'Class': predictions
    })
    
    # Sort by ID as usually required
    submission_df = submission_df.sort_values(by='ID')
    
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission_df.head())

if __name__ == "__main__":
    main()
