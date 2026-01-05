import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze():
    print("📊 Loading data...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    print(f"✅ Train shape: {train.shape}")
    print(f"✅ Test shape: {test.shape}")

    # Check for missing values
    print(f"❓ Missing values in train: {train.isnull().sum().sum()}")
    print(f"❓ Missing values in test: {test.isnull().sum().sum()}")

    # Class distribution
    print("\n📈 Class Distribution:")
    counts = train['label'].value_counts().sort_index()
    print(counts)
    
    # Plot distribution
    plt.figure(figsize=(10, 5))
    counts.plot(kind='bar')
    plt.title('Digit Distribution in Training Set')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.savefig('class_distribution.png')
    print("🖼️ Saved class_distribution.png")

    # Save a sample image
    sample = train.iloc[0, 1:].values.reshape(28, 28)
    plt.figure()
    plt.imshow(sample, cmap='gray')
    plt.title(f"Label: {train.iloc[0, 0]}")
    plt.axis('off')
    plt.savefig('sample_digit.png')
    print("🖼️ Saved sample_digit.png")

if __name__ == "__main__":
    analyze()
