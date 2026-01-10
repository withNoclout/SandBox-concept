import os
import glob
import hashlib
from collections import defaultdict
import pandas as pd

def get_file_hash(filepath):
    """Returns MD5 hash of file content."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def check_leakage():
    base_path = r"d:\SandBox-concept\Street_veiw"
    train_dir = os.path.join(base_path, "train")
    test_dir = os.path.join(base_path, "test")
    
    print("Hashing Train Images...")
    train_files = glob.glob(os.path.join(train_dir, "*.Bmp"))
    train_hashes = {}
    for f in train_files:
        h = get_file_hash(f)
        train_hashes[h] = f
        
    print(f"Hashed {len(train_files)} train images.")
    
    print("Hashing Test Images...")
    test_files = glob.glob(os.path.join(test_dir, "*.Bmp"))
    test_hashes = {}
    leak_matches = []
    
    for f in test_files:
        h = get_file_hash(f)
        if h in train_hashes:
            leak_matches.append((f, train_hashes[h]))
            
    print(f"Hashed {len(test_files)} test images.")
    
    print("-" * 30)
    print(f"Found {len(leak_matches)} EXACT pixel matches between Train and Test.")
    
    if len(leak_matches) > 0:
        print("SAMPLE LEAKS:")
        for test_f, train_f in leak_matches[:5]:
            print(f"Test: {os.path.basename(test_f)} == Train: {os.path.basename(train_f)}")
    else:
        print("No exact leakage found.")
        
if __name__ == "__main__":
    check_leakage()
