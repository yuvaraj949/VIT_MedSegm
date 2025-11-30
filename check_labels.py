import numpy as np
import os
import glob
from tqdm import tqdm

DATA_DIR = r"C:\Users\yuvar\Projects\Computer Vision\Project\New Setup\data\preprocessed\train"

def check_labels():
    files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    unique_labels = set()
    
    print(f"Checking {len(files)} files...")
    # Check first 100 files to be fast
    for f in tqdm(files[:100]):
        data = np.load(f)
        label = data['label']
        unique = np.unique(label)
        unique_labels.update(unique)
        
    print(f"Unique labels found: {sorted(list(unique_labels))}")

if __name__ == "__main__":
    check_labels()