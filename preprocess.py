import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# Configuration
BASE_DIR = r"C:\Users\yuvar\Projects\Computer Vision\Project\synapse_data\Abdomen\RawData\Training"
OUTPUT_DIR = r"C:\Users\yuvar\Projects\Computer Vision\Project\New Setup\data\preprocessed"
IMG_DIR = os.path.join(BASE_DIR, "img")
LABEL_DIR = os.path.join(BASE_DIR, "label")

TRAIN_SIZE = 18
TEST_SIZE = 12

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sort_by_number(files, prefix):
    return sorted([f for f in files if f.startswith(prefix)], key=lambda x: int(''.join(filter(str.isdigit, x))))

def preprocess_dataset():
    make_folder(os.path.join(OUTPUT_DIR, "train"))
    make_folder(os.path.join(OUTPUT_DIR, "test"))

    img_files = sort_by_number(os.listdir(IMG_DIR), "img")
    label_files = sort_by_number(os.listdir(LABEL_DIR), "label")

    assert len(img_files) == len(label_files), "Mismatch in image and label count"
    total_scans = len(img_files)
    print(f"Found {total_scans} scans.")

    # Split into train and test
    # Using first 18 for train, last 12 for test (or as per specific list if needed)
    # For now, simple split
    train_imgs = img_files[:TRAIN_SIZE]
    train_labels = label_files[:TRAIN_SIZE]
    test_imgs = img_files[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
    test_labels = label_files[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]

    print(f"Training set: {len(train_imgs)} scans")
    print(f"Testing set: {len(test_imgs)} scans")

    # Process Training Data
    print("Processing Training Data...")
    process_and_save(train_imgs, train_labels, "train")

    # Process Testing Data
    print("Processing Testing Data...")
    process_and_save(test_imgs, test_labels, "test")

def process_and_save(img_list, label_list, split):
    slice_count = 0
    for img_name, label_name in tqdm(zip(img_list, label_list), total=len(img_list)):
        img_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, label_name)

        # Load 3D volume
        itk_img = sitk.ReadImage(img_path)
        itk_label = sitk.ReadImage(label_path)

        img_array = sitk.GetArrayFromImage(itk_img)
        label_array = sitk.GetArrayFromImage(itk_label)

        # SimpleITK loads as (Depth, Height, Width)
        # We want to slice along Depth
        
        # Normalize image: clip to [-125, 275], then normalize to [0, 1]
        # This is standard for CT Abdomen
        img_array = np.clip(img_array, -125, 275)
        img_array = (img_array - (-125)) / (275 - (-125))

        for i in range(img_array.shape[0]):
            img_slice = img_array[i, :, :]
            label_slice = label_array[i, :, :]

            # Skip slices with no labels if desired (optional, but often good for training)
            # For now, keeping all slices or maybe just those with some content?
            # Roadmap doesn't specify, but usually we keep slices containing organs.
            # Let's save all for now to be safe, or maybe filter empty ones?
            # TransUNet paper usually filters empty slices for training.
            
            if split == "train":
                # Filter empty slices for training? 
                # Let's keep it simple: save all, filter in dataloader if needed, 
                # OR save only slices with labels. 
                # Common practice: save if label_slice.sum() > 0
                if np.sum(label_slice) > 0:
                     save_slice(img_slice, label_slice, split, img_name, i)
                     slice_count += 1
            else:
                # Keep all slices for testing to reconstruct volume
                save_slice(img_slice, label_slice, split, img_name, i)
                slice_count += 1
    
    print(f"Saved {slice_count} slices for {split}.")

def save_slice(img, label, split, case_name, slice_idx):
    # Resize to 224x224 if needed? Roadmap says Input: 224x224x3
    # But raw data might be 512x512.
    # We should probably resize here or in dataloader. 
    # Let's save raw size and resize in dataloader to preserve quality, 
    # OR resize here to save space. 
    # Roadmap 4.1 says "Random crop: 224x224". This implies we might want larger images saved.
    # However, TransUNet usually expects 224 input.
    # I will save as is (likely 512x512) and let the dataloader handle cropping/resizing.
    
    case_id = case_name.replace(".nii.gz", "").replace("img", "")
    filename = f"case{case_id}_slice{slice_idx:03d}.npz"
    save_path = os.path.join(OUTPUT_DIR, split, filename)
    
    np.savez(save_path, image=img, label=label)

if __name__ == "__main__":
    preprocess_dataset()
