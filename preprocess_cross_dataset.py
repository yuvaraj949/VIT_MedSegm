"""
Preprocessing for AMOS and BTCV datasets to match Synapse format
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json

class CrossDatasetPreprocessor:
    def __init__(self, dataset_name, input_dir, output_dir):
        """
        Args:
            dataset_name: 'AMOS' or 'BTCV'
            input_dir: Path to raw dataset
            output_dir: Path to save preprocessed data
        """
        self.dataset_name = dataset_name
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Synapse organ mapping (what your model was trained on)
        self.synapse_organs = {
            0: 'background',
            1: 'aorta',
            2: 'gallbladder',
            3: 'kidney_left',
            4: 'kidney_right',
            5: 'liver',
            6: 'pancreas',
            7: 'spleen',
            8: 'stomach'
        }
        
        # Dataset-specific organ mappings
        self.organ_mappings = self._get_organ_mapping()
    
    def _get_organ_mapping(self):
        """Map AMOS/BTCV labels to Synapse labels"""
        if self.dataset_name == 'AMOS':
            # AMOS has 15 organs, map to Synapse's 8
            return {
                0: 0,   # background
                1: 7,   # spleen
                2: 4,   # kidney_right
                3: 3,   # kidney_left
                4: 2,   # gallbladder
                5: 0,   # esophagus -> background (not in Synapse)
                6: 5,   # liver
                7: 8,   # stomach
                8: 1,   # aorta
                9: 0,   # inferior vena cava -> background
                10: 6,  # pancreas
                11: 0,  # adrenal_right -> background
                12: 0,  # adrenal_left -> background
                13: 0,  # duodenum -> background
                14: 0,  # bladder -> background
                15: 0   # prostate -> background
            }
        elif self.dataset_name == 'BTCV':
            # BTCV has 13 organs
            return {
                0: 0,   # background
                1: 7,   # spleen
                2: 4,   # kidney_right
                3: 3,   # kidney_left
                4: 2,   # gallbladder
                5: 0,   # esophagus -> background
                6: 5,   # liver
                7: 8,   # stomach
                8: 1,   # aorta
                9: 0,   # inferior vena cava -> background
                10: 0,  # portal vein -> background
                11: 6,  # pancreas
                12: 0,  # adrenal_right -> background
                13: 0   # adrenal_left -> background
            }
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def normalize_ct(self, image):
        """Normalize CT to [0, 1] using abdominal window"""
        # Clip to abdominal window [-125, 275] HU
        image = np.clip(image, -125, 275)
        # Normalize to [0, 1]
        image = (image - (-125)) / (275 - (-125))
        return image
    
    def remap_labels(self, label):
        """Remap dataset-specific labels to Synapse format"""
        remapped = np.zeros_like(label)
        for src_label, tgt_label in self.organ_mappings.items():
            remapped[label == src_label] = tgt_label
        return remapped
    
    def extract_slices(self, image, label, case_id):
        """Extract 2D slices from 3D volume"""
        slices_data = []
        
        # Get volume dimensions
        depth = image.shape[2]
        
        for z in range(depth):
            # Extract slice
            img_slice = image[:, :, z]
            lbl_slice = label[:, :, z]
            
            # Skip slices with no organs (only background)
            if np.sum(lbl_slice > 0) == 0:
                continue
            
            # Resize to 224x224
            from scipy.ndimage import zoom
            zoom_factor = (224 / img_slice.shape[0], 224 / img_slice.shape[1])
            img_resized = zoom(img_slice, zoom_factor, order=1)  # bilinear
            lbl_resized = zoom(lbl_slice, zoom_factor, order=0)  # nearest
            
            # Convert to 3-channel (for ResNet compatibility)
            img_3ch = np.stack([img_resized] * 3, axis=-1)
            
            slices_data.append({
                'image': img_3ch,
                'label': lbl_resized,
                'case_id': case_id,
                'slice_idx': z
            })
        
        return slices_data
    
    def process_case(self, img_path, lbl_path, case_id):
        """Process single CT case"""
        try:
            # Load NIFTI files
            img_nii = nib.load(img_path)
            lbl_nii = nib.load(lbl_path)
            
            image = img_nii.get_fdata()
            label = lbl_nii.get_fdata()
            
            # Normalize CT
            image = self.normalize_ct(image)
            
            # Remap labels to Synapse format
            label = self.remap_labels(label)
            
            # Extract 2D slices
            slices = self.extract_slices(image, label, case_id)
            
            return slices
        
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            return []
    
    def preprocess_dataset(self):
        """Main preprocessing pipeline"""
        print(f"Preprocessing {self.dataset_name} dataset...")
        
        # Find all image-label pairs
        if self.dataset_name == 'AMOS':
            img_dir = self.input_dir / 'imagesTr'
            lbl_dir = self.input_dir / 'labelsTr'
            img_files = sorted(img_dir.glob('*.nii.gz'))
        else:  # BTCV
            img_dir = self.input_dir / 'img'
            lbl_dir = self.input_dir / 'label'
            img_files = sorted(img_dir.glob('*.nii.gz'))
        
        all_slices = []
        stats = {
            'total_cases': len(img_files),
            'total_slices': 0,
            'skipped_cases': 0,
            'organ_counts': {i: 0 for i in range(9)}  # 0-8
        }
        
        for img_path in tqdm(img_files, desc=f"Processing {self.dataset_name}"):
            case_id = img_path.stem.replace('.nii', '')
            lbl_path = lbl_dir / img_path.name
            
            if not lbl_path.exists():
                print(f"Label not found for {case_id}, skipping...")
                stats['skipped_cases'] += 1
                continue
            
            # Process case
            slices = self.process_case(img_path, lbl_path, case_id)
            all_slices.extend(slices)
            stats['total_slices'] += len(slices)
            
            # Count organs
            for slice_data in slices:
                unique_labels = np.unique(slice_data['label'])
                for lbl in unique_labels:
                    if 0 <= lbl <= 8:
                        stats['organ_counts'][int(lbl)] += 1
        
        # Save preprocessed data
        output_file = self.output_dir / f'{self.dataset_name}_preprocessed.npz'
        print(f"\nSaving {len(all_slices)} slices to {output_file}...")
        
        np.savez_compressed(
            output_file,
            images=np.array([s['image'] for s in all_slices]),
            labels=np.array([s['label'] for s in all_slices]),
            case_ids=np.array([s['case_id'] for s in all_slices]),
            slice_indices=np.array([s['slice_idx'] for s in all_slices])
        )
        
        # Save statistics
        stats_file = self.output_dir / f'{self.dataset_name}_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nPreprocessing complete!")
        print(f"Total cases: {stats['total_cases']}")
        print(f"Total slices: {stats['total_slices']}")
        print(f"Skipped cases: {stats['skipped_cases']}")
        print(f"Organ distribution: {stats['organ_counts']}")
        
        return all_slices

# Usage
if __name__ == '__main__':
    # Preprocess AMOS
    amos_preprocessor = CrossDatasetPreprocessor(
        dataset_name='AMOS',
        input_dir='./datasets/AMOS',
        output_dir='./preprocessed/AMOS'
    )
    amos_preprocessor.preprocess_dataset()
    
    # Preprocess BTCV
    btcv_preprocessor = CrossDatasetPreprocessor(
        dataset_name='BTCV',
        input_dir='./datasets/BTCV',
        output_dir='./preprocessed/BTCV'
    )
    btcv_preprocessor.preprocess_dataset()
