"""
Evaluate TransUNet trained on Synapse on AMOS and BTCV datasets
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from scipy.ndimage import zoom
from transunet import TransUNet
from metrics import dice_coefficient, iou_score, hausdorff_distance_95

class CrossDatasetEvaluator:
    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: Path to trained TransUNet model (best_model.pth)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.organ_names = {
            0: 'Background',
            1: 'Aorta',
            2: 'Gallbladder',
            3: 'Kidney (L)',
            4: 'Kidney (R)',
            5: 'Liver',
            6: 'Pancreas',
            7: 'Spleen',
            8: 'Stomach'
        }
    
    def _load_model(self, model_path):
        """Load trained TransUNet model"""
        print(f"Loading model from {model_path}...")
        model = TransUNet(num_classes=14, img_dim=224).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.eval()
        print("Model loaded successfully!")
        return model
    
    def load_dataset(self, dataset_name):
        """Load preprocessed dataset"""
        if dataset_name == 'Synapse':
            # Synapse data is stored as individual .npz files in data/preprocessed/test
            data_dir = Path('./data/preprocessed/test')
            print(f"\nLoading {dataset_name} dataset from {data_dir}...")
            files = sorted(list(data_dir.glob('*.npz')))
            
            if not files:
                raise FileNotFoundError(f"No .npz files found in {data_dir}")
                
            images = []
            labels = []
            
            for f in tqdm(files, desc="Loading Synapse slices"):
                data = np.load(f)
                # Synapse preprocessing saves as 'image' and 'label'
                # Image might be 2D (H, W) or 3D (C, H, W) or (H, W, C)
                img = data['image']
                lbl = data['label']
                
                # Resize to 224x224 if needed
                if img.shape[0] != 224 or img.shape[1] != 224:
                    zoom_factor = (224 / img.shape[0], 224 / img.shape[1])
                    img = zoom(img, zoom_factor, order=1)  # Linear interpolation for image
                    lbl = zoom(lbl, zoom_factor, order=0)  # Nearest neighbor for label
                
                # Ensure image is 3-channel for TransUNet (H, W, 3)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                
                images.append(img)
                labels.append(lbl)
                
            return np.array(images), np.array(labels)
            
        else:
            # AMOS/BTCV are stored as single .npz files
            data_path = Path(f'./preprocessed/{dataset_name}/{dataset_name}_preprocessed.npz')
            print(f"\nLoading {dataset_name} dataset from {data_path}...")
            
            if not data_path.exists():
                raise FileNotFoundError(f"File not found: {data_path}")
                
            data = np.load(data_path)
            images = data['images']
            labels = data['labels']
            
            print(f"Loaded {len(images)} slices from {dataset_name}")
            return images, labels
    
    def evaluate_dataset(self, dataset_name):
        """Evaluate model on a dataset"""
        # Load dataset
        try:
            images, labels = self.load_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None
            
        if len(images) == 0:
            print(f"Dataset {dataset_name} is empty.")
            return None
        
        # Storage for per-slice results
        all_dice_scores = []
        all_iou_scores = []
        all_hd95_scores = []
        
        print(f"\nEvaluating on {dataset_name}...")
        
        # Batch evaluation
        batch_size = 16
        num_batches = (len(images) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {dataset_name}"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(images))
                
                # Get batch
                batch_images = images[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]
                
                # Convert to tensor
                batch_images = torch.from_numpy(batch_images).float().permute(0, 3, 1, 2).to(self.device)
                batch_labels = torch.from_numpy(batch_labels).long().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_images)
                predictions = torch.argmax(outputs, dim=1)
                
                # Compute metrics
                for i in range(len(batch_images)):
                    pred = predictions[i].cpu().numpy()
                    gt = batch_labels[i].cpu().numpy()
                    
                    # Per-class Dice and IoU
                    # Synapse has 8 organs (1-8). 
                    # We compute metrics for each organ individually.
                    slice_dice = []
                    slice_iou = []
                    
                    # We need to ensure we cover all 14 classes (0-13) or just the relevant 8 organs?
                    # The model output is 14 classes. The ground truth for Synapse is 0-8.
                    # We will evaluate 0-13 to match model output, but only report 1-8.
                    
                    for c in range(14): 
                        pred_c = (pred == c).astype(np.float32)
                        gt_c = (gt == c).astype(np.float32)
                        
                        slice_dice.append(dice_coefficient(pred_c, gt_c))
                        slice_iou.append(iou_score(pred_c, gt_c))
                    
                    # HD95 (only for organs present in ground truth, typically 1-8)
                    hd95_scores = []
                    for organ_id in range(1, 9):  # Skip background
                        if np.sum(gt == organ_id) > 0:
                            hd95 = hausdorff_distance_95(pred == organ_id, gt == organ_id)
                            hd95_scores.append(hd95)
                    
                    all_dice_scores.append(slice_dice)
                    all_iou_scores.append(slice_iou)
                    all_hd95_scores.append(np.mean(hd95_scores) if hd95_scores else 0)
        
        # Aggregate results
        all_dice_scores = np.array(all_dice_scores)  # Shape: (num_slices, 14)
        all_iou_scores = np.array(all_iou_scores)
        
        results = {
            'dataset': dataset_name,
            'mean_dice': np.mean(all_dice_scores[:, 1:9]),  # Organs only (1-8)
            'std_dice': np.std(all_dice_scores[:, 1:9]),
            'mean_iou': np.mean(all_iou_scores[:, 1:9]),
            'std_iou': np.std(all_iou_scores[:, 1:9]),
            'mean_hd95': np.mean(all_hd95_scores),
            'std_hd95': np.std(all_hd95_scores),
            'per_organ_dice': {},
            'per_organ_iou': {},
            'per_organ_hd95': {}
        }
        
        # Per-organ statistics
        for organ_id in range(1, 9):
            organ_name = self.organ_names[organ_id]
            results['per_organ_dice'][organ_name] = {
                'mean': np.mean(all_dice_scores[:, organ_id]),
                'std': np.std(all_dice_scores[:, organ_id])
            }
            results['per_organ_iou'][organ_name] = {
                'mean': np.mean(all_iou_scores[:, organ_id]),
                'std': np.std(all_iou_scores[:, organ_id])
            }
        
        return results
    
    def run_cross_dataset_evaluation(self):
        """Evaluate on Synapse (in-domain), AMOS, and BTCV (cross-domain)"""
        results = {}
        
        # Evaluate on each dataset
        for dataset in ['Synapse', 'AMOS', 'BTCV']:
            try:
                results[dataset] = self.evaluate_dataset(dataset)
            except Exception as e:
                print(f"Error evaluating {dataset}: {e}")
                results[dataset] = None
        
        return results
    
    def generate_report(self, results):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*80)
        print("CROSS-DATASET EVALUATION REPORT")
        print("="*80 + "\n")
        
        # Overall performance table
        print("Overall Performance (Mean ± Std)")
        print("-" * 80)
        print(f"{'Dataset':<15} {'Dice (%)':<20} {'IoU (%)':<20} {'HD95 (mm)':<20}")
        print("-" * 80)
        
        for dataset, result in results.items():
            if result:
                print(f"{dataset:<15} "
                      f"{result['mean_dice']*100:.2f}±{result['std_dice']*100:.2f}  "
                      f"{result['mean_iou']*100:.2f}±{result['std_iou']*100:.2f}  "
                      f"{result['mean_hd95']:.2f}±{result['std_hd95']:.2f}")
            else:
                 print(f"{dataset:<15} N/A (Failed or Empty)")

        print("\n" + "="*80)
        
        # Per-organ comparison
        print("\nPer-Organ Dice Scores (%)")
        print("-" * 120)
        
        # Create DataFrame for better visualization
        organ_data = []
        organ_names = []  # Initialize here
        for organ_id in range(1, 9):
            organ_name = self.organ_names[organ_id]
            organ_names.append(organ_name)
            row = {'Organ': organ_name}
            for dataset, result in results.items():
                if result and organ_name in result['per_organ_dice']:
                    dice = result['per_organ_dice'][organ_name]
                    row[dataset] = f"{dice['mean']*100:.2f}±{dice['std']*100:.2f}"
                else:
                    row[dataset] = "N/A"
            organ_data.append(row)
        
        df = pd.DataFrame(organ_data)
        print(df.to_string(index=False))
        
        # Calculate generalization gap
        if 'Synapse' in results and results['Synapse']:
            synapse_dice = results['Synapse']['mean_dice'] * 100
            
            print("\n" + "="*80)
            print("GENERALIZATION GAP ANALYSIS")
            print("="*80)
            
            for dataset in ['AMOS', 'BTCV']:
                if dataset in results and results[dataset]:
                    cross_dice = results[dataset]['mean_dice'] * 100
                    gap = synapse_dice - cross_dice
                    print(f"\n{dataset} Generalization Gap: {gap:.2f}%")
                    print(f"  Synapse (In-Domain):  {synapse_dice:.2f}%")
                    print(f"  {dataset} (Cross-Domain): {cross_dice:.2f}%")
                    print(f"  Performance Drop:     {(gap/synapse_dice)*100:.1f}% relative")
        
        return df
    
    def save_results(self, results, output_dir='./cross_dataset_results'):
        """Save results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results as JSON
        import json
        with open(output_dir / 'results.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for dataset, result in results.items():
                if result:
                    json_results[dataset] = {
                        'mean_dice': float(result['mean_dice']),
                        'std_dice': float(result['std_dice']),
                        'mean_iou': float(result['mean_iou']),
                        'std_iou': float(result['std_iou']),
                        'mean_hd95': float(result['mean_hd95']),
                        'std_hd95': float(result['std_hd95'])
                    }
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
    
    def plot_results(self, results, output_dir='./cross_dataset_results'):
        """Generate visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter out failed datasets
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("No valid results to plot.")
            return

        # 1. Overall performance bar chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        datasets = list(valid_results.keys())
        metrics = ['mean_dice', 'mean_iou', 'mean_hd95']
        titles = ['Dice Coefficient (%)', 'IoU (%)', 'HD95 (mm)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            values = [valid_results[d][metric] * (100 if 'dice' in metric or 'iou' in metric else 1) 
                     for d in datasets]
            
            axes[idx].bar(datasets, values, color=['#2ecc71', '#3498db', '#e74c3c'][:len(datasets)])
            axes[idx].set_title(title, fontsize=14, fontweight='bold')
            axes[idx].set_ylabel(title.split('(')[1].strip(')'))
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'overall_performance.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_dir / 'overall_performance.png'}")
        
        # 2. Per-organ heatmap
        organ_dice_data = []
        organ_names = []
        
        for organ_id in range(1, 9):
            organ_name = self.organ_names[organ_id]
            organ_names.append(organ_name)
            row = []
            for dataset in datasets:
                if organ_name in valid_results[dataset]['per_organ_dice']:
                    dice = valid_results[dataset]['per_organ_dice'][organ_name]['mean'] * 100
                    row.append(dice)
                else:
                    row.append(0)
            organ_dice_data.append(row)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(organ_dice_data, annot=True, fmt='.1f', cmap='RdYlGn',
                   xticklabels=datasets, yticklabels=organ_names,
                   cbar_kws={'label': 'Dice Score (%)'})
        plt.title('Per-Organ Dice Scores Across Datasets', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'per_organ_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_dir / 'per_organ_heatmap.png'}")
        
        plt.close('all')

# Usage
if __name__ == '__main__':
    # Initialize evaluator
    evaluator = CrossDatasetEvaluator(
        model_path='./models/best_model.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run cross-dataset evaluation
    results = evaluator.run_cross_dataset_evaluation()
    
    # Generate report
    df = evaluator.generate_report(results)
    
    # Save results
    evaluator.save_results(results)
    
    # Plot results
    evaluator.plot_results(results)
    
    print("\nCross-dataset evaluation complete!")
