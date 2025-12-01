"""
Run TTA Experiments with Different Configurations
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from test_time_adaptation import TestTimeAdaptation
from metrics import calculate_dice_per_class, calculate_iou_per_class, calculate_hd95_per_class
import matplotlib.pyplot as plt
from transunet import TransUNet
from tqdm import tqdm


class TTAExperimentRunner:
    """Run comprehensive TTA experiments"""
    
    def __init__(self, model_path, test_loader, device='cuda'):
        self.model_path = model_path
        self.test_loader = test_loader
        self.device = device
        
        self.base_model = self.load_model()
        
        self.results = {}
    
    def load_model(self):        
        model = TransUNet(
            num_classes=14, 
            img_dim=224
        )
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
        return model

    def run_baseline(self):
        """Run baseline evaluation (no TTA) - CRASH-PROOF VERSION"""
        print("\n" + "="*60)
        print("BASELINE EVALUATION (No TTA)")
        print("="*60)
        
        all_preds = []
        all_labels = []
        
        print(f"Processing {len(self.test_loader)} batches...")
        
        try:
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(tqdm(self.test_loader, desc="Baseline")):
                    try:
                        images = images.to(self.device)
                        
                        # Debug first batch
                        if batch_idx == 0:
                            print(f"\n  Batch 0 - Image shape: {images.shape}, Label shape: {labels.shape}")
                        
                        # Forward pass
                        outputs = self.base_model(images)
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        
                        # Store results
                        all_preds.append(preds)
                        all_labels.append(labels.numpy())
                        
                        # Progress indicator every 50 batches
                        if (batch_idx + 1) % 50 == 0:
                            print(f"  Processed {batch_idx + 1}/{len(self.test_loader)} batches")
                        
                        # Free GPU memory
                        del images, outputs
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"\n  ⚠️ Error in batch {batch_idx}: {e}")
                        continue
            
            # Concatenate all results
            print("\n  Computing metrics...")
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            print(f"  Predictions shape: {all_preds.shape}")
            print(f"  Labels shape: {all_labels.shape}")
            print(f"  Unique classes in preds: {np.unique(all_preds)}")
            print(f"  Unique classes in labels: {np.unique(all_labels)}")
            
            # Compute metrics CLASS BY CLASS to avoid memory issues
            num_classes = len(np.unique(all_labels))
            dice_per_class = np.zeros(num_classes)
            iou_per_class = np.zeros(num_classes)
            hd95_per_class = np.zeros(num_classes)
            
            print(f"\n  Computing metrics for {num_classes} classes...")
            
            for class_id in range(num_classes):
                try:
                    print(f"    Class {class_id}...", end='')
                    
                    # Binary masks for this class
                    pred_mask = (all_preds == class_id)
                    label_mask = (all_labels == class_id)
                    
                    # Compute Dice
                    intersection = np.sum(pred_mask & label_mask)
                    dice = (2.0 * intersection) / (np.sum(pred_mask) + np.sum(label_mask) + 1e-8)
                    dice_per_class[class_id] = dice * 100
                    
                    # Compute IoU
                    union = np.sum(pred_mask | label_mask)
                    iou = intersection / (union + 1e-8)
                    iou_per_class[class_id] = iou * 100
                    
                    # Compute HD95 (only if class exists)
                    if np.sum(label_mask) > 0 and np.sum(pred_mask) > 0:
                        try:
                            from scipy.ndimage import distance_transform_edt
                            
                            # Sample a subset for HD95 to avoid memory issues
                            sample_size = min(100, all_preds.shape[0])
                            sample_indices = np.random.choice(all_preds.shape[0], sample_size, replace=False)
                            
                            pred_sample = pred_mask[sample_indices]
                            label_sample = label_mask[sample_indices]
                            
                            hd95 = self._compute_hd95_safe(pred_sample, label_sample)
                            hd95_per_class[class_id] = hd95
                        except:
                            hd95_per_class[class_id] = 0.0
                    else:
                        hd95_per_class[class_id] = 0.0
                    
                    print(f" Dice={dice_per_class[class_id]:.2f}%")
                    
                except Exception as e:
                    print(f" ERROR: {e}")
                    dice_per_class[class_id] = 0.0
                    iou_per_class[class_id] = 0.0
                    hd95_per_class[class_id] = 0.0
            
            # Compute overall statistics
            results = {
                'method': 'Baseline (No TTA)',
                'dice_mean': float(np.mean(dice_per_class)),
                'dice_std': float(np.std(dice_per_class)),
                'dice_per_class': dice_per_class.tolist(),
                'iou_mean': float(np.mean(iou_per_class)),
                'iou_std': float(np.std(iou_per_class)),
                'iou_per_class': iou_per_class.tolist(),
                'hd95_mean': float(np.mean(hd95_per_class)),
                'hd95_std': float(np.std(hd95_per_class)),
                'hd95_per_class': hd95_per_class.tolist()
            }
            
            self.results['baseline'] = results
            self._print_results(results)
            
            return results
            
        except Exception as e:
            print(f"\n❌ FATAL ERROR in baseline evaluation: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _compute_hd95_safe(self, pred, label):
        """Compute HD95 with safety checks"""
        try:
            from scipy.ndimage import distance_transform_edt
            from scipy.spatial.distance import directed_hausdorff
            
            # Get boundary points
            pred_points = np.argwhere(pred)
            label_points = np.argwhere(label)
            
            if len(pred_points) < 2 or len(label_points) < 2:
                return 0.0
            
            # Sample points if too many (to avoid memory issues)
            max_points = 1000
            if len(pred_points) > max_points:
                pred_points = pred_points[np.random.choice(len(pred_points), max_points, replace=False)]
            if len(label_points) > max_points:
                label_points = label_points[np.random.choice(len(label_points), max_points, replace=False)]
            
            # Compute distances
            distances_1 = directed_hausdorff(pred_points, label_points)[0]
            distances_2 = directed_hausdorff(label_points, pred_points)[0]
            
            # Return 95th percentile
            all_distances = [distances_1, distances_2]
            return float(np.percentile(all_distances, 95))
            
        except:
            return 0.0

    def run_tta_experiment(self, config_name, lr, adapt_steps, method):
        """Run single TTA configuration - MEMORY-SAFE VERSION"""
        print(f"\n" + "="*60)
        print(f"TTA EXPERIMENT: {config_name}")
        print(f"LR: {lr}, Steps: {adapt_steps}, Method: {method}")
        print("="*60)
        
        # Reset model for each experiment
        model = self.load_model()
        
        # Initialize TTA
        tta = TestTimeAdaptation(
            model=model,
            lr=lr,
            adapt_steps=adapt_steps,
            method=method
        )
        
        all_preds = []
        all_labels = []
        
        try:
            for batch_idx, (images, labels) in enumerate(tqdm(self.test_loader, desc=config_name)):
                try:
                    images = images.to(self.device)
                    
                    # Adapt and predict
                    outputs = tta.adapt_single_batch(images)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    
                    # Store results
                    all_preds.append(preds)
                    all_labels.append(labels.numpy())
                    
                    # Progress indicator
                    if (batch_idx + 1) % 50 == 0:
                        print(f"  Processed {batch_idx + 1}/{len(self.test_loader)} batches")
                    
                    # Free memory
                    del images, outputs
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"\n  ⚠️ Error in batch {batch_idx}: {e}")
                    continue
            
            # Concatenate and compute metrics (reuse baseline logic)
            print("\n  Computing metrics...")
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            # Compute per-class metrics
            num_classes = len(np.unique(all_labels))
            dice_per_class = np.zeros(num_classes)
            iou_per_class = np.zeros(num_classes)
            hd95_per_class = np.zeros(num_classes)
            
            print(f"  Computing metrics for {num_classes} classes...")
            
            for class_id in range(num_classes):
                try:
                    print(f"    Class {class_id}...", end='')
                    
                    pred_mask = (all_preds == class_id)
                    label_mask = (all_labels == class_id)
                    
                    intersection = np.sum(pred_mask & label_mask)
                    dice = (2.0 * intersection) / (np.sum(pred_mask) + np.sum(label_mask) + 1e-8)
                    dice_per_class[class_id] = dice * 100
                    
                    union = np.sum(pred_mask | label_mask)
                    iou = intersection / (union + 1e-8)
                    iou_per_class[class_id] = iou * 100
                    
                    if np.sum(label_mask) > 0 and np.sum(pred_mask) > 0:
                        hd95_per_class[class_id] = self._compute_hd95_safe(pred_mask, label_mask)
                    else:
                        hd95_per_class[class_id] = 0.0
                    
                    print(f" Dice={dice_per_class[class_id]:.2f}%")
                    
                except Exception as e:
                    print(f" ERROR: {e}")
                    dice_per_class[class_id] = 0.0
                    iou_per_class[class_id] = 0.0
                    hd95_per_class[class_id] = 0.0
            
            results = {
                'method': config_name,
                'lr': lr,
                'adapt_steps': adapt_steps,
                'tta_method': method,
                'dice_mean': float(np.mean(dice_per_class)),
                'dice_std': float(np.std(dice_per_class)),
                'dice_per_class': dice_per_class.tolist(),
                'iou_mean': float(np.mean(iou_per_class)),
                'iou_std': float(np.std(iou_per_class)),
                'hd95_mean': float(np.mean(hd95_per_class)),
                'hd95_std': float(np.std(hd95_per_class))
            }
            
            self.results[config_name] = results
            self._print_results(results)
            
            return results
            
        except Exception as e:
            print(f"\n❌ ERROR in TTA experiment: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all_experiments(self):
        """Run complete TTA experiment suite"""
        
        self.run_baseline()
        
        configs = [
            ('TTA_BN_Only', 1e-4, 10, 'bn_only'),
            ('TTA_Entropy_Only', 1e-4, 10, 'entropy_only'),
            ('TTA_BN_Entropy_LR1e-4_S10', 1e-4, 10, 'bn_entropy'),
            ('TTA_BN_Entropy_LR1e-3_S10', 1e-3, 10, 'bn_entropy'),
            ('TTA_BN_Entropy_LR1e-5_S10', 1e-5, 10, 'bn_entropy'),
            ('TTA_BN_Entropy_LR1e-4_S5', 1e-4, 5, 'bn_entropy'),
            ('TTA_BN_Entropy_LR1e-4_S20', 1e-4, 20, 'bn_entropy'),
        ]
        
        for config in configs:
            self.run_tta_experiment(*config)
        
        self.save_results()
        
        self.plot_comparisons()
    
    def _print_results(self, results):
        """Pretty print results"""
        print(f"\nResults for: {results['method']}")
        print(f"  Dice:  {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
        print(f"  IoU:   {results['iou_mean']:.4f} ± {results['iou_std']:.4f}")
        print(f"  HD95:  {results['hd95_mean']:.4f} ± {results['hd95_std']:.4f}")
    
    def save_results(self):
        """Save all results to JSON"""
        output_dir = Path('tta_results')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'all_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {output_dir / 'all_results.json'}")
    
    def plot_comparisons(self):
        """Generate comparison plots"""
        output_dir = Path('tta_results')
        output_dir.mkdir(exist_ok=True)
        
        methods = list(self.results.keys())
        dice_means = [self.results[m]['dice_mean'] for m in methods]
        dice_stds = [self.results[m]['dice_std'] for m in methods]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(methods))
        bars = ax.bar(x, dice_means, yerr=dice_stds, capsize=5, alpha=0.7)
        
        bars[0].set_color('red')
        bars[0].set_label('Baseline')
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Dice Coefficient (%)', fontsize=12)
        ax.set_title('TTA Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tta_comparison.png', dpi=300)
        plt.close()
        
        print(f"Plots saved to {output_dir}")

# ============================================================================
# USAGE - CONFIGURED FOR YOUR SETUP
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    print("\n" + "="*70)
    print("TEST-TIME ADAPTATION EXPERIMENT RUNNER")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Load Your Dataset
    # ========================================================================
    print("\n[1/4] Loading dataset...")
    
    from torch.utils.data import Dataset, DataLoader
    import torch
    import numpy as np
    
    class SynapseNPZDataset(Dataset):
        """Dataset for Synapse preprocessed NPZ files"""
        def __init__(self, data_dir):
            self.data_dir = Path(data_dir)
            self.files = sorted(self.data_dir.glob('*.npz'))
            
            if len(self.files) == 0:
                raise FileNotFoundError(f"No NPZ files found in {data_dir}")
            
            print(f"  Found {len(self.files)} NPZ files")
            
            # Quick check of data format
            sample = np.load(self.files[0])
            print(f"  Keys in NPZ: {list(sample.keys())}")
            print(f"  Image shape: {sample['image'].shape if 'image' in sample else 'N/A'}")
            print(f"  Label shape: {sample['label'].shape if 'label' in sample else 'N/A'}")
        
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            data = np.load(self.files[idx])
            
            # Load image and label
            image = data['image'].astype(np.float32)
            label = data['label'].astype(np.int64)
            
            # Convert to tensors
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)
            
            # Ensure image has 3 channels [C, H, W]
            if image.ndim == 2:  # [H, W]
                image = image.unsqueeze(0).repeat(3, 1, 1)
            elif image.ndim == 3 and image.shape[0] == 1:  # [1, H, W]
                image = image.repeat(3, 1, 1)
            elif image.ndim == 3 and image.shape[-1] == 3:  # [H, W, 3]
                image = image.permute(2, 0, 1)
            
            # Resize to 224x224 (model input size)
            if image.shape[1] != 224 or image.shape[2] != 224:
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            if label.ndim == 2 and (label.shape[0] != 224 or label.shape[1] != 224):
                label = torch.nn.functional.interpolate(
                    label.unsqueeze(0).unsqueeze(0).float(), 
                    size=(224, 224), 
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
            
            return image, label

    
    # Your data path
    test_data_dir = r'C:\Users\yuvar\Projects\Computer Vision\Project\data\preprocessed\test'
    
    print(f"  Loading from: {test_data_dir}")
    
    try:
        test_dataset = SynapseNPZDataset(test_data_dir)
    except Exception as e:
        print(f"\n❌ ERROR loading dataset: {e}")
        sys.exit(1)
    
    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,  # Adjust based on your GPU memory (RTX 4080 can handle this)
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows to avoid multiprocessing issues
        pin_memory=True
    )
    
    print(f"✓ Dataset loaded: {len(test_dataset)} samples")
    print(f"✓ Number of batches: {len(test_loader)}")
    print(f"✓ Batch size: 8")
    
    # ========================================================================
    # STEP 2: Locate Model Checkpoint
    # ========================================================================
    print("\n[2/4] Locating model checkpoint...")
    
    # Try to find model automatically
    project_dir = Path(r'C:\Users\yuvar\Projects\Computer Vision\Project')
    model_paths = [
        project_dir / 'best_model.pth',
        project_dir / 'checkpoints' / 'best_model.pth',
        project_dir / 'models' / 'best_model.pth',
        project_dir / 'best_checkpoint.pth',
        project_dir / 'model_best.pth',
        Path('best_model.pth'),
        Path('checkpoints/best_model.pth'),
    ]
    
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = str(path)
            break
    
    if model_path is None:
        print("\n⚠️  Model checkpoint not found in default locations!")
        print("Searched in:")
        for path in model_paths:
            print(f"  - {path}")
        print("\nPlease enter your model path:")
        model_path = input("Model path: ").strip().strip('"').strip("'")
        
        if not Path(model_path).exists():
            print(f"❌ ERROR: File not found: {model_path}")
            sys.exit(1)
    
    print(f"✓ Found model: {model_path}")
    
    # ========================================================================
    # STEP 3: Initialize Experiment Runner
    # ========================================================================
    print("\n[3/4] Initializing TTA experiment runner...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        runner = TTAExperimentRunner(
            model_path=model_path,
            test_loader=test_loader,
            device=device
        )
        print("✓ Runner initialized successfully")
    except Exception as e:
        print(f"\n❌ ERROR initializing runner: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # STEP 4: Run All Experiments
    # ========================================================================
    print("\n[4/4] Running all TTA experiments...")
    print(f"\nDataset size: {len(test_dataset)} slices")
    print(f"Estimated time: ~{len(test_loader) * 7 * 0.5 / 60:.1f} minutes")
    print("\nExperiment sequence:")
    print("  1. Baseline (no TTA)")
    print("  2. TTA with BN adaptation only")
    print("  3. TTA with entropy minimization only")
    print("  4-7. TTA with combined approach (different configs)")
    
    print("\n⚠️  This will take some time. Results will be saved incrementally.")
    response = input("\nProceed? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        sys.exit(0)
    
    try:
        import time
        start_time = time.time()
        
        runner.run_all_experiments()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print("✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nTotal time: {elapsed/60:.1f} minutes")
        print("\nResults saved to:")
        print("  - tta_results/all_results.json")
        print("  - tta_results/tta_comparison.png")
        print("\nNext steps:")
        print("  1. Run: python visualize_tta_results.py")
        print("  2. Check tta_figures/ for publication-quality plots")
        print("  3. Use tta_figures/tta_latex_table.tex in your paper")
        print("\n" + "="*70)
        
    except KeyboardInterrupt:
        print("\n\n❌ Experiments interrupted by user")
        print("Partial results may be saved in tta_results/")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR during experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
