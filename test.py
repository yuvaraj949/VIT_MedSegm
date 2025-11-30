import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from transunet import TransUNet
from train import SynapseDataset
from metrics import dice_coefficient, iou_score, hausdorff_distance_95


def evaluate_model(model_path, data_dir=r"C:\Users\yuvar\Projects\Computer Vision\Project\data\preprocessed", 
                   num_classes=14, img_size=224, split="test", save_results=False, output_dir="results"):
    """
    Evaluate a trained model and return comprehensive metrics
    
    Args:
        model_path (str): Path to trained model weights (.pth file)
        data_dir (str): Path to preprocessed data directory
        num_classes (int): Number of output classes
        img_size (int): Input image size
        split (str): Dataset split to evaluate ('test' or 'val')
        save_results (bool): Whether to save results to CSV
        output_dir (str): Directory to save results if save_results=True
    
    Returns:
        dict: Dictionary containing:
            - 'mean_dice': Overall mean Dice score
            - 'mean_iou': Overall mean IoU score
            - 'mean_hd95': Overall mean HD95 score
            - 'std_dice': Standard deviation of Dice scores
            - 'std_iou': Standard deviation of IoU scores
            - 'std_hd95': Standard deviation of HD95 scores
            - 'per_class_metrics': List of dicts with per-class results
            - 'per_case_metrics': List of dicts with per-case results
            - 'num_params': Total model parameters
    """
    import os
    import numpy as np
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transunet import TransUNet
    from train import SynapseDataset
    from metrics import dice_coefficient, iou_score, hausdorff_distance_95
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Evaluating model: {os.path.basename(model_path)}")
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = TransUNet(num_classes=num_classes, img_dim=img_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {num_params/1e6:.2f}M")
    
    # Load data
    test_dataset = SynapseDataset(data_dir, split=split)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"[INFO] Loaded {len(test_dataset)} {split} samples")
    
    # Organ names (Synapse dataset standard)
    organ_names = {
        0: "Background",
        1: "Aorta",
        2: "Gallbladder",
        3: "Kidney (L)",
        4: "Kidney (R)",
        5: "Liver",
        6: "Pancreas",
        7: "Spleen",
        8: "Stomach"
    }
    
    # Initialize metrics storage
    per_class_metrics = {c: {'Dice': [], 'IoU': [], 'HD95': []} for c in range(1, num_classes)}
    per_case_metrics = []
    
    # Evaluation loop
    print("[INFO] Running evaluation...")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)):
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            
            # Forward pass
            output = model(image)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
            
            # Per-class metrics
            case_dice_scores = []
            case_iou_scores = []
            case_hd95_scores = []
            
            for class_id in range(1, num_classes):  # Skip background (0)
                pred_mask = (pred == class_id).cpu().numpy()
                gt_mask = (label == class_id).cpu().numpy()
                
                if np.sum(gt_mask) > 0:  # Only if class present in ground truth
                    try:
                        dice = dice_coefficient(pred_mask, gt_mask)
                        iou = iou_score(pred_mask, gt_mask)
                        hd95 = hausdorff_distance_95(pred_mask, gt_mask)
                        
                        per_class_metrics[class_id]['Dice'].append(dice)
                        per_class_metrics[class_id]['IoU'].append(iou)
                        per_class_metrics[class_id]['HD95'].append(hd95)
                        
                        case_dice_scores.append(dice)
                        case_iou_scores.append(iou)
                        case_hd95_scores.append(hd95)
                    except Exception as e:
                        print(f"[WARNING] Error computing metrics for class {class_id} in case {idx}: {e}")
            
            # Store per-case average
            if case_dice_scores:
                per_case_metrics.append({
                    'case_id': idx,
                    'dice': np.mean(case_dice_scores),
                    'iou': np.mean(case_iou_scores),
                    'hd95': np.mean(case_hd95_scores)
                })
    
    # Aggregate results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Overall metrics
    all_dice = []
    all_iou = []
    all_hd95 = []
    
    per_class_results = []
    for class_id in range(1, num_classes):
        if per_class_metrics[class_id]['Dice']:
            dice_mean = np.mean(per_class_metrics[class_id]['Dice'])
            dice_std = np.std(per_class_metrics[class_id]['Dice'])
            iou_mean = np.mean(per_class_metrics[class_id]['IoU'])
            iou_std = np.std(per_class_metrics[class_id]['IoU'])
            hd95_mean = np.mean(per_class_metrics[class_id]['HD95'])
            hd95_std = np.std(per_class_metrics[class_id]['HD95'])
            
            all_dice.extend(per_class_metrics[class_id]['Dice'])
            all_iou.extend(per_class_metrics[class_id]['IoU'])
            all_hd95.extend(per_class_metrics[class_id]['HD95'])
            
            organ_name = organ_names.get(class_id, f"Class_{class_id}")
            per_class_results.append({
                'Class': class_id,
                'Organ': organ_name,
                'Dice': dice_mean,
                'Dice_Std': dice_std,
                'IoU': iou_mean,
                'IoU_Std': iou_std,
                'HD95': hd95_mean,
                'HD95_Std': hd95_std,
                'Samples': len(per_class_metrics[class_id]['Dice'])
            })
    
    # Overall statistics
    mean_dice = np.mean(all_dice) if all_dice else 0.0
    std_dice = np.std(all_dice) if all_dice else 0.0
    mean_iou = np.mean(all_iou) if all_iou else 0.0
    std_iou = np.std(all_iou) if all_iou else 0.0
    mean_hd95 = np.mean(all_hd95) if all_hd95 else 0.0
    std_hd95 = np.std(all_hd95) if all_hd95 else 0.0
    
    print(f"\nOverall Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Overall Mean IoU:  {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Overall Mean HD95: {mean_hd95:.2f} ± {std_hd95:.2f}")
    
    # Display per-class results
    if per_class_results:
        df_class = pd.DataFrame(per_class_results)
        print("\nPer-Class Results:")
        print(df_class.to_markdown(index=False, floatfmt=".4f"))
    
    # Save results if requested
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save per-class results
        if per_class_results:
            df_class = pd.DataFrame(per_class_results)
            class_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(model_path))[0]}_per_class.csv")
            df_class.to_csv(class_file, index=False)
            print(f"\n[INFO] Per-class results saved to: {class_file}")
        
        # Save per-case results
        if per_case_metrics:
            df_case = pd.DataFrame(per_case_metrics)
            case_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(model_path))[0]}_per_case.csv")
            df_case.to_csv(case_file, index=False)
            print(f"[INFO] Per-case results saved to: {case_file}")
        
        # Save summary
        summary = {
            'model': os.path.basename(model_path),
            'mean_dice': mean_dice,
            'std_dice': std_dice,
            'mean_iou': mean_iou,
            'std_iou': std_iou,
            'mean_hd95': mean_hd95,
            'std_hd95': std_hd95,
            'num_params': num_params,
            'num_samples': len(test_dataset)
        }
        
        summary_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(model_path))[0]}_summary.json")
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Summary saved to: {summary_file}")
    
    # Return results dictionary
    return {
        'mean_dice': mean_dice,
        'std_dice': std_dice,
        'mean_iou': mean_iou,
        'std_iou': std_iou,
        'mean_hd95': mean_hd95,
        'std_hd95': std_hd95,
        'dice': mean_dice,  # For backward compatibility
        'iou': mean_iou,
        'hd95': mean_hd95,
        'params': num_params / 1e6,  # In millions
        'per_class_metrics': per_class_results,
        'per_case_metrics': per_case_metrics,
        'num_params': num_params
    }


def evaluate_multiple_models(model_dict, data_dir, output_dir="results", num_classes=14, img_size=224):
    """
    Evaluate multiple models and create comparison table
    
    Args:
        model_dict (dict): Dictionary of {model_name: model_path}
        data_dir (str): Path to data directory
        output_dir (str): Output directory for results
        num_classes (int): Number of classes
        img_size (int): Input image size
    
    Returns:
        pd.DataFrame: Comparison table with all models
    """
    import pandas as pd
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_results = []
    
    for model_name, model_path in model_dict.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print("="*60)
        
        if not os.path.exists(model_path):
            print(f"[WARNING] Model not found: {model_path}")
            continue
        
        try:
            results = evaluate_model(
                model_path=model_path,
                data_dir=data_dir,
                num_classes=num_classes,
                img_size=img_size,
                save_results=True,
                output_dir=output_dir
            )
            
            comparison_results.append({
                'Model': model_name,
                'Dice': results['mean_dice'],
                'Dice_Std': results['std_dice'],
                'IoU': results['mean_iou'],
                'IoU_Std': results['std_iou'],
                'HD95': results['mean_hd95'],
                'HD95_Std': results['std_hd95'],
                'Params (M)': results['params']
            })
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_name}: {e}")
    
    # Create comparison DataFrame
    if comparison_results:
        df_comparison = pd.DataFrame(comparison_results)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(df_comparison.to_markdown(index=False, floatfmt=".4f"))
        
        # Save comparison
        comparison_file = os.path.join(output_dir, "model_comparison.csv")
        df_comparison.to_csv(comparison_file, index=False)
        print(f"\n[INFO] Comparison saved to: {comparison_file}")
        
        return df_comparison
    else:
        print("\n[WARNING] No models were successfully evaluated")
        return None


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    test_dataset = SynapseDataset(args.data_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Model
    model = TransUNet(num_classes=args.num_classes, img_dim=args.img_size).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Metrics storage
    # Organ mapping (assuming standard Synapse labels 1-8, but we have 14 classes)
    # We will report all classes 1-13.
    organ_metrics = {c: {'Dice': [], 'IoU': [], 'HD95': []} for c in range(1, args.num_classes)}
    
    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            
            output = model(image)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
            
            # Compute metrics per organ
            for c in range(1, args.num_classes):
                pred_c = (pred == c).cpu().numpy()
                label_c = (label == c).cpu().numpy()
                
                if np.sum(label_c) > 0: # Only evaluate if organ is present in GT
                    dice = dice_coefficient(pred_c, label_c)
                    iou = iou_score(pred_c, label_c)
                    hd95 = hausdorff_distance_95(pred_c, label_c)
                    
                    organ_metrics[c]['Dice'].append(dice)
                    organ_metrics[c]['IoU'].append(iou)
                    organ_metrics[c]['HD95'].append(hd95)
    
    # Aggregate results
    print("\n=== Final Results ===")
    print(f"{'Class':<10} {'Dice':<10} {'IoU':<10} {'HD95':<10}")
    
    avg_dice = []
    avg_iou = []
    avg_hd95 = []
    
    for c in range(1, args.num_classes):
        if organ_metrics[c]['Dice']:
            d = np.mean(organ_metrics[c]['Dice'])
            i = np.mean(organ_metrics[c]['IoU'])
            h = np.mean(organ_metrics[c]['HD95'])
            
            print(f"{c:<10} {d:.4f}     {i:.4f}     {h:.4f}")
            
            avg_dice.append(d)
            avg_iou.append(i)
            avg_hd95.append(h)
        else:
            print(f"{c:<10} N/A        N/A        N/A")
            
    print("-" * 40)
    print(f"{'Average':<10} {np.mean(avg_dice):.4f}     {np.mean(avg_iou):.4f}     {np.mean(avg_hd95):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\yuvar\Projects\Computer Vision\Project\data\preprocessed")
    parser.add_argument("--model_path", type=str, default=r"C:\Users\yuvar\Projects\Computer Vision\Project\models\best_model.pth")
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--img_size", type=int, default=224)
    
    args = parser.parse_args()
    test(args)
