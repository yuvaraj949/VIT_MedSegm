
"""
Comprehensive Metrics for Medical Image Segmentation
Includes per-class and aggregate metrics
"""

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
import warnings

def calculate_dice(pred, target, smooth=1e-5):
    """
    Calculate Dice coefficient for binary masks
    
    Args:
        pred: Prediction mask (H, W)
        target: Ground truth mask (H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        dice: Dice coefficient [0, 1]
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def calculate_iou(pred, target, smooth=1e-5):
    """
    Calculate Intersection over Union (IoU) / Jaccard Index
    
    Args:
        pred: Prediction mask (H, W)
        target: Ground truth mask (H, W)
        smooth: Smoothing factor
    
    Returns:
        iou: IoU score [0, 1]
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


def calculate_hd95(pred, target, percentile=95):
    """
    Calculate 95th percentile Hausdorff Distance
    
    Args:
        pred: Prediction mask (H, W)
        target: Ground truth mask (H, W)
        percentile: Percentile to use (default: 95)
    
    Returns:
        hd95: 95th percentile Hausdorff distance
    """
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return 0.0
    
    try:
        # Get boundary points
        pred_points = np.argwhere(pred > 0)
        target_points = np.argwhere(target > 0)
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return 0.0
        
        # Compute directed Hausdorff distances
        forward_distances = directed_hausdorff(pred_points, target_points)[0]
        backward_distances = directed_hausdorff(target_points, pred_points)[0]
        
        # Combine and take percentile
        all_distances = np.concatenate([
            [forward_distances],
            [backward_distances]
        ])
        
        hd95 = np.percentile(all_distances, percentile)
        return hd95
    
    except Exception as e:
        warnings.warn(f"HD95 calculation failed: {e}")
        return 0.0


def calculate_dice_per_class(predictions, labels, num_classes=None):
    """
    Calculate Dice coefficient for each class
    
    Args:
        predictions: Predicted segmentation masks (N, H, W)
        labels: Ground truth masks (N, H, W)
        num_classes: Number of classes (auto-detected if None)
    
    Returns:
        dice_per_class: Array of Dice scores per class (num_classes,)
    """
    if num_classes is None:
        num_classes = max(np.max(predictions), np.max(labels)) + 1
    
    dice_scores = np.zeros(num_classes)
    
    for class_idx in range(num_classes):
        # Create binary masks for current class
        pred_binary = (predictions == class_idx).astype(np.float32)
        label_binary = (labels == class_idx).astype(np.float32)
        
        # Calculate Dice for this class across all samples
        intersection = np.sum(pred_binary * label_binary)
        union = np.sum(pred_binary) + np.sum(label_binary)
        
        if union > 0:
            dice_scores[class_idx] = (2.0 * intersection) / union
        else:
            dice_scores[class_idx] = 0.0
    
    return dice_scores


def calculate_iou_per_class(predictions, labels, num_classes=None):
    """
    Calculate IoU for each class
    
    Args:
        predictions: Predicted segmentation masks (N, H, W)
        labels: Ground truth masks (N, H, W)
        num_classes: Number of classes (auto-detected if None)
    
    Returns:
        iou_per_class: Array of IoU scores per class (num_classes,)
    """
    if num_classes is None:
        num_classes = max(np.max(predictions), np.max(labels)) + 1
    
    iou_scores = np.zeros(num_classes)
    
    for class_idx in range(num_classes):
        # Create binary masks for current class
        pred_binary = (predictions == class_idx).astype(np.float32)
        label_binary = (labels == class_idx).astype(np.float32)
        
        # Calculate IoU for this class
        intersection = np.sum(pred_binary * label_binary)
        union = np.sum(pred_binary) + np.sum(label_binary) - intersection
        
        if union > 0:
            iou_scores[class_idx] = intersection / union
        else:
            iou_scores[class_idx] = 0.0
    
    return iou_scores


def calculate_hd95_per_class(predictions, labels, num_classes=None):
    """
    Calculate HD95 for each class
    
    Args:
        predictions: Predicted segmentation masks (N, H, W)
        labels: Ground truth masks (N, H, W)
        num_classes: Number of classes (auto-detected if None)
    
    Returns:
        hd95_per_class: Array of HD95 scores per class (num_classes,)
    """
    if num_classes is None:
        num_classes = max(np.max(predictions), np.max(labels)) + 1
    
    hd95_scores = np.zeros(num_classes)
    
    for class_idx in range(num_classes):
        # Create binary masks for current class
        pred_binary = (predictions == class_idx).astype(np.uint8)
        label_binary = (labels == class_idx).astype(np.uint8)
        
        # Calculate HD95 for this class
        hd95_scores[class_idx] = calculate_hd95(pred_binary, label_binary)
    
    return hd95_scores


def calculate_metrics_batch(predictions, labels):
    """
    Calculate all metrics for a batch of predictions
    
    Args:
        predictions: Predicted masks (N, H, W)
        labels: Ground truth masks (N, H, W)
    
    Returns:
        metrics: Dictionary with all metrics
    """
    batch_size = predictions.shape[0]
    
    dice_scores = []
    iou_scores = []
    hd95_scores = []
    
    for i in range(batch_size):
        pred = predictions[i]
        label = labels[i]
        
        dice = calculate_dice(pred, label)
        iou = calculate_iou(pred, label)
        hd95 = calculate_hd95(pred, label)
        
        dice_scores.append(dice)
        iou_scores.append(iou)
        hd95_scores.append(hd95)
    
    metrics = {
        'dice_mean': np.mean(dice_scores),
        'dice_std': np.std(dice_scores),
        'iou_mean': np.mean(iou_scores),
        'iou_std': np.std(iou_scores),
        'hd95_mean': np.mean(hd95_scores),
        'hd95_std': np.std(hd95_scores)
    }
    
    return metrics


def print_metrics_summary(metrics, title="Metrics Summary"):
    """
    Pretty print metrics summary
    
    Args:
        metrics: Dictionary with metrics
        title: Title for the summary
    """
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)
    print(f"Dice:  {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"IoU:   {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")
    print(f"HD95:  {metrics['hd95_mean']:.4f} ± {metrics['hd95_std']:.4f}")
    print("="*60)


def compare_metrics(baseline_metrics, improved_metrics):
    """
    Compare two sets of metrics and print improvements
    
    Args:
        baseline_metrics: Dictionary with baseline metrics
        improved_metrics: Dictionary with improved metrics
    """
    dice_improvement = improved_metrics['dice_mean'] - baseline_metrics['dice_mean']
    iou_improvement = improved_metrics['iou_mean'] - baseline_metrics['iou_mean']
    hd95_improvement = baseline_metrics['hd95_mean'] - improved_metrics['hd95_mean']  # Lower is better
    
    print("\n" + "="*60)
    print("IMPROVEMENT SUMMARY")
    print("="*60)
    print(f"Dice:  {dice_improvement:+.4f} ({dice_improvement/baseline_metrics['dice_mean']*100:+.2f}%)")
    print(f"IoU:   {iou_improvement:+.4f} ({iou_improvement/baseline_metrics['iou_mean']*100:+.2f}%)")
    print(f"HD95:  {hd95_improvement:+.4f} ({-hd95_improvement/baseline_metrics['hd95_mean']*100:+.2f}%)")
    print("="*60)


# Additional helper functions for visualization
def get_confusion_matrix(pred, target, num_classes):
    """
    Calculate confusion matrix
    
    Args:
        pred: Predictions (N, H, W)
        target: Ground truth (N, H, W)
        num_classes: Number of classes
    
    Returns:
        confusion_matrix: Confusion matrix (num_classes, num_classes)
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    for i in range(len(pred_flat)):
        confusion_matrix[target_flat[i], pred_flat[i]] += 1
    
    return confusion_matrix


def calculate_precision_recall(pred, target):
    """
    Calculate precision and recall
    
    Args:
        pred: Predictions (binary)
        target: Ground truth (binary)
    
    Returns:
        precision, recall: Precision and recall scores
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    true_positive = np.sum((pred_flat == 1) & (target_flat == 1))
    false_positive = np.sum((pred_flat == 1) & (target_flat == 0))
    false_negative = np.sum((pred_flat == 0) & (target_flat == 1))
    
    precision = true_positive / (true_positive + false_positive + 1e-7)
    recall = true_positive / (true_positive + false_negative + 1e-7)
    
    return precision, recall


def dice_coefficient(pred, target, smooth=1e-5):
    """
    Compute Dice Similarity Coefficient.
    pred: (B, C, H, W) or (C, H, W) - Softmax output or One-hot encoded
    target: (B, H, W) or (H, W) - Integer labels
    """
    # If pred is raw logits, apply softmax/argmax? 
    # Usually metrics expect binary masks or probability maps.
    # Assuming pred is already thresholded or argmaxed for final evaluation,
    # OR pred is probability map for loss calculation.
    # For evaluation (validation/test), we typically use argmax.
    
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    # Flatten
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def iou_score(pred, target, smooth=1e-5):
    """
    Compute Intersection over Union (Jaccard Index).
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def hausdorff_distance_95(pred, target):
    """
    Compute 95th Percentile Hausdorff Distance.
    pred: Binary mask (H, W)
    target: Binary mask (H, W)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    # Check if empty
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return 0.0 # Or some max value? Usually 0 if both empty, max if one empty.
        
    # Get coordinates of non-zero pixels
    pred_coords = np.argwhere(pred)
    target_coords = np.argwhere(target)
    
    # Calculate forward and backward Hausdorff distances
    hd_1 = directed_hausdorff(pred_coords, target_coords)[0]
    hd_2 = directed_hausdorff(target_coords, pred_coords)[0]
    
    return max(hd_1, hd_2)

def compute_metrics(pred, target, num_classes=9):
    """
    Compute metrics for a batch or single image across all classes.
    pred: (B, C, H, W) logits or (B, H, W) class indices
    target: (B, H, W) class indices
    """
    # If pred is logits, take argmax
    if pred.ndim == 4:
        pred = torch.argmax(pred, dim=1)
        
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    metrics = {
        'Dice': [],
        'IoU': [],
        'HD95': []
    }
    
    # Iterate over classes (skip background class 0 usually, or include?)
    # Synapse dataset usually evaluates 8 organs (classes 1-8).
    # Class 0 is background.
    
    for c in range(1, num_classes):
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        
        metrics['Dice'].append(dice_coefficient(pred_c, target_c))
        metrics['IoU'].append(iou_score(pred_c, target_c))
        # HD95 can be slow, maybe skip for training logs?
        # metrics['HD95'].append(hausdorff_distance_95(pred_c, target_c))
        
    return {k: np.mean(v) for k, v in metrics.items() if v}
