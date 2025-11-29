import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff

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
