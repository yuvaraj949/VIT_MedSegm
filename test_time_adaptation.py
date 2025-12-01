"""
Test-Time Adaptation for TransUNet
Implements BN adaptation + Entropy Minimization for cross-dataset generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from copy import deepcopy

class TestTimeAdaptation:
    """
    Test-Time Adaptation for TransUNet
    
    Methods:
    1. Batch Normalization Statistics Adaptation
    2. Entropy Minimization
    """
    
    def __init__(self, model, lr=1e-4, adapt_steps=10, method='bn_entropy'):
        """
        Args:
            model: Pre-trained TransUNet model
            lr: Learning rate for adaptation (default: 1e-4)
            adapt_steps: Number of adaptation iterations per batch
            method: 'bn_only', 'entropy_only', or 'bn_entropy' (combined)
        """
        self.model = model
        self.lr = lr
        self.adapt_steps = adapt_steps
        self.method = method
        
        # Store original model state
        self.original_state = deepcopy(model.state_dict())
        
        # Setup optimizer (only update BN parameters for 'bn' methods)
        self.optimizer = self._setup_optimizer()
        
    def _setup_optimizer(self):
        """Create optimizer that only updates Batch Normalization parameters"""
        # Collect all BN parameters
        bn_params = []
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                bn_params.extend(list(module.parameters()))
        
        if len(bn_params) == 0:
            print("Warning: No BatchNorm parameters found!")
        
        optimizer = torch.optim.Adam(bn_params, lr=self.lr)
        return optimizer
    
    def reset_model(self):
        """Reset model to original pre-trained state"""
        self.model.load_state_dict(self.original_state)
    
    @staticmethod
    def entropy_loss(predictions):
        """
        Compute entropy loss for confident predictions
        
        Args:
            predictions: Model output logits [B, C, H, W]
        
        Returns:
            entropy: Scalar entropy loss (lower = more confident)
        """
        # Softmax to get probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Compute entropy: -sum(p * log(p))
        log_probs = F.log_softmax(predictions, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)
        
        # Average over spatial dimensions and batch
        return entropy.mean()
    
    def adapt_single_batch(self, images):
        """
        Adapt model to a single batch of test images
        
        Args:
            images: Batch of images [B, C, H, W]
        
        Returns:
            predictions: Final adapted predictions [B, C, H, W]
        """
        self.model.train()  # Enable BN update mode
        
        for step in range(self.adapt_steps):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss based on method
            if self.method == 'entropy_only':
                loss = self.entropy_loss(outputs)
            elif self.method == 'bn_only':
                # BN adaptation only (no explicit loss, just forward pass)
                loss = torch.tensor(0.0, device=images.device)
            else:  # bn_entropy (combined)
                loss = self.entropy_loss(outputs)
            
            # Backward pass (only updates BN params)
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()
        
        # Final prediction after adaptation
        self.model.eval()
        with torch.no_grad():
            final_outputs = self.model(images)
        
        return final_outputs
    
    def adapt_and_predict(self, dataloader, device='cuda'):
        """
        Adapt model on entire test set and collect predictions
        
        Args:
            dataloader: Test data loader
            device: Device to run on
        
        Returns:
            all_predictions: List of prediction tensors
            all_labels: List of ground truth tensors
        """
        all_predictions = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="TTA Progress")):
            images = images.to(device)
            labels = labels.to(device)
            
            # Adapt and predict
            predictions = self.adapt_single_batch(images)
            
            # Store results
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            
            # Optional: Reset model after each batch for independent adaptation
            # Uncomment if you want batch-independent adaptation
            # self.reset_model()
        
        return all_predictions, all_labels


def evaluate_tta(model, test_loader, device='cuda', tta_config=None):
    """
    Evaluate model with Test-Time Adaptation
    
    Args:
        model: Pre-trained model
        test_loader: Test data loader
        device: Device
        tta_config: Dict with TTA settings
    
    Returns:
        results: Dict with metrics
    """
    if tta_config is None:
        tta_config = {
            'lr': 1e-4,
            'adapt_steps': 10,
            'method': 'bn_entropy'
        }
    
    # Initialize TTA
    tta = TestTimeAdaptation(
        model=model,
        lr=tta_config['lr'],
        adapt_steps=tta_config['adapt_steps'],
        method=tta_config['method']
    )
    
    # Run adaptation
    predictions, labels = tta.adapt_and_predict(test_loader, device)
    
    # Compute metrics
    results = compute_segmentation_metrics(predictions, labels)
    
    return results


def compute_segmentation_metrics(predictions, labels):
    """
    Compute Dice, IoU, HD95 from predictions and labels
    
    Args:
        predictions: List of prediction tensors [B, C, H, W]
        labels: List of ground truth tensors [B, H, W]
    
    Returns:
        metrics: Dict with Dice, IoU, HD95
    """
    from metrics import calculate_dice, calculate_iou, calculate_hd95
    
    all_dice = []
    all_iou = []
    all_hd95 = []
    
    for pred_batch, label_batch in zip(predictions, labels):
        # Convert logits to predictions
        pred_classes = torch.argmax(pred_batch, dim=1)  # [B, H, W]
        
        for pred, label in zip(pred_classes, label_batch):
            pred_np = pred.numpy()
            label_np = label.numpy()
            
            # Compute metrics
            dice = calculate_dice(pred_np, label_np)
            iou = calculate_iou(pred_np, label_np)
            hd95 = calculate_hd95(pred_np, label_np)
            
            all_dice.append(dice)
            all_iou.append(iou)
            all_hd95.append(hd95)
    
    results = {
        'dice_mean': np.mean(all_dice),
        'dice_std': np.std(all_dice),
        'iou_mean': np.mean(all_iou),
        'iou_std': np.std(all_iou),
        'hd95_mean': np.mean(all_hd95),
        'hd95_std': np.std(all_hd95)
    }
    
    return results
