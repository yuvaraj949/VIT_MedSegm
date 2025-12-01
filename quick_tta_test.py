"""
Quick TTA Test Script - Verify everything works
Run this before full experiments
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from test_time_adaptation import TestTimeAdaptation

# Import your actual TransUNet - adjust import based on your structure
try:
    from transunet import TransUNet
except ImportError:
    print("Error: Could not import TransUNet. Please check your transunet.py file exists.")
    exit(1)

def quick_tta_sanity_check():
    """Quick test to verify TTA implementation"""
    print("\n" + "="*60)
    print("TTA SANITY CHECK")
    print("="*60)
    
    # Create dummy data
    batch_size = 4
    num_classes = 9
    img_size = 224
    
    dummy_images = torch.randn(batch_size, 3, img_size, img_size)
    dummy_labels = torch.randint(0, num_classes, (batch_size, img_size, img_size))
    
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=batch_size)
    
    # Create model - FIXED: Use correct parameter names for your TransUNet
    print("\n[Step 1] Creating TransUNet model...")
    try:
        # Try common parameter variations
        model = TransUNet(
            img_dim=img_size,
            num_classes=num_classes  # Changed from out_channels
        )
    except TypeError as e:
        print(f"First attempt failed: {e}")
        print("Trying alternative parameter names...")
        try:
            # Alternative 1: n_classes
            model = TransUNet(
                img_dim=img_size,
                n_classes=num_classes
            )
        except TypeError:
            try:
                # Alternative 2: out_channels with different structure
                model = TransUNet(
                    img_size=img_size,
                    num_classes=num_classes
                )
            except TypeError:
                # Alternative 3: Minimal parameters
                model = TransUNet()
                print("Warning: Using default TransUNet initialization")
    
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    dummy_images = dummy_images.to(device)
    
    print(f"✓ Model created and moved to {device}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test 1: Baseline inference
    print("\n--- Test 1: Baseline Inference ---")
    with torch.no_grad():
        baseline_output = model(dummy_images)
    print(f"✓ Baseline output shape: {baseline_output.shape}")
    print(f"✓ Baseline output range: [{baseline_output.min():.3f}, {baseline_output.max():.3f}]")
    
    # Test 2: TTA with BN only
    print("\n--- Test 2: TTA with BN Adaptation Only ---")
    try:
        tta_bn = TestTimeAdaptation(model, lr=1e-4, adapt_steps=5, method='bn_only')
        tta_bn_output = tta_bn.adapt_single_batch(dummy_images)
        print(f"✓ TTA-BN output shape: {tta_bn_output.shape}")
        print(f"✓ Output changed: {not torch.allclose(baseline_output, tta_bn_output, atol=1e-3)}")
    except Exception as e:
        print(f"✗ TTA-BN test failed: {e}")
        return False
    
    # Test 3: TTA with Entropy
    print("\n--- Test 3: TTA with Entropy Minimization ---")
    try:
        tta_entropy = TestTimeAdaptation(model, lr=1e-4, adapt_steps=5, method='entropy_only')
        tta_entropy_output = tta_entropy.adapt_single_batch(dummy_images)
        entropy_loss = TestTimeAdaptation.entropy_loss(tta_entropy_output)
        print(f"✓ TTA-Entropy output shape: {tta_entropy_output.shape}")
        print(f"✓ Entropy loss: {entropy_loss.item():.4f}")
    except Exception as e:
        print(f"✗ TTA-Entropy test failed: {e}")
        return False
    
    # Test 4: TTA with Combined
    print("\n--- Test 4: TTA with Combined (BN + Entropy) ---")
    try:
        tta_combined = TestTimeAdaptation(model, lr=1e-4, adapt_steps=10, method='bn_entropy')
        tta_combined_output = tta_combined.adapt_single_batch(dummy_images)
        print(f"✓ TTA-Combined output shape: {tta_combined_output.shape}")
    except Exception as e:
        print(f"✗ TTA-Combined test failed: {e}")
        return False
    
    # Compare outputs
    print("\n--- Output Comparison ---")
    print(f"Baseline vs TTA-BN L2 distance: {torch.norm(baseline_output - tta_bn_output).item():.4f}")
    print(f"Baseline vs TTA-Entropy L2 distance: {torch.norm(baseline_output - tta_entropy_output).item():.4f}")
    print(f"Baseline vs TTA-Combined L2 distance: {torch.norm(baseline_output - tta_combined_output).item():.4f}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\n[NEXT STEP] You can now run: python run_tta_experiments.py")
    
    return True

if __name__ == "__main__":
    success = quick_tta_sanity_check()
    if not success:
        print("\n" + "="*60)
        print("⚠ TESTS FAILED - Please check your TransUNet implementation")
        print("="*60)
        exit(1)
