import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from transunet import TransUNet
from train import SynapseDataset
from metrics import dice_coefficient, iou_score, hausdorff_distance_95

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
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\yuvar\Projects\Computer Vision\Project\New Setup\data\preprocessed")
    parser.add_argument("--model_path", type=str, default=r"C:\Users\yuvar\Projects\Computer Vision\Project\New Setup\models\best_model.pth")
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--img_size", type=int, default=224)
    
    args = parser.parse_args()
    test(args)
