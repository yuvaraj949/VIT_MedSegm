import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm
from transunet import TransUNet
from metrics import compute_metrics, dice_coefficient

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dataset
class SynapseDataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.sample_list = []
        
        split_dir = os.path.join(self.base_dir, split)
        if os.path.exists(split_dir):
            self.sample_list = os.listdir(split_dir)
        
        print(f"Loaded {len(self.sample_list)} samples for {split}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]
        data_path = os.path.join(self.base_dir, self.split, slice_name)
        data = np.load(data_path)
        image, label = data['image'], data['label']
        
        # Image is (H, W), Label is (H, W)
        # Convert to tensor
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        # Add channel dim to image: (1, H, W)
        image = image.unsqueeze(0)
        
        # Repeat channels to 3 for ResNet backbone
        image = image.repeat(3, 1, 1)
        
        # Resize if needed (Roadmap: 224x224)
        # We need to resize image and label.
        # Image: Bilinear, Label: Nearest
        if self.transform:
            # Apply transforms
            # Note: standard transforms might not handle dicts easily without custom wrapper
            # For simplicity, let's do manual resize here or assume transform handles it.
            # Let's use torch.nn.functional.interpolate for simplicity and consistency
            pass
            
        # Resize to 224x224
        # Image is (3, H, W), Label is (H, W) -> need (1, 1, H, W) for interpolate
        target_size = (224, 224)
        if image.shape[-2:] != target_size:
            image = torch.nn.functional.interpolate(image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            label = torch.nn.functional.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode='nearest').long().squeeze(0).squeeze(0)
        
        sample = {'image': image, 'label': label}
            
        return sample

# Loss Function
class DiceCELoss(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        
        # Dice Loss
        pred_softmax = torch.softmax(pred, dim=1)
        dice_loss = 0
        # Skip background? Usually yes or weighted less. 
        # Let's average over all classes including background for now, or skip 0.
        for c in range(1, self.num_classes):
            pred_c = pred_softmax[:, c, :, :]
            target_c = (target == c).float()
            dice_loss += (1 - dice_coefficient(pred_c, target_c))
            
        dice_loss /= (self.num_classes - 1)
        
        return 0.5 * ce_loss + 0.5 * dice_loss

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    # Data
    train_dataset = SynapseDataset(args.data_dir, split="train")
    val_dataset = SynapseDataset(args.data_dir, split="test") # Using test as val for now? Or split train?
    # Roadmap says: "Validation phase (NO test set here!)"
    # But we only split into Train (18) and Test (12).
    # We should probably split Train into Train/Val.
    # For now, let's use a subset of Train or just use Test as Val (common in simple setups, but strictly wrong).
    # Better: Split train_dataset.
    
    # Let's split train_dataset into train/val (e.g., 80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = TransUNet(num_classes=args.num_classes, img_dim=args.img_size).to(device)
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    criterion = DiceCELoss(num_classes=args.num_classes)
    
    best_dice = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = model(images)
                
                # Compute metrics
                metrics = compute_metrics(outputs, labels, num_classes=args.num_classes)
                val_dice += metrics['Dice']
        
        val_dice /= len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Dice: {val_dice:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\yuvar\Projects\Computer Vision\Project\New Setup\data\preprocessed")
    parser.add_argument("--save_dir", type=str, default=r"C:\Users\yuvar\Projects\Computer Vision\Project\New Setup\models")
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=24) # Reduced from 24 if OOM
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    train(args)
