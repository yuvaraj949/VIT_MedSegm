import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        # We don't use layer4 for TransUNet usually, or we use it as input to ViT
        # Standard TransUNet uses features from layer3 (1/16) as input to ViT?
        # Or layer4 (1/32)?
        # Roadmap says: "CNN output: (H/16, W/16, 1024)" -> This corresponds to Layer3 output.
        # Layer3 output channels: 1024.
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)      # 1/4
        
        x1 = self.layer1(x)      # 1/4, 256
        x2 = self.layer2(x1)     # 1/8, 512
        x3 = self.layer3(x2)     # 1/16, 1024
        
        return x1, x2, x3

class ViTBlock(nn.Module):
    def __init__(self, hidden_dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, N, C)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=768, num_layers=12, heads=12, mlp_dim=3072, num_patches=196):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.blocks = nn.ModuleList([
            ViTBlock(hidden_dim, heads, mlp_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, C, H, W) -> flatten -> (B, N, C)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, N, C)
        
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        # Reshape back to (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class TransUNet(nn.Module):
    def __init__(self, num_classes=9, img_dim=224):
        super().__init__()
        self.encoder = ResNetEncoder()
        
        # ViT Parameters
        # Input to ViT is 1/16 of 224 = 14x14 patches.
        # Num patches = 14*14 = 196.
        self.vit = ViT(input_dim=1024, hidden_dim=768, num_layers=12, heads=12, mlp_dim=3072, num_patches=196)
        
        # Decoder
        # ViT output: 768 channels, 14x14
        # Skip connections from ResNet:
        # x3: 1024 channels, 14x14 (We use this? Or just ViT output?)
        # Usually TransUNet uses ViT output as the bottleneck features.
        # And skips from x2 (512, 28x28), x1 (256, 56x56).
        
        # Decoder 1: 14x14 -> 28x28. Input: 768. Skip: x2 (512). Output: 512?
        self.decoder1 = DecoderBlock(768, 512, 256)
        
        # Decoder 2: 28x28 -> 56x56. Input: 256. Skip: x1 (256). Output: 128?
        self.decoder2 = DecoderBlock(256, 256, 128)
        
        # Decoder 3: 56x56 -> 112x112. Input: 128. Skip: None (or low level?). 
        # ResNet stem reduces to 1/4 (56x56).
        # We need to upsample to 224x224.
        
        self.decoder3 = DecoderBlock(128, 0, 64) # 56 -> 112
        self.decoder4 = DecoderBlock(64, 0, 16)  # 112 -> 224
        
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1, x2, x3 = self.encoder(x)
        # x1: (B, 256, 56, 56)
        # x2: (B, 512, 28, 28)
        # x3: (B, 1024, 14, 14)
        
        # ViT
        x_vit = self.vit(x3) # (B, 768, 14, 14)
        
        # Decoder
        d1 = self.decoder1(x_vit, x2) # 14->28, cat 512 -> 256
        d2 = self.decoder2(d1, x1)    # 28->56, cat 256 -> 128
        d3 = self.decoder3(d2)        # 56->112 -> 64
        d4 = self.decoder4(d3)        # 112->224 -> 16
        
        out = self.final_conv(d4)
        return out

if __name__ == "__main__":
    # Test
    model = TransUNet(num_classes=9)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (1, 9, 224, 224)
    print("TransUNet implementation verified.")
