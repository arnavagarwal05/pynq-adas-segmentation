"""
SimpleFCN Tiny — for PYNQ-Z2 HLS deployment
Channels: 3→8→16→32→16→8→4
~11K params, fits easily in Z2 BRAM
"""

import torch
import torch.nn as nn


class SimpleFCN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2, bias=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=True),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(16, 8, 2, stride=2, bias=True)
        self.head = nn.Conv2d(8, num_classes, 1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = SimpleFCN(4)
    total = sum(p.numel() for p in model.parameters())
    print(f"SimpleFCN Tiny: {total:,} params ({total/1024:.1f} KB at INT8)")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {n:35s} {str(list(p.shape)):20s} = {p.numel():>6,d}")
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print(f"\nInput: {x.shape} → Output: {y.shape}")
    assert y.shape == (1, 4, 128, 128)
    print("OK")
