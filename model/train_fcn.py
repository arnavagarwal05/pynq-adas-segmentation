"""
Training script for SimpleFCN on Cityscapes + IDD (4-class ADAS).

Usage:
    # Step 1: Train on Cityscapes
    python train_fcn.py --cityscapes-dir ~/emb/cityscapes \
        --epochs 100 --batch-size 64 --lr 5e-4

    # Step 2: Joint fine-tune on Cityscapes + IDD
    python train_fcn.py --cityscapes-dir ~/emb/cityscapes \
        --idd-dir ~/emb/idd/idd20kII \
        --finetune checkpoints_fcn/best.pth \
        --epochs 50 --batch-size 64 --lr 2e-4 \
        --save-dir checkpoints_fcn_idd
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from simple_fcn import SimpleFCN
from cityscapes_dataset import CityscapesADAS, NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES

try:
    from idd_dataset import IDDADAS, CombinedADAS
    HAS_IDD = True
except ImportError:
    HAS_IDD = False


class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]
        for t, p in zip(target, pred):
            if t < self.num_classes and p < self.num_classes:
                self.confusion_matrix[t, p] += 1

    def compute(self):
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) +
                 self.confusion_matrix.sum(axis=0) - intersection)
        iou = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if union[i] > 0:
                iou[i] = intersection[i] / union[i]
        miou = np.mean(iou)
        return iou, miou

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    pbar = tqdm(dataloader, desc="Train")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    metrics = SegmentationMetrics(NUM_CLASSES, IGNORE_INDEX)
    pbar = tqdm(dataloader, desc="Val  ")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        num_batches += 1
        preds = outputs.argmax(dim=1)
        metrics.update(preds, labels)
    iou, miou = metrics.compute()
    avg_loss = total_loss / num_batches
    return avg_loss, iou, miou


def main():
    parser = argparse.ArgumentParser(description="Train SimpleFCN for ADAS (4-class)")
    parser.add_argument("--cityscapes-dir", type=str, default=None)
    parser.add_argument("--idd-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints_fcn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--finetune", type=str, default=None)
    args = parser.parse_args()

    if args.cityscapes_dir is None and args.idd_dir is None:
        parser.error("Provide at least one of --cityscapes-dir or --idd-dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    img_size = (128, 128)
    print(f"Input resolution: {img_size[0]}x{img_size[1]} (fixed for FPGA)")

    train_datasets = []
    val_dataset = None

    if args.cityscapes_dir:
        cs_train = CityscapesADAS(args.cityscapes_dir, split="train", img_size=img_size, augment=True)
        cs_val = CityscapesADAS(args.cityscapes_dir, split="val", img_size=img_size, augment=False)
        train_datasets.append(cs_train)
        val_dataset = cs_val
        print(f"Cityscapes: {len(cs_train)} train, {len(cs_val)} val")

    if args.idd_dir:
        if not HAS_IDD:
            raise RuntimeError("idd_dataset.py not found")
        idd_train = IDDADAS(args.idd_dir, split="train", img_size=img_size, augment=True)
        idd_val = IDDADAS(args.idd_dir, split="val", img_size=img_size, augment=False)
        train_datasets.append(idd_train)
        val_dataset = idd_val
        print(f"IDD: {len(idd_train)} train, {len(idd_val)} val")

    if len(train_datasets) > 1:
        train_dataset = CombinedADAS(*train_datasets)
        print(f"Combined: {len(train_dataset)} images")
    else:
        train_dataset = train_datasets[0]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = SimpleFCN(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleFCN parameters: {total_params:,} ({total_params / 1024:.1f} KB at INT8)")

    class_weights = torch.FloatTensor([0.553, 2.4767, 0.9009, 1.1235]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    start_epoch = 0
    best_miou = 0.0

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded weights from: {args.finetune} (mIoU={checkpoint.get('best_miou', '?')})")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_miou", 0.0)
        print(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou:.4f}")

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Classes: {CLASS_NAMES}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, iou_per_class, miou = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"\nEpoch [{epoch+1}/{args.epochs}] ({elapsed:.0f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  mIoU:       {miou:.4f}")
        for c, name in enumerate(CLASS_NAMES):
            print(f"    {name:12s}: {iou_per_class[c]:.4f}")

        is_best = miou > best_miou
        if is_best:
            best_miou = miou

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": best_miou,
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "model_type": "SimpleFCN",
            "img_size": img_size,
        }

        torch.save(checkpoint, os.path.join(args.save_dir, "latest.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_dir, "best.pth"))
            print(f"  * New best mIoU: {best_miou:.4f}")
        print()

    print(f"Training complete. Best mIoU: {best_miou:.4f}")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'best.pth')}")


if __name__ == "__main__":
    main()
