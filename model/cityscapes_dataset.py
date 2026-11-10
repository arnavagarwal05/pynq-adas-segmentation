"""
Cityscapes Dataset with class remapping for ADAS.

Remaps 19 Cityscapes trainId classes → 4 ADAS super-classes:
  0: Road/Traversable  (road, sidewalk)
  1: Obstacle           (person, rider, car, truck, bus, train, motorcycle, bicycle)
  2: Structure          (building, wall, fence, pole, traffic light, traffic sign)
  3: Background         (vegetation, terrain, sky)
  255: Ignore           (unlabeled / void)

Cityscapes trainId mapping (19 classes):
  0  = road          → 0 (Road)
  1  = sidewalk      → 0 (Road)
  2  = building      → 2 (Structure)
  3  = wall          → 2 (Structure)
  4  = fence         → 2 (Structure)
  5  = pole          → 2 (Structure)
  6  = traffic light → 2 (Structure)
  7  = traffic sign  → 2 (Structure)
  8  = vegetation    → 3 (Background)
  9  = terrain       → 3 (Background)
  10 = sky           → 3 (Background)
  11 = person        → 1 (Obstacle)
  12 = rider         → 1 (Obstacle)
  13 = car           → 1 (Obstacle)
  14 = truck         → 1 (Obstacle)
  15 = bus           → 1 (Obstacle)
  16 = train         → 1 (Obstacle)
  17 = motorcycle    → 1 (Obstacle)
  18 = bicycle       → 1 (Obstacle)
  255 = ignore       → 255 (Ignore)
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ──────────────────────────────────────────────────────────
#  Class Remapping Config
# ──────────────────────────────────────────────────────────

NUM_CLASSES = 4
IGNORE_INDEX = 255

CLASS_NAMES = ["Road", "Obstacle", "Structure", "Background"]

# Colors for visualization (RGB)
CLASS_COLORS = [
    (128, 64, 128),   # Road - purple
    (220, 20, 60),    # Obstacle - red
    (70, 70, 70),     # Structure - dark gray
    (107, 142, 35),   # Background - green
]

# Cityscapes trainId → ADAS class mapping
# Index = cityscapes trainId, Value = ADAS class id
TRAINID_TO_ADAS = np.array([
    0,    # 0  road         → Road
    0,    # 1  sidewalk     → Road
    2,    # 2  building     → Structure
    2,    # 3  wall         → Structure
    2,    # 4  fence        → Structure
    2,    # 5  pole         → Structure
    2,    # 6  traffic light→ Structure
    2,    # 7  traffic sign → Structure
    3,    # 8  vegetation   → Background
    3,    # 9  terrain      → Background
    3,    # 10 sky          → Background
    1,    # 11 person       → Obstacle
    1,    # 12 rider        → Obstacle
    1,    # 13 car          → Obstacle
    1,    # 14 truck        → Obstacle
    1,    # 15 bus          → Obstacle
    1,    # 16 train        → Obstacle
    1,    # 17 motorcycle   → Obstacle
    1,    # 18 bicycle      → Obstacle
], dtype=np.uint8)


def remap_cityscapes_label(label_img):
    """
    Remap a Cityscapes label image (trainId encoding) to ADAS 4-class encoding.
    
    Args:
        label_img: numpy array (H, W) with Cityscapes trainId values (0-18, 255)
    
    Returns:
        numpy array (H, W) with ADAS class values (0-3, 255)
    """
    remapped = np.full_like(label_img, IGNORE_INDEX, dtype=np.uint8)
    for train_id in range(19):
        remapped[label_img == train_id] = TRAINID_TO_ADAS[train_id]
    return remapped


# ──────────────────────────────────────────────────────────
#  Dataset Class
# ──────────────────────────────────────────────────────────

class CityscapesADAS(Dataset):
    """
    Cityscapes dataset with automatic remapping to 4 ADAS classes.
    
    Expected directory structure:
        root/
        ├── leftImg8bit/
        │   ├── train/
        │   │   ├── aachen/
        │   │   │   ├── aachen_000000_000019_leftImg8bit.png
        │   │   │   └── ...
        │   │   └── ...
        │   └── val/
        └── gtFine/
            ├── train/
            │   ├── aachen/
            │   │   ├── aachen_000000_000019_gtFine_labelTrainIds.png
            │   │   └── ...
            │   └── ...
            └── val/
    
    IMPORTANT: You need to run Cityscapes' `createTrainIdLabelImgs.py` first 
    to generate the `_labelTrainIds.png` files from the polygon annotations.
    
    Args:
        root:       Path to cityscapes root directory
        split:      'train' or 'val'
        img_size:   (H, W) tuple for resizing. Default (512, 1024)
        augment:    Apply training augmentations
    """
    def __init__(self, root, split="train", img_size=(512, 1024), augment=False):
        super().__init__()
        self.root = root
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.images = []
        self.labels = []

        img_dir = os.path.join(root, "leftImg8bit", split)
        label_dir = os.path.join(root, "gtFine", split)

        if not os.path.isdir(img_dir):
            raise RuntimeError(f"Image directory not found: {img_dir}")
        if not os.path.isdir(label_dir):
            raise RuntimeError(f"Label directory not found: {label_dir}")

        for city in sorted(os.listdir(img_dir)):
            city_img_dir = os.path.join(img_dir, city)
            city_label_dir = os.path.join(label_dir, city)
            if not os.path.isdir(city_img_dir):
                continue

            for fname in sorted(os.listdir(city_img_dir)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue
                img_path = os.path.join(city_img_dir, fname)
                # Construct label path
                label_fname = fname.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
                label_path = os.path.join(city_label_dir, label_fname)

                if os.path.isfile(label_path):
                    self.images.append(img_path)
                    self.labels.append(label_path)

        print(f"[CityscapesADAS] {split}: Found {len(self.images)} images")

        # Normalization (ImageNet stats)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and label
        img = Image.open(self.images[idx]).convert("RGB")
        label = Image.open(self.labels[idx])  # Single channel, trainId encoding

        # Augmentations (applied to both image and label)
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

            # Random crop (take a random 512x1024 crop from the 1024x2048 image)
            if img.size[0] > self.img_size[1] and img.size[1] > self.img_size[0]:
                i = torch.randint(0, img.size[1] - self.img_size[0] + 1, (1,)).item()
                j = torch.randint(0, img.size[0] - self.img_size[1] + 1, (1,)).item()
                img = img.crop((j, i, j + self.img_size[1], i + self.img_size[0]))
                label = label.crop((j, i, j + self.img_size[1], i + self.img_size[0]))
            else:
                img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
                label = label.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        else:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            label = label.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        # Convert label to numpy and remap
        label = np.array(label, dtype=np.uint8)
        label = remap_cityscapes_label(label)

        # Convert image to tensor and normalize
        img = transforms.ToTensor()(img)  # [3, H, W], float32 [0, 1]
        img = self.normalize(img)

        label = torch.from_numpy(label).long()  # [H, W]

        return img, label


# ──────────────────────────────────────────────────────────
#  Utility: Compute class weights (for imbalanced classes)
# ──────────────────────────────────────────────────────────

def compute_class_weights(dataset, num_classes=NUM_CLASSES):
    """
    Compute median frequency balancing weights.
    Useful for handling class imbalance (road pixels >> obstacle pixels).
    
    Run this once on the training set, then pass weights to CrossEntropyLoss.
    """
    print("Computing class weights (this may take a while)...")
    class_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_np = label.numpy()
        for c in range(num_classes):
            class_counts[c] += np.sum(label_np == c)
        total_pixels += np.sum(label_np != IGNORE_INDEX)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} images")

    freq = class_counts / total_pixels
    median_freq = np.median(freq)
    weights = median_freq / freq

    print(f"\nClass distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}: {freq[i]*100:6.2f}%  weight={weights[i]:.4f}")

    return torch.FloatTensor(weights)


# ──────────────────────────────────────────────────────────
#  Utility: Colorize prediction for visualization
# ──────────────────────────────────────────────────────────

def colorize_prediction(pred):
    """
    Convert a class prediction map to an RGB image for visualization.
    
    Args:
        pred: numpy array (H, W) with class ids (0-3)
    
    Returns:
        numpy array (H, W, 3) RGB image
    """
    h, w = pred.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in enumerate(CLASS_COLORS):
        color_img[pred == c] = color
    return color_img


# ──────────────────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py /path/to/cityscapes")
        print("\nThis script will:")
        print("  1. Load the dataset with class remapping")
        print("  2. Print class distribution stats")
        print("  3. Save a sample visualization")
        sys.exit(0)

    root = sys.argv[1]

    # Test dataset loading
    dataset = CityscapesADAS(root, split="train", augment=True)
    print(f"\nDataset size: {len(dataset)}")

    # Load one sample
    img, label = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Label unique values: {torch.unique(label).tolist()}")

    # Show class distribution for this sample
    for c, name in enumerate(CLASS_NAMES):
        count = (label == c).sum().item()
        total = (label != IGNORE_INDEX).sum().item()
        print(f"  {name}: {count/total*100:.1f}%")

    # Optionally compute full class weights
    weights = compute_class_weights(dataset)
    print(f"\nClass weights: {weights}")
