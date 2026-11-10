"""
IDD (Indian Driving Dataset) utilities:
1. Convert polygon JSON annotations → PNG label images
2. Dataset class with remapping to 4 ADAS classes

IDD Label → ADAS mapping:
  0: Road/Traversable  → road, drivable fallback, sidewalk, curb
  1: Obstacle           → person, rider, car, truck, bus, autorickshaw, motorcycle, 
                          bicycle, animal, vehicle fallback, caravan
  2: Structure          → building, wall, fence, pole, polegroup, traffic sign, 
                          traffic light, billboard, obs-str-bar-fallback, bridge, guard rail
  3: Background         → vegetation, sky, fallback background, non-drivable fallback, out of roi
  255: Ignore           → unlabeled

Usage:
    # Step 1: Convert JSON → PNG (run once)
    python idd_dataset.py --convert --idd-dir ~/emb/idd/idd20kII

    # Step 2: Use in training (imported by train_idd.py)
"""

import os
import json
import glob
import argparse
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ──────────────────────────────────────────────────────────
#  IDD Label → ADAS Class Mapping
# ──────────────────────────────────────────────────────────

NUM_CLASSES = 4
IGNORE_INDEX = 255

CLASS_NAMES = ["Road", "Obstacle", "Structure", "Background"]

CLASS_COLORS = [
    (128, 64, 128),   # Road - purple
    (220, 20, 60),    # Obstacle - red
    (70, 70, 70),     # Structure - dark gray
    (107, 142, 35),   # Background - green
]

IDD_LABEL_TO_ADAS = {
    # Road / Traversable (0)
    "road": 0,
    "drivable fallback": 0,
    "sidewalk": 0,
    "curb": 0,

    # Obstacle (1)
    "person": 1,
    "rider": 1,
    "car": 1,
    "truck": 1,
    "bus": 1,
    "autorickshaw": 1,
    "motorcycle": 1,
    "bicycle": 1,
    "animal": 1,
    "vehicle fallback": 1,
    "caravan": 1,

    # Structure (2)
    "building": 2,
    "wall": 2,
    "fence": 2,
    "pole": 2,
    "polegroup": 2,
    "traffic sign": 2,
    "traffic light": 2,
    "billboard": 2,
    "obs-str-bar-fallback": 2,
    "bridge": 2,
    "guard rail": 2,

    # Background (3)
    "vegetation": 3,
    "sky": 3,
    "fallback background": 3,
    "non-drivable fallback": 3,
    "out of roi": 3,
}


# ──────────────────────────────────────────────────────────
#  JSON → PNG Conversion
# ──────────────────────────────────────────────────────────

def convert_json_to_png(json_path, output_path):
    """
    Convert a single IDD polygon JSON annotation to a PNG label image.
    Pixels are filled with ADAS class IDs (0-3), default IGNORE_INDEX (255).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    h = data['imgHeight']
    w = data['imgWidth']

    # Start with ignore
    label_img = Image.new('L', (w, h), IGNORE_INDEX)
    draw = ImageDraw.Draw(label_img)

    # Draw polygons in order (later objects occlude earlier ones, matching IDD convention)
    for obj in data.get('objects', []):
        label_name = obj.get('label', '')
        adas_class = IDD_LABEL_TO_ADAS.get(label_name, IGNORE_INDEX)

        polygon = obj.get('polygon', [])
        if len(polygon) < 3:
            continue

        # Convert [[x,y], [x,y], ...] to [(x,y), (x,y), ...]
        poly_tuples = [(int(p[0]), int(p[1])) for p in polygon]
        draw.polygon(poly_tuples, fill=adas_class)

    label_img.save(output_path)


def convert_all_annotations(idd_dir):
    """
    Convert all JSON annotations in an IDD directory to PNG label images.
    Saves PNG files alongside the JSON files with suffix _labelADAS.png
    """
    json_files = glob.glob(os.path.join(idd_dir, 'gtFine', '**', '*.json'), recursive=True)
    print(f"Found {len(json_files)} JSON annotation files")

    for json_path in tqdm(json_files, desc="Converting"):
        output_path = json_path.replace('_gtFine_polygons.json', '_gtFine_labelADAS.png')
        convert_json_to_png(json_path, output_path)

    # Verify
    png_files = glob.glob(os.path.join(idd_dir, 'gtFine', '**', '*_labelADAS.png'), recursive=True)
    print(f"Created {len(png_files)} PNG label files")


# ──────────────────────────────────────────────────────────
#  Dataset Class
# ──────────────────────────────────────────────────────────

class IDDADAS(Dataset):
    """
    IDD dataset with ADAS 4-class labels.
    
    Requires running convert_all_annotations() first to generate PNG labels.
    
    Args:
        idd_dir:    Path to IDD root (e.g., ~/emb/idd/idd20kII)
        split:      'train' or 'val'
        img_size:   (H, W) tuple
        augment:    Apply training augmentations
    """
    def __init__(self, idd_dir, split="train", img_size=(256, 320), augment=False):
        super().__init__()
        self.idd_dir = idd_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.images = []
        self.labels = []

        img_dir = os.path.join(idd_dir, "leftImg8bit", split)
        label_dir = os.path.join(idd_dir, "gtFine", split)

        if not os.path.isdir(img_dir):
            raise RuntimeError(f"Image directory not found: {img_dir}")

        # IDD structure: leftImg8bit/split/sequence_id/frameXXXXX_leftImg8bit.jpg (or .png)
        for seq_dir in sorted(os.listdir(img_dir)):
            seq_img_dir = os.path.join(img_dir, seq_dir)
            seq_label_dir = os.path.join(label_dir, seq_dir)

            if not os.path.isdir(seq_img_dir):
                continue

            for fname in sorted(os.listdir(seq_img_dir)):
                # Handle both .jpg and .png images
                if not (fname.endswith('.jpg') or fname.endswith('.png')):
                    continue

                img_path = os.path.join(seq_img_dir, fname)

                # Construct label path
                base_name = fname.replace('_leftImg8bit.jpg', '').replace('_leftImg8bit.png', '')
                label_fname = base_name + '_gtFine_labelADAS.png'
                label_path = os.path.join(seq_label_dir, label_fname)

                if os.path.isfile(label_path):
                    self.images.append(img_path)
                    self.labels.append(label_path)

        print(f"[IDDADAS] {split}: Found {len(self.images)} image-label pairs")

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = Image.open(self.labels[idx])

        if self.augment:
            if torch.rand(1).item() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        label = label.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        label = np.array(label, dtype=np.uint8)
        img = transforms.ToTensor()(img)
        img = self.normalize(img)
        label = torch.from_numpy(label).long()

        return img, label


# ──────────────────────────────────────────────────────────
#  Combined Dataset (Cityscapes + IDD)
# ──────────────────────────────────────────────────────────

class CombinedADAS(Dataset):
    """Combine Cityscapes and IDD datasets for training."""
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative = np.cumsum(self.lengths)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        for i, cum_len in enumerate(self.cumulative):
            if idx < cum_len:
                local_idx = idx - (self.cumulative[i - 1] if i > 0 else 0)
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {idx} out of range")


# ──────────────────────────────────────────────────────────
#  Main: Convert annotations
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert", action="store_true", help="Convert JSON → PNG labels")
    parser.add_argument("--idd-dir", type=str, required=True, help="Path to IDD root (e.g., idd20kII)")
    parser.add_argument("--idd-dir2", type=str, default=None, help="Path to second IDD dir (e.g., IDD_Segmentation)")
    args = parser.parse_args()

    if args.convert:
        print(f"Converting annotations in: {args.idd_dir}")
        convert_all_annotations(args.idd_dir)

        if args.idd_dir2:
            print(f"\nConverting annotations in: {args.idd_dir2}")
            convert_all_annotations(args.idd_dir2)

        print("\nDone! You can now use IDDADAS dataset class for training.")
    else:
        # Quick test
        dataset = IDDADAS(args.idd_dir, split="val", img_size=(256, 320))
        if len(dataset) > 0:
            img, label = dataset[0]
            print(f"Image shape: {img.shape}")
            print(f"Label shape: {label.shape}")
            print(f"Unique labels: {torch.unique(label).tolist()}")