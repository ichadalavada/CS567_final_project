#!/usr/bin/env python3
"""
Scooter Dataset Preprocessing to Pickle Format
Converts images and labels to pickle files for efficient training
Supports automatic resizing for MobileNet deployment

File structure:
  scooter_data/
    data/
      RGB/                        <- images (named like 191209_15153700000540.png)
        <image_name>.png
      yolo_format/                <- split files + per-image annotation txts
        train.txt                 <- list of image stems for training
        val.txt
        test.txt
        <image_name>.txt          <- YOLO annotations for that image
"""

import os
import cv2
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


SCOOTER_CLASS_MAPPING = {
    0: 3,  # Bump -> misc
    1: 3,  # Column -> misc
    2: 3,  # Dent -> misc
    3: 3,  # Fence -> misc
    4: 1,  # People -> pedestrian
    5: 0,  # Vehicle -> vehicle
    6: 3,  # Wall -> misc
    7: 3,  # Weed -> misc
    8: 3,  # Zebra -> misc
    9: 3,  # Crossing -> misc
    10: 4, # Traffic Cone -> traffic
    11: 4, # Traffic Sign -> traffic
}

NEW_CLASSES = {
    0: "vehicle",
    1: "pedestrian",
    2: "cyclist",
    3: "misc",
    4: "traffic"
}


def read_split_file(split_file):
    """Read a train/val/test split file — returns list of image stems."""
    stems = []
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Handle full paths or bare stems
                stem = Path(line).stem
                stems.append(stem)
    return stems


def read_labels(label_file, mapping=None):
    """Read YOLO format labels and remap class IDs."""
    if mapping is None:
        mapping = SCOOTER_CLASS_MAPPING

    annotations = []
    try:
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    new_class_id = mapping.get(class_id, 3)  # default misc
                    annotations.append([new_class_id] + coords)
    except Exception as e:
        print(f"  Warning reading {label_file}: {e}")
    return annotations


def load_image(image_path, target_size=None):
    """Load and optionally resize image (returns RGB numpy array)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    return img


def process_split(split_name, stems, images_dir, labels_dir, target_size, stats):
    """Load images + labels for one split, return data dict."""
    data = {'images': [], 'labels': [], 'image_names': []}
    missing_images = 0
    missing_labels = 0

    for stem in tqdm(stems, desc=f"  Loading {split_name}"):
        # Try common image extensions
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            missing_images += 1
            continue

        img = load_image(img_path, target_size)
        if img is None:
            missing_images += 1
            continue

        label_file = labels_dir / (stem + '.txt')
        annotations = []
        if label_file.exists():
            annotations = read_labels(label_file)
        else:
            missing_labels += 1

        data['images'].append(img)
        data['labels'].append(annotations)
        data['image_names'].append(stem)

        stats[split_name]['images'] += 1
        stats[split_name]['annotations'] += len(annotations)

    if missing_images:
        print(f"  ⚠ {missing_images} images not found for {split_name}")
    if missing_labels:
        print(f"  ⚠ {missing_labels} images had no label file (stored as empty) for {split_name}")

    return data


def preprocess_dataset(base_path=".", target_size=None):
    base_path = Path(base_path)

    images_dir  = base_path / "data" / "RGB"
    labels_dir  = base_path / "data" / "yolo_format"   # annotation txts live here
    splits_dir  = base_path / "data"   # train/val/test.txt live here

    # Validate
    for d in [images_dir, labels_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Expected directory not found: {d}")

    pkl_suffix = f"_{target_size[0]}" if target_size else ""
    output_dir = base_path / f"pkl{pkl_suffix}"
    output_dir.mkdir(exist_ok=True)

    stats = {
        'train': {'images': 0, 'annotations': 0},
        'val':   {'images': 0, 'annotations': 0},
        'test':  {'images': 0, 'annotations': 0},
        'target_size': target_size,
    }

    for split in ['train', 'val', 'test']:
        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            print(f"No {split}.txt found — skipping {split} split.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} set...")
        print(f"{'='*60}")

        stems = read_split_file(split_file)
        print(f"  Found {len(stems)} entries in {split}.txt")

        data = process_split(split, stems, images_dir, labels_dir, target_size, stats)

        pkl_path = output_dir / f"{split}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✓ Saved → {pkl_path}")

    # Metadata
    metadata = {
        'classes': NEW_CLASSES,
        'num_classes': len(NEW_CLASSES),
        'target_size': target_size,
        'train_images': stats['train']['images'],
        'train_annotations': stats['train']['annotations'],
        'val_images': stats['val']['images'],
        'val_annotations': stats['val']['annotations'],
        'test_images': stats['test']['images'],
        'test_annotations': stats['test']['annotations'],
        'format': 'YOLO normalized coordinates',
        'class_mapping': SCOOTER_CLASS_MAPPING,
    }
    meta_path = output_dir / "metadata.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"\n  ✓ Saved metadata → {meta_path}")

    return stats, output_dir


def print_statistics(stats):
    print(f"\n{'='*60}")
    print("PREPROCESSING STATISTICS")
    print(f"{'='*60}")
    total_imgs = total_anns = 0
    for split in ['train', 'val', 'test']:
        imgs = stats[split]['images']
        anns = stats[split]['annotations']
        if imgs == 0:
            continue
        print(f"\n{split.capitalize()} Set:")
        print(f"  Images:      {imgs}")
        print(f"  Annotations: {anns}")
        print(f"  Avg obj/img: {anns/imgs:.1f}")
        total_imgs += imgs
        total_anns += anns
    if total_imgs:
        print(f"\nTotal: {total_imgs} images, {total_anns} annotations "
              f"({total_anns/total_imgs:.1f} avg)")
    if stats['target_size']:
        print(f"Target size: {stats['target_size'][0]}×{stats['target_size'][1]}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess scooter dataset to pickle format")
    parser.add_argument("--path", default="~/Desktop/ml_final_project/scooter_data",
                        help="Path to scooter_data root")
    parser.add_argument("--size", type=int, choices=[224, 192, 160, 128],
                        help="Target square image size (omit to keep originals)")
    args = parser.parse_args()

    base_path = Path(args.path).expanduser()
    target_size = (args.size, args.size) if args.size else None

    print(f"\n{'='*60}")
    print("Scooter Dataset → Pickle Preprocessor")
    print(f"{'='*60}")
    print(f"Root:        {base_path}")
    print(f"Target size: {target_size or 'original'}")

    stats, output_dir = preprocess_dataset(base_path, target_size)
    print_statistics(stats)

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}/")
    print("  ├── train.pkl")
    print("  ├── val.pkl")
    print("  ├── test.pkl")
    print("  └── metadata.pkl")


if __name__ == "__main__":
    main()