#!/usr/bin/env python3
"""
Split IDD_val.pkl into subsets (1%, 5%, 10%) and merge with Caltech YOLO train dataset.
Outputs caltech_idd_1%_train.pkl, caltech_idd_5%_train.pkl, caltech_idd_10%_train.pkl
to merged_datasets/.
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List
import random

# Works whether run from project root or from inside caltech_yolo/
sys.path.insert(0, str(Path(__file__).parent))
from train_and_evaluate_merged import create_caltech_pkl_dataset


def load_pickle(filepath: str) -> Dict:
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Dict, filepath: str) -> None:
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def split_dataset(data: Dict, percentages: List[float], seed: int = 42) -> List[Dict]:
    """Split a dataset into multiple subsets by percentage."""
    random.seed(seed)
    np.random.seed(seed)

    total_samples = len(data['images'])
    indices = list(range(total_samples))
    random.shuffle(indices)

    subsets = []
    for percentage in percentages:
        num_samples = max(1, int(total_samples * percentage))
        subset_indices = indices[:num_samples]

        subset = {
            'images':      np.array([data['images'][i] for i in subset_indices]),
            'labels':      [data['labels'][i] for i in subset_indices],
            'image_names': [data['image_names'][i] for i in subset_indices],
        }
        subsets.append(subset)

    return subsets


def merge_datasets(caltech_train: Dict, idd_subset: Dict) -> Dict:
    """Merge Caltech train dataset with an IDD subset."""
    caltech_images = caltech_train['images']
    idd_images = idd_subset['images']

    # Ensure both are proper (N, H, W, C) numpy arrays before stacking
    if not isinstance(caltech_images, np.ndarray) or caltech_images.ndim != 4:
        caltech_images = np.array(caltech_images, dtype=np.uint8)
    if not isinstance(idd_images, np.ndarray) or idd_images.ndim != 4:
        idd_images = np.array(idd_images, dtype=np.uint8)

    return {
        'images':      np.concatenate([caltech_images, idd_images], axis=0),
        'labels':      list(caltech_train['labels']) + list(idd_subset['labels']),
        'image_names': list(caltech_train['image_names']) + list(idd_subset['image_names']),
    }


def main():
    base_path    = Path('/Users/ishachadalavada/Desktop/ml_final_project')
    idd_val_path = base_path / 'IDD' / 'idd_val.pkl'
    caltech_dir  = base_path / 'caltech_yolo'
    output_dir   = base_path / 'merged_datasets'

    output_dir.mkdir(exist_ok=True)

    percentages      = [0.01, 0.05, 0.10]
    percentage_names = ['1%', '5%', '10%']

    print("Building Caltech train dataset from YOLO format (train split)...")
    caltech_train_raw = create_caltech_pkl_dataset(
        caltech_dir, split='train', target_size=(224, 224)
    )

    caltech_images = np.array(caltech_train_raw['images'], dtype=np.uint8) \
        if caltech_train_raw['images'] else np.empty((0, 224, 224, 3), dtype=np.uint8)

    caltech_train = {
        'images':      caltech_images,
        'labels':      caltech_train_raw['labels'],
        'image_names': caltech_train_raw['image_names'],
    }
    print(f"Caltech train size: {len(caltech_train['images'])} images")

    print("\nLoading IDD_val dataset...")
    idd_val = load_pickle(str(idd_val_path))
    print(f"IDD_val size: {len(idd_val['images'])} images")

    print(f"\nSplitting IDD_val into {percentage_names} subsets...")
    idd_subsets = split_dataset(idd_val, percentages, seed=42)

    for idd_subset, name in zip(idd_subsets, percentage_names):
        print(f"\nMerging Caltech train + IDD {name} ({len(idd_subset['images'])} IDD images)...")
        merged = merge_datasets(caltech_train, idd_subset)

        output_file = output_dir / f'caltech_idd_{name}_train.pkl'
        save_pickle(merged, str(output_file))

        print(f"  Saved -> {output_file}")
        print(f"  Total: {len(merged['images'])} images "
              f"({len(caltech_train['images'])} Caltech + {len(idd_subset['images'])} IDD)")

    print("\nAll merged datasets created successfully!")
    print(f"\nOutput files in: {output_dir}")
    for name in percentage_names:
        print(f"  - caltech_idd_{name}_train.pkl")


if __name__ == "__main__":
    main()
