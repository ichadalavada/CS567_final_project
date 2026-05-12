#!/usr/bin/env python3
"""
Script to split the remaining 90% of IDD_val.pkl (not used in merge_idd_data) 
into train and validation sets.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Set
import random


def load_pickle(filepath: str) -> Dict:
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Dict, filepath: str) -> None:
    """Save data to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def get_used_indices(total_samples: int, percentage: float = 0.10, seed: int = 42) -> Set[int]:
    """
    Get the indices that are used in merge_idd_data.py for the given percentage
    
    Args:
        total_samples: Total number of samples in the dataset
        percentage: Percentage being used (default 10%)
        seed: Random seed (must match merge_idd_data.py)
    
    Returns:
        Set of indices that are used
    """
    random.seed(seed)
    np.random.seed(seed)
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    num_samples = max(1, int(total_samples * percentage))
    used_indices = set(indices[:num_samples])
    
    return used_indices


def split_remainder_dataset(data: Dict, used_indices: Set[int], 
                           train_split: float = 0.9, seed: int = 42) -> Tuple[Dict, Dict]:
    """
    Split the remaining dataset (excluding used indices) into train and val sets
    
    Args:
        data: Dictionary with 'images', 'labels', 'image_names'
        used_indices: Set of indices already used in merge_idd_data
        train_split: Proportion of remaining data to use for training (default 90%)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dict, val_dict)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    total_samples = len(data['images'])
    all_indices = set(range(total_samples))
    remaining_indices = list(all_indices - used_indices)
    
    # Shuffle remaining indices
    random.shuffle(remaining_indices)
    
    # Split into train and val
    num_train = int(len(remaining_indices) * train_split)
    train_indices = remaining_indices[:num_train]
    val_indices = remaining_indices[num_train:]
    
    # Create datasets
    train_data = {
        'images': np.array([data['images'][i] for i in train_indices]),
        'labels': [data['labels'][i] for i in train_indices],
        'image_names': [data['image_names'][i] for i in train_indices]
    }
    
    val_data = {
        'images': np.array([data['images'][i] for i in val_indices]),
        'labels': [data['labels'][i] for i in val_indices],
        'image_names': [data['image_names'][i] for i in val_indices]
    }
    
    return train_data, val_data


def main():
    # Define paths
    idd_val_path = Path('/Users/ishachadalavada/Desktop/ml_final_project/IDD/idd_val.pkl')
    output_dir = Path('/Users/ishachadalavada/Desktop/ml_final_project/merged_datasets')
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print("Loading IDD_val dataset...")
    idd_val = load_pickle(str(idd_val_path))
    total_samples = len(idd_val['images'])
    print(f"Total IDD_val dataset size: {total_samples} images")
    
    # Get indices already used in merge_idd_data (top 10%)
    print("\nIdentifying 10% of samples already used in merge_idd_data...")
    used_indices = get_used_indices(total_samples, percentage=0.10, seed=42)
    print(f"Found {len(used_indices)} images already used (10%)")
    
    # Split remaining 90% into train (90% of remaining) and val (10% of remaining)
    print("\nSplitting remaining 90% into train/val sets...")
    print("  - Train: 90% of remaining data")
    print("  - Val: 10% of remaining data")
    
    train_data, val_data = split_remainder_dataset(
        idd_val, 
        used_indices, 
        train_split=0.9, 
        seed=42
    )
    
    print(f"\nTrain set: {len(train_data['images'])} images")
    print(f"Val set: {len(val_data['images'])} images")
    print(f"Total remaining: {len(train_data['images']) + len(val_data['images'])} images")
    
    # Save datasets
    print("\nSaving datasets...")
    train_file = output_dir / 'idd_remainder_train.pkl'
    val_file = output_dir / 'idd_remainder_val.pkl'
    
    save_pickle(train_data, str(train_file))
    save_pickle(val_data, str(val_file))
    
    print(f"  ✓ Saved train set to {train_file}")
    print(f"  ✓ Saved val set to {val_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original IDD_val size: {total_samples}")
    print(f"\nUsed in merge_idd_data (10%): {len(used_indices)}")
    print(f"\nRemaining 90% split:")
    print(f"  - Train set: {len(train_data['images'])} images (81% of original)")
    print(f"  - Val set: {len(val_data['images'])} images (9% of original)")
    print("="*60)


if __name__ == "__main__":
    main()
