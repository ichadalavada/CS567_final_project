#!/usr/bin/env python3
"""
Script to split IDD_val.pkl into subsets (1%, 5%, 10%) and merge with Scooter train dataset
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random


def load_pickle(filepath: str) -> Dict:
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Dict, filepath: str) -> None:
    """Save data to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def split_dataset(data: Dict, percentages: List[float], seed: int = 42) -> List[Dict]:
    """
    Split a dataset into multiple subsets by percentage
    
    Args:
        data: Dictionary with 'images', 'labels', 'image_names'
        percentages: List of percentages to split (e.g., [0.01, 0.05, 0.10])
        seed: Random seed for reproducibility
    
    Returns:
        List of dictionaries, one for each percentage
    """
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
            'images': np.array([data['images'][i] for i in subset_indices]),
            'labels': [data['labels'][i] for i in subset_indices],
            'image_names': [data['image_names'][i] for i in subset_indices]
        }
        subsets.append(subset)
    
    return subsets


def merge_datasets(scooter_train: Dict, idd_subset: Dict) -> Dict:
    """
    Merge Scooter train dataset with IDD subset
    
    Args:
        scooter_train: Scooter train dataset
        idd_subset: IDD subset dataset
    
    Returns:
        Merged dataset
    """
    merged = {
        'images': np.vstack([scooter_train['images'], idd_subset['images']]),
        'labels': scooter_train['labels'] + idd_subset['labels'],
        'image_names': scooter_train['image_names'] + idd_subset['image_names']
    }
    return merged


def main():
    # Define paths
    idd_val_path = Path('/Users/ishachadalavada/Desktop/ml_final_project/IDD/idd_val.pkl')
    scooter_train_path = Path('/Users/ishachadalavada/Desktop/ml_final_project/scooter_data/pkl_224/train.pkl')
    output_dir = Path('/Users/ishachadalavada/Desktop/ml_final_project/merged_datasets')
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Define percentages to split
    percentages = [.20, .30, .50]  # 20%, 30%, 50% of IDD_val
    percentage_names = ['20%', '30%', '50%']
    
    print("Loading datasets...")
    idd_val = load_pickle(str(idd_val_path))
    scooter_train = load_pickle(str(scooter_train_path))
    
    print(f"IDD_val dataset size: {len(idd_val['images'])} images")
    print(f"Scooter train dataset size: {len(scooter_train['images'])} images")
    
    # Split IDD_val into subsets
    print(f"\nSplitting IDD_val into {percentages} subsets...")
    idd_subsets = split_dataset(idd_val, percentages, seed=42)
    
    # Merge each subset with Scooter train and save
    for i, (percentage, name) in enumerate(zip(percentages, percentage_names)):
        print(f"\nProcessing {name} subset ({len(idd_subsets[i]['images'])} images from IDD)...")
        
        # Merge datasets
        merged = merge_datasets(scooter_train, idd_subsets[i])
        
        # Create output path
        output_file = output_dir / f'scooter_idd_{name}_train.pkl'
        
        # Save merged dataset
        save_pickle(merged, str(output_file))
        
        print(f"  ✓ Saved to {output_file}")
        print(f"    Total merged size: {len(merged['images'])} images")
        print(f"    ({len(scooter_train['images'])} Scooter + {len(idd_subsets[i]['images'])} IDD)")
    
    # Also save the individual IDD subsets if needed
    print(f"\nSaving individual IDD subsets...")
    for i, (percentage, name) in enumerate(zip(percentages, percentage_names)):
        subset_file = output_dir / f'idd_{name}.pkl'
        save_pickle(idd_subsets[i], str(subset_file))
        print(f"  ✓ Saved IDD {name} subset to {subset_file}")
    
    print("\n✓ All datasets created successfully!")
    print(f"\nOutput files in: {output_dir}")
    print("  - scooter_idd_1%_train.pkl")
    print("  - scooter_idd_5%_train.pkl")
    print("  - scooter_idd_10%_train.pkl")
    print("  - idd_1%.pkl")
    print("  - idd_5%.pkl")
    print("  - idd_10%.pkl")


if __name__ == "__main__":
    main()
