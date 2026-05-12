#!/usr/bin/env python3
"""
KITTI Dataset Preprocessing to Pickle Format
Converts images and labels to pickle files for efficient training
Supports automatic resizing for MobileNet deployment
"""

import os
import cv2
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse


# Class mapping (8 classes -> 4 classes)
KITTI_CLASS_MAPPING = {
    0: 0,  # car -> vehicle
    1: 0,  # van -> vehicle
    2: 0,  # truck -> vehicle
    3: 1,  # pedestrian -> pedestrian
    4: 1,  # Person_sitting -> pedestrian
    5: 2,  # cyclist -> cyclist
    6: 0,  # tram -> vehicle
    7: 3   # misc -> misc
}

NEW_CLASSES = {
    0: "vehicle",
    1: "pedestrian",
    2: "cyclist",
    3: "misc",
    4: "traffic"
}


def read_labels(label_file):
    """Read YOLO format labels from file
    Returns list of [class_id, x_center, y_center, width, height]
    """
    annotations = []
    try:
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    annotations.append([class_id] + coords)
    except Exception as e:
        print(f"Error reading {label_file}: {e}")
        return []
    
    return annotations


def remap_labels(annotations, mapping=None):
    """Remap class IDs using the mapping dictionary"""
    if mapping is None:
        mapping = KITTI_CLASS_MAPPING
    
    remapped = []
    for ann in annotations:
        class_id = int(ann[0])
        new_class_id = mapping[class_id]
        remapped.append([new_class_id] + ann[1:])
    
    return remapped


def load_image(image_path, target_size=None):
    """Load and optionally resize image"""
    img = cv2.imread(str(image_path))
    
    if img is None:
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if specified
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    return img


def preprocess_dataset(base_path=".", target_size=None, output_suffix=""):
    """
    Preprocess KITTI dataset and save to pickle files
    
    Args:
        base_path: Path to dataset root
        target_size: Tuple (width, height) for resizing, None to keep original
        output_suffix: Suffix for output directory (e.g., "_224" for "pkl_224")
    
    Returns:
        dict with statistics
    """
    
    base_path = Path(base_path)
    
    # Define paths
    train_images_dir = base_path / "kitti" /  "images" / "train"
    train_labels_dir = base_path / "kitti" / "kitti_data" / "labels_4class" / "train"
    val_images_dir   = base_path / "kitti" / "images" / "val"
    val_labels_dir = base_path / "kitti" / "kitti_data" / "labels_4class" / "val"

    # Create output directory
    pkl_suffix = f"_{target_size[0]}" if target_size else ""
    output_dir = base_path / f"pkl{pkl_suffix}"
    output_dir.mkdir(exist_ok=True)
    
    stats = {
        'train': {'images': 0, 'annotations': 0},
        'val': {'images': 0, 'annotations': 0},
        'target_size': target_size,
    }
    
    # Process train set
    print("\n" + "="*70)
    print(f"Processing TRAIN set...")
    print("="*70)
    
    train_data = {'images': [], 'labels': [], 'image_names': []}
    
    train_image_files = sorted(train_images_dir.glob("*.png"))
    print(f"Found {len(train_image_files)} training images")
    
    for img_path in tqdm(train_image_files, desc="Loading train images"):
        # Load image
        img = load_image(img_path, target_size)
        if img is None:
            continue
        
        # Load labels
        label_file = train_labels_dir / img_path.stem / ".txt"
        if not label_file.exists():
            label_file = train_labels_dir / (img_path.stem + ".txt")
        
        annotations = []
        if label_file.exists():
            annotations = read_labels(label_file)
        
        train_data['images'].append(img)
        train_data['labels'].append(annotations)
        train_data['image_names'].append(img_path.stem)
        
        stats['train']['images'] += 1
        stats['train']['annotations'] += len(annotations)
    
    # Save train pickle
    train_pkl_path = output_dir / "train.pkl"
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved train data: {train_pkl_path}")
    
    # Process val set
    print("\n" + "="*70)
    print(f"Processing VALIDATION set...")
    print("="*70)
    
    val_data = {'images': [], 'labels': [], 'image_names': []}
    
    val_image_files = sorted(val_images_dir.glob("*.png"))
    print(f"Found {len(val_image_files)} validation images")
    
    for img_path in tqdm(val_image_files, desc="Loading val images"):
        # Load image
        img = load_image(img_path, target_size)
        if img is None:
            continue
        
        # Load labels
        label_file = val_labels_dir / img_path.stem / ".txt"
        if not label_file.exists():
            label_file = val_labels_dir / (img_path.stem + ".txt")
        
        annotations = []
        if label_file.exists():
            annotations = read_labels(label_file)
        
        val_data['images'].append(img)
        val_data['labels'].append(annotations)
        val_data['image_names'].append(img_path.stem)
        
        stats['val']['images'] += 1
        stats['val']['annotations'] += len(annotations)
    
    # Save val pickle
    val_pkl_path = output_dir / "val.pkl"
    with open(val_pkl_path, 'wb') as f:
        pickle.dump(val_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved val data: {val_pkl_path}")
    
    # Save metadata
    metadata = {
        'classes': NEW_CLASSES,
        'num_classes': 4,
        'target_size': target_size,
        'original_size': (375, 1242),
        'train_images': stats['train']['images'],
        'train_annotations': stats['train']['annotations'],
        'val_images': stats['val']['images'],
        'val_annotations': stats['val']['annotations'],
        'format': 'YOLO normalized coordinates',
    }
    
    metadata_path = output_dir / "metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata: {metadata_path}")
    
    return stats, output_dir


def print_statistics(stats):
    """Print preprocessing statistics"""
    
    print("\n" + "="*70)
    print("PREPROCESSING STATISTICS")
    print("="*70)
    
    print(f"\nTrain Set:")
    print(f"  Images: {stats['train']['images']}")
    print(f"  Annotations: {stats['train']['annotations']}")
    print(f"  Avg objects/image: {stats['train']['annotations']/stats['train']['images']:.1f}")
    
    print(f"\nValidation Set:")
    print(f"  Images: {stats['val']['images']}")
    print(f"  Annotations: {stats['val']['annotations']}")
    print(f"  Avg objects/image: {stats['val']['annotations']/stats['val']['images']:.1f}")
    
    total_images = stats['train']['images'] + stats['val']['images']
    total_annotations = stats['train']['annotations'] + stats['val']['annotations']
    
    print(f"\nTotal:")
    print(f"  Images: {total_images}")
    print(f"  Annotations: {total_annotations}")
    print(f"  Avg objects/image: {total_annotations/total_images:.1f}")
    
    if stats['target_size']:
        print(f"\nImage Size:")
        print(f"  Target: {stats['target_size'][0]}x{stats['target_size'][1]}")
    else:
        print(f"\nImage Size:")
        print(f"  Original: 375x1242 (variable)")


def create_loader_example(output_dir, target_size=None):
    """Create example data loader script for training"""
    
    if target_size:
        size_str = f"{target_size[0]}x{target_size[1]}"
    else:
        size_str = "original"
    
    loader_code = f'''#!/usr/bin/env python3
"""
Example data loader for KITTI pickle dataset
Demonstrates how to load and use the preprocessed pickle files
"""

import pickle
import numpy as np
from pathlib import Path

class KITTIPickleDataset:
    """Load KITTI data from pickle files"""
    
    def __init__(self, pkl_dir="pkl{'' if target_size is None else f'_{target_size[0]}'}"):
        pkl_dir = Path(pkl_dir)
        
        # Load metadata
        with open(pkl_dir / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.classes = self.metadata['classes']
        self.num_classes = self.metadata['num_classes']
        self.target_size = self.metadata['target_size']
        
        # Load train split
        with open(pkl_dir / "train.pkl", 'rb') as f:
            train_data = pickle.load(f)
        
        self.train_images = train_data['images']
        self.train_labels = train_data['labels']
        self.train_names = train_data['image_names']
        
        # Load val split
        with open(pkl_dir / "val.pkl", 'rb') as f:
            val_data = pickle.load(f)
        
        self.val_images = val_data['images']
        self.val_labels = val_data['labels']
        self.val_names = val_data['image_names']
    
    def get_train_batch(self, indices):
        """Get a batch of training data"""
        images = np.array([self.train_images[i] for i in indices])
        labels = [self.train_labels[i] for i in indices]
        return images, labels
    
    def get_val_batch(self, indices):
        """Get a batch of validation data"""
        images = np.array([self.val_images[i] for i in indices])
        labels = [self.val_labels[i] for i in indices]
        return images, labels
    
    def __repr__(self):
        info = f"""
KITTI Pickle Dataset
  Train images: {{len(self.train_images)}}
  Val images: {{len(self.val_images)}}
  Image size: {{self.target_size if self.target_size else 'original (375x1242)'}}
  Format: YOLO normalized coordinates
"""
        return info

# Example usage
if __name__ == "__main__":
    # Load dataset
    dataset = KITTIPickleDataset()
    print(dataset)
    
    # Get a batch
    batch_indices = list(range(5))
    images, labels = dataset.get_train_batch(batch_indices)
    
    print(f"Batch shape: {{images.shape}}")
    print(f"Batch labels: {{len(labels)}} samples")
    print(f"First sample annotations: {{labels[0]}}")
'''
    
    loader_path = Path(output_dir) / "data_loader.py"
    with open(loader_path, 'w') as f:
        f.write(loader_code)
    
    return loader_path


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess KITTI dataset to pickle format"
    )
    parser.add_argument(
        "--path",
        default="/Users/ishachadalavada/Desktop/ml_final_project",
        help="Path to KITTI dataset"
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[224, 192, 160, 128],
        help="Target image size for resizing (None to keep original)"
    )
    
    args = parser.parse_args()
    base_path = Path(args.path)
    
    # Check if preprocessed labels exist
    if not (base_path / "kitti" / "kitti_data" / "labels_4class").exists():
        print("Error: labels_4class directory not found!")
        print("Please run preprocess_kitti.py first!")
        return 1
    
    print("\n" + "="*70)
    print("KITTI to Pickle Preprocessor")
    print("="*70)
    
    target_size = (args.size, args.size) if args.size else None
    
    # Run preprocessing
    stats, output_dir = preprocess_dataset(base_path, target_size)
    
    # Print statistics
    print_statistics(stats)
    
    # Create example loader
    loader_path = create_loader_example(output_dir, target_size)
    print(f"\n✓ Created example loader: {loader_path.name}")
    
    print("\n" + "="*70)
    print("PICKLE PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"""
Output directory: {output_dir.name}
  ├── train.pkl        (training images & labels)
  ├── val.pkl          (validation images & labels)
  ├── metadata.pkl     (dataset metadata)
  └── data_loader.py   (example usage)

Ready for training with fast data loading!
Use data_loader.py as reference for your training pipeline.
""")
    
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
