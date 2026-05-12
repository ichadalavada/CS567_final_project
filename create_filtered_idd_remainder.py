#!/usr/bin/env python3
"""
Create a filtered version of idd_remainder_test.pkl:
  - Drop class 4
  - Keep classes 0, 1, 2, 3 only
  - Keep max 500 samples from each class
"""

import pickle
import numpy as np
from collections import Counter
from pathlib import Path

INPUT_PKL = "IDD/idd_remainder_test.pkl"
OUTPUT_PKL = "IDD/idd_remainder_test_filtered.pkl"
KEEP_CLASSES = [0, 1, 2, 3]
MAX_SAMPLES_PER_CLASS = 500

print(f"Loading {INPUT_PKL}...")
with open(INPUT_PKL, "rb") as f:
    data = pickle.load(f)

images = data["images"]
labels_raw = data["labels"]
image_names = data["image_names"]

print(f"  Total samples: {len(images)}")

# Extract class labels (first bbox's class)
class_labels = []
for img_bboxes in labels_raw:
    if len(img_bboxes) > 0:
        class_id = int(img_bboxes[0][0])
        class_labels.append(class_id)
    else:
        class_labels.append(0)

print(f"  Original class distribution: {dict(sorted(Counter(class_labels).items()))}")

# Filter to keep only specified classes, max N samples per class
filtered_indices = []
class_counts = {c: 0 for c in KEEP_CLASSES}

for idx, class_id in enumerate(class_labels):
    if class_id in KEEP_CLASSES and class_counts[class_id] < MAX_SAMPLES_PER_CLASS:
        filtered_indices.append(idx)
        class_counts[class_id] += 1

print(f"\n  Filtered to {len(filtered_indices)} samples")
print(f"  New class distribution: {dict(sorted(class_counts.items()))}")

# Create filtered dataset
indices = np.array(filtered_indices)
filtered_data = {
    "images": images[indices],
    "labels": [labels_raw[i] for i in filtered_indices],
    "image_names": [image_names[i] for i in filtered_indices],
}

print(f"\nSaving filtered dataset to {OUTPUT_PKL}...")
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(filtered_data, f)

print(f"✓ Done! New file: {OUTPUT_PKL}")
print(f"  Shapes: images {filtered_data['images'].shape}, labels {len(filtered_data['labels'])}")
