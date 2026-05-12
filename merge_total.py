#!/usr/bin/env python3
"""
Merge KITTI + Caltech + Scooter + Once train data with 1%, 5%, and 10% of IDD.
Outputs total/total_data/total_{1%,5%,10%}_train.pkl
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

BASE = Path("/Users/ishachadalavada/Desktop/ml_final_project")
sys.path.insert(0, str(BASE / "caltech_yolo"))
from train_and_evaluate_merged import create_caltech_pkl_dataset

OUT_DIR = BASE / "total" / "total_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KITTI_TRAIN   = BASE / "kitti"   / "kitti_pkl_224"  / "train.pkl"
SCOOTER_TRAIN = BASE / "scooter_data" / "pkl_224"   / "train.pkl"
ONCE_TRAIN    = BASE / "once"    / "once_data"       / "train.pkl"
MERGED_DIR    = BASE / "merged_datasets"

IDD_SPLITS = {"1%": "idd_1%.pkl", "5%": "idd_5%.pkl", "10%": "idd_10%.pkl"}

NUM_CLASSES = 4  # vehicle, pedestrian, cyclist, misc — drop class 4 (traffic)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def to_4d_uint8(imgs):
    arr = np.array(imgs, dtype=np.uint8) if not isinstance(imgs, np.ndarray) else imgs
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim != 4:
        raise ValueError(f"Expected (N,H,W,C), got shape {arr.shape}")
    return arr


def filter_classes(data):
    """Drop samples whose dominant class >= NUM_CLASSES (traffic etc.)."""
    images, labels, names = [], [], []
    for img, anns, name in zip(data["images"], data["labels"], data["image_names"]):
        if len(anns) == 0:
            continue
        class_ids = [ann[0] for ann in anns]
        most_common = Counter(class_ids).most_common(1)[0][0]
        if most_common < 0 or most_common >= NUM_CLASSES:
            continue
        images.append(img)
        labels.append(anns)
        names.append(name)
    arr = np.array(images, dtype=np.uint8)
    return {"images": arr, "labels": labels, "image_names": names}


def load_source(path, label):
    raw = load_pkl(path)
    filtered = filter_classes(raw)
    n = len(filtered["images"])
    print(f"  {label}: {n} samples (after class filter)")
    return filtered


def merge_datasets(parts):
    images = np.concatenate([to_4d_uint8(p["images"]) for p in parts], axis=0)
    labels = sum([list(p["labels"]) for p in parts], [])
    names  = sum([list(p["image_names"]) for p in parts], [])
    return {"images": images, "labels": labels, "image_names": names}


def main():
    print("Loading KITTI train...")
    kitti = load_source(KITTI_TRAIN, "KITTI")

    print("\nLoading Caltech YOLO train...")
    cal_raw = create_caltech_pkl_dataset(BASE / "caltech_yolo", split="train", target_size=(224, 224))
    caltech = filter_classes(cal_raw)
    print(f"  Caltech: {len(caltech['images'])} samples (after class filter)")

    print("\nLoading Scooter train...")
    scooter = load_source(SCOOTER_TRAIN, "Scooter")

    print("\nLoading Once train...")
    once = load_source(ONCE_TRAIN, "Once")

    base_parts = [kitti, caltech, scooter, once]
    base_total = sum(len(p["images"]) for p in base_parts)
    print(f"\nBase (KITTI+Caltech+Scooter+Once): {base_total} samples")

    for pct, idd_file in IDD_SPLITS.items():
        print(f"\n{'─'*55}")
        print(f"Merging with IDD {pct}...")
        idd = load_source(MERGED_DIR / idd_file, f"IDD {pct}")

        merged = merge_datasets(base_parts + [idd])
        total = len(merged["images"])
        print(f"  Total: {total} samples")

        # Class distribution summary
        dominant = []
        for anns in merged["labels"]:
            ids = [ann[0] for ann in anns]
            dominant.append(Counter(ids).most_common(1)[0][0])
        dist = dict(sorted(Counter(dominant).items()))
        print(f"  Class distribution: {dist}")

        out_path = OUT_DIR / f"total_{pct}_train.pkl"
        save_pkl(merged, out_path)
        print(f"  Saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
