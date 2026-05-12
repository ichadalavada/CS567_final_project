"""
Enhanced CNN training for Caltech Pedestrian Detection Dataset (YOLO format).
Trains on caltech + IDD merged data and evaluates on Caltech test set.
Includes class imbalance fixes: inverse-frequency weights (capped), oversampling, focal loss.

Five-class mapping (unified across all datasets):
  0: vehicle
  1: pedestrian        <- Caltech primary focus
  2: cyclist
  3: misc
  4: traffic
"""

import pickle
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2


# ────────────────────────────────────────────────────────────────────────────
# UNIFIED CLASS MAPPING
# ────────────────────────────────────────────────────────────────────────────
NEW_CLASSES = {
    0: "vehicle",
    1: "pedestrian",
    2: "cyclist",
    3: "misc",
    4: "traffic"
}

NUM_CLASSES = len(NEW_CLASSES)  # 5

# Caltech YOLO dataset has only class 0 = "person" → map to pedestrian (1)
CALTECH_YOLO_TO_NEW = {
    0: 1,  # person → pedestrian
}


# ────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────────────────────

def load_pkl_dataset(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_metadata():
    """Return hardcoded 5-class metadata (Caltech has no metadata.pkl)."""
    return {
        "num_classes": NUM_CLASSES,
        "classes": NEW_CLASSES,
    }


def create_caltech_pkl_dataset(base_path, split='train', target_size=(224, 224)):
    """
    Load Caltech YOLO-format dataset into a pkl-compatible dict.

    Directory layout expected:
      <base_path>/datasets/images/<split>/caltechpedestriandataset/<set_id>/<stem>.png
      <base_path>/datasets/labels/<split>/caltechpedestriandataset/<set_id>/<stem>.txt

    YOLO label files: one line per object → "class_id cx cy w h"
    Caltech class 0 (person) is remapped to NEW_CLASSES class 1 (pedestrian).

    Returns dict with keys 'images', 'labels', 'image_names' matching IDD pkl format.
    """
    base_path = Path(base_path)
    lbl_base = base_path / "datasets" / "labels" / split / "caltechpedestriandataset"
    img_base = base_path / "datasets" / "images" / split / "caltechpedestriandataset"

    data = {'images': [], 'labels': [], 'image_names': []}

    label_files = sorted(f for f in lbl_base.rglob("*.txt") if not f.name.endswith(".cache"))

    print(f"\n  Processing {split} split ({len(label_files)} label files)...")

    loaded, skipped = 0, 0
    for lbl_path in label_files:
        rel = lbl_path.relative_to(lbl_base)           # e.g. set00/set00_V001_0043.txt
        img_path = img_base / rel.with_suffix('.png')

        if not img_path.exists():
            skipped += 1
            continue

        try:
            lines = lbl_path.read_text().strip().splitlines()
            annotations = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                yolo_cls = int(parts[0])
                new_cls  = CALTECH_YOLO_TO_NEW.get(yolo_cls, 3)
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                annotations.append([new_cls, cx, cy, w, h])

            if not annotations:
                skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if target_size:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

            data['images'].append(img)
            data['labels'].append(annotations)
            data['image_names'].append(lbl_path.stem)
            loaded += 1

        except Exception:
            skipped += 1
            continue

    print(f"    Loaded {loaded} images  (skipped {skipped})")
    return data


def load_all_test_data():
    """Load Caltech test set from YOLO labels (sets 06-10)."""
    caltech_base = Path(".")
    caltech_test = create_caltech_pkl_dataset(caltech_base, split='test', target_size=(224, 224))
    return caltech_test, None


# ────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ────────────────────────────────────────────────────────────────────────────

def convert_to_classification(data, num_classes=5):
    """Convert detection data to classification format, filtering invalid labels."""
    images = []
    labels = []
    invalid_count = 0

    for img, anns in zip(data["images"], data["labels"]):
        if len(anns) == 0:
            continue

        class_ids = [ann[0] for ann in anns]
        most_common = Counter(class_ids).most_common(1)[0][0]

        if most_common < 0 or most_common >= num_classes:
            invalid_count += 1
            continue

        images.append(img)
        labels.append(most_common)

    if invalid_count > 0:
        print(f"    Warning: Filtered out {invalid_count} samples with invalid class indices")

    return np.array(images), np.array(labels)


# ────────────────────────────────────────────────────────────────────────────
# CLASS IMBALANCE FIXES
# ────────────────────────────────────────────────────────────────────────────

def compute_class_weights(y_train, max_weight=10.0):
    """Inverse-frequency weighting with a hard cap."""
    counts = Counter(y_train.tolist())
    total = len(y_train)
    n_cls = len(counts)

    class_weight_dict = {}
    for cls, count in counts.items():
        w = total / (n_cls * count)
        w = min(w, max_weight)
        class_weight_dict[cls] = w

    print(f"\n  Class weights (capped at {max_weight}):")
    for cls, w in sorted(class_weight_dict.items()):
        print(f"    class {cls} ({NEW_CLASSES.get(cls, '?')}): {w:.3f}  (n={counts[cls]})")

    return class_weight_dict


def oversample_minority_classes(X_train, y_train, max_samples_per_class=5000):
    """Upsample minority classes with a hard cap to prevent memory overflow."""
    counts = Counter(y_train.tolist())
    max_count = max(counts.values())
    target_count = min(max_count, max_samples_per_class)

    print(f"\n  Oversampling to {target_count} samples per class (capped at {max_samples_per_class})...")
    if max_count > max_samples_per_class:
        print(f"  Warning: Original max was {max_count}, using cap to prevent memory overflow")

    X_balanced = []
    y_balanced = []

    for cls in sorted(counts.keys()):
        idx = np.where(y_train == cls)[0]
        X_cls, y_cls = X_train[idx], y_train[idx]

        if len(idx) < target_count:
            X_cls, y_cls = resample(X_cls, y_cls, replace=True,
                                    n_samples=target_count, random_state=42)
        elif len(idx) > target_count:
            X_cls, y_cls = resample(X_cls, y_cls, replace=False,
                                    n_samples=target_count, random_state=42)

        X_balanced.append(X_cls)
        y_balanced.append(y_cls)

    X_out = np.concatenate(X_balanced, axis=0)
    y_out = np.concatenate(y_balanced, axis=0)

    perm = np.random.permutation(len(X_out))
    print(f"  After balancing: {len(X_out)} samples ({len(X_out)/len(y_train):.2f}x original)")
    print(f"  New class distribution: {dict(sorted(Counter(y_out.tolist()).items()))}")

    mem_gb = (X_out.nbytes + y_out.nbytes) / (1024**3)
    print(f"  Estimated memory usage: {mem_gb:.2f} GB")

    return X_out[perm], y_out[perm]


def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true   = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred   = tf.clip_by_value(y_pred, 1e-7, 1.0)
        n_cls    = tf.shape(y_pred)[1]
        y_onehot = tf.one_hot(y_true, n_cls)

        ce           = -tf.reduce_sum(y_onehot * tf.math.log(y_pred), axis=1)
        p_t          = tf.reduce_sum(y_onehot * y_pred, axis=1)
        focal_weight = tf.pow(1.0 - p_t, gamma)

        return tf.reduce_mean(alpha * focal_weight * ce)
    return loss_fn


# ────────────────────────────────────────────────────────────────────────────
# MODEL
# ────────────────────────────────────────────────────────────────────────────

def create_mobilenet(input_shape=(224, 224, 3), num_classes=5, use_focal_loss=True):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    loss = focal_loss(gamma=2.0, alpha=0.25) if use_focal_loss \
           else "sparse_categorical_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=loss,
        metrics=["accuracy"]
    )
    return model


# ────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ────────────────────────────────────────────────────────────────────────────

def train_model(train_data, merged_dataset_name, metadata, epochs=15, batch_size=32,
                max_samples_per_class=5000, use_oversampling=True):
    print(f"\n{'='*60}")
    print(f"  Training on {merged_dataset_name}")
    print(f"{'='*60}")

    num_classes = metadata["num_classes"]

    X_train, y_train = convert_to_classification(train_data, num_classes=num_classes)
    X_train = X_train.astype("float32") / 255.0

    print(f"  Original class distribution: {dict(sorted(Counter(y_train.tolist()).items()))}")

    class_weight_dict = compute_class_weights(y_train, max_weight=10.0)

    if use_oversampling:
        X_train, y_train = oversample_minority_classes(
            X_train, y_train, max_samples_per_class=max_samples_per_class
        )
    else:
        print("  Skipping oversampling (relying on class weights + focal loss)")

    print(f"  Training data shape: {X_train.shape}")

    model = create_mobilenet(
        input_shape=X_train.shape[1:],
        num_classes=num_classes,
        use_focal_loss=True
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    print(f"\n  Training for up to {epochs} epochs (early stopping enabled)...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    return model, history


def evaluate_on_testset(model, X_test, y_test, test_name, metadata):
    CLASS_NAMES = list(metadata["classes"].values())
    X_test = np.asarray(X_test, dtype=np.float32)

    try:
        model.trainable = False
    except Exception:
        pass

    try:
        y_probs = model.predict(X_test, verbose=0, batch_size=32)
    except Exception as e:
        print(f"  Warning: batch_size=32 failed: {e}, trying batch_size=8...")
        try:
            y_probs = model.predict(X_test, verbose=0, batch_size=8)
        except Exception as e2:
            print(f"  Warning: batch_size=8 failed: {e2}, falling back to manual batching...")
            y_probs_list = []
            for i in range(0, len(X_test), 8):
                batch = X_test[i:i+8]
                y_probs_list.append(model(batch, training=False).numpy())
            y_probs = np.vstack(y_probs_list)

    y_pred = np.argmax(y_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    present_labels = sorted(set(y_test) | set(y_pred))
    present_names  = [CLASS_NAMES[i] for i in present_labels if i < len(CLASS_NAMES)]

    print(f"\n{'='*60}")
    print(f"  Evaluation on {test_name}")
    print(f"{'='*60}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Total samples: {len(y_test)}")

    print(f"\n  Class distribution in {test_name}:")
    for cls_id, count in sorted(Counter(y_test.tolist()).items()):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        print(f"    {name:15s}: {count:4d} samples")

    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))

    return accuracy, y_pred, y_probs, present_labels, present_names


def plot_confusion_matrix(y_test, y_pred, test_name, class_names, save_path):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {test_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved confusion matrix -> {save_path}")
    plt.close()


def create_accuracy_matrix(results_df, save_path="accuracy_matrix_caltech.png"):
    pivot_data = results_df.pivot_table(
        index="Training Data",
        columns="Test Data",
        values="Accuracy (%)"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGn",
                cbar_kws={"label": "Accuracy (%)"}, vmin=0, vmax=100)
    plt.title("Model Accuracy Matrix\n(Trained on Caltech+IDD mixes, tested on Caltech test set)")
    plt.ylabel("Training Dataset")
    plt.xlabel("Test Dataset")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nAccuracy matrix saved -> {save_path}")
    plt.close()
    return pivot_data


def plot_accuracy_comparison(results_df, save_path="accuracy_comparison_caltech.png"):
    _, ax = plt.subplots(figsize=(10, 5))

    caltech_data = results_df[results_df["Test Data"] == "Caltech"]
    ax.bar(caltech_data["Training Data"], caltech_data["Accuracy (%)"],
           color="steelblue", alpha=0.7)
    ax.set_title("Caltech Test Set Performance", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim([0, 105])
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(caltech_data["Accuracy (%)"]):
        ax.text(i, v + 2, f"{v:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Accuracy comparison plot saved -> {save_path}")
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    base_path  = Path("/Users/ishachadalavada/Desktop/ml_final_project")
    merged_dir = base_path / "merged_datasets"

    datasets_to_train = [
        ("1%",  merged_dir / "caltech_idd_1%_train.pkl"),
        ("5%",  merged_dir / "caltech_idd_5%_train.pkl"),
        ("10%", merged_dir / "caltech_idd_10%_train.pkl"),
    ]

    caltech_yolo_dir = base_path / "caltech_yolo"
    caltech_test_raw, _ = load_all_test_data() if Path(".").resolve() == caltech_yolo_dir.resolve() \
        else (create_caltech_pkl_dataset(caltech_yolo_dir, split='test', target_size=(224, 224)), None)
    caltech_test = caltech_test_raw

    metadata = load_metadata()
    num_classes = metadata["num_classes"]

    X_caltech_test, y_caltech_test = convert_to_classification(caltech_test, num_classes=num_classes)
    X_caltech_test = X_caltech_test.astype("float32") / 255.0

    print(f"Caltech test samples: {len(y_caltech_test)}")
    print(f"Class distribution: {dict(sorted(Counter(y_caltech_test.tolist()).items()))}")

    results = []

    for dataset_name, dataset_path in datasets_to_train:
        print(f"\n\n{'#'*60}")
        print(f"# Processing {dataset_name} merged dataset")
        print(f"{'#'*60}")

        try:
            train_data = load_pkl_dataset(str(dataset_path))

            num_train_samples = len(train_data['images'])
            print(f"  Dataset size: {num_train_samples} samples")

            batch_size = 32
            if num_train_samples < 100:
                batch_size = max(8, num_train_samples // 10)
                print(f"  Small dataset detected, using batch_size={batch_size}")

            model, _ = train_model(
                train_data,
                f"Caltech + IDD ({dataset_name})",
                metadata,
                epochs=15,
                batch_size=batch_size,
                max_samples_per_class=5000,
                use_oversampling=True
            )

            accuracy_caltech, y_pred_caltech, _, labels_caltech, _ = evaluate_on_testset(
                model, X_caltech_test, y_caltech_test,
                f"Caltech Test Set (n={len(y_caltech_test)})",
                metadata
            )
            results.append({
                "Training Data": f"Caltech + IDD ({dataset_name})",
                "Test Data": "Caltech",
                "Accuracy (%)": accuracy_caltech * 100,
                "Samples": len(y_caltech_test)
            })
            plot_confusion_matrix(
                y_caltech_test, y_pred_caltech,
                f"Caltech Test - Trained on Caltech+IDD {dataset_name}",
                [list(metadata["classes"].values())[i] for i in labels_caltech],
                f"confusion_matrix_caltech_{dataset_name}.png"
            )

            model_save_path = f"mobilenet_caltech_idd_{dataset_name}_classifier.keras"
            model.save(model_save_path)
            print(f"\nModel saved -> {model_save_path}")

        except Exception as e:
            print(f"\n  Error processing {dataset_name} dataset: {e}")
            import traceback
            traceback.print_exc()
            continue

    results_df = pd.DataFrame(results)

    print(f"\n\n{'='*80}")
    print(f"  ACCURACY SUMMARY TABLE")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))

    results_df.to_csv("accuracy_results_caltech.csv", index=False)
    print(f"\nResults saved -> accuracy_results_caltech.csv")

    if not results_df.empty:
        pivot_table = create_accuracy_matrix(results_df)
        print(f"\n{'='*80}")
        print(f"  ACCURACY MATRIX (Pivot View)")
        print(f"{'='*80}\n")
        print(pivot_table)

        try:
            plot_accuracy_comparison(results_df)
        except Exception as e:
            print(f"  Warning: Could not create comparison plot: {e}")

    print(f"\n{'='*80}")
    print(f"  ALL TRAINING AND EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  - accuracy_results_caltech.csv")
    print(f"  - accuracy_matrix_caltech.png")
    print(f"  - accuracy_comparison_caltech.png")
    print(f"  - confusion_matrix_caltech_*.png")
    print(f"  - mobilenet_caltech_idd_*_classifier.keras")


if __name__ == "__main__":
    main()
