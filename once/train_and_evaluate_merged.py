"""
Enhanced CNN training using merged ONCE + IDD datasets.
Trains on 1%, 5%, 10% IDD-augmented data and evaluates on ONCE validation set.
Includes class imbalance fixes: inverse-frequency weights (capped), oversampling, focal loss.
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


def load_pkl_dataset(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_all_test_data():
    once_val_path = Path("once/once_data/merged_val.pkl")
    if not once_val_path.exists():
        raise FileNotFoundError(f"Test file not found: {once_val_path}")
    print("Loading test datasets...")
    once_val = load_pkl_dataset(str(once_val_path))
    return once_val, None


def load_metadata():
    metadata_path = "once/once_data/metadata.pkl"
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return metadata


def convert_to_classification(data, num_classes=4):
    """
    Convert detection data to classification format, filtering invalid labels
    """
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
        print(f"    ⚠️  Filtered out {invalid_count} samples with invalid class indices")

    return np.array(images), np.array(labels)


# ── FIX 1: Inverse-frequency class weights, capped ───────────────────────────
def compute_class_weights(y_train, max_weight=10.0):
    """
    Inverse-frequency weighting with a hard cap.
    Without the cap, a class with 7 samples gets weight ~1130,
    which explodes the loss and destabilizes training entirely.
    max_weight=10 is a safe upper bound for most imbalanced datasets.
    """
    counts = Counter(y_train.tolist())
    total  = len(y_train)
    n_cls  = len(counts)

    class_weight_dict = {}
    for cls, count in counts.items():
        w = total / (n_cls * count)
        w = min(w, max_weight)          # ← cap here
        class_weight_dict[cls] = w

    print(f"\n  Class weights (capped at {max_weight}):")
    for cls, w in sorted(class_weight_dict.items()):
        print(f"    class {cls}: {w:.3f}  (n={counts[cls]})")

    return class_weight_dict


# ── FIX 2: Oversample minority classes (WITH MEMORY CAP) ──────────────────────
def oversample_minority_classes(X_train, y_train, max_samples_per_class=5000):
    """
    Upsample minority classes with a hard cap to prevent memory overflow.
    
    Args:
        max_samples_per_class: Maximum samples per class (default 5000)
                               Prevents memory issues with large datasets
    """
    counts = Counter(y_train.tolist())
    max_count = max(counts.values())
    
    # Cap the target to prevent memory overflow
    target_count = min(max_count, max_samples_per_class)
    
    print(f"\n  Oversampling to {target_count} samples per class (capped at {max_samples_per_class})...")
    if max_count > max_samples_per_class:
        print(f"  ⚠️  Original max was {max_count}, using cap to prevent memory overflow")

    X_balanced = []
    y_balanced = []

    for cls in sorted(counts.keys()):
        idx = np.where(y_train == cls)[0]
        X_cls = X_train[idx]
        y_cls = y_train[idx]

        if len(idx) < target_count:
            # Resample with replacement
            X_cls, y_cls = resample(
                X_cls, y_cls,
                replace=True,
                n_samples=target_count,
                random_state=42
            )
        elif len(idx) > target_count:
            # Downsample majority class if needed
            X_cls, y_cls = resample(
                X_cls, y_cls,
                replace=False,
                n_samples=target_count,
                random_state=42
            )

        X_balanced.append(X_cls)
        y_balanced.append(y_cls)

    X_out = np.concatenate(X_balanced, axis=0)
    y_out = np.concatenate(y_balanced, axis=0)

    perm = np.random.permutation(len(X_out))
    print(f"  After balancing: {len(X_out)} samples ({len(X_out)/len(y_train):.2f}x original)")
    print(f"  New class distribution: {dict(sorted(Counter(y_out.tolist()).items()))}")
    
    # Estimate memory usage
    mem_gb = (X_out.nbytes + y_out.nbytes) / (1024**3)
    print(f"  Estimated memory usage: {mem_gb:.2f} GB")

    return X_out[perm], y_out[perm]


# ── FIX 3: Focal loss ─────────────────────────────────────────────────────────
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


def create_mobilenet(input_shape=(224, 224, 3), num_classes=4, use_focal_loss=True):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
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


def evaluate_on_testset(model, X_test, y_test, test_name, metadata):
    CLASS_NAMES = list(metadata["classes"].values())
    X_test = np.asarray(X_test, dtype=np.float32)

    try:
        model.trainable = False
    except:
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
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix — {test_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved confusion matrix → {save_path}")
    plt.close()


def train_model(train_data, merged_dataset_name, metadata, epochs=15, batch_size=32, 
                max_samples_per_class=5000, use_oversampling=True):
    print(f"\n{'='*60}")
    print(f"  Training on {merged_dataset_name}")
    print(f"{'='*60}")

    num_classes = metadata["num_classes"]

    X_train, y_train = convert_to_classification(train_data, num_classes=num_classes)

    print(f"  Original class distribution: {dict(sorted(Counter(y_train.tolist()).items()))}")

    # Cap majority classes on uint8 BEFORE float32 conversion to avoid ~26 GB allocation
    counts = Counter(y_train.tolist())
    if max(counts.values()) > max_samples_per_class:
        rng = np.random.default_rng(42)
        keep = []
        for cls, idx in ((c, np.where(y_train == c)[0]) for c in counts):
            keep.append(rng.choice(idx, min(len(idx), max_samples_per_class), replace=False))
        keep = np.concatenate(keep)
        rng.shuffle(keep)
        X_train, y_train = X_train[keep], y_train[keep]
        print(f"  Pre-capped to {len(X_train)} samples ({dict(sorted(Counter(y_train.tolist()).items()))})")

    X_train = X_train.astype("float32") / 255.0

    # Fix 1: capped class weights
    class_weight_dict = compute_class_weights(y_train, max_weight=10.0)

    # Fix 2: oversample minority classes (with memory cap)
    if use_oversampling:
        X_train, y_train = oversample_minority_classes(X_train, y_train, 
                                                       max_samples_per_class=max_samples_per_class)
    else:
        print(f"  ⚠️  Skipping oversampling (relying on class weights + focal loss)")

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
    
    # FIXED: Added return statement
    return model, history


def create_accuracy_matrix(results_df, save_path="accuracy_matrix.png"):
    pivot_data = results_df.pivot_table(
        index="Training Data",
        columns="Test Data",
        values="Accuracy (%)"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".2f",
        cmap="YlGn",
        cbar_kws={"label": "Accuracy (%)"},
        vmin=0,
        vmax=100
    )
    plt.title("Model Accuracy Matrix\n(Trained on various ONCE+IDD mixes, tested on ONCE val set)")
    plt.ylabel("Training Dataset")
    plt.xlabel("Test Dataset")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Accuracy matrix saved → {save_path}")
    plt.close()

    return pivot_data


def plot_accuracy_comparison(results_df, save_path="accuracy_comparison.png"):
    fig, ax = plt.subplots(figsize=(10, 5))

    once_data = results_df[results_df["Test Data"] == "ONCE"]
    ax.bar(once_data["Training Data"], once_data["Accuracy (%)"], color="steelblue", alpha=0.7)
    ax.set_title("ONCE Validation Set Performance", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim([0, 105])
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(once_data["Accuracy (%)"]):
        ax.text(i, v + 2, f"{v:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Accuracy comparison plot saved → {save_path}")
    plt.close()


def main():
    base_path  = Path("/Users/ishachadalavada/Desktop/ml_final_project")
    merged_dir = base_path / "merged_datasets"

    datasets_to_train = [
        ("1%",  merged_dir / "once_caltech_idd_1%_train.pkl"),
        ("5%",  merged_dir / "once_caltech_idd_5%_train.pkl"),
        ("10%", merged_dir / "once_caltech_idd_10%_train.pkl"),
    ]

    once_val, _ = load_all_test_data()
    metadata = load_metadata()

    num_classes = metadata["num_classes"]
    X_once_val, y_once_val = convert_to_classification(once_val, num_classes=num_classes)
    X_once_val = X_once_val.astype("float32") / 255.0

    print(f"ONCE validation samples: {len(y_once_val)}")

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

            # Train with memory-safe oversampling
            # max_samples_per_class=5000 caps each class at 5k samples (4 classes × 5k = 20k total)
            # This uses ~2-3 GB RAM instead of 60+ GB
            # To disable oversampling entirely, set use_oversampling=False
            model, history = train_model(
                train_data,
                f"ONCE + Caltech + IDD ({dataset_name})",
                metadata,
                epochs=15,
                batch_size=batch_size,
                max_samples_per_class=5000,  # Adjust this based on your RAM (4GB→2000, 8GB→3000, 16GB→5000, 32GB→10000)
                use_oversampling=True
            )

            accuracy_once, y_pred_once, _, labels_once, _ = evaluate_on_testset(
                model, X_once_val, y_once_val,
                f"ONCE Val Set (n={len(y_once_val)})",
                metadata
            )
            results.append({
                "Training Data": f"ONCE + Caltech + IDD ({dataset_name})",
                "Test Data": "ONCE",
                "Accuracy (%)": accuracy_once * 100,
                "Samples": len(y_once_val)
            })
            plot_confusion_matrix(
                y_once_val, y_pred_once,
                f"ONCE Val - Trained on ONCE+Caltech+IDD {dataset_name}",
                [list(metadata["classes"].values())[i] for i in labels_once],
                f"confusion_matrix_once_caltech_{dataset_name}.png"
            )

            model_save_path = f"once/mobilenet_once_caltech_idd_{dataset_name}_classifier.keras"
            model.save(model_save_path)
            print(f"\n✓ Model saved → {model_save_path}")

        except Exception as e:
            print(f"\n  ❌ Error processing {dataset_name} dataset: {e}")
            import traceback
            traceback.print_exc()
            continue

    results_df = pd.DataFrame(results)

    print(f"\n\n{'='*80}")
    print(f"  ACCURACY SUMMARY TABLE")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))

    results_df.to_csv("accuracy_results_once_caltech.csv", index=False)
    print(f"\n✓ Results saved → accuracy_results_once_caltech.csv")

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
    print(f"  ✓ ALL TRAINING AND EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  - accuracy_results_once_caltech.csv")
    print(f"  - accuracy_matrix.png")
    print(f"  - accuracy_comparison.png")
    print(f"  - confusion_matrix_once_caltech_*.png")
    print(f"  - mobilenet_once_caltech_idd_*_classifier.keras")


if __name__ == "__main__":
    main()