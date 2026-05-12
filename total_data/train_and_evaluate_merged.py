"""
CNN training on merged Total datasets (KITTI + Caltech + Scooter + Once + IDD).
Trains on 1%, 5%, 10% IDD-augmented total data; evaluates on IDD remainder test.
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

BASE_PATH    = Path("/Users/ishachadalavada/Desktop/ml_final_project")
TOTAL_DIR    = BASE_PATH / "total" / "total_data"
IDD_TEST_PATH = BASE_PATH / "IDD" / "idd_remainder_test.pkl"

CLASS_NAMES = ["vehicle", "pedestrian", "cyclist", "misc"]
NUM_CLASSES = 4


def load_pkl_dataset(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def convert_to_classification(data):
    images, labels = [], []
    invalid_count = 0
    for img, anns in zip(data["images"], data["labels"]):
        if len(anns) == 0:
            continue
        class_ids = [ann[0] for ann in anns]
        most_common = Counter(class_ids).most_common(1)[0][0]
        if most_common < 0 or most_common >= NUM_CLASSES:
            invalid_count += 1
            continue
        images.append(img)
        labels.append(most_common)
    if invalid_count > 0:
        print(f"    Filtered out {invalid_count} samples with class >= {NUM_CLASSES}")
    return np.array(images), np.array(labels)


def compute_class_weights(y_train, max_weight=10.0):
    counts = Counter(y_train.tolist())
    total  = len(y_train)
    n_cls  = len(counts)
    class_weight_dict = {}
    for cls, count in counts.items():
        w = min(total / (n_cls * count), max_weight)
        class_weight_dict[cls] = w
    print(f"\n  Class weights (capped at {max_weight}):")
    for cls, w in sorted(class_weight_dict.items()):
        print(f"    {CLASS_NAMES[cls]:12s}: {w:.3f}  (n={counts[cls]})")
    return class_weight_dict


def oversample_minority_classes(X_train, y_train, max_samples_per_class=5000):
    counts    = Counter(y_train.tolist())
    max_count = max(counts.values())
    target    = min(max_count, max_samples_per_class)

    print(f"\n  Oversampling to {target} samples per class (cap={max_samples_per_class})...")
    if max_count > max_samples_per_class:
        print(f"  Original max was {max_count}, capping to avoid memory overflow")

    X_balanced, y_balanced = [], []
    for cls in sorted(counts.keys()):
        idx   = np.where(y_train == cls)[0]
        X_cls = X_train[idx]
        y_cls = y_train[idx]
        if len(idx) != target:
            X_cls, y_cls = resample(X_cls, y_cls, replace=len(idx) < target,
                                    n_samples=target, random_state=42)
        X_balanced.append(X_cls)
        y_balanced.append(y_cls)

    X_out = np.concatenate(X_balanced, axis=0)
    y_out = np.concatenate(y_balanced, axis=0)
    perm  = np.random.permutation(len(X_out))
    print(f"  After balancing: {len(X_out)} samples  "
          f"({len(X_out)/len(y_train):.2f}x original)")
    print(f"  New distribution: {dict(sorted(Counter(y_out.tolist()).items()))}")
    print(f"  Estimated memory: {(X_out.nbytes + y_out.nbytes) / 1024**3:.2f} GB")
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


def create_mobilenet(input_shape=(224, 224, 3), use_focal_loss=True):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False,
                             weights="imagenet")
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    loss = focal_loss(gamma=2.0, alpha=0.25) if use_focal_loss \
           else "sparse_categorical_crossentropy"
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=loss, metrics=["accuracy"])
    return model


def evaluate_on_testset(model, X_test, y_test, test_name):
    X_test = np.asarray(X_test, dtype=np.float32)
    try:
        y_probs = model.predict(X_test, verbose=0, batch_size=32)
    except Exception:
        y_probs_list = []
        for i in range(0, len(X_test), 8):
            y_probs_list.append(model(X_test[i:i+8], training=False).numpy())
        y_probs = np.vstack(y_probs_list)

    y_pred = np.argmax(y_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    present_labels = sorted(set(y_test) | set(y_pred))
    present_names  = [CLASS_NAMES[i] for i in present_labels if i < NUM_CLASSES]

    print(f"\n{'='*60}")
    print(f"  Evaluation on {test_name}")
    print(f"{'='*60}")
    print(f"  Accuracy: {accuracy*100:.2f}%  (n={len(y_test)})")
    print(f"\n  Class distribution:")
    for cls_id, count in sorted(Counter(y_test.tolist()).items()):
        name = CLASS_NAMES[cls_id] if cls_id < NUM_CLASSES else f"class_{cls_id}"
        print(f"    {name:12s}: {count:4d}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, labels=present_labels,
                                target_names=present_names, zero_division=0))
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
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved confusion matrix → {save_path}")
    plt.close()


def train_model(train_data, dataset_label, epochs=15, batch_size=32,
                max_samples_per_class=5000, use_oversampling=True):
    print(f"\n{'='*60}")
    print(f"  Training on {dataset_label}")
    print(f"{'='*60}")

    X_train, y_train = convert_to_classification(train_data)

    # Pre-cap on uint8 before float32 conversion to avoid memory overflow
    counts = Counter(y_train.tolist())
    if max(counts.values()) > max_samples_per_class:
        rng  = np.random.default_rng(42)
        keep = []
        for _, idx in ((c, np.where(y_train == c)[0]) for c in counts):
            keep.append(rng.choice(idx, min(len(idx), max_samples_per_class), replace=False))
        keep = np.concatenate(keep)
        rng.shuffle(keep)
        X_train, y_train = X_train[keep], y_train[keep]

    X_train = X_train.astype("float32") / 255.0
    print(f"  Original distribution: {dict(sorted(Counter(y_train.tolist()).items()))}")

    class_weight_dict = compute_class_weights(y_train, max_weight=10.0)

    if use_oversampling:
        X_train, y_train = oversample_minority_classes(
            X_train, y_train, max_samples_per_class=max_samples_per_class)
    else:
        print("  Skipping oversampling (relying on class weights + focal loss)")

    print(f"  Training shape: {X_train.shape}")

    model = create_mobilenet(input_shape=X_train.shape[1:], use_focal_loss=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    print(f"\n  Training for up to {epochs} epochs (early stopping enabled)...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def create_accuracy_matrix(results_df, save_path="accuracy_matrix_total.png"):
    pivot_data = results_df.pivot_table(
        index="Training Data", columns="Test Data", values="Accuracy (%)")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGn",
                cbar_kws={"label": "Accuracy (%)"}, vmin=0, vmax=100)
    plt.title("Model Accuracy Matrix\n(Total datasets + IDD splits, tested on IDD remainder)")
    plt.ylabel("Training Dataset")
    plt.xlabel("Test Dataset")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nAccuracy matrix saved → {save_path}")
    plt.close()
    return pivot_data


def plot_accuracy_comparison(results_df, save_path="accuracy_comparison_total.png"):
    idd_data = results_df[results_df["Test Data"] == "IDD remainder"]
    if idd_data.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(idd_data["Training Data"], idd_data["Accuracy (%)"],
                  color="steelblue", alpha=0.8)
    for bar, v in zip(bars, idd_data["Accuracy (%)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Total Ensemble — IDD Remainder Test Performance", fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Accuracy comparison saved → {save_path}")
    plt.close()


def main():
    datasets_to_train = [
        ("1%",  TOTAL_DIR / "total_1%_train.pkl"),
        ("5%",  TOTAL_DIR / "total_5%_train.pkl"),
        ("10%", TOTAL_DIR / "total_10%_train.pkl"),
    ]

    print("Loading IDD remainder test set...")
    idd_test = load_pkl_dataset(str(IDD_TEST_PATH))
    X_idd_test, y_idd_test = convert_to_classification(idd_test)
    X_idd_test = X_idd_test.astype("float32") / 255.0
    print(f"  IDD test samples: {len(y_idd_test)}")
    print(f"  Class distribution: {dict(sorted(Counter(y_idd_test.tolist()).items()))}")

    results    = []
    models_dict = {}

    for dataset_name, dataset_path in datasets_to_train:
        print(f"\n\n{'#'*60}")
        print(f"# Total + IDD {dataset_name}")
        print(f"{'#'*60}")

        try:
            train_data = load_pkl_dataset(str(dataset_path))
            n = len(train_data["images"])
            print(f"  Dataset size: {n} samples")

            batch_size = 32 if n >= 100 else max(8, n // 10)

            model, _ = train_model(
                train_data,
                f"Total + IDD ({dataset_name})",
                epochs=15,
                batch_size=batch_size,
                max_samples_per_class=5000,
                use_oversampling=True,
            )
            models_dict[dataset_name] = model

            accuracy, y_pred, _, labels, _ = evaluate_on_testset(
                model, X_idd_test, y_idd_test,
                f"IDD Remainder Test (n={len(y_idd_test)})",
            )

            results.append({
                "Training Data": f"Total + IDD ({dataset_name})",
                "Test Data": "IDD remainder",
                "Accuracy (%)": accuracy * 100,
                "Samples": len(y_idd_test),
            })

            plot_confusion_matrix(
                y_idd_test, y_pred,
                f"IDD Test — Total+IDD {dataset_name}",
                [CLASS_NAMES[i] for i in labels],
                str(TOTAL_DIR / f"confusion_matrix_total_{dataset_name}.png"),
            )

            model_path = str(TOTAL_DIR / f"mobilenet_total_idd_{dataset_name}_classifier.keras")
            model.save(model_path)
            print(f"\nModel saved → {model_path}")

        except Exception as e:
            print(f"\n  Error processing {dataset_name}: {e}")
            import traceback; traceback.print_exc()
            continue

    results_df = pd.DataFrame(results)

    print(f"\n\n{'='*80}")
    print(f"  ACCURACY SUMMARY")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))

    csv_path = str(TOTAL_DIR / "accuracy_results_total.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    if not results_df.empty:
        create_accuracy_matrix(results_df)
        try:
            plot_accuracy_comparison(results_df)
        except Exception as e:
            print(f"  Warning: could not create comparison plot: {e}")

    print(f"\n{'='*80}")
    print(f"  ALL TRAINING AND EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated files (in {TOTAL_DIR}):")
    print(f"  - accuracy_results_total.csv")
    print(f"  - accuracy_matrix_total.png")
    print(f"  - accuracy_comparison_total.png")
    print(f"  - confusion_matrix_total_*.png")
    print(f"  - mobilenet_total_idd_*_classifier.keras")


if __name__ == "__main__":
    main()
