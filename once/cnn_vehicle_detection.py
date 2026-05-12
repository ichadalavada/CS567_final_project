import pickle
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

def load_pkl_dataset(pkl_dir=None, use_merged=True, merged_percentage=None):
    """
    Load ONCE dataset from pickle files
    
    Args:
        pkl_dir: Path to pickle directory (default: once/pkl_224)
        use_merged: Whether to use merged ONCE+IDD training data
        merged_percentage: Percentage of IDD to merge ('1%', '5%', '10%')
    """
    if pkl_dir is None:
        pkl_dir = "once/pkl_224"
    
    pkl_dir = Path(pkl_dir)

    if use_merged and merged_percentage:
        # Load merged ONCE+IDD training data
        merged_train_path = Path("merged_datasets") / f"once_idd_{merged_percentage}_train.pkl"
        print(f"Loading merged dataset: {merged_train_path}")
        
        if not merged_train_path.exists():
            print(f"  ⚠ Merged dataset not found at {merged_train_path}")
            print(f"  Falling back to standard ONCE dataset")
            use_merged = False
        else:
            with open(merged_train_path, "rb") as f:
                train_data = pickle.load(f)
            
            # Use ONCE validation data for validation
            with open(pkl_dir / "val.pkl", "rb") as f:
                val_data = pickle.load(f)
            
            with open(pkl_dir / "metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
            
            print(f"✓ Loaded merged train data: {len(train_data['images'])} images")
            print(f"✓ Loaded ONCE val data: {len(val_data['images'])} images")
            return train_data, val_data, metadata

    # Load standard ONCE data
    print("Loading ONCE dataset from pkl files...")
    with open(pkl_dir / "train.pkl", "rb") as f:
        train_data = pickle.load(f)

    with open(pkl_dir / "val.pkl", "rb") as f:
        val_data = pickle.load(f)

    with open(pkl_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    print(f"✓ Loaded ONCE train data: {len(train_data['images'])} images")
    print(f"✓ Loaded ONCE val data: {len(val_data['images'])} images")

    return train_data, val_data, metadata

def convert_to_classification(data):
    images = []
    labels = []

    for img, anns in zip(data["images"], data["labels"]):
        if len(anns) == 0:
            continue
        class_ids = [ann[0] for ann in anns]
        most_common = Counter(class_ids).most_common(1)[0][0]
        images.append(img)
        labels.append(most_common)

    return np.array(images), np.array(labels)


# ── FIX 1: Compute inverse-frequency class weights ────────────────────────────
def compute_class_weights(y_train):
    """
    Inverse-frequency weighting: minority classes get higher loss weight.
    This penalizes the model more for misclassifying rare classes.
    """
    counts = Counter(y_train.tolist())
    total  = len(y_train)
    n_cls  = len(counts)

    class_weight_dict = {
        cls: total / (n_cls * count)
        for cls, count in counts.items()
    }

    print("\n  Class weights (higher = rarer class penalized more):")
    for cls, w in sorted(class_weight_dict.items()):
        print(f"    class {cls}: {w:.3f}  (n={counts[cls]})")

    return class_weight_dict


# ── FIX 2: Oversample minority classes to balance training set ────────────────
def oversample_minority_classes(X_train, y_train):
    """
    Upsample all classes to match the majority class size.
    Ensures the model sees equal numbers of each class per epoch.
    """
    counts   = Counter(y_train.tolist())
    max_count = max(counts.values())

    X_balanced, y_balanced = [], []

    for cls in sorted(counts.keys()):
        idx = np.where(y_train == cls)[0]
        X_cls = X_train[idx]
        y_cls = y_train[idx]

        if len(idx) < max_count:
            # Resample with replacement to reach majority class size
            X_cls, y_cls = resample(
                X_cls, y_cls,
                replace=True,
                n_samples=max_count,
                random_state=42
            )

        X_balanced.append(X_cls)
        y_balanced.append(y_cls)

    X_out = np.concatenate(X_balanced, axis=0)
    y_out = np.concatenate(y_balanced, axis=0)

    # Shuffle
    perm = np.random.permutation(len(X_out))
    print(f"\n  After oversampling: {len(X_out)} samples "
          f"({len(X_out)//len(y_train):.1f}x original)")
    print(f"  New class distribution: {dict(sorted(Counter(y_out.tolist()).items()))}")

    return X_out[perm], y_out[perm]


# ── FIX 3: Focal loss — down-weights easy/confident predictions ───────────────
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss focuses training on hard, misclassified examples.
    gamma: focusing parameter (higher = more focus on hard examples)
    alpha: base class balancing weight
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        # One-hot encode ground truth
        n_cls   = tf.shape(y_pred)[1]
        y_onehot = tf.one_hot(y_true, n_cls)

        # Cross-entropy per sample
        ce = -tf.reduce_sum(y_onehot * tf.math.log(y_pred), axis=1)

        # Focal weight: (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_onehot * y_pred, axis=1)
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
    x = layers.Dense(256, activation="relu")(x)   # wider head
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


def evaluate_model(model, X_val, y_val, metadata):
    CLASS_NAMES = list(metadata["classes"].values())

    y_probs = model.predict(X_val, verbose=0)
    y_pred  = np.argmax(y_probs, axis=1)

    present_labels = sorted(set(y_val) | set(y_pred))
    present_names  = [CLASS_NAMES[i] for i in present_labels if i < len(CLASS_NAMES)]

    acc = np.mean(y_val == y_pred)
    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*55}")
    print(f"  Validation Accuracy: {acc*100:.2f}%")

    print(f"\n  Class distribution in val set:")
    for cls_id, count in sorted(Counter(y_val.tolist()).items()):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        print(f"    {name:15s}: {count} samples")

    print(f"\n  Classification Report:")
    print(classification_report(
        y_val, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))

    cm = confusion_matrix(y_val, y_pred, labels=present_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=present_names,
        yticklabels=present_names
    )
    plt.title("Confusion Matrix — MobileNet (Class-Balanced)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix_balanced.png", dpi=150)
    plt.show()
    print("  ✓ Confusion matrix saved → confusion_matrix_balanced.png")

    return y_pred, y_probs


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history_balanced.png", dpi=150)
    plt.show()


def main():
    print("="*55)
    print("  ONCE DATASET CNN CLASSIFIER")
    print("="*55)
    
    # Load ONCE dataset
    print("\nLoading dataset...")
    train_data, val_data, metadata = load_pkl_dataset(
        pkl_dir="once/pkl_224",
        use_merged=False,  # Set to True if using merged ONCE+IDD data
        merged_percentage=None  # Use '1%', '5%', or '10%' if merging
    )
    print(f"Metadata: {metadata}")

    X_train, y_train = convert_to_classification(train_data)
    X_val,   y_val   = convert_to_classification(val_data)

    print(f"\nOriginal class distribution (train):")
    print(f"  {dict(sorted(Counter(y_train.tolist()).items()))}")

    # Normalize before oversampling (saves memory vs. doing it after)
    X_train = X_train.astype("float32") / 255.0
    X_val   = X_val.astype("float32")   / 255.0

    # ── Apply fixes ───────────────────────────────────────────────────────────

    # Fix 1: class weights (used even with oversampling as a safety net)
    class_weight_dict = compute_class_weights(y_train)

    # Fix 2: oversample minority classes
    X_train, y_train = oversample_minority_classes(X_train, y_train)

    num_classes = metadata["num_classes"]
    print(f"\nInput shape: {X_train.shape[1:]}, Classes: {num_classes}")

    # Fix 3: focal loss enabled in model
    model = create_mobilenet(
        input_shape=X_train.shape[1:],
        num_classes=num_classes,
        use_focal_loss=True        # ← focal loss active
    )
    model.summary()

    # Early stopping to prevent overfitting on oversampled data
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        class_weight=class_weight_dict,   # ← Fix 1 applied here
        callbacks=callbacks
    )

    loss, acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"\nValidation Accuracy: {acc:.4f}")
    print(f"Validation Loss:     {loss:.4f}")

    plot_training_history(history)
    y_pred, y_probs = evaluate_model(model, X_val, y_val, metadata)

    model.save("mobilenet_once_classifier_balanced.keras")
    print("✓ Model saved → mobilenet_once_classifier_balanced.keras")


if __name__ == "__main__":
    main()