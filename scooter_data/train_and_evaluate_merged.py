"""
Enhanced CNN training using merged Scooter + IDD datasets.
Trains on 1%, 5%, 10% IDD-augmented data and evaluates on both Scooter and IDD test sets.
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
    """Load a single pickle dataset file"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_all_test_data():
    """Load Scooter validation (as test) and IDD test data"""
    scooter_test_path = Path("pkl_224/val.pkl")
    idd_test_path = Path("/Users/ishachadalavada/Desktop/ml_final_project/IDD/idd_test.pkl")
    
    print("Loading test datasets...")
    scooter_test = load_pkl_dataset(str(scooter_test_path))
    
    # Load IDD test data
    try:
        idd_test = load_pkl_dataset(str(idd_test_path))
    except Exception as e:
        print(f"  Warning: Could not load IDD test data: {e}")
        idd_test = None
    
    return scooter_test, idd_test


def load_metadata():
    """Load Scooter metadata"""
    with open("pkl_224/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return metadata


def convert_to_classification(data, num_classes=4):
    """Convert detection data to classification format, filtering invalid labels"""
    images = []
    labels = []
    invalid_count = 0

    for img, anns in zip(data["images"], data["labels"]):
        if len(anns) == 0:
            continue  # skip images with no objects

        class_ids = [ann[0] for ann in anns]
        most_common = Counter(class_ids).most_common(1)[0][0]
        
        # Filter out invalid class indices
        if most_common < 0 or most_common >= num_classes:
            invalid_count += 1
            continue  # Skip samples with invalid class IDs

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
    """Create MobileNetV2-based classification model"""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False  # freeze backbone

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
    """Evaluate model on a test set and return metrics"""
    
    CLASS_NAMES = list(metadata["classes"].values())[:4]  # drop class 4 (traffic)
    
    # Ensure data is float32 and properly formatted
    X_test = np.asarray(X_test, dtype=np.float32)
    
    # Ensure model is in inference mode and compiled
    try:
        model.trainable = False
    except:
        pass
    
    # Get predictions with error handling
    try:
        y_probs = model.predict(X_test, verbose=0, batch_size=32)
    except Exception as e:
        print(f"  Warning: model.predict() failed with batch_size=32: {e}")
        print(f"  Trying batch_size=8...")
        try:
            y_probs = model.predict(X_test, verbose=0, batch_size=8)
        except Exception as e2:
            print(f"  Warning: model.predict() failed with batch_size=8: {e2}")
            print(f"  Trying batch-wise call operation...")
            # Fallback: predict using call
            batch_size = 8
            y_probs_list = []
            for i in range(0, len(X_test), batch_size):
                batch = X_test[i:i+batch_size]
                try:
                    batch_probs = model(batch, training=False).numpy()
                    y_probs_list.append(batch_probs)
                except Exception as e3:
                    print(f"    Batch {i//batch_size} failed: {e3}")
                    raise
            y_probs = np.vstack(y_probs_list)
    
    y_pred = np.argmax(y_probs, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get present labels
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names = [CLASS_NAMES[i] for i in present_labels if i < len(CLASS_NAMES)]
    
    print(f"\n{'='*60}")
    print(f"  Evaluation on {test_name}")
    print(f"{'='*60}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Total samples: {len(y_test)}")
    
    # Class distribution
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
    """Plot and save confusion matrix"""
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


def train_model(train_data, merged_dataset_name, epochs=15, batch_size=32,
                max_samples_per_class=5000, use_oversampling=True):
    """Train a model on merged dataset"""

    print(f"\n{'='*60}")
    print(f"  Training on {merged_dataset_name}")
    print(f"{'='*60}")

    num_classes = 4  # drop class 4 (traffic)
    
    # Convert to classification format
    X_train, y_train = convert_to_classification(train_data, num_classes=num_classes)
    
    # Normalize
    X_train = X_train.astype("float32") / 255.0
    
    print(f"  Original class distribution: {dict(sorted(Counter(y_train.tolist()).items()))}")
    
    # Fix 1: capped class weights
    class_weight_dict = compute_class_weights(y_train, max_weight=10.0)
    
    # Fix 2: oversample minority classes (with memory cap)
    if use_oversampling:
        X_train, y_train = oversample_minority_classes(X_train, y_train, 
                                                       max_samples_per_class=max_samples_per_class)
    else:
        print(f"  ⚠️  Skipping oversampling (relying on class weights + focal loss)")
    
    print(f"  Training data shape: {X_train.shape}")
    print(f"  Training labels shape: {y_train.shape}")
    
    # Create model with focal loss
    model = create_mobilenet(
        input_shape=X_train.shape[1:],
        num_classes=num_classes,
        use_focal_loss=True
    )
    
    # Add callbacks
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


def create_accuracy_matrix(results_df, save_path="accuracy_matrix.png"):
    """Create and visualize accuracy matrix"""
    
    # Pivot data for heatmap
    pivot_data = results_df.pivot_table(
        index="Training Data",
        columns="Test Data",
        values="Accuracy (%)"
    )
    
    # Plot heatmap
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
    plt.title("Model Accuracy Matrix\n(Trained on various Scooter+IDD mixes, tested on Scooter & IDD)")
    plt.ylabel("Training Dataset")
    plt.xlabel("Test Dataset")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Accuracy matrix saved → {save_path}")
    plt.close()
    
    return pivot_data


def plot_accuracy_comparison(results_df, save_path="accuracy_comparison.png"):
    """Create bar plot comparing accuracies across datasets"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scooter results
    scooter_data = results_df[results_df["Test Data"] == "Scooter"]
    axes[0].bar(scooter_data["Training Data"], scooter_data["Accuracy (%)"], color="steelblue", alpha=0.7)
    axes[0].set_title("Scooter Test Set Performance", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim([0, 105])
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(scooter_data["Accuracy (%)"]):
        axes[0].text(i, v + 2, f"{v:.2f}%", ha="center", va="bottom", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    
    # IDD results
    idd_data = results_df[results_df["Test Data"] == "IDD"]
    if len(idd_data) > 0:
        axes[1].bar(idd_data["Training Data"], idd_data["Accuracy (%)"], color="coral", alpha=0.7)
        axes[1].set_title("IDD Test Set Performance", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim([0, 105])
        axes[1].grid(axis="y", alpha=0.3)
        for i, v in enumerate(idd_data["Accuracy (%)"]):
            axes[1].text(i, v + 2, f"{v:.2f}%", ha="center", va="bottom", fontweight="bold")
        axes[1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Accuracy comparison plot saved → {save_path}")
    plt.close()


def main():
    # Setup paths
    base_path = Path("/Users/ishachadalavada/Desktop/ml_final_project")
    merged_dir = base_path / "merged_datasets"
    scooter_dir = Path(".")
    
    # Define datasets to train on
    datasets_to_train = [
        ("1%", merged_dir / "scooter_idd_1%_train.pkl"),
        ("5%", merged_dir / "scooter_idd_5%_train.pkl"),
        ("10%", merged_dir / "scooter_idd_10%_train.pkl"),
    ]
    
    # Load test data
    scooter_test, idd_test = load_all_test_data()
    metadata = load_metadata()
    
    # Convert test data to classification format (cap at 4 classes, drop traffic)
    num_classes = 4
    X_scooter_test, y_scooter_test = convert_to_classification(scooter_test, num_classes=num_classes)
    X_scooter_test = X_scooter_test.astype("float32") / 255.0
    
    if idd_test is not None and len(idd_test.get('images', [])) > 0:
        X_idd_test, y_idd_test = convert_to_classification(idd_test, num_classes=num_classes)
        X_idd_test = X_idd_test.astype("float32") / 255.0
        has_idd = len(X_idd_test) > 0
    else:
        X_idd_test = None
        y_idd_test = None
        has_idd = False
    
    print(f"Scooter test samples: {len(y_scooter_test)}")
    if has_idd:
        print(f"IDD test samples: {len(y_idd_test)}")
    else:
        print(f"IDD test data: NOT AVAILABLE (will skip IDD evaluation)")
    
    # Store results for accuracy matrix
    results = []
    models_dict = {}
    
    # Train models on different datasets
    for dataset_name, dataset_path in datasets_to_train:
        print(f"\n\n{'#'*60}")
        print(f"# Processing {dataset_name} merged dataset")
        print(f"{'#'*60}")
        
        try:
            # Load merged training data
            train_data = load_pkl_dataset(str(dataset_path))
            
            # Check dataset size
            num_train_samples = len(train_data['images'])
            print(f"  Dataset size: {num_train_samples} samples")
            
            # Adjust batch size if dataset is too small
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
                f"Scooter + IDD ({dataset_name})",
                epochs=15,
                batch_size=batch_size,
                max_samples_per_class=5000,  # Adjust this based on your RAM (4GB→2000, 8GB→3000, 16GB→5000, 32GB→10000)
                use_oversampling=True
            )
            
            models_dict[dataset_name] = model
            
            # Evaluate on Scooter test
            accuracy_scooter, y_pred_scooter, _, labels_scooter, names_scooter = evaluate_on_testset(
                model, X_scooter_test, y_scooter_test, 
                f"Scooter Test Set (n={len(y_scooter_test)})",
                metadata
            )
            
            results.append({
                "Training Data": f"Scooter + IDD ({dataset_name})",
                "Test Data": "Scooter",
                "Accuracy (%)": accuracy_scooter * 100,
                "Samples": len(y_scooter_test)
            })
            
            # Save Scooter confusion matrix
            plot_confusion_matrix(
                y_scooter_test, y_pred_scooter,
                f"Scooter Test - Trained on Scooter+IDD {dataset_name}",
                [list(metadata["classes"].values())[i] for i in labels_scooter],
                f"confusion_matrix_scooter_{dataset_name}.png"
            )
            
            # Evaluate on IDD test if available
            if has_idd:
                try:
                    accuracy_idd, y_pred_idd, _, labels_idd, names_idd = evaluate_on_testset(
                        model, X_idd_test, y_idd_test,
                        f"IDD Test Set (n={len(y_idd_test)})",
                        metadata
                    )
                    
                    results.append({
                        "Training Data": f"Scooter + IDD ({dataset_name})",
                        "Test Data": "IDD",
                        "Accuracy (%)": accuracy_idd * 100,
                        "Samples": len(y_idd_test)
                    })
                    
                    # Save IDD confusion matrix
                    plot_confusion_matrix(
                        y_idd_test, y_pred_idd,
                        f"IDD Test - Trained on Scooter+IDD {dataset_name}",
                        [list(metadata["classes"].values())[i] for i in labels_idd],
                        f"confusion_matrix_idd_{dataset_name}.png"
                    )
                except Exception as e:
                    print(f"\n  ⚠️  Error evaluating on IDD test set: {e}")
                    print(f"  Skipping IDD evaluation for {dataset_name}")
            else:
                print(f"\n  ⓘ IDD test data not available, skipping IDD evaluation")
            
            # Save model
            model_save_path = f"mobilenet_scooter_idd_{dataset_name}_classifier.keras"
            model.save(model_save_path)
            print(f"\n✓ Model saved → {model_save_path}")
        
        except Exception as e:
            print(f"\n  ❌ Error processing {dataset_name} dataset: {e}")
            print(f"  Skipping to next dataset...")
            import traceback
            traceback.print_exc()
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Print summary table
    print(f"\n\n{'='*80}")
    print(f"  ACCURACY SUMMARY TABLE")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))
    
    # Save CSV
    results_df.to_csv("accuracy_results_scooter.csv", index=False)
    print(f"\n✓ Results saved → accuracy_results_scooter.csv")
    
    # Create accuracy matrix visualization
    if not results_df.empty:
        pivot_table = create_accuracy_matrix(results_df)
        
        print(f"\n{'='*80}")
        print(f"  ACCURACY MATRIX (Pivot View)")
        print(f"{'='*80}\n")
        print(pivot_table)
        
        # Create comparison plot
        try:
            plot_accuracy_comparison(results_df)
        except Exception as e:
            print(f"  Warning: Could not create comparison plot: {e}")
    
    print(f"\n{'='*80}")
    print(f"  ✓ ALL TRAINING AND EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  - accuracy_results_scooter.csv")
    print(f"  - accuracy_matrix.png")
    print(f"  - accuracy_comparison.png")
    print(f"  - confusion_matrix_scooter_*.png")
    if has_idd:
        print(f"  - confusion_matrix_idd_*.png")
    print(f"  - mobilenet_scooter_idd_*_classifier.keras")


if __name__ == "__main__":
    main()