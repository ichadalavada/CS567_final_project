import pickle
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_pkl_dataset(pkl_dir="scooter_data/pkl_224", use_merged=False, merged_percentage=None):
    pkl_dir = Path(pkl_dir)

    if use_merged and merged_percentage:
        # Load merged Scooter+IDD training data
        merged_train_path = Path("../merged_datasets") / f"scooter_idd_{merged_percentage}_train.pkl"
        print(f"Loading merged dataset: {merged_train_path}")
        
        with open(merged_train_path, "rb") as f:
            train_data = pickle.load(f)
        
        # Use validation data from Scooter for validation
        with open(pkl_dir / "val.pkl", "rb") as f:
            val_data = pickle.load(f)
        
        with open(pkl_dir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        print(f"✓ Loaded merged train data: {len(train_data['images'])} images")
        print(f"✓ Loaded Scooter val data: {len(val_data['images'])} images")
    else:
        # Load regular Scooter data
        with open(pkl_dir / "train.pkl", "rb") as f:
            train_data = pickle.load(f)

        with open(pkl_dir / "val.pkl", "rb") as f:
            val_data = pickle.load(f)

        with open(pkl_dir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

    return train_data, val_data, metadata




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




def create_mobilenet(input_shape=(224, 224, 3), num_classes=4):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False  # freeze backbone

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def evaluate_model(model, X_val, y_val, metadata, test_name="Validation"):
    """Full evaluation with classification report and confusion matrix."""
    
    CLASS_NAMES = list(metadata["classes"].values())  # pulls from your metadata dict
    num_classes  = metadata["num_classes"]

    # Get predictions
    y_probs = model.predict(X_val, verbose=0)
    y_pred  = np.argmax(y_probs, axis=1)

    # Only report on classes that actually appear
    present_labels = sorted(set(y_val) | set(y_pred))
    present_names  = [CLASS_NAMES[i] for i in present_labels if i < len(CLASS_NAMES)]

    # Accuracy
    acc = accuracy_score(y_val, y_pred)
    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS — {test_name}")
    print(f"{'='*55}")
    print(f"  Accuracy: {acc*100:.2f}%")

    # Class distribution
    print(f"\n  Class distribution in {test_name}:")
    for cls_id, count in sorted(Counter(y_val.tolist()).items()):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        print(f"    {name:15s}: {count} samples")

    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(
        y_val, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))

    # Confusion matrix plot
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
    plt.title(f"Confusion Matrix — {test_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{test_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()
    print(f"  ✓ Confusion matrix saved")

    return y_pred, y_probs, acc

def main(use_merged=False, merged_percentage="5%", eval_on_idd=False):
    """
    Train and evaluate MobileNetV2 model
    
    Args:
        use_merged: If True, train on merged Scooter+IDD dataset
        merged_percentage: Which merged dataset to use ('1%', '5%', '10%')
        eval_on_idd: If True, also evaluate on IDD test data
    """
    
    # Load datasets
    train_data, val_data, metadata = load_pkl_dataset(
        "pkl_224",
        use_merged=use_merged,
        merged_percentage=merged_percentage
    )
    num_classes = metadata["num_classes"]

    X_train, y_train = convert_to_classification(train_data, num_classes=num_classes)
    X_val, y_val = convert_to_classification(val_data, num_classes=num_classes)

    # Normalize images
    X_train = X_train.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0

    

    print(f"\nTraining data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")

    model = create_mobilenet(
        input_shape=X_train.shape[1:],
        num_classes=num_classes
    )

    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32
    )

    loss, acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"\nValidation Accuracy: {acc:.4f}")
    print(f"Validation Loss:     {loss:.4f}")

    # ── Full evaluation on Scooter ───────────────────────────────────────
    y_pred, y_probs, acc_scooter = evaluate_model(
        model, X_val, y_val, metadata, 
        test_name="Scooter Validation"
    )

    # ── Optional evaluation on IDD test data ───────────────────────────
    if eval_on_idd:
        try:
            print("\nLoading IDD test data...")
            with open("../IDD/idd_test.pkl", "rb") as f:
                idd_test_data = pickle.load(f)
            
            X_idd_test, y_idd_test = convert_to_classification(idd_test_data)
            X_idd_test = X_idd_test.astype("float32") / 255.0
            
            print(f"IDD test data: {X_idd_test.shape}")
            
            y_pred_idd, y_probs_idd, acc_idd = evaluate_model(
                model, X_idd_test, y_idd_test, metadata,
                test_name="IDD Test"
            )
            
            print(f"\n{'='*55}")
            print(f"  CROSS-DOMAIN EVALUATION SUMMARY")
            print(f"{'='*55}")
            print(f"  Scooter Validation Accuracy: {acc_scooter*100:.2f}%")
            print(f"  IDD Test Accuracy:          {acc_idd*100:.2f}%")
            print(f"{'='*55}")
            
        except Exception as e:
            print(f"Could not evaluate on IDD: {e}")
    
    # Save model
    if use_merged:
        model_path = f"mobilenet_scooter_idd_{merged_percentage}_classifier.keras"
    else:
        model_path = "mobilenet_scooter_classifier.keras"
    
    model.save(model_path)
    print(f"✓ Model saved → {model_path}")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    use_merged = "--merged" in sys.argv
    eval_on_idd = "--eval-idd" in sys.argv
    
    # Get merged percentage if specified
    merged_percentage = "5%"  # default
    for arg in sys.argv:
        if arg.startswith("--percent="):
            merged_percentage = arg.split("=")[1]
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Use Merged Data: {use_merged}")
    if use_merged:
        print(f"  Merged Percentage: {merged_percentage}")
    print(f"  Evaluate on IDD: {eval_on_idd}")
    print(f"{'='*60}\n")
    
    main(use_merged=use_merged, merged_percentage=merged_percentage, eval_on_idd=eval_on_idd)