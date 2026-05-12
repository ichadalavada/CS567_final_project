"""
Test a Keras model on the IDD remainder test dataset.

Usage:
    python test_idd_remainder.py --model path/to/model.keras
    python test_idd_remainder.py --model scooter_data/mobilenet_scooter_classifier.keras
    python test_idd_remainder.py --model scooter_data/mobilenet_scooter_classifier.keras --pkl IDD/idd_remainder_test.pkl
"""

import argparse
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate a Keras model on IDD remainder test set.")
parser.add_argument("--model", required=True, help="Path to Keras model (.keras or .h5)")
parser.add_argument("--pkl", default="IDD/idd_remainder_test_filtered.pkl", help="Path to test pkl file (default: filtered version)")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference (default: 32)")
parser.add_argument("--plot", action="store_true", help="Generate confusion matrix plot")
args = parser.parse_args()

# ── Load Keras model ──────────────────────────────────────────────────────────
print(f"\n[1/3] Loading Keras model from: {args.model}")
if not Path(args.model).exists():
    raise FileNotFoundError(f"Model file not found: {args.model}")

# Load model without compiling to avoid custom loss function issues
try:
    keras_model = tf.keras.models.load_model(args.model, compile=False)
except Exception as e:
    print(f"   Warning: Could not load with compile=False. Trying standard load...")
    try:
        keras_model = tf.keras.models.load_model(args.model)
    except Exception as e2:
        print(f"   Error loading model: {e2}")
        raise

print(f"   Model loaded successfully")
keras_model.summary()

# ── Load IDD remainder test data ──────────────────────────────────────────────
pkl_path = args.pkl
print(f"\n[2/3] Loading test data from: {pkl_path}")
if not Path(pkl_path).exists():
    raise FileNotFoundError(f"Test data file not found: {pkl_path}")

with open(pkl_path, "rb") as f:
    pkl_data = pickle.load(f)

X_test = pkl_data["images"]  # shape: (8282, 224, 224, 3), uint8
labels_raw = pkl_data["labels"]  # list of 8282 label lists (bounding boxes)
image_names = pkl_data["image_names"]  # list of image names

# Extract class label from bounding boxes (use the first bbox or most common class)
# Each bbox is [class_id, center_x, center_y, width, height]
y_test = []
for img_bboxes in labels_raw:
    if len(img_bboxes) > 0:
        # Get the class of the first bounding box
        class_id = int(img_bboxes[0][0])
        y_test.append(class_id)
    else:
        # No bounding boxes, assign to "unknown" class (e.g., -1 or 0)
        y_test.append(0)

X_test = np.array(X_test, dtype=np.float32) / 255.0  # Normalize to [0, 1]
y_test = np.array(y_test, dtype=int)

print(f"   X_test shape : {X_test.shape}")
print(f"   y_test shape : {y_test.shape}")
print(f"   Data type (after normalization): {X_test.dtype}")
print(f"   Pixel range: [{X_test.min():.4f}, {X_test.max():.4f}]")
print(f"   Unique classes in labels: {np.unique(y_test)}")
print(f"   Class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# ── Run inference ─────────────────────────────────────────────────────────────
print(f"\n[3/3] Running inference (batch_size={args.batch_size})...")

# Determine the number of output classes from the model
output_shape = keras_model.output_shape
if isinstance(output_shape, list):
    num_output_classes = output_shape[-1][-1] if len(output_shape[-1]) > 1 else int(output_shape[-1][-1])
else:
    num_output_classes = output_shape[-1] if len(output_shape) > 1 else int(output_shape[-1])

print(f"   Model output classes: {num_output_classes}")
print(f"   Unique labels in test set: {np.unique(y_test)}")
print(f"   Max label in test set: {np.max(y_test)}")

# Filter out samples with invalid class labels (outside model's output range)
valid_idx = y_test < num_output_classes
X_test_valid = X_test[valid_idx]
y_test_valid = y_test[valid_idx]

invalid_count = np.sum(~valid_idx)
if invalid_count > 0:
    print(f"   Warning: Filtered out {invalid_count} samples with invalid class labels")
    print(f"   Proceeding with {len(y_test_valid)}/{len(y_test)} valid samples")

y_pred_probs = keras_model.predict(X_test_valid, batch_size=args.batch_size, verbose=1)

# Convert probabilities → class indices
if y_pred_probs.shape[-1] > 1:
    # Multi-class (softmax output)
    y_pred = np.argmax(y_pred_probs, axis=1)
else:
    # Binary (sigmoid output)
    y_pred = (y_pred_probs.squeeze() >= 0.5).astype(int)

# ── Compute metrics ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS - IDD Remainder Test Set")
print("=" * 70)

accuracy = np.mean(y_pred == y_test_valid)
print(f"\n  Overall Accuracy : {accuracy * 100:.2f}%  ({np.sum(y_pred == y_test_valid)}/{len(y_test_valid)} correct)")

# Per-class breakdown
num_classes = len(np.unique(y_test_valid))
print(f"\n  Class distribution and per-class accuracy:")
print(f"  {'Class':>10} | {'Samples':>8} | {'Correct':>8} | {'Accuracy':>10}")
print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

for cls in sorted(np.unique(y_test_valid)):
    count = np.sum(y_test_valid == cls)
    correct = np.sum((y_test_valid == cls) & (y_pred == cls))
    class_acc = correct / count if count > 0 else 0
    print(f"  {cls:>10} | {count:>8} | {correct:>8} | {class_acc*100:>9.2f}%")

# Classification report
print(f"\n  Detailed Classification Report:")
print(classification_report(y_test_valid, y_pred, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test_valid, y_pred)
print(f"\n  Confusion Matrix shape: {cm.shape}")

# Try model.evaluate if available
print(f"\n  Keras model.evaluate():")
try:
    y_test_eval = y_test_valid if y_test_valid.ndim == 1 else np.eye(num_output_classes)[y_test_valid]
    results = keras_model.evaluate(X_test_valid, y_test_eval,
                                   batch_size=args.batch_size, verbose=0)
    metric_names = keras_model.metrics_names if hasattr(keras_model, 'metrics_names') else ['loss', 'accuracy']
    for name, val in zip(metric_names, results if isinstance(results, (list, tuple)) else [results]):
        print(f"    {name}: {val:.4f}")
except Exception as e:
    print(f"    (Skipped — {e})")

print("=" * 70)

# ── Optional: Generate confusion matrix plot ──────────────────────────────────
if args.plot:
    print(f"\n  Generating confusion matrix plot...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f"Confusion Matrix - IDD Remainder Test\nAccuracy: {accuracy*100:.2f}% (n={len(y_test_valid)})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    output_file = f"confusion_matrix_{Path(args.model).stem}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_file}")
    plt.close()

print("\nDone!")
