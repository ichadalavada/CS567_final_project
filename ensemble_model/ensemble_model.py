#!/usr/bin/env python3
"""
Ensemble of three Keras CNN models for IDD test dataset classification
Tests multiple ensemble strategies (averaging, voting, weighted)
"""

import pickle
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras


# Config
TEST_PKL_PATH = "IDD/idd_remainder_test.pkl"
MODEL_1_PATH  = "kitti/mobilenet_kitti_idd_1%_classifier.keras"
MODEL_2_PATH  = "scooter_data/mobilenet_scooter_idd_1%_classifier.keras"
MODEL_3_PATH  = "once/mobilenet_once_caltech_idd_1%_classifier.keras"

CLASS_NAMES   = ["vehicle", "pedestrian", "cyclist", "misc"]
NUM_CLASSES   = 4


def load_test_pkl(pkl_path):
    """Load single IDD test pickle file"""
    with open(pkl_path, "rb") as f:
        test_data = pickle.load(f)
    return test_data


def convert_to_classification(data):
    images, labels = [], []
    for img, anns in zip(data["images"], data["labels"]):
        if len(anns) == 0:
            continue
        if not isinstance(img, np.ndarray) or img.ndim != 3:
            continue
        class_ids = [ann[0] for ann in anns]
        most_common = Counter(class_ids).most_common(1)[0][0]
        if most_common >= NUM_CLASSES:   # drop class 4 (traffic) and above
            continue
        images.append(img)
        labels.append(most_common)
    return np.stack(images).astype("float32") / 255.0, np.array(labels)


def sample_to_size(X, y, n=1000, seed=42):
    """Resample each class to exactly n examples, oversampling if needed."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        idx.append(rng.choice(c_idx, n, replace=len(c_idx) < n))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def get_probs(model, X, model_idx, target_classes):
    """Get softmax probabilities from a model, normalizing output as needed."""
    preds = model.predict(X, verbose=0)
    print(f"  Model {model_idx} raw output shape: {preds.shape}")

    # 1D output (class indices) -> one-hot
    if preds.ndim == 1:
        print(f"  Model {model_idx}: converting class indices to one-hot")
        n_cls = int(preds.max()) + 1
        one_hot = np.zeros((len(preds), n_cls), dtype="float32")
        one_hot[np.arange(len(preds)), preds.astype(int)] = 1.0
        preds = one_hot

    # Logits -> softmax
    elif preds.ndim == 2 and (preds.max() > 1.0 or preds.min() < 0.0):
        print(f"  Model {model_idx}: applying softmax to logits")
        preds = np.exp(preds) / np.exp(preds).sum(axis=1, keepdims=True)

    # Pad or truncate to target number of classes
    current = preds.shape[1]
    if current < target_classes:
        print(f"  Model {model_idx}: padding {current} to {target_classes} classes")
        pad = np.zeros((preds.shape[0], target_classes - current), dtype="float32")
        preds = np.concatenate([preds, pad], axis=1)
    elif current > target_classes:
        print(f"  Model {model_idx}: truncating {current} to {target_classes} classes")
        preds = preds[:, :target_classes]

    return preds


def ensemble_predict(models, X, num_classes=NUM_CLASSES, strategy="average"):
    print(f"\n  Running ensemble ({strategy}) over {len(models)} models...")
    all_probs = [
        get_probs(model, X, i + 1, num_classes)
        for i, model in enumerate(models)
    ]

    if strategy == "average":
        mean_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
        return np.argmax(mean_probs, axis=1), mean_probs

    elif strategy == "vote":
        all_preds = np.stack([np.argmax(p, axis=1) for p in all_probs], axis=1)
        voted = np.apply_along_axis(
            lambda row: Counter(row).most_common(1)[0][0], 1, all_preds
        )
        return voted, None

    elif strategy == "weighted":
        # Tune these - must sum to 1 and match number of models
        weights = [0.4, 0.3, 0.3]
        assert len(weights) == len(models), "One weight per model required"
        weighted_probs = sum(w * p for w, p in zip(weights, all_probs))
        return np.argmax(weighted_probs, axis=1), weighted_probs

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def evaluate(y_true, y_pred, class_names, strategy_name=""):
    acc = np.mean(y_true == y_pred)
    print(f"\n{'='*55}")
    print(f"  Ensemble Results ({strategy_name})")
    print(f"{'='*55}")
    print(f"  Accuracy: {acc*100:.2f}%")

    present_labels = sorted(set(y_true) | set(y_pred))
    present_names  = [class_names[i] for i in present_labels if i < len(class_names)]

    print("\n  Classification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=present_labels))


def main():
    # Load test data
    print("Loading test data...")
    test_data = load_test_pkl(TEST_PKL_PATH)
    X_test, y_test = convert_to_classification(test_data)
    X_test, y_test = sample_to_size(X_test, y_test, n=500)
    print(f"  Test samples: {len(X_test)}")
    print(f"  Class distribution: {Counter(y_test.tolist())}")

    # Load models
    print("\nLoading models...")
    model_1 = keras.models.load_model(MODEL_1_PATH, compile=False)
    model_2 = keras.models.load_model(MODEL_2_PATH, compile=False)
    model_3 = keras.models.load_model(MODEL_3_PATH, compile=False)
    models  = [model_1, model_2, model_3]
    print(f"  Loaded {len(models)} models")

    # Individual model baselines
    print("\nIndividual model accuracies on test set:")
    for i, model in enumerate(models, 1):
        preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
        acc = np.mean(y_test == preds)
        print(f"  Model {i}: {acc*100:.2f}%")

    # Ensemble strategies
    for strategy in ["average", "vote", "weighted"]:
        y_pred, probs = ensemble_predict(models, X_test,
                                         num_classes=NUM_CLASSES,
                                         strategy=strategy)
        evaluate(y_test, y_pred, CLASS_NAMES, strategy_name=strategy)


if __name__ == "__main__":
    main()
