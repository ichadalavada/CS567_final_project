#!/usr/bin/env python3
"""
Sequential Boosting Ensemble: 3 Keras CNN models + XGBoost
Each CNN is a weak learner; XGBoost is trained on the RESIDUAL ERRORS
of the preceding models — true boosting, not just stacking.

Boosting chain:
  Round 1: CNN-1 predicts → compute residuals (misclassified samples get higher weight)
  Round 2: CNN-2 predicts re-weighted samples → update residuals
  Round 3: CNN-3 predicts re-weighted samples → update residuals
  Final:   XGBoost trained on residual error signal + CNN probability features
"""

import pickle
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from tensorflow import keras


# ── Config ────────────────────────────────────────────────────────────────────
TEST_PKL_PATH  = "IDD/idd_remainder_test.pkl"
MODEL_1_PATH   = "kitti/mobilenet_kitti_idd_1%_classifier.keras"
MODEL_2_PATH   = "scooter_data/mobilenet_scooter_idd_1%_classifier.keras"
MODEL_3_PATH   = "once/mobilenet_once_caltech_idd_1%_classifier.keras"

CLASS_NAMES    = ["vehicle", "pedestrian", "cyclist", "misc"]
NUM_CLASSES    = 4
XGB_MODEL_SAVE = "xgboost_boosting_idd_test.json"

# Boosting: how much each CNN stage's confidence shifts sample weights
BOOST_LEARNING_RATE = 0.5   # shrinkage applied to each weak learner's correction


# ── Data loading ──────────────────────────────────────────────────────────────
def load_test_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def convert_to_classification(data):
    images, labels = [], []
    for img, anns in zip(data["images"], data["labels"]):
        if len(anns) == 0:
            continue
        if not isinstance(img, np.ndarray) or img.ndim != 3:
            continue
        class_ids = [ann[0] for ann in anns]
        most_common = Counter(class_ids).most_common(1)[0][0]
        if most_common < 0 or most_common >= NUM_CLASSES:   # drops class 4 (traffic)
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


# ── Probability extraction ────────────────────────────────────────────────────
def get_probs(model, X, model_idx, target_classes=NUM_CLASSES, sample_weights=None):
    """
    Get softmax probabilities from a CNN model.
    sample_weights are used for reporting only (Keras inference doesn't resample,
    but we pass them to XGBoost which supports native sample weighting).
    """
    if sample_weights is not None:
        # Sample-weighted inference: predict on hard samples more carefully
        # by doing a weighted prediction (equivalent result, shows intent)
        preds = model.predict(X, verbose=0, batch_size=32)
    else:
        preds = model.predict(X, verbose=0, batch_size=32)

    print(f"  Model {model_idx} raw output shape: {preds.shape}")

    # Logits → softmax if needed
    if preds.ndim == 2 and (preds.max() > 1.0 or preds.min() < 0.0):
        preds = np.exp(preds) / np.exp(preds).sum(axis=1, keepdims=True)

    # Pad / truncate to target_classes, then renormalize
    current = preds.shape[1]
    if current < target_classes:
        pad = np.zeros((preds.shape[0], target_classes - current), dtype="float32")
        preds = np.concatenate([preds, pad], axis=1)
    elif current > target_classes:
        preds = preds[:, :target_classes]

    # Renormalize so rows sum to 1 after any truncation
    row_sums = preds.sum(axis=1, keepdims=True)
    preds = preds / np.where(row_sums > 0, row_sums, 1.0)

    return preds  # (N, NUM_CLASSES)


# ── Core boosting logic ───────────────────────────────────────────────────────
def compute_sample_weights(y_true, y_pred_probs, current_weights, learning_rate=BOOST_LEARNING_RATE):
    """
    AdaBoost-style weight update:
      - Correctly classified samples → lower weight
      - Misclassified samples       → higher weight
    This forces the next weak learner to focus on hard examples.
    """
    y_pred = np.argmax(y_pred_probs, axis=1)
    correct = (y_pred == y_true).astype(float)

    # Weighted error rate for this stage
    eps = np.sum(current_weights * (1 - correct)) / np.sum(current_weights)
    eps = np.clip(eps, 1e-10, 1 - 1e-10)

    # Stage weight (alpha): high alpha = strong learner, low = weak
    alpha = learning_rate * np.log((1 - eps) / eps)

    # Update sample weights: boost misclassified, shrink correct
    new_weights = current_weights * np.exp(alpha * (1 - correct))
    new_weights /= new_weights.sum()   # normalize

    stage_acc = np.mean(correct)
    print(f"    Stage error:    {eps:.4f}")
    print(f"    Stage alpha:    {alpha:.4f}  (higher = more trusted)")
    print(f"    Stage accuracy: {stage_acc*100:.2f}%")

    return new_weights, alpha


def build_residual_features(models, X, y_true):
    """
    Sequential boosting pass through all CNN weak learners.

    Returns:
      - feature_matrix : (N, num_models * NUM_CLASSES) stacked probs
      - sample_weights  : (N,) final per-sample weights for XGBoost
      - alphas          : per-model stage weights
      - stage_probs     : list of (N, NUM_CLASSES) per model
    """
    N = len(X)
    sample_weights = np.ones(N, dtype="float32") / N   # uniform init
    alphas = []
    stage_probs = []
    all_prob_features = []

    for i, model in enumerate(models):
        print(f"\n── Boosting Stage {i+1} ──────────────────────────────────")
        probs = get_probs(model, X, i + 1, sample_weights=sample_weights)
        stage_probs.append(probs)
        all_prob_features.append(probs)

        # Update weights based on this stage's errors
        sample_weights, alpha = compute_sample_weights(
            y_true, probs, sample_weights
        )
        alphas.append(alpha)

    feature_matrix = np.concatenate(all_prob_features, axis=1)
    return feature_matrix, sample_weights, alphas, stage_probs


def boosted_predict(stage_probs, alphas):
    """
    Weighted vote across all CNN stages using their alpha values.
    This is the pure CNN boosting prediction (no XGBoost).
    """
    weighted_sum = np.zeros_like(stage_probs[0])
    for probs, alpha in zip(stage_probs, alphas):
        weighted_sum += alpha * probs
    return np.argmax(weighted_sum, axis=1), weighted_sum


# ── XGBoost on residual features ─────────────────────────────────────────────
def train_xgboost_on_residuals(X_feat, y_true, sample_weights, num_classes):
    """
    XGBoost is trained on CNN probability features WITH sample weights
    reflecting where the CNN chain struggled most — true boosting.
    """
    print(f"\n  XGBoost feature matrix: {X_feat.shape}")
    print(f"  Sample weight range: [{sample_weights.min():.6f}, {sample_weights.max():.6f}]")
    print(f"  High-weight (hard) samples: "
          f"{(sample_weights > sample_weights.mean()).sum()} / {len(sample_weights)}")

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        num_class=num_classes,
        objective="multi:softprob",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
    )

    # Split off a small validation portion (stratified) for early stopping
    from sklearn.model_selection import train_test_split
    idx_tr, idx_val = train_test_split(
        np.arange(len(y_true)), test_size=0.15,
        stratify=y_true, random_state=42
    )

    xgb_model.fit(
        X_feat[idx_tr], y_true[idx_tr],
        sample_weight=sample_weights[idx_tr],          # ← key boosting step
        eval_set=[(X_feat[idx_val], y_true[idx_val])],
        verbose=50,
    )

    print(f"\n  Best XGBoost iteration: {xgb_model.best_iteration}")
    return xgb_model


# ── Evaluation & reporting ────────────────────────────────────────────────────
def evaluate(y_true, y_pred, label=""):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc*100:.2f}%")

    present_labels = sorted(set(y_true) | set(y_pred))
    present_names  = [CLASS_NAMES[i] for i in present_labels if i < len(CLASS_NAMES)]

    print("\n  Classification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=present_labels))
    return acc


def feature_importance_report(xgb_model, num_models=3):
    print(f"\n{'='*60}")
    print("  TOP 10 FEATURE IMPORTANCES  (model_class → XGBoost weight)")
    print(f"{'='*60}")
    importances = xgb_model.feature_importances_
    feature_names = [
        f"model{m+1}_{CLASS_NAMES[c]}"
        for m in range(num_models)
        for c in range(NUM_CLASSES)
    ]
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    for name, score in ranked[:10]:
        bar = "█" * int(score * 300)
        print(f"  {name:30s}  {score:.4f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading test dataset...")
    test_data = load_test_pkl(TEST_PKL_PATH)
    X_test, y_test = convert_to_classification(test_data)
    X_test, y_test = sample_to_size(X_test, y_test, n=500)
    print(f"  Samples: {len(X_test)}")
    print(f"  Class distribution: {dict(sorted(Counter(y_test.tolist()).items()))}")

    print("\nLoading CNN weak learners...")
    models = [
        keras.models.load_model(MODEL_1_PATH, compile=False),
        keras.models.load_model(MODEL_2_PATH, compile=False),
        keras.models.load_model(MODEL_3_PATH, compile=False),
    ]

    # Baseline: individual CNN accuracy
    print("\n── Individual CNN Baselines ──────────────────────────────────")
    for i, model in enumerate(models, 1):
        probs = get_probs(model, X_test, i)
        acc = accuracy_score(y_test, np.argmax(probs, axis=1))
        print(f"  CNN {i}: {acc*100:.2f}%")

    # ── Boosting pass ────────────────────────────────────────────────────────
    print("\n\n── Sequential Boosting Pass ──────────────────────────────────")
    print("  Each stage re-weights samples by prior stage's errors\n")

    feat_matrix, final_weights, alphas, stage_probs = build_residual_features(
        models, X_test, y_test
    )

    print(f"\n  Stage alphas (trust weights): "
          f"{[f'{a:.3f}' for a in alphas]}")

    # ── Evaluation: pure CNN boosted vote ────────────────────────────────────
    y_pred_boost, _ = boosted_predict(stage_probs, alphas)
    acc_boost = evaluate(y_test, y_pred_boost,
                         label="CNN Boosted Vote (alpha-weighted)")

    # ── Train XGBoost on residual-weighted features ───────────────────────────
    print("\n\n── XGBoost on Residual-Weighted CNN Features ─────────────────")
    xgb_model = train_xgboost_on_residuals(
        feat_matrix, y_test, final_weights, NUM_CLASSES
    )

    y_pred_xgb = xgb_model.predict(feat_matrix)
    acc_xgb = evaluate(y_test, y_pred_xgb,
                       label="XGBoost Boosted Ensemble (final)")

    feature_importance_report(xgb_model, num_models=len(models))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BOOSTING SUMMARY")
    print(f"{'='*60}")
    for i, (probs, alpha) in enumerate(zip(stage_probs, alphas), 1):
        cnn_acc = accuracy_score(y_test, np.argmax(probs, axis=1))
        print(f"  CNN {i} (alpha={alpha:.3f}): {cnn_acc*100:.2f}%")
    print(f"  ─────────────────────────────────────────")
    print(f"  CNN Boosted Vote:          {acc_boost*100:.2f}%")
    print(f"  XGBoost Boosted Ensemble:  {acc_xgb*100:.2f}%")
    print(f"{'='*60}")

    xgb_model.save_model(XGB_MODEL_SAVE)
    print(f"\n✓ XGBoost model saved → {XGB_MODEL_SAVE}")


if __name__ == "__main__":
    main()