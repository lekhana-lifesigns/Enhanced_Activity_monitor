#!/usr/bin/env python3
"""
Validation harness for trained GRU temporal model vs decision tree baseline.

Compares:
  1. Per-class precision / recall / F1
  2. Cohen's kappa (inter-rater agreement)
  3. Temporal consistency (class-flip rate within 1-second windows)
  4. Confusion matrix

Usage:
    python evaluation/validate_temporal_model.py --data_dir data/training_collection
    python evaluation/validate_temporal_model.py --data_dir data/training_collection --model models/temporal/gru_optimized.pth
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("validate")

LABEL_NAMES = ["calm", "agitation", "restlessness", "delirium", "convulsion", "pain_response"]


def load_test_data(data_dir, window_size=48, feature_dim=9, test_ratio=0.1, seed=42):
    """Load .npz files and return the test split."""
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {data_dir}")

    features_list = []
    labels_list = []

    for fp in npz_files:
        try:
            data = np.load(fp, allow_pickle=True)
            feat = data["features"]
            label = int(data["label"])
            data.close()
            if feat.shape == (window_size, feature_dim) and 0 <= label <= 5:
                if not (np.isnan(feat).any() or np.isinf(feat).any()):
                    features_list.append(feat.astype(np.float32))
                    labels_list.append(label)
        except Exception:
            continue

    n = len(features_list)
    log.info("Loaded %d valid samples from %d files", n, len(npz_files))

    # Use last test_ratio as test set (deterministic with seed)
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    test_start = int(n * (1.0 - test_ratio))
    test_idx = indices[test_start:]

    test_features = [features_list[i] for i in test_idx]
    test_labels = [labels_list[i] for i in test_idx]
    log.info("Test set: %d samples", len(test_features))
    return test_features, test_labels


def predict_gru(model_path, test_features, device="cpu"):
    """Run GRU model predictions on test set."""
    import torch
    from pipeline.pose.temporal_model_enhanced import OptimizedTemporalModel

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = OptimizedTemporalModel(
        input_dim=config.get("input_dim", 9),
        hidden_dim=config.get("hidden_dim", 64),
        num_classes=config.get("num_classes", 6),
        num_heads=config.get("num_heads", 2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for feat in test_features:
            x = torch.FloatTensor(feat).unsqueeze(0).to(device)
            logits = model(x)
            preds.append(int(logits.argmax(dim=1).item()))
    return preds


def predict_decision_tree(test_features):
    """Run decision tree heuristic predictions on test set (last-frame features only)."""
    preds = []
    for feat_window in test_features:
        # Use last frame features for decision tree
        feat = feat_window[-1]
        motion_energy = float(feat[0])
        jerk_index = float(feat[1])
        motor_entropy = float(feat[6])
        motion_variability = float(feat[8])

        # Simplified decision tree thresholds (from activity_decision_tree.py)
        if motion_energy > 0.8 and motion_variability > 0.3:
            pred = 4  # convulsion
        elif motion_energy > 0.5 and motor_entropy > 0.6:
            pred = 1  # agitation
        elif motion_energy > 0.1 and motor_entropy > 0.4:
            pred = 2  # restlessness
        elif motor_entropy > 0.5 and motion_variability > 0.2:
            pred = 3  # delirium
        elif jerk_index > 0.3:
            pred = 5  # pain_response
        else:
            pred = 0  # calm
        preds.append(pred)
    return preds


def confusion_matrix(preds, targets, n_classes=6):
    """Compute confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(preds, targets):
        if 0 <= p < n_classes and 0 <= t < n_classes:
            cm[t][p] += 1
    return cm


def per_class_metrics(cm):
    """Compute per-class precision, recall, F1 from confusion matrix."""
    n = cm.shape[0]
    metrics = {}
    for i in range(n):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        metrics[LABEL_NAMES[i]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(cm[i, :].sum()),
        }
    return metrics


def cohens_kappa(preds_a, preds_b, n_classes=6):
    """Compute Cohen's kappa between two sets of predictions."""
    n = len(preds_a)
    if n == 0:
        return 0.0
    # Observed agreement
    p_o = sum(1 for a, b in zip(preds_a, preds_b) if a == b) / n
    # Expected agreement
    counts_a = np.bincount(preds_a, minlength=n_classes).astype(float) / n
    counts_b = np.bincount(preds_b, minlength=n_classes).astype(float) / n
    p_e = float(np.sum(counts_a * counts_b))
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def temporal_consistency(preds, fps=15):
    """Measure class-flip rate within 1-second windows.

    Returns the fraction of consecutive prediction pairs that change class.
    Lower = more temporally stable.
    """
    if len(preds) < 2:
        return 0.0
    flips = sum(1 for i in range(1, len(preds)) if preds[i] != preds[i - 1])
    return flips / (len(preds) - 1)


def main():
    parser = argparse.ArgumentParser(description="Validate GRU temporal model")
    parser.add_argument("--data_dir", default="data/training_collection")
    parser.add_argument("--model", default="models/temporal/gru_optimized.pth")
    parser.add_argument("--feature_dim", type=int, default=145, help="Feature dimension (145 with graph, 9 without)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="evaluation/validation_results.json")
    args = parser.parse_args()

    test_features, test_labels = load_test_data(args.data_dir, feature_dim=args.feature_dim)
    if len(test_features) < 10:
        log.error("Too few test samples (%d). Need at least 10.", len(test_features))
        sys.exit(1)

    results = {"n_samples": len(test_features)}

    # Decision tree baseline
    dt_preds = predict_decision_tree(test_features)
    dt_cm = confusion_matrix(dt_preds, test_labels)
    dt_accuracy = sum(1 for p, t in zip(dt_preds, test_labels) if p == t) / len(test_labels)
    dt_metrics = per_class_metrics(dt_cm)
    dt_flip_rate = temporal_consistency(dt_preds)

    results["decision_tree"] = {
        "accuracy": round(dt_accuracy, 4),
        "per_class": dt_metrics,
        "flip_rate": round(dt_flip_rate, 4),
        "confusion_matrix": dt_cm.tolist(),
    }
    log.info("Decision Tree: accuracy=%.1f%%, flip_rate=%.3f", dt_accuracy * 100, dt_flip_rate)

    # GRU model (if available)
    if os.path.exists(args.model):
        gru_preds = predict_gru(args.model, test_features, device=args.device)
        gru_cm = confusion_matrix(gru_preds, test_labels)
        gru_accuracy = sum(1 for p, t in zip(gru_preds, test_labels) if p == t) / len(test_labels)
        gru_metrics = per_class_metrics(gru_cm)
        gru_flip_rate = temporal_consistency(gru_preds)
        kappa = cohens_kappa(gru_preds, dt_preds)

        results["gru_model"] = {
            "model_path": args.model,
            "accuracy": round(gru_accuracy, 4),
            "per_class": gru_metrics,
            "flip_rate": round(gru_flip_rate, 4),
            "confusion_matrix": gru_cm.tolist(),
        }
        results["agreement"] = {
            "cohens_kappa": round(kappa, 4),
            "interpretation": (
                "almost_perfect" if kappa > 0.81 else
                "substantial" if kappa > 0.61 else
                "moderate" if kappa > 0.41 else
                "fair" if kappa > 0.21 else "slight"
            ),
        }
        log.info("GRU Model: accuracy=%.1f%%, flip_rate=%.3f", gru_accuracy * 100, gru_flip_rate)
        log.info("Cohen's Kappa (GRU vs DT): %.3f (%s)", kappa, results["agreement"]["interpretation"])

        # Print confusion matrix
        log.info("GRU Confusion Matrix:")
        header = "            " + "  ".join(f"{n[:5]:>6}" for n in LABEL_NAMES)
        log.info(header)
        for i, row in enumerate(gru_cm):
            row_str = "  ".join(f"{v:6d}" for v in row)
            log.info("  %-12s %s", LABEL_NAMES[i], row_str)
    else:
        log.warning("No trained GRU model at %s — skipping GRU evaluation", args.model)
        results["gru_model"] = None

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
