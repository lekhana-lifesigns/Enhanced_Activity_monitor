# scripts/train_enhanced_temporal.py
"""
Training script for optimized GRU temporal model.
Trains on .npz files collected by TrainingDataCollector.

Usage:
    python scripts/train_enhanced_temporal.py --data_dir data/training_collection
    python scripts/train_enhanced_temporal.py --data_dir data/training_collection --epochs 100 --device cuda
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pose.temporal_model_enhanced import OptimizedTemporalModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_temporal")

LABEL_NAMES = ["calm", "agitation", "restlessness", "delirium", "convulsion", "pain_response"]


class ActivityDataset(Dataset):
    """Dataset for temporal activity classification from collected .npz files."""

    def __init__(self, file_paths, window_size=48, feature_dim=9):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.samples = []
        self.labels = []

        for fp in file_paths:
            try:
                data = np.load(fp, allow_pickle=True)
                features = data["features"]
                label = int(data["label"])
                data.close()

                if features.shape != (window_size, feature_dim):
                    continue
                if np.isnan(features).any() or np.isinf(features).any():
                    continue
                if not 0 <= label <= 5:
                    continue

                self.samples.append(features.astype(np.float32))
                self.labels.append(label)
            except Exception:
                continue

        log.info("Loaded %d valid samples from %d files", len(self.samples), len(file_paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), torch.LongTensor([self.labels[idx]])[0]

    def get_class_weights(self):
        """Compute inverse-frequency class weights for imbalanced data."""
        counts = np.bincount(self.labels, minlength=6).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)
        return torch.FloatTensor(weights)

    def get_class_distribution(self):
        """Return class distribution string for logging."""
        counts = np.bincount(self.labels, minlength=6)
        parts = [f"{LABEL_NAMES[i]}: {counts[i]}" for i in range(6)]
        return ", ".join(parts)


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification (Lin et al., 2017)."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def split_files(data_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Load all .npz files and split into train/val/test."""
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    log.info("Found %d .npz files in %s", len(npz_files), data_dir)

    # Read labels for stratified split
    labels = []
    valid_files = []
    for fp in npz_files:
        try:
            data = np.load(fp, allow_pickle=True)
            label = int(data["label"])
            data.close()
            if 0 <= label <= 5:
                valid_files.append(fp)
                labels.append(label)
        except Exception:
            continue

    rng = np.random.RandomState(seed)
    indices = np.arange(len(valid_files))
    rng.shuffle(indices)

    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_files = [valid_files[i] for i in train_idx]
    val_files = [valid_files[i] for i in val_idx]
    test_files = [valid_files[i] for i in test_idx]

    log.info("Split: train=%d, val=%d, test=%d", len(train_files), len(val_files), len(test_files))
    return train_files, val_files, test_files


def evaluate(model, loader, criterion, device):
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy, all_preds, all_targets


def train_model(model, train_loader, val_loader, num_epochs=100, device="cpu",
                class_weights=None, patience=15, max_grad_norm=1.0,
                save_dir="models/temporal"):
    """Train the optimized temporal model with focal loss, gradient clipping, early stopping."""

    if class_weights is not None:
        criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping — prevents GRU gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)

        scheduler.step()

        train_acc = 100.0 * train_correct / max(train_total, 1)
        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validation
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)

        log.info(
            "Epoch %d/%d | Train Loss: %.4f Acc: %.1f%% | Val Loss: %.4f Acc: %.1f%% | LR: %.6f",
            epoch + 1, num_epochs, avg_train_loss, train_acc, val_loss, val_acc,
            optimizer.param_groups[0]['lr']
        )

        # Confusion matrix every 10 epochs
        if (epoch + 1) % 10 == 0 and val_targets:
            _log_confusion_matrix(val_preds, val_targets)

        # Early stopping with best model save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "gru_optimized.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': {
                    'input_dim': model.input_dim,
                    'hidden_dim': model.hidden_dim,
                    'num_classes': model.num_classes,
                    'architecture': 'OptimizedTemporalModel',
                }
            }, save_path)
            log.info("Saved best model (epoch %d, val_loss=%.4f) to %s", epoch + 1, val_loss, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info("Early stopping at epoch %d (best was epoch %d)", epoch + 1, best_epoch)
                break

    return best_val_loss, best_epoch


def _log_confusion_matrix(preds, targets):
    """Log confusion matrix and per-class metrics."""
    cm = np.zeros((6, 6), dtype=int)
    for p, t in zip(preds, targets):
        if 0 <= p < 6 and 0 <= t < 6:
            cm[t][p] += 1

    log.info("Confusion Matrix:")
    header = "            " + "  ".join(f"{n[:4]:>5}" for n in LABEL_NAMES)
    log.info(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:5d}" for v in row)
        log.info(f"  {LABEL_NAMES[i]:<12} {row_str}")

    # Per-class accuracy
    for i in range(6):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = 100.0 * correct / max(total, 1)
        log.info("  %s: %d/%d (%.1f%%)", LABEL_NAMES[i], correct, total, acc)


def main():
    parser = argparse.ArgumentParser(description="Train optimized GRU temporal model")
    parser.add_argument("--data_dir", default="data/training_collection", help="Directory with .npz files")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="GRU hidden dimension")
    parser.add_argument("--num_heads", type=int, default=2, help="Attention heads")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--feature_dim", type=int, default=145, help="Feature dimension (145 with graph features, 9 without)")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    parser.add_argument("--save_dir", default="models/temporal", help="Model save directory")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("Using device: %s", device)

    # Load and split data
    train_files, val_files, test_files = split_files(args.data_dir)

    train_dataset = ActivityDataset(train_files, window_size=48, feature_dim=args.feature_dim)
    val_dataset = ActivityDataset(val_files, window_size=48, feature_dim=args.feature_dim)
    test_dataset = ActivityDataset(test_files, window_size=48, feature_dim=args.feature_dim)

    if len(train_dataset) == 0:
        log.error("No valid training samples found. Collect data first with collect_training_data: true")
        sys.exit(1)

    log.info("Train distribution: %s", train_dataset.get_class_distribution())
    class_weights = train_dataset.get_class_weights()
    log.info("Class weights: %s", class_weights.numpy().round(2))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create optimized model
    model = OptimizedTemporalModel(
        input_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=6,
        num_heads=args.num_heads,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    log.info("Model: OptimizedTemporalModel (%d params, %.1f KB)", param_count, param_count * 4 / 1024)

    # Train
    best_loss, best_epoch = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs, device=device,
        class_weights=class_weights, patience=args.patience,
        save_dir=args.save_dir,
    )

    # Final test evaluation
    if len(test_dataset) > 0:
        # Load best model for test
        best_path = os.path.join(args.save_dir, "gru_optimized.pth")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

        criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
        test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
        log.info("TEST RESULTS: Loss=%.4f, Accuracy=%.1f%%", test_loss, test_acc)
        _log_confusion_matrix(test_preds, test_targets)

    log.info("Training complete. Best model at epoch %d (val_loss=%.4f)", best_epoch, best_loss)


if __name__ == "__main__":
    main()
