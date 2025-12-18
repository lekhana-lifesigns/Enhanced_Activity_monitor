# scripts/train_enhanced_temporal.py
"""
Training script for enhanced temporal model (Phase 1)
Trains the upgraded GRU + attention model on activity data
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pose.temporal_model_enhanced import EnhancedTemporalModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_temporal")


class ActivityDataset(Dataset):
    """Dataset for temporal activity classification."""
    
    def __init__(self, data_dir="data/training", window_size=48, feature_dim=13):
        self.data_dir = data_dir
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.samples = []
        self.labels = []
        
        # Load data (placeholder - replace with actual data loading)
        self._load_data()
    
    def _load_data(self):
        """Load training data."""
        # TODO: Implement actual data loading from files
        # For now, create dummy data for demonstration
        log.warning("Using dummy data - replace with actual data loading")
        
        # Generate dummy sequences
        num_samples = 1000
        for i in range(num_samples):
            # Random feature sequence
            seq = np.random.randn(self.window_size, self.feature_dim).astype(np.float32)
            self.samples.append(seq)
            
            # Random label (0-5 for 6 activity classes)
            label = np.random.randint(0, 6)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), torch.LongTensor([self.labels[idx]])[0]


def train_model(model, train_loader, val_loader, num_epochs=50, device="cpu"):
    """Train the enhanced temporal model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        log.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%"
        )
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "models/temporal/gru_enhanced.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            log.info(f"Saved best model to {save_path}")


def main():
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Create dataset
    train_dataset = ActivityDataset(data_dir="data/training")
    val_dataset = ActivityDataset(data_dir="data/validation")  # TODO: Separate validation set
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = EnhancedTemporalModel(
        input_dim=13,
        hidden_dim1=128,
        hidden_dim2=256,
        num_classes=6
    ).to(device)
    
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    train_model(model, train_loader, val_loader, num_epochs=50, device=device)
    
    log.info("Training complete!")


if __name__ == "__main__":
    main()

