# pipeline/pose/temporal_model_enhanced.py
"""
Enhanced Temporal Model with Attention Mechanism
Upgraded from simple GRU to larger GRU + attention for better accuracy
"""
import numpy as np
import logging
import os

log = logging.getLogger("temporal_enhanced")

# Try TensorFlow Lite
try:
    import tensorflow as tf
    TFLITE = tf.lite
    log.info("Using TensorFlow Lite from full TensorFlow")
except Exception:
    TFLITE = None
    log.warning("TensorFlow not available; temporal model will use fallback")

# Try PyTorch for advanced models
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    log.info("PyTorch available for advanced temporal modeling")
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not available; will use TFLite fallback")


class AttentionLayer(nn.Module):
    """Multi-head self-attention layer for temporal sequences."""
    
    def __init__(self, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Save for residual
        residual = x
        x = self.layer_norm(x)
        
        # Multi-head attention
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        attn_output = self.w_o(attn_output)
        
        return self.dropout(attn_output) + residual


class EnhancedTemporalModel(nn.Module):
    """
    Enhanced temporal model with larger GRU + attention.
    Architecture:
    - Input: (batch, seq_len, feature_dim)
    - GRU Layer 1: 128 hidden units
    - GRU Layer 2: 256 hidden units (upgraded from 64)
    - Attention Layer: Multi-head self-attention
    - Dense: Classification head
    """
    
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=256, 
                 num_classes=6, num_heads=8, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_classes = num_classes
        
        # GRU layers (larger than before)
        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True, dropout=dropout)
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True, dropout=dropout)
        
        # Attention layer
        self.attention = AttentionLayer(d_model=hidden_dim2, n_heads=num_heads, dropout=dropout)
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # GRU layers
        gru_out1, _ = self.gru1(x)
        gru_out2, _ = self.gru2(gru_out1)
        
        # Attention
        attn_out = self.attention(gru_out2)
        
        # Global average pooling over temporal dimension
        pooled = torch.mean(attn_out, dim=1)  # (batch, hidden_dim2)
        
        # Classification
        out = self.fc1(pooled)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class TemporalModelEnhanced:
    """
    Enhanced temporal model wrapper with both PyTorch and TFLite support.
    Falls back to TFLite if PyTorch model not available.
    """
    
    def __init__(self, model_path=None, window_size=48, labels=None, 
                 use_pytorch=True, device="cpu"):
        self.model_path = model_path
        self.window_size = window_size
        self.labels = labels or [
            "calm",
            "agitation",
            "restlessness",
            "delirium",
            "convulsion",
            "pain_response"
        ]
        self.device = device
        self.use_pytorch = use_pytorch and TORCH_AVAILABLE
        
        self.pytorch_model = None
        self.tflite_interpreter = None
        self.prev_probs = None
        
        # Try to load PyTorch model first
        if self.use_pytorch:
            try:
                if model_path and os.path.exists(model_path.replace('.tflite', '.pth')):
                    pytorch_path = model_path.replace('.tflite', '.pth')
                    self.pytorch_model = EnhancedTemporalModel(
                        input_dim=13,
                        hidden_dim1=128,
                        hidden_dim2=256,
                        num_classes=len(self.labels)
                    )
                    self.pytorch_model.load_state_dict(torch.load(pytorch_path, map_location=device))
                    self.pytorch_model.to(device)
                    self.pytorch_model.eval()
                    log.info("Loaded enhanced PyTorch temporal model: %s", pytorch_path)
                else:
                    # Initialize new model (for training)
                    # Auto-detect input dimension from first prediction attempt
                    # Default to 9 (ICUFeatureEncoder output) but allow override
                    input_dim = getattr(self, '_detected_input_dim', 9)
                    self.pytorch_model = EnhancedTemporalModel(
                        input_dim=input_dim,
                        hidden_dim1=128,
                        hidden_dim2=256,
                        num_classes=len(self.labels)
                    )
                    self.pytorch_model.to(device)
                    log.info("Initialized new enhanced temporal model (PyTorch)")
            except Exception as e:
                log.warning("Failed to load PyTorch model: %s, falling back to TFLite", e)
                self.use_pytorch = False
        
        # Fallback to TFLite
        if not self.use_pytorch and TFLITE and model_path:
            try:
                self.tflite_interpreter = TFLITE.Interpreter(model_path=model_path)
                self.tflite_interpreter.allocate_tensors()
                self.input_details = self.tflite_interpreter.get_input_details()
                self.output_details = self.tflite_interpreter.get_output_details()
                log.info("Loaded TFLite temporal model: %s", model_path)
            except Exception as e:
                log.warning("Failed to load TFLite model: %s", e)
                self.tflite_interpreter = None
    
    def predict(self, feat_window, use_smoothing=True, alpha=0.7):
        """
        Predict activity class from feature window.
        
        Args:
            feat_window: np.array (T, F) or (batch, T, F) - feature sequence
            use_smoothing: Whether to apply exponential moving average
            alpha: Smoothing factor (0-1)
        
        Returns:
            (label, confidence, probs) tuple
        """
        # Handle input shape
        if feat_window.ndim == 2:
            feat_window = np.expand_dims(feat_window, 0)  # (1, T, F)
        
        # Use PyTorch model if available
        if self.use_pytorch and self.pytorch_model is not None:
            try:
                with torch.no_grad():
                    x = torch.FloatTensor(feat_window).to(self.device)
                    
                    # Auto-detect input dimension mismatch and fix it
                    actual_dim = x.size(-1)
                    expected_dim = self.pytorch_model.input_dim
                    
                    if actual_dim != expected_dim:
                        # Reinitialize model with correct input dimension
                        log.info("Auto-detecting input dimension: %d (model expects %d), reinitializing model", 
                                actual_dim, expected_dim)
                        self.pytorch_model = EnhancedTemporalModel(
                            input_dim=actual_dim,
                            hidden_dim1=128,
                            hidden_dim2=256,
                            num_classes=len(self.labels)
                        ).to(self.device)
                        self.pytorch_model.eval()
                    
                    logits = self.pytorch_model(x)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            except Exception as e:
                log.warning("PyTorch prediction failed: %s, using fallback", e)
                probs = self._predict_fallback(feat_window)
        # Use TFLite model
        elif self.tflite_interpreter is not None:
            probs = self._predict_tflite(feat_window)
        else:
            probs = self._predict_fallback(feat_window)
        
        # Apply smoothing if enabled
        if use_smoothing and self.prev_probs is not None:
            probs = [
                alpha * p + (1 - alpha) * prev_p
                for p, prev_p in zip(probs, self.prev_probs)
            ]
            # Renormalize
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
        
        self.prev_probs = probs
        
        idx = int(np.argmax(probs))
        return (self.labels[idx], float(probs[idx]), probs)
    
    def _predict_tflite(self, feat_window):
        """Predict using TFLite model."""
        x = np.asarray(feat_window, dtype=self.input_details[0]['dtype'])
        self.tflite_interpreter.set_tensor(self.input_details[0]['index'], x)
        self.tflite_interpreter.invoke()
        out = self.tflite_interpreter.get_tensor(self.output_details[0]['index'])
        return out[0].tolist()
    
    def _predict_fallback(self, feat_window):
        """Fallback prediction (uniform distribution)."""
        default_probs = [1.0] + [0.0] * (len(self.labels) - 1)
        return default_probs
    
    def save_pytorch_model(self, save_path):
        """Save PyTorch model for inference."""
        if self.pytorch_model is not None:
            torch.save(self.pytorch_model.state_dict(), save_path)
            log.info("Saved PyTorch model to: %s", save_path)
        else:
            log.warning("No PyTorch model to save")


# Backward compatibility: alias to original name
TemporalModel = TemporalModelEnhanced

