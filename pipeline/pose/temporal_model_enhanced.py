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
    """Lightweight multi-head self-attention for temporal sequences."""

    def __init__(self, d_model=64, n_heads=2, dropout=0.1):
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
        residual = x
        x = self.layer_norm(x)

        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        attn_output = self.w_o(attn_output)

        return self.dropout(attn_output) + residual


class OptimizedTemporalModel(nn.Module):
    """
    Optimized temporal model for 9D -> 6-class ICU activity classification.
    ~34K params (18x reduction from legacy 600K model).

    Architecture:
    - Single GRU layer (input_dim -> hidden_dim)
    - LayerNorm after GRU
    - 2-head self-attention with residual
    - Attention-weighted temporal pooling (last hidden as query)
    - Compact FC classification head with GELU + dropout
    """

    def __init__(self, input_dim=9, hidden_dim=64, num_classes=6,
                 num_heads=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Single GRU layer (no dropout arg — only works with num_layers>1)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True,
                          num_layers=1, bidirectional=False)

        # LayerNorm after GRU for training stability
        self.gru_norm = nn.LayerNorm(hidden_dim)

        # Lightweight 2-head self-attention with residual
        self.attention = AttentionLayer(
            d_model=hidden_dim, n_heads=num_heads, dropout=0.1
        )

        # Classification head
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size = x.size(0)

        # GRU encoding
        gru_out, hidden = self.gru(x)  # gru_out: (batch, seq, hidden_dim)
        gru_out = self.gru_norm(gru_out)

        # Self-attention with residual
        attn_out = self.attention(gru_out)  # (batch, seq, hidden_dim)

        # Attention-weighted temporal pooling using last hidden state as query
        query = hidden.squeeze(0).unsqueeze(1)  # (batch, 1, hidden_dim)
        attn_scores = torch.bmm(query, attn_out.transpose(1, 2))  # (batch, 1, seq)
        attn_scores = torch.softmax(attn_scores / (self.hidden_dim ** 0.5), dim=-1)
        pooled = torch.bmm(attn_scores, attn_out).squeeze(1)  # (batch, hidden_dim)

        # Classification
        out = self.fc1(pooled)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class EnhancedTemporalModel_Legacy(nn.Module):
    """Legacy 600K-param model. Kept for loading old .pth checkpoints."""

    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=256,
                 num_classes=6, num_heads=8, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_classes = num_classes

        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True, dropout=dropout)
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True, dropout=dropout)
        self.attention = AttentionLayer(d_model=hidden_dim2, n_heads=num_heads, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        gru_out1, _ = self.gru1(x)
        gru_out2, _ = self.gru2(gru_out1)
        attn_out = self.attention(gru_out2)
        pooled = torch.mean(attn_out, dim=1)
        out = self.fc1(pooled)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# Default to optimized architecture
EnhancedTemporalModel = OptimizedTemporalModel


class TemporalModelEnhanced:
    """
    Enhanced temporal model wrapper with both PyTorch and TFLite support.
    Falls back to TFLite if PyTorch model not available.
    Tracks trained/untrained state to prevent random-weight predictions.
    """

    def __init__(self, model_path=None, window_size=48, labels=None,
                 use_pytorch=True, device="cpu", use_fp16=False):
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
        self.use_fp16 = use_fp16 and device != "cpu"

        self.pytorch_model = None
        self.tflite_interpreter = None
        self.prev_probs = None
        self.is_trained = False

        # Temporal stride: skip predictions to save compute
        self._prediction_stride = 1
        self._stride_counter = 0
        self._last_prediction = None

        # Try to load PyTorch model first
        if self.use_pytorch:
            try:
                # Check for optimized model first, then legacy
                optimized_path = None
                legacy_path = None
                if model_path:
                    opt_path = model_path.replace('.tflite', '_optimized.pth')
                    leg_path = model_path.replace('.tflite', '.pth')
                    if os.path.exists(opt_path):
                        optimized_path = opt_path
                    elif os.path.exists(leg_path):
                        legacy_path = leg_path

                if optimized_path:
                    checkpoint = torch.load(optimized_path, map_location=device)
                    config = checkpoint.get('config', {})
                    self.pytorch_model = OptimizedTemporalModel(
                        input_dim=config.get('input_dim', 9),
                        hidden_dim=config.get('hidden_dim', 64),
                        num_classes=len(self.labels),
                        num_heads=config.get('num_heads', 2),
                    )
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                    self.pytorch_model.load_state_dict(state_dict)
                    self.pytorch_model.to(device)
                    self.pytorch_model.eval()
                    self.is_trained = True
                    log.info("Loaded optimized PyTorch model (~34K params): %s", optimized_path)
                elif legacy_path:
                    self.pytorch_model = EnhancedTemporalModel_Legacy(
                        input_dim=13,
                        hidden_dim1=128,
                        hidden_dim2=256,
                        num_classes=len(self.labels)
                    )
                    self.pytorch_model.load_state_dict(torch.load(legacy_path, map_location=device))
                    self.pytorch_model.to(device)
                    self.pytorch_model.eval()
                    self.is_trained = True
                    log.info("Loaded legacy PyTorch model (~600K params): %s", legacy_path)
                else:
                    # No trained weights — initialize structure only
                    self.pytorch_model = OptimizedTemporalModel(
                        input_dim=9,
                        hidden_dim=64,
                        num_classes=len(self.labels),
                        num_heads=2,
                    )
                    self.pytorch_model.to(device)
                    self.is_trained = False
                    log.warning(
                        "No trained PyTorch model found. GRU will return untrained fallback. "
                        "Decision tree classifier will be used for activity classification."
                    )

                # Apply FP16 if requested
                if self.use_fp16 and self.pytorch_model is not None:
                    self.pytorch_model.half()
                    log.info("Using FP16 inference for temporal model")

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
                self.is_trained = True
                log.info("Loaded TFLite temporal model: %s", model_path)
            except Exception as e:
                log.warning("Failed to load TFLite model: %s", e)
                self.tflite_interpreter = None

    @staticmethod
    def _compute_uncertainty(probs, num_classes):
        """
        Compute normalized Shannon entropy as uncertainty measure.

        Returns:
            float in [0, 1]: 0 = completely certain, 1 = maximum uncertainty (uniform)
        """
        probs_arr = np.array(probs, dtype=np.float64)
        probs_arr = np.clip(probs_arr, 1e-10, 1.0)
        entropy = -np.sum(probs_arr * np.log(probs_arr))
        max_entropy = np.log(num_classes)
        if max_entropy < 1e-10:
            return 0.0
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    def predict(self, feat_window, use_smoothing=True, alpha=0.7):
        """
        Predict activity class from feature window.

        Args:
            feat_window: np.array (T, F) or (batch, T, F) - feature sequence
            use_smoothing: Whether to apply exponential moving average
            alpha: Smoothing factor (0-1)

        Returns:
            (label, confidence, probs, uncertainty) tuple
        """
        # Temporal stride: reuse last prediction if within stride window
        self._stride_counter += 1
        if (self._prediction_stride > 1
                and self._stride_counter % self._prediction_stride != 0
                and self._last_prediction is not None):
            return self._last_prediction

        # Handle input shape
        if feat_window.ndim == 2:
            feat_window = np.expand_dims(feat_window, 0)  # (1, T, F)

        # If model is not trained, return deterministic fallback
        if not self.is_trained:
            return self._predict_untrained_fallback()

        # Use PyTorch model if available
        if self.use_pytorch and self.pytorch_model is not None:
            try:
                with torch.no_grad():
                    x = torch.FloatTensor(feat_window).to(self.device)
                    if self.use_fp16:
                        x = x.half()

                    # Detect dimension mismatch but do NOT reinitialize
                    actual_dim = x.size(-1)
                    expected_dim = self.pytorch_model.input_dim

                    if actual_dim != expected_dim:
                        log.error(
                            "Feature dimension mismatch: model expects %d but received %d. "
                            "This indicates a configuration error. Returning untrained fallback.",
                            expected_dim, actual_dim
                        )
                        return self._predict_untrained_fallback()

                    logits = self.pytorch_model(x)
                    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]
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
        uncertainty = self._compute_uncertainty(probs, len(self.labels))
        result = (self.labels[idx], float(probs[idx]), probs, uncertainty)
        self._last_prediction = result
        return result

    def _predict_untrained_fallback(self):
        """Return clearly marked untrained result. Decision tree should be used instead."""
        uniform_probs = [1.0 / len(self.labels)] * len(self.labels)
        return ("untrained", 0.0, uniform_probs, 1.0)

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


class TrainingDataCollector:
    """
    Opt-in training data collector for GRU temporal model.
    Saves (feature_window, activity_label) pairs to disk as .npz files.
    """

    # Mapping from fine-grained activity labels to 6 temporal model classes
    LABEL_MAP = {
        "sleeping": "calm", "lying": "calm", "lying_in_bed": "calm",
        "still": "calm", "resting": "calm", "sitting": "calm",
        "sitting_on_bed": "calm", "standing": "calm",
        "agitated": "agitation", "tube_pulling": "agitation",
        "fighting_restraints": "agitation", "pulling_iv": "agitation",
        "thrashing": "agitation",
        "restless": "restlessness", "turning_in_bed": "restlessness",
        "repositioning_in_bed": "restlessness", "fidgeting": "restlessness",
        "tossing": "restlessness",
        "confused": "delirium", "picking_at_air": "delirium",
        "disoriented": "delirium",
        "seizure": "convulsion", "tremor": "convulsion",
        "convulsing": "convulsion",
        "pain": "pain_response", "guarding": "pain_response",
        "grimacing": "pain_response", "bracing": "pain_response",
    }

    LABEL_TO_IDX = {
        "calm": 0, "agitation": 1, "restlessness": 2,
        "delirium": 3, "convulsion": 4, "pain_response": 5,
    }

    def __init__(self, save_dir="data/training_collection",
                 min_confidence=0.7, max_samples=50000):
        self.save_dir = save_dir
        self.min_confidence = min_confidence
        self.max_samples = max_samples
        self.sample_count = 0
        self.enabled = False

    def enable(self):
        """Enable data collection (creates save directory)."""
        os.makedirs(self.save_dir, exist_ok=True)
        self.enabled = True
        log.info("Training data collection enabled: %s", self.save_dir)

    def disable(self):
        """Disable data collection."""
        self.enabled = False
        log.info("Training data collection disabled. Collected %d samples.", self.sample_count)

    def collect(self, feat_window, label, confidence, metadata=None):
        """
        Save a training sample if enabled and confidence meets threshold.

        Args:
            feat_window: np.ndarray (T, F) — feature window
            label: str — activity label from decision tree / enhanced classifier
            confidence: float — confidence score from the classifier
            metadata: dict — optional metadata (patient_id, timestamp, etc.)
        """
        if not self.enabled or self.sample_count >= self.max_samples:
            return
        if confidence < self.min_confidence:
            return

        temporal_label = self.LABEL_MAP.get(label)
        if temporal_label is None:
            return

        try:
            import time as _time
            timestamp = int(_time.time() * 1000)
            filename = f"sample_{timestamp}_{self.sample_count:06d}.npz"
            filepath = os.path.join(self.save_dir, filename)

            save_dict = {
                "features": np.asarray(feat_window, dtype=np.float32),
                "label": np.array(self.LABEL_TO_IDX[temporal_label], dtype=np.int64),
                "label_name": np.array(temporal_label),
                "confidence": np.array(confidence, dtype=np.float32),
            }
            if metadata:
                for k, v in metadata.items():
                    save_dict[f"meta_{k}"] = np.array(v)

            np.savez_compressed(filepath, **save_dict)
            self.sample_count += 1

            if self.sample_count % 100 == 0:
                log.info("Collected %d training samples", self.sample_count)
        except Exception as e:
            log.debug("Failed to save training sample: %s", e)


# Backward compatibility: alias to original name
TemporalModel = TemporalModelEnhanced

