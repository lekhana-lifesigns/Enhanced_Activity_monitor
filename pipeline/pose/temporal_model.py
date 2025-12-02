# pipeline/temporal_model.py
import numpy as np, logging
log = logging.getLogger("temporal")
"""try:
    import tflite_runtime.interpreter as tflite_rt
    TFLITE = tflite_rt
except Exception:
    try:
        import tensorflow as tf
        TFLITE = tf.lite
    except Exception:
        TFLITE = None"""
# Try full TensorFlow (works on Windows)
try:
    import tensorflow as tf
    TFLITE = tf.lite
    log.info("Using TensorFlow Lite from full TensorFlow")
except Exception:
    TFLITE = None
    log.warning("TensorFlow not available; will use MediaPipe fallback")


class TemporalModel:
    def __init__(self, model_path=None, window_size=48, labels=None):
        self.model_path = model_path
        self.window_size = window_size
        # Clinical labels for ICU-grade monitoring
        self.labels = labels or [
            "calm",
            "agitation",
            "restlessness",
            "delirium",
            "convulsion",
            "pain_response"
        ]
        self.interpreter = None
        self.prev_probs = None  # For prediction smoothing
        if model_path and TFLITE:
            try:
                self.interpreter = TFLITE.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                log.info("Loaded temporal model %s", model_path)
            except Exception:
                log.exception("Failed to load temporal model")
                self.interpreter = None

    def predict(self, feat_window, use_smoothing=True, alpha=0.7):
        """
        Predict activity class from feature window.
        
        Args:
            feat_window: np.array (T,F) - feature sequence
            use_smoothing: Whether to apply exponential moving average
            alpha: Smoothing factor (0-1), higher = more weight to current prediction
        
        Returns:
            (label, confidence, probs) tuple
        """
        # feat_window: np.array (T,F)
        if self.interpreter is None:
            default_probs = [1.0] + [0.0] * (len(self.labels) - 1)
            return ("calm", 1.0, default_probs)
        
        x = np.asarray(feat_window, dtype=self.input_details[0]['dtype'])
        if x.ndim == 2:
            x = np.expand_dims(x, 0)   # (1,T,F)
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        probs = out[0].tolist()
        
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
