import os, logging, cv2, numpy as np
log = logging.getLogger("pose")

# Prefer tflite_runtime (fast for Raspberry Pi)
"""try:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path="movenet.tflite")

    TFLITE = tflite_rt
except Exception:
    try:
        import tensorflow as tf
        TFLITE = tf.lite
    except Exception:
        TFLITE = None
"""
# Try full TensorFlow (works on Windows)
try:
    import tensorflow as tf
    TFLITE = tf.lite
    log.info("Using TensorFlow Lite from full TensorFlow")
except Exception:
    TFLITE = None
    log.warning("TensorFlow not available; will use MediaPipe fallback")

# MediaPipe fallback
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False


class PoseEstimator:
    """
    Unified PoseEstimator:
    Priority:
        1. MoveNet (TFLite)
        2. MediaPipe fallback
    Output format:
        list of (x, y, score) in normalized coordinates
    """
    def __init__(self, model_path=None, input_size=192):
        self.model_path = model_path
        self.input_size = int(input_size)
        self.interpreter = None
        self.use_mediapipe = False

        # Try MoveNet TFLite first
        if model_path and os.path.exists(model_path) and TFLITE:
            try:
                self.interpreter = TFLITE.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                log.info(f"Loaded MoveNet TFLite model: {model_path}")
            except Exception as e:
                log.error(f"Failed to load MoveNet model: {e}")
                self.interpreter = None

        # Fallback to MediaPipe
        if self.interpreter is None and MP_AVAILABLE:
            self.use_mediapipe = True
            self.mp_pose = mp.solutions.pose
            self.pose_proc = self.mp_pose.Pose(
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4
            )
            log.info("Using MediaPipe pose fallback")

        if self.interpreter is None and not self.use_mediapipe:
            log.warning("⚠ No pose estimator available!")

    # -------------------------
    #       PREPROCESS
    # -------------------------
    def preprocess(self, crop):
        s = self.input_size
        img = cv2.resize(crop, (s, s))

        dtype = self.input_details[0]["dtype"]
        if dtype == np.uint8:
            arr = img.astype(np.uint8)
        else:
            arr = img.astype(np.float32) / 255.0

        return np.expand_dims(arr, axis=0)

    # -------------------------
    #       MoveNet Inference
    # -------------------------
    def _infer_movenet(self, crop):
        inp = self.preprocess(crop)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()

        out = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Many MoveNet models output shape: (1, 1, 17, 3)
        kps = out.reshape(-1, 3)

        # Return as list of (x, y, score)
        return [(float(x), float(y), float(score)) for (y, x, score) in kps]

    # -------------------------
    #       MediaPipe Inference
    # -------------------------
    def _infer_mediapipe(self, crop):
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = self.pose_proc.process(img_rgb)

        if not res.pose_landmarks:
            return None

        kps = []
        for lm in res.pose_landmarks.landmark:
            score = float(lm.visibility) if hasattr(lm, "visibility") else 1.0
            kps.append((float(lm.x), float(lm.y), score))

        return kps

    # -------------------------
    #       PUBLIC API
    # -------------------------
    def infer(self, crop):
        if self.interpreter:
            return self._infer_movenet(crop)
        if self.use_mediapipe:
            return self._infer_mediapipe(crop)
        return None
