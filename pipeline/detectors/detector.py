# pipeline/detector.py
import os, logging, cv2, numpy as np
log = logging.getLogger("detector")

# -----------------------------------------
# 1. Try TensorFlow Lite (works on Windows)
# -----------------------------------------
try:
    import tensorflow as tf
    TFLITE = tf.lite
    log.info("Using TensorFlow Lite (from full TensorFlow)")
except Exception:
    TFLITE = None
    log.warning("TensorFlow not available")

# -----------------------------------------
# 2. Optional Coral support (Linux only)
# -----------------------------------------
try:
    from pycoral.utils.edgetpu import make_interpreter as coral_make_interpreter
    EDGETPU_OK = True
    log.info("Coral TPU support enabled")
except Exception:
    coral_make_interpreter = None
    EDGETPU_OK = False
    log.info("Coral TPU not available (this is normal on Windows)")


class Detector:
    def __init__(self, model_path=None, input_size=(320, 320), use_edgetpu=False):
        self.model_path = model_path
        self.input_size = tuple(input_size)

        # EdgeTPU allowed only if pycoral is present
        self.use_edgetpu = use_edgetpu and EDGETPU_OK

        self.interpreter = None

        # ----------------------------------------------------
        # Load model if available
        # ----------------------------------------------------
        if model_path and os.path.exists(model_path) and TFLITE:
            if self.use_edgetpu:
                log.info("Loading EdgeTPU detector...")
                self.interpreter = coral_make_interpreter(model_path)
            else:
                log.info("Loading TFLite detector...")
                self.interpreter = TFLITE.Interpreter(model_path=model_path)

            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            log.info("Detector loaded: %s (EdgeTPU=%s)",
                     model_path, self.use_edgetpu)

        else:
            log.warning("Detector model not loaded — detector disabled")
            self.input_details = None
            self.output_details = None

    # ----------------------------------------------------
    # Preprocess image
    # ----------------------------------------------------
    def preprocess(self, frame):
        W, H = self.input_size
        img = cv2.resize(frame, (W, H))

        if self.input_details:
            dtype = self.input_details[0]["dtype"]
        else:
            dtype = np.float32

        if dtype == np.uint8:
            arr = img.astype(np.uint8)
        else:
            arr = img.astype(np.float32) / 255.0

        arr = np.expand_dims(arr, axis=0)
        return arr

    # ----------------------------------------------------
    # Inference
    # ----------------------------------------------------
    def infer(self, frame, score_thr=0.3):
        # If detector missing → use full frame
        if not self.interpreter:
            h, w = frame.shape[:2]
            return [{
                "bbox": [0, 0, w, h],
                "score": 1.0,
                "class": 0
            }]

        inp = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        self.interpreter.invoke()

        # Standard TFLite OD API: boxes, classes, scores, num
        try:
            boxes = self.interpreter.get_tensor(self.output_details[0]["index"])
            classes = self.interpreter.get_tensor(self.output_details[1]["index"])
            scores = self.interpreter.get_tensor(self.output_details[2]["index"])

            num = boxes.shape[1] if boxes.ndim >= 3 else boxes.shape[0]
            H, W = frame.shape[:2]

            dets = []
            for i in range(int(num)):
                s = float(scores[0][i])
                if s < score_thr:
                    continue

                # boxes: [ymin, xmin, ymax, xmax]
                y1, x1, y2, x2 = boxes[0][i]

                x1 = int(x1 * W)
                y1 = int(y1 * H)
                x2 = int(x2 * W)
                y2 = int(y2 * H)

                dets.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": s,
                    "class": int(classes[0][i])
                })

            return dets

        except Exception as e:
            log.error("Detector parse error: %s", e)
            h, w = frame.shape[:2]
            return [{
                "bbox": [0, 0, w, h],
                "score": 1.0,
                "class": 0
            }]
