"""# pipeline/inference_pipeline.py
import time, logging, numpy as np
from .camera import Camera
from pipeline.detectors.detector import Detector
from .pose_estimator import PoseEstimator
from .feature_extractor import ICUFeatureEncoder
from .temporal_model import TemporalModel
from pipeline.display.display import ICUMonitorDisplay
from .decision_engine import apply_rules

log = logging.getLogger("infpipe")

class InferencePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        camres = tuple(cfg.get("camera_resolution", (1280,720)))
        fps = cfg.get("camera_fps", 15)
        self.camera = Camera(index=cfg.get("camera_idx",0), resolution=camres, fps=fps)
        models = cfg.get("models", {})
        self.det = Detector(model_path=models.get("detector"), input_size=tuple(models.get("det_input_size", (320,320))), use_edgetpu=cfg.get("use_edgetpu", False))
        self.pose = PoseEstimator(model_path=models.get("pose"), input_size=models.get("pose_input_size",192))
        self.temporal = TemporalModel(model_path=models.get("temporal"), window_size=cfg.get("window_size",48))
        
        # ICU Feature Encoder
        window_size = cfg.get("window_size", 48)
        self.feature_encoder = ICUFeatureEncoder(window_size=window_size, fps=fps)
        
        self.window = []   # list of feat arrays
        self.prev_kps = None
        self.prev_prev_kps = None
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.fps = 0.0

    def run_once(self):
        st = time.time()
        frame = self.camera.read()
        dets = self.det.infer(frame)
        if not dets:
            log.warning("No person detected — skipping frame")
            return None
        det = max(dets, key=lambda d: d["score"])
        x,y,w,h = det["bbox"]
        x1 = max(0, x); y1 = max(0,y); x2 = min(frame.shape[1], x+x+w); y2 = min(frame.shape[0], y+y+h)
        crop = frame[y1:y2, x1:x2] if (y2>y1 and x2>x1) else frame
        kps = self.pose.infer(crop)    # normalized to crop
        
        # Extract ICU-grade features
        feat = self.feature_encoder.extract_feature_vector(
            kps, 
            prev_kps=self.prev_kps,
            prev_prev_kps=self.prev_prev_kps
        )
        
        if feat is not None:
            self.window.append(feat)
            if len(self.window) > self.temporal.window_size:
                self.window.pop(0)
        
        # Update keypoint history
        self.prev_prev_kps = self.prev_kps
        self.prev_kps = kps
        
        # Temporal model prediction
        label, conf, probs = ("normal", 1.0, [1.0])
        if len(self.window) >= max(8, self.temporal.window_size // 4):
            feat_win = np.stack(self.window[-self.temporal.window_size:])   # (T,F)
            label, conf, probs = self.temporal.predict(feat_win)
        
        inference_ms = (time.time() - st) * 1000.0
        
        # Calculate FPS
        self.fps_frame_count += 1
        elapsed = time.time() - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.last_fps_time = time.time()
        
        # Enhanced decision engine with clinical features
        decision = apply_rules(label, probs, kps, features=feat)
        
        result = {
            "ts": time.time(),
            "label": decision["label"],
            "confidence": decision.get("confidence", conf),
            "probs": probs,
            "bbox": det["bbox"],
            "kps": kps,
            "inference_ms": inference_ms,
            "decision": decision,
            "features": feat.tolist() if feat is not None else None
        }
        log.info("FPS: %.2f", self.fps)
        return result

    def run_once_and_publish(self, publish_fn):
        res = self.run_once()
        publish_fn(res)
        return res
"""

# pipeline/inference_pipeline.py
import time
import logging
import numpy as np
from .camera import Camera
from pipeline.detectors.detector import Detector
from .pose_estimator import PoseEstimator
from .feature_extractor import ICUFeatureEncoder
from .temporal_model import TemporalModel
from pipeline.display.display import ICUMonitorDisplay
from .decision_engine import apply_rules

log = logging.getLogger("infpipe")


class InferencePipeline:
    """
    Inference pipeline for enhanced activity monitor.
    Robust handling for missing detections / pose failures, optional display,
    sliding-window temporal prediction and clinical decision engine.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        camres = tuple(cfg.get("camera_resolution", (1280, 720)))
        fps = cfg.get("camera_fps", 15)

        self.camera = Camera(index=cfg.get("camera_idx", 0), resolution=camres, fps=fps)
        models = cfg.get("models", {})
        self.det = Detector(
            model_path=models.get("detector"),
            input_size=tuple(models.get("det_input_size", (320, 320))),
            use_edgetpu=cfg.get("use_edgetpu", False),
        )
        self.pose = PoseEstimator(model_path=models.get("pose"), input_size=models.get("pose_input_size", 192))
        self.temporal = TemporalModel(model_path=models.get("temporal"), window_size=cfg.get("window_size", 48))

        # ICU Feature Encoder
        window_size = cfg.get("window_size", 48)
        self.feature_encoder = ICUFeatureEncoder(window_size=window_size, fps=fps)

        self.window = []  # list of feat arrays
        self.prev_kps = None
        self.prev_prev_kps = None

        # FPS counters
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.fps = 0.0

        # Display (optional)
        self.enable_display = bool(cfg.get("enable_display", True))
        self.display = ICUMonitorDisplay() if self.enable_display else None
        # --- DISPLAY ENABLE TOGGLE ---
        self.display_enabled = cfg.get("enable_display", False)

        if self.display_enabled:
            self.display = ICUMonitorDisplay(title="ICU Live Monitor")


        # Control flag (display or external stop request)
        self.stop_requested = False

    def _parse_bbox(self, bbox, frame_shape):
        """
        Accepts either (x,y,w,h) or (x1,y1,x2,y2). Returns safely-clamped ints.
        """
        h, w = frame_shape[0], frame_shape[1]
        try:
            bbox = [int(v) for v in bbox]
        except Exception:
            return 0, 0, w, h

        if len(bbox) == 4:
            a, b, c, d = bbox
            # Detect common formats:
            # if c > a and d > b AND likely (x1,y1,x2,y2)
            if c > a and d > b and (c - a > 0 and d - b > 0) and (c > 1 and d > 1) and (c <= w and d <= h):
                x1, y1, x2, y2 = a, b, c, d
            else:
                # assume (x,y,w,h)
                x1, y1 = a, b
                x2, y2 = a + c, b + d
        else:
            # fallback full frame
            x1, y1, x2, y2 = 0, 0, w, h

        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0, 0, w, h
        return x1, y1, x2, y2

    def run_once(self):
        """
        Run single inference step. Returns result dict or None (skip).
        Non-fatal exceptions are logged and return None.
        """
        if self.stop_requested:
            return None

        st = time.time()
        try:
            frame = self.camera.read()
            if frame is None:
                log.warning("Camera read returned None")
                return None

            dets = self.det.infer(frame)
            if not dets:
                log.debug("No person detected — skipping frame")
                return None

            # choose highest-score detection
            det = max(dets, key=lambda d: d.get("score", 0.0))
            bbox = det.get("bbox", [0, 0, frame.shape[1], frame.shape[0]])
            x1, y1, x2, y2 = self._parse_bbox(bbox, frame.shape)
            crop = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else frame

            # pose inference (pose returns normalized / crop coords)
            kps = self.pose.infer(crop)
            if not kps:
                log.debug("Pose not detected — skipping frame")
                return None

            # Extract ICU-grade features
            feat = self.feature_encoder.extract_feature_vector(kps, prev_kps=self.prev_kps, prev_prev_kps=self.prev_prev_kps)

            # validate features
            if feat is not None:
                feat = np.asarray(feat, dtype=np.float32)
                if np.isnan(feat).any():
                    log.warning("Feature vector contains NaN — skipping append")
                else:
                    self.window.append(feat)
                    # keep window length bounded
                    if len(self.window) > self.temporal.window_size:
                        self.window = self.window[-self.temporal.window_size :]

            # update kps history
            self.prev_prev_kps = self.prev_kps
            self.prev_kps = kps

            # Temporal model prediction when enough frames
            label, conf, probs = ("normal", 1.0, [1.0])
            if len(self.window) >= max(8, self.temporal.window_size // 4):
                feat_win = np.stack(self.window[-self.temporal.window_size :])  # (T,F)
                label, conf, probs = self.temporal.predict(feat_win)

            inference_ms = (time.time() - st) * 1000.0

            # FPS calculation
            self.fps_frame_count += 1
            elapsed = time.time() - self.last_fps_time
            if elapsed >= 1.0:
                self.fps = self.fps_frame_count / elapsed
                self.fps_frame_count = 0
                self.last_fps_time = time.time()

            # Apply decision engine (blends ML + clinical features)
            try:
                decision = apply_rules(label, probs, kps, features=feat)
            except Exception:
                log.exception("Decision engine error - falling back to ML label")
                decision = {"label": label, "confidence": conf}

            # Prepare result
            result = {
                "ts": time.time(),
                "label": decision.get("label", label),
                "confidence": float(decision.get("confidence", conf)),
                "probs": probs,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "kps": kps,
                "inference_ms": float(inference_ms),
                "decision": decision,
                "features": feat.tolist() if feat is not None else None,
                "fps": round(self.fps, 2),
            }

            # Display overlay if enabled
            if self.display_enabled:
                frame_vis = frame.copy()
            

                # Draw outputs
                self.display.draw_bbox(frame_vis, det["bbox"], label=decision["label"])
                self.display.draw_skeleton(frame_vis, kps)
                self.display.draw_metrics(frame_vis, {
                    "FPS": round(self.fps, 1),
                    "Activity": decision["label"],
                    "Conf": round(decision.get("confidence", conf), 2),
                    "Latency(ms)": round(inference_ms, 1)
        })

            if not self.display.show(frame_vis):
                self.camera.release()
                self.display.close()
                exit(0)

            if self.display:
                metrics = {
                    "FPS": result["fps"],
                    "Label": result["label"],
                    "Conf": round(result["confidence"], 2),
                    "Latency(ms)": round(result["inference_ms"], 1),
                }
                try:
                    # draw in original frame coords (we provide kps in crop coords - PoseEstimator should document this)
                    display_frame = frame.copy()
                    # If pose returns normalized coords relative to crop, convert back to frame coords
                    try:
                        # expect kps as list of (x_norm, y_norm, score)
                        kps_frame = []
                        for kp in kps:
                            if kp is None:
                                continue
                            kx, ky = kp[0], kp[1]
                            # if values are in [0,1], map to crop -> frame
                            if 0.0 <= kx <= 1.0 and 0.0 <= ky <= 1.0:
                                fx = int(x1 + kx * (x2 - x1))
                                fy = int(y1 + ky * (y2 - y1))
                            else:
                                fx = int(x1 + kx)
                                fy = int(y1 + ky)
                            kps_frame.append((fx, fy))
                        display_frame = self.display.draw_skeleton(display_frame, kps_frame)
                    except Exception:
                        # best-effort: draw nothing if conversion fails
                        display_frame = self.display.draw_skeleton(display_frame, [])
                    display_frame = self.display.draw_metrics(display_frame, metrics)
                    if not self.display.show(display_frame):
                        log.info("Display requested exit")
                        self.stop_requested = True
                        return None
                except SystemExit:
                    self.stop_requested = True
                    return None
                except Exception:
                    log.exception("Display rendering error (continuing)")

            log.debug("FPS: %.2f  Label: %s  Latency: %.1fms", self.fps, result["label"], result["inference_ms"])
            return result

        except Exception as e:
            log.exception("Inference pipeline error (skipping frame): %s", e)
            return None

    def run_once_and_publish(self, publish_fn):
        """
        Runs one cycle and publishes via provided callback.
        publish_fn should accept a single dict (the result). If result is None, nothing is published.
        """
        res = self.run_once()
        if res is None:
            return None
        try:
            publish_fn(res)
        except Exception:
            log.exception("Publish function raised an exception")
        return res
   

