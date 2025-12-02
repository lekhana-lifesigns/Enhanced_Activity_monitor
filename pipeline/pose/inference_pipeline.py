# pipeline/inference_pipeline.py
import time, logging, numpy as np
from .camera import Camera
from pipeline.detectors.detector import Detector
from .pose_estimator import PoseEstimator
from .feature_extractor import ICUFeatureEncoder
from .temporal_model import TemporalModel
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
