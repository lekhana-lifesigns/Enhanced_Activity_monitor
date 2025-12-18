import logging

log = logging.getLogger("yolo")

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class YOLODetector:
    """Wrapper for Ultralytics YOLO model with basic tracking support. Requires 'ultralytics' package.
    Returns detections list with keys: bbox (x1,y1,x2,y2), score, class, label, track_id (if available)
    """

    def __init__(self, model="yolo11n.pt", conf=0.45, device="cpu"):
        self.model_path = model
        self.conf = conf
        self.device = device
        self.model = None
        if YOLO is not None:
            try:
                self.model = YOLO(model)
                log.info("YOLO model loaded: %s", model)
            except Exception as e:
                log.exception("Failed to load YOLO model: %s", e)
                self.model = None
        else:
            log.warning("Ultralytics YOLO not available; install with 'pip install ultralytics'")

    def infer(self, frame, score_thr=None):
        if self.model is None:
            h, w = frame.shape[:2]
            return [{"bbox": [0, 0, w, h], "score": 1.0, "class": 0, "label": "person", "track_id": -1}]

        conf = score_thr if score_thr is not None else self.conf
        results = self.model(frame, conf=conf, device=self.device)
        # results can be a list; take first
        res = results[0]
        dets = []
        if getattr(res, 'boxes', None) is None:
            return dets
        for box in res.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            tid = int(box.id[0]) if getattr(box, 'id', None) is not None else -1
            label = self.model.names.get(cls, str(cls)) if hasattr(self.model, 'names') else str(cls)
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            dets.append({"bbox": [x1, y1, x2 - x1, y2 - y1], "score": round(conf, 3), "class": cls, "label": label, "track_id": tid})
        return dets