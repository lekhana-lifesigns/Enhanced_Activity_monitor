import logging
from pipeline.detectors.detector import Detector

log = logging.getLogger("hybrid")

class HybridDetector:
    """Lightweight hybrid wrapper: use TFLite detector + simple tracking (centroid matching).
    This is intentionally small and not production-grade—use for single-bed setups.
    """
    def __init__(self, model_path=None, input_size=(320,320), conf_threshold=0.4):
        self.detector = Detector(model_path=model_path, input_size=input_size)
        self.conf_threshold = conf_threshold
        self.next_track_id = 1
        self.tracks = {}  # simple: track_id -> last centroid
        self.has_model = self.detector.interpreter is not None

    def _centroid(self, bbox):
        x, y, w, h = bbox
        return (x + w/2.0, y + h/2.0)

    def infer(self, frame):
        dets = self.detector.infer(frame, score_thr=self.conf_threshold)
        out = []
        for d in dets:
            bbox = d.get('bbox')
            centroid = self._centroid(bbox)
            # naive assignment: if any track within 50px -> assign
            assigned = -1
            for tid, cent in list(self.tracks.items()):
                dist = ((centroid[0]-cent[0])**2 + (centroid[1]-cent[1])**2)**0.5
                if dist < 50:
                    assigned = tid
                    self.tracks[tid] = centroid
                    break
            if assigned == -1:
                assigned = self.next_track_id
                self.next_track_id += 1
                self.tracks[assigned] = centroid
            out.append({"bbox": bbox, "score": d.get('score',1.0), "class": d.get('class',0), "label": d.get('label','person'), "track_id": assigned})
        return out
