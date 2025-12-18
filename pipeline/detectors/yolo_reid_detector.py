# pipeline/detectors/yolo_reid_detector.py
"""
YOLO11 ReID-based Detection and Tracking
Two-stage approach: Detection first, then detailed pose/activity analysis
Uses ReID (Re-identification) for robust patient tracking
"""
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import os

log = logging.getLogger("YOLOReIDDetector")


class YOLOReIDDetector:
    """
    YOLO11 detector with ReID-based tracking.
    Two-stage pipeline:
    1. Fast person detection + ReID tracking
    2. Detailed pose/activity analysis on tracked person
    """
    def __init__(self,
                 detection_model="yolo11n.pt",  # Fast detection model
                 pose_model="yolo11n-pose.pt",  # Detailed pose model
                 tracker="bytetrack.yaml",  # Can use "strongsort.yaml" for ReID
                 conf=0.5,
                 device="cpu",
                 use_reid=True):
        """
        Args:
            detection_model: YOLO detection model (fast, for bbox detection)
            pose_model: YOLO pose model (for detailed keypoint analysis)
            tracker: Tracker config ("bytetrack.yaml" or "strongsort.yaml" for ReID)
            conf: Confidence threshold
            device: "cpu" or "cuda"
            use_reid: Enable ReID-based tracking (requires StrongSORT)
        """
        log.info("Loading YOLO11 ReID detector...")
        
        # Stage 1: Fast detection model (for bbox detection)
        try:
            self.det_model = YOLO(detection_model)
            log.info("Detection model loaded: %s", detection_model)
        except Exception as e:
            log.error("Failed to load detection model: %s", e)
            raise
        
        # Stage 2: Pose model (for detailed analysis)
        try:
            self.pose_model = YOLO(pose_model)
            log.info("Pose model loaded: %s", pose_model)
        except Exception as e:
            log.warning("Failed to load pose model %s: %s", pose_model, e)
            self.pose_model = None
        
        self.tracker = tracker
        self.conf = conf
        self.device = device
        self.use_reid = use_reid
        
        # Set tracker - use ByteTrack by default (StrongSORT requires separate installation)
        # ByteTrack works well for tracking, StrongSORT adds ReID features but needs config file
        if tracker == "strongsort.yaml":
            # Check if StrongSORT config exists
            import os
            from ultralytics.utils import ROOT
            strongsort_path = os.path.join(ROOT, "trackers", "strongsort.yaml")
            if os.path.exists(strongsort_path):
                self.tracker = "strongsort.yaml"
                log.info("Using StrongSORT tracker for ReID-based tracking")
            else:
                log.warning("StrongSORT config not found, falling back to ByteTrack")
                self.tracker = "bytetrack.yaml"
                # ByteTrack doesn't have true ReID, but works for tracking
                if use_reid:
                    log.info("ByteTrack used for tracking (ReID features limited)")
        else:
            self.tracker = tracker
            if use_reid and tracker == "bytetrack.yaml":
                log.info("Using ByteTrack tracker (ReID features limited)")

    def infer_detection(self, frame):
        """
        Stage 1: Fast person detection + ReID tracking.
        Returns bounding boxes with track IDs.
        
        Returns:
            [
              {
                'bbox': [x,y,w,h],
                'score': float,
                'track_id': int,
                'label': str
              }
            ]
        """
        try:
            # Fast detection with tracking
            # Wrap in try-except to handle tracker config errors gracefully
            try:
                results = self.det_model.track(
                    source=frame,
                    conf=self.conf,
                    tracker=self.tracker,
                    persist=True,
                    device=self.device,
                    verbose=False
                )[0]
            except FileNotFoundError as e:
                # Tracker config file not found - fallback to ByteTrack
                if "strongsort.yaml" in str(e) or "strongsort" in str(e).lower():
                    log.warning("StrongSORT config not found, falling back to ByteTrack")
                    self.tracker = "bytetrack.yaml"
                    results = self.det_model.track(
                        source=frame,
                        conf=self.conf,
                        tracker=self.tracker,
                        persist=True,
                        device=self.device,
                        verbose=False
                    )[0]
                else:
                    raise

            dets = []

            if results.boxes is None:
                return dets

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                tid = int(box.id[0]) if box.id is not None else -1
                label = self.det_model.names.get(cls, "unknown")

                # Only return person detections
                if label == "person" or cls == 0:
                    dets.append({
                        "bbox": [x1, y1, x2, y2],  # Proper format (x1,y1,x2,y2)
                        "score": round(conf, 3),
                        "track_id": tid,
                        "label": label,
                        "class": cls
                    })

            return dets

        except Exception as e:
            log.exception("Detection inference error: %s", e)
            return []

    def infer_pose_detailed(self, crop, track_id=None):
        """
        Stage 2: Detailed pose estimation on detected person crop.
        Only runs on tracked patient for efficiency.
        
        Args:
            crop: Cropped image of detected person
            track_id: Track ID for logging
        
        Returns:
            List of (x, y, confidence) tuples - 17 COCO keypoints
            Returns None if no pose detected
        """
        if self.pose_model is None:
            return None
        
        try:
            results = self.pose_model.predict(
                source=crop,
                conf=self.conf,
                device=self.device,
                verbose=False
            )[0]

            if results.keypoints is None or len(results.keypoints.data) == 0:
                return None

            # Get first person's keypoints
            kp_data = results.keypoints.data[0]  # Shape: (17, 3)
            h, w = crop.shape[:2]

            # Convert to normalized coordinates relative to crop
            kps = []
            for kp in kp_data:
                x_norm = float(kp[0]) / w if w > 0 else 0.0
                y_norm = float(kp[1]) / h if h > 0 else 0.0
                conf_kp = float(kp[2]) if len(kp) > 2 else 0.0
                kps.append((x_norm, y_norm, conf_kp))

            return kps

        except Exception as e:
            log.exception("Pose inference error: %s", e)
            return None

    def infer(self, frame):
        """
        Unified inference: Detection + tracking (ReID-based).
        Returns detections with track IDs for patient selection.
        
        Returns:
            [
              {
                'bbox': [x,y,w,h],
                'score': float,
                'track_id': int,
                'label': str,
                'reid_enabled': bool
              }
            ]
        """
        dets = self.infer_detection(frame)
        
        # Add ReID info
        for det in dets:
            det["reid_enabled"] = self.use_reid
        
        return dets

