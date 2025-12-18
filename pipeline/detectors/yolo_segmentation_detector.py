# pipeline/detectors/yolo_segmentation_detector.py
"""
YOLO11 Segmentation Detector
Provides instant segmentation masks along with bounding boxes
"""
import cv2
import numpy as np
from ultralytics import YOLO
import logging

log = logging.getLogger("YOLOSegDetector")


class YOLOSegmentationDetector:
    """
    YOLO11 detector with instant segmentation support.
    Provides both bounding boxes and segmentation masks.
    """
    
    def __init__(self,
                 model="yolo11n-seg.pt",  # Segmentation model
                 tracker="bytetrack.yaml",
                 conf=0.5,
                 device="cpu"):
        """
        Args:
            model: YOLO segmentation model (yolo11n-seg.pt, yolo11s-seg.pt, etc.)
            tracker: Tracker config
            conf: Confidence threshold
            device: "cpu" or "cuda"
        """
        log.info("Loading YOLO11 segmentation model: %s", model)
        
        try:
            self.model = YOLO(model)
            log.info("Segmentation model loaded: %s", model)
        except Exception as e:
            log.error("Failed to load segmentation model: %s", e)
            # Fallback to detection-only model
            try:
                fallback_model = model.replace("-seg", "")
                log.warning("Falling back to detection model: %s", fallback_model)
                self.model = YOLO(fallback_model)
                self.has_segmentation = False
            except Exception as e2:
                log.error("Failed to load fallback model: %s", e2)
                raise
        
        self.tracker = tracker
        self.conf = conf
        self.device = device
        self.has_segmentation = hasattr(self.model.model, 'seg') or 'seg' in str(type(self.model.model))
        
        if not self.has_segmentation:
            log.warning("Model does not support segmentation. Masks will not be available.")
        
        # TODO-001: Detection Frame Skipping
        self.frame_skip_interval = 1  # Process every N frames (1 = every frame)
        self.frame_counter = 0
        self.last_detections = None  # Cache for skipped frames
        self.enable_frame_skipping = False  # Disabled by default, enable via config
    
    def infer(self, frame, frame_skip_interval=1, enable_frame_skipping=False):
        """
        Infer detections with segmentation masks.
        TODO-001: Detection Frame Skipping - Process every N frames, interpolate for skipped frames.
        
        Args:
            frame: Input frame
            frame_skip_interval: Process every N frames (1 = every frame)
            enable_frame_skipping: Enable frame skipping optimization
        
        Returns:
            [
              {
                'bbox': [x1, y1, x2, y2],  # Proper format (x1,y1,x2,y2)
                'score': float,
                'track_id': int,
                'label': str,
                'mask': np.ndarray (H, W) or None,  # Binary segmentation mask
                'mask_area': float,  # Area of mask in pixels
                'segmentation_confidence': float
              }
            ]
        """
        # TODO-001: Frame skipping logic
        self.frame_counter += 1
        should_skip = (enable_frame_skipping and 
                      self.frame_skip_interval > 1 and 
                      self.frame_counter % self.frame_skip_interval != 0)
        
        if should_skip and self.last_detections is not None:
            # Return cached detections (tracker will handle interpolation)
            log.debug("Skipping detection frame %d (interval: %d)", self.frame_counter, self.frame_skip_interval)
            return self.last_detections
        
        try:
            # Run segmentation inference
            results = self.model.track(
                source=frame,
                conf=self.conf,
                tracker=self.tracker,
                persist=True,
                device=self.device,
                verbose=False
            )[0]
            
            dets = []
            
            if results.boxes is None:
                return dets
            
            h, w = frame.shape[:2]
            
            # Get masks if available
            masks = None
            if results.masks is not None and len(results.masks.data) > 0:
                masks = results.masks
            
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                tid = int(box.id[0]) if box.id is not None else -1
                label = self.model.names.get(cls, "unknown")
                
                # Only return person detections
                if label == "person" or cls == 0:
                    det = {
                        "bbox": [x1, y1, x2, y2],  # Proper format (x1,y1,x2,y2)
                        "score": round(conf, 3),
                        "track_id": tid,
                        "label": label,
                        "class": cls
                    }
                    
                    # Add segmentation mask if available
                    if masks is not None and i < len(masks.data):
                        try:
                            mask = masks.data[i].cpu().numpy()
                            
                            # Resize mask to frame size if needed
                            if mask.shape != (h, w):
                                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                            
                            # Convert to binary mask
                            mask_binary = (mask > 0.5).astype(np.uint8)
                            
                            # Calculate mask area
                            mask_area = float(np.sum(mask_binary))
                            
                            det["mask"] = mask_binary
                            det["mask_area"] = mask_area
                            det["segmentation_confidence"] = float(np.mean(mask)) if mask.max() > 0 else 0.0
                            
                        except Exception as e:
                            log.debug("Failed to extract mask: %s", e)
                            det["mask"] = None
                            det["mask_area"] = 0.0
                            det["segmentation_confidence"] = 0.0
                    else:
                        det["mask"] = None
                        det["mask_area"] = 0.0
                        det["segmentation_confidence"] = 0.0
                    
                    dets.append(det)
            
            return dets
            
        except Exception as e:
            log.exception("Segmentation inference error: %s", e)
            return []

