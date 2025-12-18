# analytics/bed_detection.py
# Bed Detection and Identification Module
# Detects and tracks beds in the camera frame for context-aware activity monitoring

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

log = logging.getLogger("bed_detection")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    log.warning("YOLO not available - bed detection will use fallback methods")


class BedDetector:
    """
    Detects and tracks beds in camera frames.
    Uses YOLO for object detection (beds are typically detected as 'bed' class).
    Falls back to geometric methods if YOLO unavailable.
    """
    
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.3):
        """
        Initialize bed detector.
        
        Args:
            model_path: Path to YOLO model (if None, uses default YOLO11n)
            conf_threshold: Confidence threshold for bed detection
        """
        self.conf_threshold = conf_threshold
        self.model = None
        self.bed_history = []  # Track bed positions over time
        self.max_history = 30  # Keep last 30 detections
        
        if YOLO_AVAILABLE:
            try:
                if model_path and model_path.endswith('.pt'):
                    self.model = YOLO(model_path)
                    log.info("Bed detector loaded custom model: %s", model_path)
                else:
                    # Use default YOLO11n which can detect furniture including beds
                    self.model = YOLO("yolo11n.pt")
                    log.info("Bed detector initialized with YOLO11n")
            except Exception as e:
                log.warning("Failed to load YOLO model for bed detection: %s", e)
                self.model = None
        else:
            log.warning("YOLO not available - using geometric fallback for bed detection")
    
    def detect_beds(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect beds in the frame.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            List of bed detections, each with:
            - bbox: [x1, y1, x2, y2] bounding box
            - confidence: Detection confidence (0-1)
            - center: (cx, cy) center point
            - area: Bounding box area
        """
        if frame is None or frame.size == 0:
            return []
        
        beds = []
        
        # Method 1: YOLO detection (preferred)
        if self.model is not None:
            try:
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            label = result.names.get(cls, "")
                            conf = float(box.conf[0])
                            
                            # YOLO COCO classes: 'bed' is class 59 in COCO dataset
                            # Also check for 'couch', 'sofa' as they might be beds
                            if label in ["bed", "couch", "sofa"] or cls == 59:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                bed_info = {
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": conf,
                                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                                    "area": (x2 - x1) * (y2 - y1),
                                    "label": label,
                                    "class": cls
                                }
                                beds.append(bed_info)
                                
            except Exception as e:
                log.debug("YOLO bed detection failed: %s", e)
        
        # Method 2: Geometric fallback (if YOLO fails or unavailable)
        if not beds:
            beds = self._geometric_bed_detection(frame)
        
        # Update history
        if beds:
            self.bed_history.append(beds[0])  # Track primary bed
            if len(self.bed_history) > self.max_history:
                self.bed_history.pop(0)
        
        return beds
    
    def _geometric_bed_detection(self, frame: np.ndarray) -> List[Dict]:
        """
        Fallback geometric method for bed detection.
        Looks for large horizontal rectangular regions (typical bed shape).
        
        Args:
            frame: Input frame
        
        Returns:
            List of potential bed detections
        """
        try:
            h, w = frame.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            beds = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (beds are typically large objects)
                min_area = (w * h) * 0.1  # At least 10% of frame
                max_area = (w * h) * 0.8   # At most 80% of frame
                
                if min_area < area < max_area:
                    # Get bounding box
                    x, y, bw, bh = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (beds are typically wider than tall)
                    aspect_ratio = bw / (bh + 1e-6)
                    
                    # Beds typically have aspect ratio > 1.2 (wider than tall)
                    if aspect_ratio > 1.2:
                        bed_info = {
                            "bbox": [x, y, x + bw, y + bh],
                            "confidence": 0.5,  # Lower confidence for geometric method
                            "center": (x + bw // 2, y + bh // 2),
                            "area": area,
                            "label": "bed_geometric",
                            "class": -1
                        }
                        beds.append(bed_info)
            
            return beds[:1]  # Return at most one bed from geometric detection
            
        except Exception as e:
            log.debug("Geometric bed detection failed: %s", e)
            return []
    
    def get_primary_bed(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Get the primary (most likely) bed in the frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Primary bed detection or None
        """
        beds = self.detect_beds(frame)
        
        if not beds:
            return None
        
        # Return bed with highest confidence
        primary_bed = max(beds, key=lambda b: b.get("confidence", 0.0))
        return primary_bed
    
    def is_person_near_bed(self, person_bbox: List[int], bed_bbox: Optional[List[int]] = None, 
                          frame: Optional[np.ndarray] = None, threshold: float = 0.3) -> bool:
        """
        Check if a person is near a bed.
        
        Args:
            person_bbox: Person bounding box [x1, y1, x2, y2]
            bed_bbox: Bed bounding box [x1, y1, x2, y2] (if None, detects bed)
            frame: Input frame (required if bed_bbox is None)
            threshold: Distance threshold (fraction of frame size)
        
        Returns:
            True if person is near bed
        """
        if bed_bbox is None:
            if frame is None:
                return False
            bed = self.get_primary_bed(frame)
            if bed is None:
                return False
            bed_bbox = bed["bbox"]
        
        if len(person_bbox) < 4 or len(bed_bbox) < 4:
            return False
        
        # Calculate person center
        px1, py1, px2, py2 = person_bbox[:4]
        person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        
        # Calculate bed center
        bx1, by1, bx2, by2 = bed_bbox[:4]
        bed_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)
        
        # Calculate distance
        if frame is not None:
            h, w = frame.shape[:2]
            max_distance = max(w, h) * threshold
        else:
            # Fallback: use bed size as reference
            bed_width = bx2 - bx1
            bed_height = by2 - by1
            max_distance = max(bed_width, bed_height) * threshold
        
        distance = np.sqrt((person_center[0] - bed_center[0])**2 + 
                          (person_center[1] - bed_center[1])**2)
        
        return distance < max_distance
    
    def is_person_on_bed(self, person_bbox: List[int], bed_bbox: Optional[List[int]] = None,
                        frame: Optional[np.ndarray] = None, overlap_threshold: float = 0.5) -> bool:
        """
        Check if a person is on the bed (overlapping with bed).
        
        Args:
            person_bbox: Person bounding box [x1, y1, x2, y2]
            bed_bbox: Bed bounding box [x1, y1, x2, y2] (if None, detects bed)
            frame: Input frame (required if bed_bbox is None)
            overlap_threshold: Minimum IoU threshold
        
        Returns:
            True if person overlaps significantly with bed
        """
        if bed_bbox is None:
            if frame is None:
                return False
            bed = self.get_primary_bed(frame)
            if bed is None:
                return False
            bed_bbox = bed["bbox"]
        
        if len(person_bbox) < 4 or len(bed_bbox) < 4:
            return False
        
        # Calculate IoU (Intersection over Union)
        px1, py1, px2, py2 = person_bbox[:4]
        bx1, by1, bx2, by2 = bed_bbox[:4]
        
        # Intersection
        ix1 = max(px1, bx1)
        iy1 = max(py1, by1)
        ix2 = min(px2, bx2)
        iy2 = min(py2, by2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return False  # No intersection
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        
        # Union
        person_area = (px2 - px1) * (py2 - py1)
        bed_area = (bx2 - bx1) * (by2 - by1)
        union = person_area + bed_area - intersection
        
        if union == 0:
            return False
        
        iou = intersection / union
        return iou >= overlap_threshold
    
    def get_bed_region_for_zoom(self, bed_bbox: Optional[List[int]] = None,
                                frame: Optional[np.ndarray] = None,
                                padding: float = 0.2) -> Optional[Tuple[int, int, int, int]]:
        """
        Get optimal region for camera zoom to focus on bed area.
        
        Args:
            bed_bbox: Bed bounding box [x1, y1, x2, y2] (if None, detects bed)
            frame: Input frame (required if bed_bbox is None)
            padding: Padding around bed (fraction of bed size)
        
        Returns:
            Zoom region as (x1, y1, x2, y2) or None
        """
        if bed_bbox is None:
            if frame is None:
                return None
            bed = self.get_primary_bed(frame)
            if bed is None:
                return None
            bed_bbox = bed["bbox"]
        
        if len(bed_bbox) < 4:
            return None
        
        x1, y1, x2, y2 = bed_bbox[:4]
        
        # Add padding
        width = x2 - x1
        height = y2 - y1
        
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        
        # Get frame bounds
        if frame is not None:
            h, w = frame.shape[:2]
        else:
            # Use bed bbox to estimate frame size
            w = int(x2 * 1.2)
            h = int(y2 * 1.2)
        
        # Calculate zoom region
        zoom_x1 = max(0, x1 - pad_x)
        zoom_y1 = max(0, y1 - pad_y)
        zoom_x2 = min(w, x2 + pad_x)
        zoom_y2 = min(h, y2 + pad_y)
        
        return (zoom_x1, zoom_y1, zoom_x2, zoom_y2)

