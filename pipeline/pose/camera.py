# pipeline/pose/camera.py
import cv2
import time
import logging
import numpy as np
from typing import Optional, Tuple, List

log = logging.getLogger("camera")

class Camera:
    def __init__(self, index=0, resolution=(1280, 720), fps=15, enable_zoom=True):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        self.original_resolution = tuple(resolution)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        time.sleep(0.2)  # allow warm-up

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {index} could not be opened")
        log.info(f"Camera {index} opened with resolution {resolution} at {fps} FPS")
        self.last=time.time()
        self.fps=fps
        
        # Zoom settings
        self.enable_zoom = enable_zoom
        self.zoom_level = 1.0  # 1.0 = no zoom, >1.0 = zoomed in
        self.zoom_center = None  # (x, y) center point for zoom
        self.zoom_region = None  # (x1, y1, x2, y2) region to zoom to
        self.auto_zoom_enabled = False
        self.zoom_smoothing = 0.1  # Smoothing factor for zoom transitions
        
        # Check if camera supports optical zoom
        self.optical_zoom_available = self._check_optical_zoom_support()
        
        # Digital zoom (always available)
        self.digital_zoom_enabled = True

    def _check_optical_zoom_support(self) -> bool:
        """Check if camera supports optical zoom."""
        try:
            # Try to get zoom property (some cameras support this)
            zoom = self.cap.get(cv2.CAP_PROP_ZOOM)
            if zoom >= 0:
                log.info("Camera supports optical zoom (current: %.2f)", zoom)
                return True
        except:
            pass
        return False

    def set_optical_zoom(self, zoom_level: float) -> bool:
        """
        Set optical zoom level (if supported by camera).
        
        Args:
            zoom_level: Zoom level (1.0 = no zoom, higher = more zoom)
        
        Returns:
            True if successful
        """
        if not self.optical_zoom_available:
            log.debug("Optical zoom not available on this camera")
            return False
        
        try:
            # CAP_PROP_ZOOM is not standard, but some cameras support it
            # Alternative: CAP_PROP_SETTINGS might open camera settings dialog
            success = self.cap.set(cv2.CAP_PROP_ZOOM, zoom_level)
            if success:
                self.zoom_level = zoom_level
                log.info("Optical zoom set to %.2f", zoom_level)
            return success
        except Exception as e:
            log.debug("Failed to set optical zoom: %s", e)
            return False

    def set_digital_zoom(self, zoom_level: float, center: Optional[Tuple[int, int]] = None):
        """
        Set digital zoom level (crops and resizes frame).
        
        Args:
            zoom_level: Zoom level (1.0 = no zoom, >1.0 = zoomed in, <1.0 = zoomed out)
            center: (x, y) center point for zoom. If None, uses frame center.
        """
        if zoom_level < 0.1:
            zoom_level = 0.1
        if zoom_level > 10.0:
            zoom_level = 10.0
        
        self.zoom_level = zoom_level
        self.zoom_center = center
        self.digital_zoom_enabled = True
        log.debug("Digital zoom set to %.2f", zoom_level)

    def set_zoom_region(self, region: Tuple[int, int, int, int]):
        """
        Set zoom to a specific region of the frame.
        
        Args:
            region: (x1, y1, x2, y2) region to zoom to
        """
        if len(region) < 4:
            log.warning("Invalid zoom region format")
            return
        
        x1, y1, x2, y2 = region[:4]
        
        # Validate region
        if x2 <= x1 or y2 <= y1:
            log.warning("Invalid zoom region: x2<=x1 or y2<=y1")
            return
        
        self.zoom_region = (x1, y1, x2, y2)
        
        # Calculate zoom level based on region size
        # Get current frame to determine dimensions
        ret, frame = self.cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            region_width = x2 - x1
            region_height = y2 - y1
            
            zoom_w = w / region_width
            zoom_h = h / region_height
            self.zoom_level = min(zoom_w, zoom_h)
            
            # Set center to region center
            self.zoom_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            log.info("Zoom region set: (%d,%d) to (%d,%d), zoom level: %.2f", 
                    x1, y1, x2, y2, self.zoom_level)
        else:
            self.zoom_region = region
            log.info("Zoom region set: (%d,%d) to (%d,%d)", x1, y1, x2, y2)

    def apply_digital_zoom(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply digital zoom to a frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Zoomed frame
        """
        if frame is None or frame.size == 0:
            return frame
        
        if not self.digital_zoom_enabled or self.zoom_level <= 1.0:
            return frame
        
        h, w = frame.shape[:2]
        
        # If zoom region is specified, use it
        if self.zoom_region is not None:
            x1, y1, x2, y2 = self.zoom_region
            
            # Clamp to frame bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(x1, min(x2, w))
            y2 = max(y1, min(y2, h))
            
            # Crop region
            cropped = frame[y1:y2, x1:x2]
            
            # Resize back to original resolution
            if cropped.size > 0:
                zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
                return zoomed
            else:
                return frame
        
        # Otherwise, use zoom level and center point
        if self.zoom_center is None:
            center_x, center_y = w // 2, h // 2
        else:
            center_x, center_y = self.zoom_center
            # Clamp to frame bounds
            center_x = max(0, min(center_x, w))
            center_y = max(0, min(center_y, h))
        
        # Calculate crop size
        crop_width = int(w / self.zoom_level)
        crop_height = int(h / self.zoom_level)
        
        # Calculate crop bounds
        x1 = max(0, center_x - crop_width // 2)
        y1 = max(0, center_y - crop_height // 2)
        x2 = min(w, x1 + crop_width)
        y2 = min(h, y1 + crop_height)
        
        # Adjust if we hit boundaries
        if x2 - x1 < crop_width:
            x1 = max(0, x2 - crop_width)
        if y2 - y1 < crop_height:
            y1 = max(0, y2 - crop_height)
        
        # Crop and resize
        cropped = frame[y1:y2, x1:x2]
        
        if cropped.size > 0:
            zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            return zoomed
        else:
            return frame

    def auto_zoom_to_person(self, person_bbox: Optional[List[int]], 
                           frame: Optional[np.ndarray] = None,
                           target_size_ratio: float = 0.4) -> bool:
        """
        Automatically zoom to keep person at target size in frame.
        
        Args:
            person_bbox: Person bounding box [x1, y1, x2, y2]
            frame: Current frame (if None, will read from camera)
            target_size_ratio: Target person size as fraction of frame (0.0-1.0)
        
        Returns:
            True if zoom was adjusted
        """
        if frame is None:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False
        
        if person_bbox is None or len(person_bbox) < 4:
            # No person detected, zoom out
            if self.zoom_level > 1.0:
                self.set_digital_zoom(1.0)
                return True
            return False
        
        h, w = frame.shape[:2]
        px1, py1, px2, py2 = person_bbox[:4]
        
        # Calculate person size
        person_width = px2 - px1
        person_height = py2 - py1
        person_size = max(person_width, person_height)
        frame_size = max(w, h)
        
        current_ratio = person_size / frame_size
        
        # Calculate desired zoom level
        if current_ratio < target_size_ratio:
            # Person too small, zoom in
            desired_zoom = target_size_ratio / current_ratio
            desired_zoom = min(desired_zoom, 3.0)  # Max 3x zoom
        else:
            # Person too large, zoom out
            desired_zoom = target_size_ratio / current_ratio
            desired_zoom = max(desired_zoom, 0.5)  # Min 0.5x zoom
        
        # Smooth zoom transition
        smoothed_zoom = self.zoom_level + (desired_zoom - self.zoom_level) * self.zoom_smoothing
        
        # Set center to person center
        person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
        
        self.set_digital_zoom(smoothed_zoom, person_center)
        return True

    def auto_zoom_to_bed(self, bed_region: Optional[Tuple[int, int, int, int]],
                         padding: float = 0.2) -> bool:
        """
        Automatically zoom to bed region.
        
        Args:
            bed_region: Bed region (x1, y1, x2, y2)
            padding: Padding around bed (fraction)
        
        Returns:
            True if zoom was set
        """
        if bed_region is None or len(bed_region) < 4:
            return False
        
        self.set_zoom_region(bed_region)
        return True

    def reset_zoom(self):
        """Reset zoom to default (no zoom)."""
        self.zoom_level = 1.0
        self.zoom_center = None
        self.zoom_region = None
        self.digital_zoom_enabled = False
        log.info("Zoom reset to default")

    def read(self):
        """Reads a valid frame, retries if needed. Returns None on failure."""
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Apply digital zoom if enabled
                if self.digital_zoom_enabled and self.zoom_level > 1.0:
                    frame = self.apply_digital_zoom(frame)
                return frame

            time.sleep(0.05)

        log.warning("Camera read failed after retries, returning None")
        return None  # Graceful failure instead of exception

    def release(self):
        """Safely release camera."""
        try:
            if self.cap.isOpened():
                self.cap.release()
        except:
            pass
