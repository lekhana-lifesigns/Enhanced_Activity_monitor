# analytics/support_surface_detector.py
"""
Support Surface Detector for Posture Recognition.
Enterprise-grade implementation that infers support surfaces from depth + stability,
NOT furniture classification.

A support surface is detected when:
1. Horizontally stable region exists
2. Depth variance is low within the region
3. Human pelvis remains at constant relative depth for > N frames

This enables proper sitting detection regardless of furniture type (couch, chair, bed, stool).
"""

import time
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

log = logging.getLogger("support_surface_detector")

# Constants
MIN_KEYPOINT_CONFIDENCE = 0.3
PELVIS_STABILITY_FRAMES = 8  # ~0.5s at 15 FPS
PELVIS_DEPTH_VARIANCE_THRESHOLD = 0.15  # meters
PELVIS_HEIGHT_VARIANCE_THRESHOLD = 0.08  # normalized units
SURFACE_PERSISTENCE_TIME = 0.8  # seconds


@dataclass
class SupportSurface:
    """Detected support surface."""
    surface_id: str
    surface_type: str  # "ground", "elevated", "unknown"
    estimated_height: float  # meters from ground
    depth_estimate: float  # meters from camera
    confidence: float
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    stability_score: float = 0.0

    def is_active(self, timeout: float = 2.0) -> bool:
        """Check if surface is still actively detected."""
        return time.time() - self.last_seen < timeout

    def update(self, height: float, depth: float, confidence: float):
        """Update surface with new observation."""
        # Exponential moving average for smoothing
        alpha = 0.3
        self.estimated_height = alpha * height + (1 - alpha) * self.estimated_height
        self.depth_estimate = alpha * depth + (1 - alpha) * self.depth_estimate
        self.confidence = alpha * confidence + (1 - alpha) * self.confidence
        self.last_seen = time.time()


@dataclass
class PelvisObservation:
    """Single pelvis observation for tracking."""
    timestamp: float
    y_position: float  # normalized vertical position
    depth_estimate: float  # estimated depth
    confidence: float


class SupportSurfaceDetector:
    """
    Detects support surfaces from pose keypoints and depth estimation.

    Core principle: A surface supports the human if the pelvis remains
    at a stable position (both vertically and in depth) over time.

    This is furniture-agnostic - works for couch, chair, bed, bench, etc.
    """

    def __init__(
        self,
        stability_window: int = PELVIS_STABILITY_FRAMES,
        depth_variance_threshold: float = PELVIS_DEPTH_VARIANCE_THRESHOLD,
        height_variance_threshold: float = PELVIS_HEIGHT_VARIANCE_THRESHOLD,
        min_persistence_time: float = SURFACE_PERSISTENCE_TIME,
    ):
        """
        Args:
            stability_window: Number of frames for stability calculation
            depth_variance_threshold: Max depth variance for stable surface
            height_variance_threshold: Max height variance for stable surface
            min_persistence_time: Min time surface must persist to be valid
        """
        self.stability_window = stability_window
        self.depth_variance_threshold = depth_variance_threshold
        self.height_variance_threshold = height_variance_threshold
        self.min_persistence_time = min_persistence_time

        # Pelvis tracking per person
        self._pelvis_history: Dict[str, deque] = {}  # person_id -> observations
        self._lock = threading.RLock()

        # Detected surfaces
        self._surfaces: Dict[str, SupportSurface] = {}
        self._surface_counter = 0

        log.info(
            "SupportSurfaceDetector initialized (window=%d, depth_var=%.2f, height_var=%.2f)",
            stability_window, depth_variance_threshold, height_variance_threshold
        )

    def _get_pelvis_position(
        self,
        kps: List[Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Extract pelvis position from keypoints.

        Args:
            kps: Keypoints [(x, y, confidence), ...]

        Returns:
            (x, y, confidence) of pelvis center or None
        """
        if not kps or len(kps) < 13:
            return None

        # COCO format: 11=left_hip, 12=right_hip
        left_hip = kps[11] if len(kps) > 11 else (0, 0, 0)
        right_hip = kps[12] if len(kps) > 12 else (0, 0, 0)

        valid_hips = []
        if left_hip[2] > MIN_KEYPOINT_CONFIDENCE:
            valid_hips.append(left_hip)
        if right_hip[2] > MIN_KEYPOINT_CONFIDENCE:
            valid_hips.append(right_hip)

        if not valid_hips:
            return None

        # Average hip positions
        x = np.mean([h[0] for h in valid_hips])
        y = np.mean([h[1] for h in valid_hips])
        conf = np.mean([h[2] for h in valid_hips])

        return (float(x), float(y), float(conf))

    def _estimate_depth_from_keypoints(
        self,
        kps: List[Tuple[float, float, float]],
        frame_height: int = 720
    ) -> float:
        """
        Estimate depth using monocular cues (size-based estimation).

        Args:
            kps: Keypoints
            frame_height: Frame height in pixels

        Returns:
            Estimated depth in meters
        """
        if not kps:
            return 2.0  # Default 2 meters

        # Get vertical extent of visible keypoints
        y_coords = [kp[1] for kp in kps if kp[2] > MIN_KEYPOINT_CONFIDENCE]
        if len(y_coords) < 2:
            return 2.0

        apparent_height = max(y_coords) - min(y_coords)

        if apparent_height < 0.01:
            return 5.0  # Very small = far away

        # Assuming average person height of 1.7m and typical camera setup
        reference_height = 1.7

        if apparent_height <= 1.0:  # Normalized coordinates
            focal_length_normalized = 0.7
            depth = (reference_height * focal_length_normalized) / apparent_height
        else:  # Pixel coordinates
            depth = (reference_height * 500) / apparent_height

        return float(np.clip(depth, 0.5, 10.0))

    def update(
        self,
        person_id: str,
        kps: List[Tuple[float, float, float]],
        depth_map: Optional[np.ndarray] = None,
        frame_shape: Tuple[int, int] = (720, 1280),
    ) -> Optional[SupportSurface]:
        """
        Update support surface detection with new keypoint observation.

        Args:
            person_id: Unique identifier for tracked person
            kps: Keypoints [(x, y, confidence), ...]
            depth_map: Optional depth map for more accurate depth
            frame_shape: (height, width) of frame

        Returns:
            Detected support surface or None
        """
        with self._lock:
            pelvis = self._get_pelvis_position(kps)
            if pelvis is None:
                return self._get_current_surface(person_id)

            px, py, pconf = pelvis

            # Estimate depth
            if depth_map is not None:
                # Use depth map if available
                h, w = depth_map.shape[:2]
                px_pixel = int(px * w) if px <= 1.0 else int(px)
                py_pixel = int(py * h) if py <= 1.0 else int(py)
                px_pixel = np.clip(px_pixel, 0, w - 1)
                py_pixel = np.clip(py_pixel, 0, h - 1)
                depth = float(depth_map[py_pixel, px_pixel])
            else:
                # Use keypoint-based estimation
                depth = self._estimate_depth_from_keypoints(kps, frame_shape[0])

            # Create observation
            obs = PelvisObservation(
                timestamp=time.time(),
                y_position=py,
                depth_estimate=depth,
                confidence=pconf
            )

            # Initialize history if needed
            if person_id not in self._pelvis_history:
                self._pelvis_history[person_id] = deque(maxlen=self.stability_window * 2)

            self._pelvis_history[person_id].append(obs)

            # Check for support surface
            return self._detect_surface(person_id)

    def _detect_surface(self, person_id: str) -> Optional[SupportSurface]:
        """
        Detect if pelvis is supported by a surface based on stability.

        Returns:
            Detected support surface or None
        """
        history = self._pelvis_history.get(person_id, deque())

        if len(history) < self.stability_window:
            return self._get_current_surface(person_id)

        # Get recent observations
        recent = list(history)[-self.stability_window:]

        # Calculate variances
        heights = [obs.y_position for obs in recent]
        depths = [obs.depth_estimate for obs in recent]
        confidences = [obs.confidence for obs in recent]

        height_var = np.var(heights)
        depth_var = np.var(depths)
        avg_confidence = np.mean(confidences)

        # Check stability criteria
        height_stable = height_var < self.height_variance_threshold
        depth_stable = depth_var < self.depth_variance_threshold

        if height_stable and depth_stable and avg_confidence > 0.4:
            # Pelvis is stable - support surface detected
            avg_height = np.mean(heights)
            avg_depth = np.mean(depths)

            # Determine surface type based on pelvis height
            # Higher pelvis (lower y in normalized coords) = elevated surface
            if avg_height < 0.55:  # Upper half of frame
                surface_type = "elevated"  # Chair, couch, bed
            elif avg_height > 0.85:
                surface_type = "ground"  # Floor sitting
            else:
                surface_type = "elevated"  # Most common case

            # Calculate stability score
            stability_score = 1.0 - (height_var / self.height_variance_threshold +
                                    depth_var / self.depth_variance_threshold) / 2
            stability_score = np.clip(stability_score, 0.0, 1.0)

            # Calculate confidence
            time_span = recent[-1].timestamp - recent[0].timestamp
            time_factor = min(1.0, time_span / self.min_persistence_time)
            surface_confidence = stability_score * avg_confidence * time_factor

            # Get or create surface
            surface_key = f"{person_id}_surface"

            if surface_key in self._surfaces:
                surface = self._surfaces[surface_key]
                surface.update(avg_height, avg_depth, surface_confidence)
                surface.stability_score = stability_score
            else:
                self._surface_counter += 1
                surface = SupportSurface(
                    surface_id=f"SURFACE_{self._surface_counter}",
                    surface_type=surface_type,
                    estimated_height=avg_height,
                    depth_estimate=avg_depth,
                    confidence=surface_confidence,
                    stability_score=stability_score
                )
                self._surfaces[surface_key] = surface
                log.debug("New support surface detected: %s (type=%s, conf=%.2f)",
                         surface.surface_id, surface_type, surface_confidence)

            return surface

        return self._get_current_surface(person_id)

    def _get_current_surface(self, person_id: str) -> Optional[SupportSurface]:
        """Get current active surface for person if still valid."""
        surface_key = f"{person_id}_surface"
        surface = self._surfaces.get(surface_key)
        if surface and surface.is_active():
            return surface
        return None

    def is_pelvis_supported(self, person_id: str) -> Tuple[bool, float]:
        """
        Check if person's pelvis is supported by a surface.

        Returns:
            (is_supported, confidence)
        """
        surface = self._get_current_surface(person_id)
        if surface and surface.is_active():
            return True, surface.confidence
        return False, 0.0

    def get_surface_info(self, person_id: str) -> Dict[str, Any]:
        """Get detailed surface information for a person."""
        surface = self._get_current_surface(person_id)
        if surface is None:
            return {
                "supported": False,
                "surface_id": None,
                "surface_type": None,
                "confidence": 0.0,
                "stability_score": 0.0,
            }

        return {
            "supported": True,
            "surface_id": surface.surface_id,
            "surface_type": surface.surface_type,
            "estimated_height_m": surface.estimated_height,
            "depth_m": surface.depth_estimate,
            "confidence": surface.confidence,
            "stability_score": surface.stability_score,
            "age_seconds": time.time() - surface.first_seen,
        }

    def clear_person(self, person_id: str):
        """Clear tracking data for a person."""
        with self._lock:
            self._pelvis_history.pop(person_id, None)
            self._surfaces.pop(f"{person_id}_surface", None)

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        with self._lock:
            active_surfaces = sum(1 for s in self._surfaces.values() if s.is_active())
            return {
                "tracked_persons": len(self._pelvis_history),
                "total_surfaces_detected": self._surface_counter,
                "active_surfaces": active_surfaces,
                "stability_window": self.stability_window,
            }


# Singleton instance
_detector: Optional[SupportSurfaceDetector] = None
_detector_lock = threading.Lock()


def get_support_surface_detector(**kwargs) -> SupportSurfaceDetector:
    """Get or create global support surface detector."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = SupportSurfaceDetector(**kwargs)
    return _detector
