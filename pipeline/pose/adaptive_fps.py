# pipeline/pose/adaptive_fps.py
"""
Adaptive FPS Controller for Performance Optimization
Dynamically adjusts frame rates based on activity levels and system load.
"""

import time
import logging
import threading
import numpy as np
from collections import deque
from typing import Optional, Dict, Any

log = logging.getLogger("adaptive_fps")


class AdaptiveFPSController:
    """
    Adaptive frame rate controller that adjusts FPS based on:
    1. Activity levels (higher activity = higher FPS)
    2. System load (higher load = lower FPS)
    3. Motion detection (motion = higher FPS)
    4. Processing latency (high latency = lower FPS)
    """

    def __init__(self, base_fps: float = 15.0, min_fps: float = 5.0, max_fps: float = 30.0,
                 adaptation_window: int = 30, activity_threshold: float = 0.1):
        """
        Initialize adaptive FPS controller.

        Args:
            base_fps: Baseline FPS when no adaptation needed
            min_fps: Minimum allowed FPS
            max_fps: Maximum allowed FPS
            adaptation_window: Number of frames to consider for adaptation
            activity_threshold: Activity level threshold for FPS changes
        """
        self.base_fps = base_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.current_fps = base_fps
        self.target_fps = base_fps

        # Activity tracking
        self.activity_window = deque(maxlen=adaptation_window)
        self.activity_threshold = activity_threshold

        # Performance tracking
        self.latency_window = deque(maxlen=adaptation_window)
        self.load_window = deque(maxlen=adaptation_window)

        # Motion tracking
        self.motion_window = deque(maxlen=adaptation_window)

        # Adaptation parameters
        self.activity_fps_multiplier = 2.0  # FPS multiplier during high activity
        self.motion_fps_multiplier = 1.5  # FPS multiplier during motion
        self.high_load_fps_divider = 2.0  # FPS divider during high system load
        self.high_latency_fps_divider = 1.5  # FPS divider during high latency

        # Thresholds
        self.high_load_threshold = 0.8  # CPU/GPU utilization threshold
        self.high_latency_threshold = 100.0  # Latency threshold in ms
        self.motion_threshold = 0.05  # Motion detection threshold

        # Adaptation smoothing
        self.fps_smoothing_alpha = 0.3  # EMA smoothing factor for FPS changes
        self.last_adaptation_time = time.time()
        self.adaptation_interval = 2.0  # Minimum seconds between adaptations

        # Threading
        self.lock = threading.Lock()
        self.enabled = True

    def update_activity(self, activity_score: float):
        """
        Update activity level measurement.

        Args:
            activity_score: Activity score (0.0 = no activity, 1.0 = high activity)
        """
        with self.lock:
            self.activity_window.append(activity_score)

    def update_motion(self, motion_score: float):
        """
        Update motion detection measurement.

        Args:
            motion_score: Motion score (0.0 = no motion, 1.0 = high motion)
        """
        with self.lock:
            self.motion_window.append(motion_score)

    def update_performance(self, latency_ms: float, system_load: float = None):
        """
        Update performance measurements.

        Args:
            latency_ms: Processing latency in milliseconds
            system_load: System load (0.0-1.0) - if None, estimated from latency
        """
        with self.lock:
            self.latency_window.append(latency_ms)

            if system_load is not None:
                self.load_window.append(system_load)
            else:
                # Estimate load from latency (rough approximation)
                estimated_load = min(1.0, latency_ms / 200.0)  # Assume 200ms = 100% load
                self.load_window.append(estimated_load)

    def calculate_target_fps(self) -> float:
        """
        Calculate optimal FPS based on current conditions.

        Returns:
            Target FPS
        """
        with self.lock:
            if not self.enabled or len(self.activity_window) == 0:
                return self.base_fps

            # Get current averages
            avg_activity = np.mean(self.activity_window)
            avg_motion = np.mean(self.motion_window) if self.motion_window else 0.0
            avg_latency = np.mean(self.latency_window) if self.latency_window else 0.0
            avg_load = np.mean(self.load_window) if self.load_window else 0.0

            # Start with base FPS
            target_fps = self.base_fps

            # Activity-based adaptation
            if avg_activity > self.activity_threshold:
                activity_multiplier = 1.0 + (avg_activity - self.activity_threshold) * (self.activity_fps_multiplier - 1.0)
                target_fps *= activity_multiplier

            # Motion-based adaptation
            if avg_motion > self.motion_threshold:
                motion_multiplier = 1.0 + (avg_motion - self.motion_threshold) * (self.motion_fps_multiplier - 1.0)
                target_fps *= motion_multiplier

            # Load-based adaptation (reduce FPS when system is overloaded)
            if avg_load > self.high_load_threshold:
                load_reduction = 1.0 / (1.0 + (avg_load - self.high_load_threshold) * (self.high_load_fps_divider - 1.0))
                target_fps *= load_reduction

            # Latency-based adaptation (reduce FPS when processing is slow)
            if avg_latency > self.high_latency_threshold:
                latency_reduction = 1.0 / (1.0 + (avg_latency - self.high_latency_threshold) / 100.0 * (self.high_latency_fps_divider - 1.0))
                target_fps *= latency_reduction

            # Clamp to limits
            target_fps = np.clip(target_fps, self.min_fps, self.max_fps)

            return float(target_fps)

    def adapt_fps(self) -> float:
        """
        Perform FPS adaptation and return new target FPS.

        Returns:
            New target FPS
        """
        current_time = time.time()

        # Rate limit adaptations
        if current_time - self.last_adaptation_time < self.adaptation_interval:
            return self.current_fps

        # Calculate new target
        new_target = self.calculate_target_fps()

        # Smooth the transition
        if abs(new_target - self.target_fps) > 0.1:  # Only adapt if change is significant
            self.target_fps = new_target
            self.last_adaptation_time = current_time

        # Apply exponential moving average for smooth transitions
        self.current_fps = self.fps_smoothing_alpha * self.target_fps + \
                           (1 - self.fps_smoothing_alpha) * self.current_fps

        log.debug(f"Adaptive FPS: {self.current_fps:.1f}")

        return self.current_fps

    def should_skip_frame(self, current_fps: float) -> bool:
        """
        Determine if current frame should be skipped based on target FPS.

        Args:
            current_fps: Current actual FPS

        Returns:
            True if frame should be skipped
        """
        if not self.enabled:
            return False

        # Calculate frame interval for target FPS
        target_interval = 1.0 / self.current_fps if self.current_fps > 0 else 0.1

        # Add some jitter to avoid synchronized skipping
        jitter = np.random.uniform(0.8, 1.2)
        effective_interval = target_interval * jitter

        # Check if enough time has passed since last processed frame
        # This is a simplified check - in practice, you'd track last processed time
        return current_fps > self.current_fps * 1.1  # Skip if running significantly faster than target

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current adaptation statistics.

        Returns:
            Dict with adaptation statistics
        """
        with self.lock:
            return {
                "current_fps": self.current_fps,
                "target_fps": self.target_fps,
                "base_fps": self.base_fps,
                "min_fps": self.min_fps,
                "max_fps": self.max_fps,
                "avg_activity": float(np.mean(self.activity_window)) if self.activity_window else 0.0,
                "avg_motion": float(np.mean(self.motion_window)) if self.motion_window else 0.0,
                "avg_latency": float(np.mean(self.latency_window)) if self.latency_window else 0.0,
                "avg_load": float(np.mean(self.load_window)) if self.load_window else 0.0,
                "enabled": self.enabled
            }

    def reset(self):
        """Reset adaptation state."""
        with self.lock:
            self.activity_window.clear()
            self.motion_window.clear()
            self.latency_window.clear()
            self.load_window.clear()
            self.current_fps = self.base_fps
            self.target_fps = self.base_fps

    def enable(self):
        """Enable adaptive FPS."""
        self.enabled = True
        log.info("Adaptive FPS enabled")

    def disable(self):
        """Disable adaptive FPS."""
        self.enabled = False
        self.current_fps = self.base_fps
        self.target_fps = self.base_fps
        log.info("Adaptive FPS disabled")


class ActivityScorer:
    """
    Calculates activity scores from pose keypoints and temporal features.
    """

    def __init__(self):
        self.prev_keypoints = None
        self.activity_history = deque(maxlen=10)

    def calculate_activity_score(self, keypoints: np.ndarray,
                                  temporal_features: Dict[str, Any]) -> float:
        """
        Calculate activity score from keypoints and temporal features.

        Args:
            keypoints: Current keypoints array
            temporal_features: Temporal analysis results

        Returns:
            Activity score (0.0-1.0)
        """
        if keypoints is None:
            return 0.0

        activity_score = 0.0

        # Motion-based activity (keypoint movement)
        if self.prev_keypoints is not None:
            motion_score = self._calculate_motion_score(keypoints, self.prev_keypoints)
            activity_score += motion_score * 0.6  # 60% weight

        # Temporal activity (from temporal model)
        temporal_score = self._extract_temporal_activity(temporal_features)
        activity_score += temporal_score * 0.4  # 40% weight

        # Update history
        self.activity_history.append(activity_score)
        self.prev_keypoints = keypoints.copy()

        # Smooth activity score
        smoothed_score = np.mean(self.activity_history)

        return float(np.clip(smoothed_score, 0.0, 1.0))

    def _calculate_motion_score(self, current_kps: np.ndarray, prev_kps: np.ndarray) -> float:
        """Calculate motion score from keypoint differences."""
        if current_kps.shape != prev_kps.shape:
            return 0.0

        # Calculate Euclidean distance for each keypoint
        distances = []
        for i in range(len(current_kps)):
            if current_kps[i, 2] > 0.3 and prev_kps[i, 2] > 0.3:  # Only use confident keypoints
                dist = np.linalg.norm(current_kps[i, :2] - prev_kps[i, :2])
                distances.append(dist)

        if not distances:
            return 0.0

        # Average motion across keypoints
        avg_motion = np.mean(distances)

        # Normalize to 0-1 scale (rough heuristic: 50 pixels = high motion)
        motion_score = np.clip(avg_motion / 50.0, 0.0, 1.0)

        return motion_score

    def _extract_temporal_activity(self, temporal_features: Dict[str, Any]) -> float:
        """Extract activity level from temporal analysis."""
        if not temporal_features:
            return 0.0

        # Map activity labels to scores
        label_scores = {
            "calm": 0.0,
            "restlessness": 0.3,
            "agitation": 0.7,
            "delirium": 0.9,
            "convulsion": 1.0
        }

        label = temporal_features.get("label", "calm")
        confidence = temporal_features.get("confidence", 0.0)

        base_score = label_scores.get(label.lower(), 0.0)

        # Modulate by confidence
        activity_score = base_score * confidence

        return activity_score
