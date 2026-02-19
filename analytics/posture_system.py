# analytics/posture_system.py
"""
Unified Posture Detection System - Enterprise Grade.
Integrates SupportSurfaceDetector, PostureStateMachine, and StateCache
for robust, real-time posture detection.

Architecture:
    Frame + Keypoints
           │
           ▼
    ┌─────────────────────┐
    │ SupportSurfaceDetector │  ─── Detects support surfaces from depth + stability
    └─────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │  PostureStateMachine  │  ─── Classifies posture with temporal hysteresis
    └─────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │      StateCache       │  ─── Authoritative state (RAM), async DB persistence
    └─────────────────────┘
           │
           ▼
       Final Output: Posture, Support, Activity, Confidence

Expected output for person sitting on couch:
    Posture: SITTING
    Support: SURFACE_1 (elevated)
    Subtype: leaning_back
    Confidence: 0.71
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from analytics.support_surface_detector import (
    SupportSurfaceDetector,
    get_support_surface_detector,
)
from analytics.posture_state_machine import (
    PostureStateMachine,
    PostureState,
    PostureOutput,
    get_posture_state_machine,
)
from analytics.state_cache import (
    StateCache,
    get_state_cache,
)

log = logging.getLogger("posture_system")


class PostureSystem:
    """
    Unified posture detection system combining:
    1. Support surface detection (furniture-agnostic)
    2. Posture state machine with temporal hysteresis
    3. State cache as authoritative source of truth

    Usage:
        system = PostureSystem()

        # Per-frame update
        result = system.process_frame(
            person_id="person_1",
            keypoints=[(0.5, 0.3, 0.9), ...],  # COCO format
            frame_shape=(720, 1280),
            depth_map=None  # Optional
        )

        print(result)
        # {
        #     "posture": "SITTING",
        #     "confidence": 0.71,
        #     "support": "SURFACE_1 (elevated)",
        #     "subtype": "leaning_back",
        #     "activity": "RESTING",
        #     "temporal_stability": 0.85
        # }
    """

    def __init__(
        self,
        stability_window: int = 8,
        state_persistence_time: float = 0.8,
        enable_depth_estimation: bool = True,
    ):
        """
        Args:
            stability_window: Frames for pelvis stability detection
            state_persistence_time: Time before confirming state change
            enable_depth_estimation: Whether to use depth-based features
        """
        self.enable_depth_estimation = enable_depth_estimation

        # Initialize components
        self.surface_detector = SupportSurfaceDetector(
            stability_window=stability_window,
        )
        self.state_machine = PostureStateMachine(
            state_persistence_time=state_persistence_time,
        )
        self.state_cache = get_state_cache()

        # Ensure cache is started
        if not self.state_cache._running:
            self.state_cache.start()

        self._lock = threading.Lock()

        log.info(
            "PostureSystem initialized (stability_window=%d, persistence=%.1fs)",
            stability_window, state_persistence_time
        )

    def process_frame(
        self,
        person_id: str,
        keypoints: List[Tuple[float, float, float]],
        frame_shape: Tuple[int, int] = (720, 1280),
        depth_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Process a single frame for posture detection.

        Args:
            person_id: Unique identifier for tracked person
            keypoints: List of (x, y, confidence) in COCO format
                       Coordinates can be normalized (0-1) or pixel coords
            frame_shape: (height, width) of frame
            depth_map: Optional depth map for improved depth estimation

        Returns:
            Dict with posture state, confidence, support surface, etc.
        """
        with self._lock:
            # Step 1: Detect support surface
            surface = self.surface_detector.update(
                person_id=person_id,
                kps=keypoints,
                depth_map=depth_map,
                frame_shape=frame_shape,
            )

            pelvis_supported = surface is not None and surface.confidence > 0.5
            surface_id = surface.surface_id if surface else None
            surface_type = surface.surface_type if surface else None

            # Step 2: Update posture state machine
            posture_output = self.state_machine.update(
                person_id=person_id,
                kps=keypoints,
                pelvis_supported=pelvis_supported,
                support_surface_id=surface_id,
            )

            # Step 3: Update state cache (authoritative source)
            state_changed = self.state_cache.update_posture(
                person_id=person_id,
                posture=posture_output.state.value,
                confidence=posture_output.confidence,
                subtype=posture_output.subtype,
                support_surface_id=surface_id,
                support_surface_type=surface_type,
            )

            # Step 4: Infer activity from posture
            activity = self._infer_activity(posture_output, pelvis_supported)
            self.state_cache.update_activity(
                person_id=person_id,
                activity=activity,
                confidence=posture_output.confidence * 0.9,
            )

            # Build result
            result = {
                "posture": posture_output.state.value,
                "confidence": round(posture_output.confidence, 2),
                "subtype": posture_output.subtype,
                "support_surface_id": surface_id,
                "support_surface_type": surface_type,
                "support": f"{surface_id} ({surface_type})" if surface_id else None,
                "activity": activity,
                "temporal_stability": round(posture_output.temporal_stability, 2),
                "time_in_state": round(posture_output.time_in_state, 1),
                "raw_state": posture_output.raw_state.value,
                "state_changed": state_changed,
            }

            return result

    def _infer_activity(
        self,
        posture: PostureOutput,
        pelvis_supported: bool
    ) -> str:
        """
        Infer high-level activity from posture.

        Simple rule-based inference:
        - SITTING + leaning_back → RESTING
        - SITTING + upright → ACTIVE_SITTING (working, etc.)
        - LYING → RESTING or SLEEPING
        - STANDING → ACTIVE
        """
        if posture.state == PostureState.LYING:
            return "RESTING"

        if posture.state == PostureState.SITTING:
            if posture.subtype in ("leaning_back", "slouched"):
                return "RESTING"
            else:
                return "SITTING"

        if posture.state == PostureState.STANDING:
            return "ACTIVE"

        if posture.state == PostureState.TRANSITIONING:
            return "TRANSITIONING"

        return "UNKNOWN"

    def get_state(self, person_id: str) -> Dict[str, Any]:
        """
        Get current state for a person from cache.
        Always instant (reads from RAM).
        """
        return self.state_cache.get_display_state(person_id)

    def get_posture(self, person_id: str) -> str:
        """Quick access to current posture."""
        state = self.state_cache.get_person_state(person_id)
        return state.posture if state else "UNKNOWN"

    def get_confidence(self, person_id: str) -> float:
        """Quick access to current confidence."""
        state = self.state_cache.get_person_state(person_id)
        return state.posture_confidence if state else 0.0

    def remove_person(self, person_id: str):
        """Remove person from all tracking."""
        self.surface_detector.clear_person(person_id)
        self.state_machine.clear_person(person_id)
        self.state_cache.remove_person(person_id)

    def cleanup_stale(self, timeout: float = 30.0):
        """Clean up stale persons."""
        self.state_cache.cleanup_stale()

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "surface_detector": self.surface_detector.get_stats(),
            "state_machine": self.state_machine.get_stats(),
            "state_cache": self.state_cache.get_stats(),
        }

    def get_display_overlay(self, person_id: str) -> Dict[str, str]:
        """
        Get formatted strings for display overlay.

        Returns:
            Dict with formatted display strings
        """
        state = self.state_cache.get_display_state(person_id)

        posture_str = state["posture"]
        if state.get("posture_subtype"):
            posture_str += f" ({state['posture_subtype']})"

        confidence_str = f"{state['posture_confidence']:.2f}"

        support_str = state.get("support") or "None"

        activity_str = state.get("activity", "UNKNOWN")

        return {
            "posture_line": f"Posture: {posture_str}",
            "confidence_line": f"Confidence: {confidence_str}",
            "support_line": f"Support: {support_str}",
            "activity_line": f"Activity: {activity_str}",
        }


# Module-level singleton
_posture_system: Optional[PostureSystem] = None
_system_lock = threading.Lock()


def get_posture_system(**kwargs) -> PostureSystem:
    """Get or create global posture system."""
    global _posture_system
    if _posture_system is None:
        with _system_lock:
            if _posture_system is None:
                _posture_system = PostureSystem(**kwargs)
    return _posture_system


def process_keypoints(
    person_id: str,
    keypoints: List[Tuple[float, float, float]],
    frame_shape: Tuple[int, int] = (720, 1280),
    depth_map: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Convenience function for processing keypoints.
    Uses the global posture system singleton.

    Args:
        person_id: Unique person identifier
        keypoints: COCO format keypoints [(x, y, conf), ...]
        frame_shape: (height, width)
        depth_map: Optional depth map

    Returns:
        Posture detection result dict
    """
    system = get_posture_system()
    return system.process_frame(
        person_id=person_id,
        keypoints=keypoints,
        frame_shape=frame_shape,
        depth_map=depth_map,
    )
