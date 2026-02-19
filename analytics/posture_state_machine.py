# analytics/posture_state_machine.py
"""
Posture State Machine with Temporal Hysteresis.
Enterprise-grade implementation for stable posture detection.

Key features:
1. Temporal hysteresis - posture must persist ≥ 0.8s before state change
2. Confidence decoupled from detector confidence
3. Smooth transitions with decay for temporary occlusions
4. Support surface integration for sitting detection

Posture States:
- STANDING: Upright, pelvis not supported
- SITTING: Pelvis supported, torso upright (works for couch, chair, bed edge)
- LYING: Horizontal orientation, typically on bed/couch
- TRANSITIONING: Between states (prevents flicker)
- UNKNOWN: Insufficient data (graceful fallback)
"""

import time
import logging
import threading
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

log = logging.getLogger("posture_state_machine")

# Constants
MIN_KEYPOINT_CONFIDENCE = 0.3
STATE_PERSISTENCE_TIME = 0.8  # seconds before confirming state change
UNKNOWN_DECAY_TIME = 1.5  # seconds before switching to UNKNOWN
CONFIDENCE_SMOOTHING_ALPHA = 0.3


class PostureState(Enum):
    """Posture states."""
    STANDING = "STANDING"
    SITTING = "SITTING"
    LYING = "LYING"
    TRANSITIONING = "TRANSITIONING"
    UNKNOWN = "UNKNOWN"


@dataclass
class PostureObservation:
    """Single posture observation."""
    timestamp: float
    raw_state: PostureState
    confidence: float
    support_surface: bool
    torso_angle: float  # degrees from vertical
    hip_knee_angle: float  # thigh angle for sitting detection
    visibility_ratio: float  # fraction of keypoints visible


@dataclass
class PostureOutput:
    """Final posture output with full context."""
    state: PostureState
    confidence: float
    support_surface_id: Optional[str]
    subtype: Optional[str]  # e.g., "upright", "slouched", "leaning_back"
    temporal_stability: float
    raw_state: PostureState
    time_in_state: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "confidence": round(self.confidence, 2),
            "support_surface_id": self.support_surface_id,
            "subtype": self.subtype,
            "temporal_stability": round(self.temporal_stability, 2),
            "raw_state": self.raw_state.value,
            "time_in_state": round(self.time_in_state, 2),
        }


class PostureStateMachine:
    """
    State machine for robust posture detection with temporal hysteresis.

    Core logic:
    - if pelvis_supported and torso_upright: SITTING
    - if not pelvis_supported and torso_upright: STANDING
    - if torso_horizontal: LYING
    - Apply temporal hysteresis to prevent flicker

    Confidence calculation:
    - f(temporal_stability, support_surface_consistency, joint_visibility, state_persistence)
    """

    def __init__(
        self,
        state_persistence_time: float = STATE_PERSISTENCE_TIME,
        unknown_decay_time: float = UNKNOWN_DECAY_TIME,
        observation_window: int = 15,  # ~1 second at 15 FPS
    ):
        """
        Args:
            state_persistence_time: Time before confirming state change
            unknown_decay_time: Time before falling back to UNKNOWN
            observation_window: Number of observations to track
        """
        self.state_persistence_time = state_persistence_time
        self.unknown_decay_time = unknown_decay_time
        self.observation_window = observation_window

        # Per-person state tracking
        self._person_states: Dict[str, "_PersonState"] = {}
        self._lock = threading.RLock()

        log.info(
            "PostureStateMachine initialized (persistence=%.1fs, unknown_decay=%.1fs)",
            state_persistence_time, unknown_decay_time
        )

    def _get_person_state(self, person_id: str) -> "_PersonState":
        """Get or create person state tracker."""
        if person_id not in self._person_states:
            self._person_states[person_id] = _PersonState(
                observation_window=self.observation_window
            )
        return self._person_states[person_id]

    def _compute_torso_angle(
        self,
        kps: List[Tuple[float, float, float]]
    ) -> Tuple[float, float]:
        """
        Compute torso angle from vertical.

        Returns:
            (angle_degrees, confidence)
        """
        if not kps or len(kps) < 13:
            return 0.0, 0.0

        # Get shoulder and hip keypoints
        # COCO: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
        left_shoulder = kps[5] if len(kps) > 5 else (0, 0, 0)
        right_shoulder = kps[6] if len(kps) > 6 else (0, 0, 0)
        left_hip = kps[11] if len(kps) > 11 else (0, 0, 0)
        right_hip = kps[12] if len(kps) > 12 else (0, 0, 0)

        # Calculate shoulder and hip centers
        shoulder_points = []
        hip_points = []

        if left_shoulder[2] > MIN_KEYPOINT_CONFIDENCE:
            shoulder_points.append(left_shoulder)
        if right_shoulder[2] > MIN_KEYPOINT_CONFIDENCE:
            shoulder_points.append(right_shoulder)
        if left_hip[2] > MIN_KEYPOINT_CONFIDENCE:
            hip_points.append(left_hip)
        if right_hip[2] > MIN_KEYPOINT_CONFIDENCE:
            hip_points.append(right_hip)

        if not shoulder_points or not hip_points:
            return 0.0, 0.0

        shoulder_center = (
            np.mean([p[0] for p in shoulder_points]),
            np.mean([p[1] for p in shoulder_points])
        )
        hip_center = (
            np.mean([p[0] for p in hip_points]),
            np.mean([p[1] for p in hip_points])
        )

        # Calculate angle from vertical
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]

        # In image coordinates, y increases downward
        # For upright person: dy should be negative (shoulder above hip)
        angle = math.degrees(math.atan2(abs(dx), -dy))  # Angle from vertical

        # Confidence based on keypoint visibility
        avg_conf = np.mean([p[2] for p in shoulder_points + hip_points])

        return float(angle), float(avg_conf)

    def _compute_hip_knee_angle(
        self,
        kps: List[Tuple[float, float, float]]
    ) -> Tuple[float, float]:
        """
        Compute hip-knee angle (thigh angle from vertical).
        Key for distinguishing sitting from standing.

        Returns:
            (angle_degrees, confidence)
        """
        if not kps or len(kps) < 14:
            return 0.0, 0.0

        # COCO: 11=left_hip, 12=right_hip, 13=left_knee, 14=right_knee
        left_hip = kps[11] if len(kps) > 11 else (0, 0, 0)
        right_hip = kps[12] if len(kps) > 12 else (0, 0, 0)
        left_knee = kps[13] if len(kps) > 13 else (0, 0, 0)
        right_knee = kps[14] if len(kps) > 14 else (0, 0, 0)

        angles = []
        confs = []

        # Left side
        if left_hip[2] > MIN_KEYPOINT_CONFIDENCE and left_knee[2] > MIN_KEYPOINT_CONFIDENCE:
            dx = left_knee[0] - left_hip[0]
            dy = left_knee[1] - left_hip[1]
            angle = abs(math.degrees(math.atan2(dx, dy)))  # From vertical
            angles.append(angle)
            confs.append((left_hip[2] + left_knee[2]) / 2)

        # Right side
        if right_hip[2] > MIN_KEYPOINT_CONFIDENCE and right_knee[2] > MIN_KEYPOINT_CONFIDENCE:
            dx = right_knee[0] - right_hip[0]
            dy = right_knee[1] - right_hip[1]
            angle = abs(math.degrees(math.atan2(dx, dy)))
            angles.append(angle)
            confs.append((right_hip[2] + right_knee[2]) / 2)

        if not angles:
            return 0.0, 0.0

        return float(np.mean(angles)), float(np.mean(confs))

    def _compute_visibility_ratio(
        self,
        kps: List[Tuple[float, float, float]]
    ) -> float:
        """Compute fraction of visible keypoints."""
        if not kps:
            return 0.0
        visible = sum(1 for kp in kps if kp[2] > MIN_KEYPOINT_CONFIDENCE)
        return visible / max(len(kps), 1)

    def _classify_raw_state(
        self,
        torso_angle: float,
        hip_knee_angle: float,
        pelvis_supported: bool,
        torso_conf: float,
        hip_knee_conf: float
    ) -> Tuple[PostureState, float]:
        """
        Classify raw posture state from angles and support.

        Core logic:
        - pelvis_supported AND torso_upright → SITTING
        - NOT pelvis_supported AND torso_upright → STANDING
        - torso_horizontal → LYING

        Returns:
            (state, confidence)
        """
        # Thresholds
        TORSO_UPRIGHT_THRESHOLD = 35.0  # degrees from vertical
        TORSO_LYING_THRESHOLD = 60.0  # degrees
        HIP_KNEE_SITTING_THRESHOLD = 45.0  # degrees (thigh roughly horizontal)

        # Check lying first (most distinctive)
        if torso_angle > TORSO_LYING_THRESHOLD and torso_conf > 0.3:
            return PostureState.LYING, torso_conf * 0.9

        # Check sitting: pelvis supported + torso relatively upright
        # OR hip-knee angle indicates sitting (even without clear support)
        is_torso_upright = torso_angle < TORSO_UPRIGHT_THRESHOLD
        is_sitting_pose = hip_knee_angle > HIP_KNEE_SITTING_THRESHOLD

        if pelvis_supported and is_torso_upright:
            # Clear sitting with support surface
            conf = min(torso_conf, hip_knee_conf) * 0.95 if hip_knee_conf > 0.3 else torso_conf * 0.8
            return PostureState.SITTING, conf

        if is_sitting_pose and is_torso_upright:
            # Sitting pose even without detected support
            # (support may be out of frame or not yet stable)
            conf = hip_knee_conf * 0.75
            return PostureState.SITTING, conf

        if is_torso_upright and not pelvis_supported and hip_knee_angle < 30:
            # Standing: upright torso, no support, legs straight
            return PostureState.STANDING, torso_conf * 0.9

        # Ambiguous - might be transitioning
        if torso_conf > 0.3 or hip_knee_conf > 0.3:
            return PostureState.TRANSITIONING, max(torso_conf, hip_knee_conf) * 0.5

        return PostureState.UNKNOWN, 0.0

    def _classify_sitting_subtype(
        self,
        kps: List[Tuple[float, float, float]],
        torso_angle: float
    ) -> str:
        """Classify sitting subtype: upright, slouched, leaning_forward, leaning_back."""
        if not kps or len(kps) < 7:
            return "upright"

        # Get spine angle (shoulder-hip vector)
        # Positive angle = leaning forward, negative = leaning back
        left_shoulder = kps[5] if len(kps) > 5 else (0, 0, 0)
        right_shoulder = kps[6] if len(kps) > 6 else (0, 0, 0)
        left_hip = kps[11] if len(kps) > 11 else (0, 0, 0)
        right_hip = kps[12] if len(kps) > 12 else (0, 0, 0)

        # Calculate forward lean
        shoulder_x = 0
        hip_x = 0
        count = 0

        if left_shoulder[2] > MIN_KEYPOINT_CONFIDENCE:
            shoulder_x += left_shoulder[0]
            count += 1
        if right_shoulder[2] > MIN_KEYPOINT_CONFIDENCE:
            shoulder_x += right_shoulder[0]
            count += 1

        if count > 0:
            shoulder_x /= count

        count = 0
        if left_hip[2] > MIN_KEYPOINT_CONFIDENCE:
            hip_x += left_hip[0]
            count += 1
        if right_hip[2] > MIN_KEYPOINT_CONFIDENCE:
            hip_x += right_hip[0]
            count += 1

        if count > 0:
            hip_x /= count

        # Forward lean detection (shoulder ahead of hip in normalized coords)
        lean = shoulder_x - hip_x

        if torso_angle > 20:
            if lean > 0.03:
                return "leaning_forward"
            else:
                return "slouched"
        elif lean < -0.03:
            return "leaning_back"  # Typical relaxed couch position
        else:
            return "upright"

    def update(
        self,
        person_id: str,
        kps: List[Tuple[float, float, float]],
        pelvis_supported: bool = False,
        support_surface_id: Optional[str] = None,
    ) -> PostureOutput:
        """
        Update posture state with new observation.

        Args:
            person_id: Unique person identifier
            kps: Keypoints [(x, y, confidence), ...]
            pelvis_supported: Whether support surface detector found support
            support_surface_id: ID of support surface if any

        Returns:
            PostureOutput with final state after hysteresis
        """
        with self._lock:
            person_state = self._get_person_state(person_id)

            # Compute pose features
            torso_angle, torso_conf = self._compute_torso_angle(kps)
            hip_knee_angle, hip_knee_conf = self._compute_hip_knee_angle(kps)
            visibility_ratio = self._compute_visibility_ratio(kps)

            # Classify raw state
            raw_state, raw_conf = self._classify_raw_state(
                torso_angle=torso_angle,
                hip_knee_angle=hip_knee_angle,
                pelvis_supported=pelvis_supported,
                torso_conf=torso_conf,
                hip_knee_conf=hip_knee_conf
            )

            # Create observation
            obs = PostureObservation(
                timestamp=time.time(),
                raw_state=raw_state,
                confidence=raw_conf,
                support_surface=pelvis_supported,
                torso_angle=torso_angle,
                hip_knee_angle=hip_knee_angle,
                visibility_ratio=visibility_ratio
            )

            # Update person state with temporal hysteresis
            final_state = person_state.update(
                obs,
                persistence_time=self.state_persistence_time,
                unknown_decay_time=self.unknown_decay_time
            )

            # Classify subtype if sitting
            subtype = None
            if final_state.state == PostureState.SITTING:
                subtype = self._classify_sitting_subtype(kps, torso_angle)

            return PostureOutput(
                state=final_state.state,
                confidence=final_state.confidence,
                support_surface_id=support_surface_id if pelvis_supported else None,
                subtype=subtype,
                temporal_stability=final_state.temporal_stability,
                raw_state=raw_state,
                time_in_state=final_state.time_in_state
            )

    def get_state(self, person_id: str) -> Optional[PostureOutput]:
        """Get current state for a person without updating."""
        with self._lock:
            if person_id not in self._person_states:
                return None
            person_state = self._person_states[person_id]
            return person_state.get_output()

    def clear_person(self, person_id: str):
        """Clear state for a person."""
        with self._lock:
            self._person_states.pop(person_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get state machine statistics."""
        with self._lock:
            state_counts = {}
            for ps in self._person_states.values():
                state_name = ps.confirmed_state.value
                state_counts[state_name] = state_counts.get(state_name, 0) + 1

            return {
                "tracked_persons": len(self._person_states),
                "state_distribution": state_counts,
                "persistence_time": self.state_persistence_time,
            }


@dataclass
class _FinalState:
    """Internal final state representation."""
    state: PostureState
    confidence: float
    temporal_stability: float
    time_in_state: float


class _PersonState:
    """Per-person state tracking with hysteresis."""

    def __init__(self, observation_window: int = 15):
        self.observation_window = observation_window
        self.observations: deque = deque(maxlen=observation_window)

        # Confirmed state (after hysteresis)
        self.confirmed_state = PostureState.UNKNOWN
        self.confirmed_state_time = time.time()
        self.confirmed_confidence = 0.0

        # Pending state (waiting for persistence)
        self.pending_state: Optional[PostureState] = None
        self.pending_state_start: float = 0.0

        # Last valid observation time
        self.last_valid_observation = time.time()

    def update(
        self,
        obs: PostureObservation,
        persistence_time: float,
        unknown_decay_time: float
    ) -> _FinalState:
        """Update state with new observation and apply hysteresis."""
        self.observations.append(obs)
        now = obs.timestamp

        # Track last valid observation
        if obs.raw_state != PostureState.UNKNOWN:
            self.last_valid_observation = now

        # Check for unknown decay (no valid observations for too long)
        time_since_valid = now - self.last_valid_observation
        if time_since_valid > unknown_decay_time:
            self._confirm_state(PostureState.UNKNOWN, 0.0, now)
            return self._make_final_state(now)

        # If current raw state matches confirmed state, reinforce it
        if obs.raw_state == self.confirmed_state:
            self.pending_state = None
            # Smooth confidence update
            alpha = CONFIDENCE_SMOOTHING_ALPHA
            self.confirmed_confidence = (
                alpha * obs.confidence +
                (1 - alpha) * self.confirmed_confidence
            )
            return self._make_final_state(now)

        # Raw state differs from confirmed state
        if obs.raw_state == PostureState.UNKNOWN:
            # Don't immediately switch to unknown - use decay
            return self._make_final_state(now)

        # New potential state detected
        if self.pending_state != obs.raw_state:
            # Start new pending state
            self.pending_state = obs.raw_state
            self.pending_state_start = now
        else:
            # Same pending state - check persistence
            pending_duration = now - self.pending_state_start
            if pending_duration >= persistence_time:
                # State persisted long enough - confirm it
                self._confirm_state(obs.raw_state, obs.confidence, now)

        return self._make_final_state(now)

    def _confirm_state(self, state: PostureState, confidence: float, timestamp: float):
        """Confirm a state change."""
        if state != self.confirmed_state:
            log.debug("State transition: %s -> %s (conf=%.2f)",
                     self.confirmed_state.value, state.value, confidence)
        self.confirmed_state = state
        self.confirmed_state_time = timestamp
        self.confirmed_confidence = confidence
        self.pending_state = None

    def _make_final_state(self, now: float) -> _FinalState:
        """Create final state output."""
        # Calculate temporal stability from recent observations
        if len(self.observations) < 3:
            temporal_stability = 0.3
        else:
            recent = list(self.observations)[-10:]
            matching = sum(1 for o in recent if o.raw_state == self.confirmed_state)
            temporal_stability = matching / len(recent)

        # Calculate time in current state
        time_in_state = now - self.confirmed_state_time

        # Adjust confidence based on temporal stability
        adjusted_confidence = self.confirmed_confidence * (0.5 + 0.5 * temporal_stability)

        return _FinalState(
            state=self.confirmed_state,
            confidence=adjusted_confidence,
            temporal_stability=temporal_stability,
            time_in_state=time_in_state
        )

    def get_output(self) -> PostureOutput:
        """Get current state as PostureOutput."""
        final = self._make_final_state(time.time())
        return PostureOutput(
            state=final.state,
            confidence=final.confidence,
            support_surface_id=None,
            subtype=None,
            temporal_stability=final.temporal_stability,
            raw_state=self.confirmed_state,
            time_in_state=final.time_in_state
        )


# Singleton instance
_state_machine: Optional[PostureStateMachine] = None
_sm_lock = threading.Lock()


def get_posture_state_machine(**kwargs) -> PostureStateMachine:
    """Get or create global posture state machine."""
    global _state_machine
    if _state_machine is None:
        with _sm_lock:
            if _state_machine is None:
                _state_machine = PostureStateMachine(**kwargs)
    return _state_machine
