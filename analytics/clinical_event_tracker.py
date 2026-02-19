# analytics/clinical_event_tracker.py
"""
Clinical Event Tracker for ICU Patient Monitoring
Integrates: Sleep tracking, Agitation detection, Tube-pulling detection,
Vitals + CV inference, Medical intervention detection, Care classification,
and Patient progress reporting.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger("clinical_event_tracker")


class CareType(Enum):
    """Classification of care/interaction types."""
    SELF_CARE = "self_care"           # Patient self-initiated activity
    NURSE_CARE = "nurse_care"         # Nursing intervention
    DOCTOR_CARE = "doctor_care"       # Physician intervention
    FAMILY_CARE = "family_care"       # Family/visitor interaction
    MEDICAL_EMERGENCY = "emergency"   # Emergency intervention
    NO_INTERACTION = "none"           # No active care


class AlertSeverity(Enum):
    """Alert severity levels for clinical escalation."""
    CRITICAL = "critical"   # Immediate physician response (30s SLA)
    HIGH = "high"           # Urgent nursing response (2min SLA)
    MEDIUM = "medium"       # Nursing attention needed (5min SLA)
    LOW = "low"             # Informational (10min SLA)
    INFO = "info"           # Documentation only


class SleepStage(Enum):
    """Sleep stage classification."""
    AWAKE = "awake"
    LIGHT_SLEEP = "light_sleep"
    DEEP_SLEEP = "deep_sleep"
    REM_SLEEP = "rem_sleep"
    RESTLESS = "restless"
    UNKNOWN = "unknown"


@dataclass
class VitalSignReading:
    """Container for vital sign data."""
    timestamp: float
    breath_rate: float = 0.0
    breath_confidence: float = 0.0
    hrv_proxy: float = 0.0
    hrv_confidence: float = 0.0
    motion_energy: float = 0.0
    source: str = "cv_inference"  # cv_inference, device, manual


@dataclass
class ClinicalEvent:
    """Represents a clinical event for tracking and reporting."""
    timestamp: float
    event_type: str
    severity: AlertSeverity
    description: str
    activity: str
    confidence: float
    care_type: CareType = CareType.NO_INTERACTION
    vitals: Optional[VitalSignReading] = None
    intervention_detected: bool = False
    requires_acknowledgment: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class PatientProgress:
    """Patient progress summary for reporting."""
    patient_id: str
    period_start: float
    period_end: float
    sleep_quality_score: float = 0.0
    agitation_trend: str = "stable"
    mobility_trend: str = "stable"
    intervention_count: int = 0
    critical_events: int = 0
    improvement_indicators: List[str] = field(default_factory=list)
    concern_indicators: List[str] = field(default_factory=list)


class SleepTracker:
    """
    Advanced sleep tracking using CV inference and motion analysis.
    Tracks sleep stages, quality, and disturbances.
    """

    def __init__(self, fps: float = 15.0):
        self.fps = fps
        self.motion_history = deque(maxlen=int(fps * 300))  # 5 min buffer
        self.breath_history = deque(maxlen=int(fps * 60))   # 1 min for breathing
        self.sleep_state = SleepStage.UNKNOWN
        self.sleep_start_time = None
        self.disturbance_count = 0
        self.last_position_change = 0.0

        # Thresholds
        self.stillness_threshold = 0.01    # Motion energy for stillness
        self.deep_sleep_threshold = 0.005  # Very low motion
        self.restless_threshold = 0.05     # Restless movement
        self.position_change_cooldown = 30.0  # Seconds between position changes

    def update(self, motion_energy: float, posture: str,
               breath_rate: Optional[float] = None,
               timestamp: Optional[float] = None) -> Dict:
        """
        Update sleep tracking with current frame data.

        Args:
            motion_energy: Current motion energy (0-1)
            posture: Current posture (supine, prone, lateral, etc.)
            breath_rate: Estimated breath rate if available
            timestamp: Current timestamp

        Returns:
            Sleep tracking result with stage, quality metrics
        """
        ts = timestamp or time.time()
        self.motion_history.append(motion_energy)

        if breath_rate:
            self.breath_history.append(breath_rate)

        # Determine if in bed posture
        bed_postures = ['supine', 'prone', 'left_lateral', 'right_lateral', 'fetal', 'side']
        in_bed = posture in bed_postures

        if not in_bed:
            self.sleep_state = SleepStage.AWAKE
            self.sleep_start_time = None
            return self._create_result(ts, in_bed)

        # Analyze motion over windows
        recent_motion = list(self.motion_history)
        if len(recent_motion) < 30:
            return self._create_result(ts, in_bed)

        # Short-term motion (5 seconds)
        short_window = recent_motion[-int(self.fps * 5):]
        short_avg = np.mean(short_window)

        # Medium-term motion (1 minute)
        med_window = recent_motion[-int(self.fps * 60):] if len(recent_motion) >= self.fps * 60 else recent_motion
        med_avg = np.mean(med_window)

        # Long-term motion (5 minutes)
        long_avg = np.mean(recent_motion)

        # Classify sleep stage
        prev_state = self.sleep_state

        if short_avg > self.restless_threshold:
            self.sleep_state = SleepStage.RESTLESS
            self.disturbance_count += 1
        elif short_avg < self.deep_sleep_threshold and med_avg < self.stillness_threshold:
            self.sleep_state = SleepStage.DEEP_SLEEP
            if self.sleep_start_time is None:
                self.sleep_start_time = ts
        elif med_avg < self.stillness_threshold:
            self.sleep_state = SleepStage.LIGHT_SLEEP
            if self.sleep_start_time is None:
                self.sleep_start_time = ts
        else:
            self.sleep_state = SleepStage.AWAKE
            self.sleep_start_time = None

        # Detect position changes
        motion_variance = np.var(short_window)
        if motion_variance > 0.01 and (ts - self.last_position_change) > self.position_change_cooldown:
            self.last_position_change = ts

        return self._create_result(ts, in_bed)

    def _create_result(self, timestamp: float, in_bed: bool) -> Dict:
        """Create sleep tracking result."""
        sleep_duration = 0.0
        if self.sleep_start_time and self.sleep_state in [SleepStage.LIGHT_SLEEP, SleepStage.DEEP_SLEEP]:
            sleep_duration = timestamp - self.sleep_start_time

        # Calculate sleep quality (0-100)
        quality = 100.0
        if self.disturbance_count > 0:
            quality -= min(50, self.disturbance_count * 5)

        # Penalize for too many position changes
        motion_list = list(self.motion_history)
        if len(motion_list) > 100:
            restless_ratio = sum(1 for m in motion_list if m > self.restless_threshold) / len(motion_list)
            quality -= restless_ratio * 30

        quality = max(0, quality)

        return {
            "sleep_stage": self.sleep_state.value,
            "in_bed": in_bed,
            "sleep_duration_seconds": sleep_duration,
            "sleep_quality_score": round(quality, 1),
            "disturbance_count": self.disturbance_count,
            "is_sleeping": self.sleep_state in [SleepStage.LIGHT_SLEEP, SleepStage.DEEP_SLEEP],
            "is_restless": self.sleep_state == SleepStage.RESTLESS,
            "timestamp": timestamp
        }

    def get_sleep_summary(self, hours: int = 8) -> Dict:
        """Get sleep summary for the past N hours."""
        return {
            "total_sleep_time": 0,  # Would need persistent storage
            "deep_sleep_percent": 0,
            "light_sleep_percent": 0,
            "disturbances": self.disturbance_count,
            "quality_score": self._create_result(time.time(), True).get("sleep_quality_score", 0)
        }


class AgitationDetector:
    """
    Advanced agitation detection combining motion, posture, and behavioral patterns.
    Includes specific detection for tube-pulling behavior.
    """

    def __init__(self, fps: float = 15.0):
        self.fps = fps
        self.motion_history = deque(maxlen=int(fps * 60))   # 1 min
        self.jerk_history = deque(maxlen=int(fps * 30))     # 30 sec
        self.hand_proximity_history = deque(maxlen=int(fps * 10))  # 10 sec
        self.agitation_score = 0.0
        self.tube_pull_risk = 0.0

        # Thresholds
        self.agitation_thresholds = {
            "mild": 0.3,
            "moderate": 0.5,
            "severe": 0.7,
            "critical": 0.85
        }

        # Sustained agitation tracking -- require persistence before alerting
        self._sustained_agitation_buffer = deque(maxlen=int(fps * 3))  # 3 seconds
        self._last_agitation_alert_time = 0.0
        self._agitation_alert_cooldown = 30.0  # Minimum 30s between agitation alerts

        # Tube pulling detection zones (COCO keypoints)
        self.face_keypoints = [0, 1, 2, 3, 4]  # Nose, eyes, ears
        self.neck_keypoints = [5, 6]            # Shoulders (proxy for neck area)
        self.wrist_keypoints = [9, 10]          # Wrists

    def detect(self, keypoints: np.ndarray, motion_energy: float,
               jerk_index: float, emotions: Optional[Dict] = None,
               self_contact: Optional[Dict] = None,
               timestamp: Optional[float] = None) -> Dict:
        """
        Detect agitation level and tube-pulling behavior.

        Args:
            keypoints: Current keypoints (17, 3) with confidence
            motion_energy: Current motion energy
            jerk_index: Movement jerkiness (0-1)
            emotions: Emotion detection results
            self_contact: Self-contact detection results
            timestamp: Current timestamp

        Returns:
            Agitation detection result
        """
        ts = timestamp or time.time()
        self.motion_history.append(motion_energy)
        self.jerk_history.append(jerk_index)

        # Calculate hand proximity to face/neck
        hand_proximity = self._calculate_hand_proximity(keypoints)
        self.hand_proximity_history.append(hand_proximity)

        # Compute agitation score from multiple signals
        motion_score = self._compute_motion_agitation()
        jerk_score = self._compute_jerk_agitation()
        emotion_score = self._compute_emotion_agitation(emotions)
        proximity_score = np.mean(list(self.hand_proximity_history)) if self.hand_proximity_history else 0

        # Weighted combination
        self.agitation_score = (
            motion_score * 0.35 +
            jerk_score * 0.25 +
            emotion_score * 0.20 +
            proximity_score * 0.20
        )

        # Sustained agitation check: require ~3 seconds of elevated score
        # before reporting agitation.  This filters transient spikes from
        # normal repositioning or brief emotional expressions.
        self._sustained_agitation_buffer.append(self.agitation_score)
        sustained_score = float(np.mean(list(self._sustained_agitation_buffer)))

        # Tube pulling detection
        tube_pull_result = self._detect_tube_pulling(keypoints, self_contact, motion_energy)
        self.tube_pull_risk = tube_pull_result["risk_score"]

        # Determine agitation level from SUSTAINED score
        level = "none"
        for lvl, threshold in sorted(self.agitation_thresholds.items(), key=lambda x: x[1], reverse=True):
            if sustained_score >= threshold:
                level = lvl
                break

        # Apply alert cooldown -- suppress repeated agitation alerts
        is_agitated = sustained_score >= self.agitation_thresholds["mild"]
        requires_intervention = level in ["severe", "critical"] or tube_pull_result["detected"]
        if is_agitated and requires_intervention:
            if (ts - self._last_agitation_alert_time) < self._agitation_alert_cooldown:
                requires_intervention = False
            else:
                self._last_agitation_alert_time = ts

        return {
            "agitation_score": round(sustained_score, 3),
            "agitation_level": level,
            "is_agitated": is_agitated,
            "tube_pull_risk": round(self.tube_pull_risk, 3),
            "tube_pull_detected": tube_pull_result["detected"],
            "tube_pull_type": tube_pull_result.get("type", None),
            "components": {
                "motion": round(motion_score, 3),
                "jerk": round(jerk_score, 3),
                "emotion": round(emotion_score, 3),
                "hand_proximity": round(proximity_score, 3)
            },
            "requires_intervention": requires_intervention,
            "timestamp": ts
        }

    def _calculate_hand_proximity(self, keypoints: np.ndarray) -> float:
        """Calculate how close hands are to face/neck area."""
        if keypoints is None or len(keypoints) < 17:
            return 0.0

        try:
            # Get face center (nose)
            nose = keypoints[0, :2]
            nose_conf = keypoints[0, 2]

            # Get wrist positions
            left_wrist = keypoints[9, :2]
            left_conf = keypoints[9, 2]
            right_wrist = keypoints[10, :2]
            right_conf = keypoints[10, 2]

            if nose_conf < 0.3:
                return 0.0

            proximities = []
            for wrist, conf in [(left_wrist, left_conf), (right_wrist, right_conf)]:
                if conf > 0.3:
                    dist = np.linalg.norm(wrist - nose)
                    # Normalize: closer = higher score
                    # Assume face width is ~0.15-0.2 of frame
                    proximity = max(0, 1.0 - dist / 0.3)
                    proximities.append(proximity)

            return max(proximities) if proximities else 0.0
        except Exception:
            return 0.0

    def _compute_motion_agitation(self) -> float:
        """Compute agitation from motion patterns."""
        if len(self.motion_history) < 10:
            return 0.0

        motion = list(self.motion_history)
        avg = np.mean(motion)
        variance = np.var(motion)

        # High average + high variance = agitation
        return min(1.0, avg * 1.5 + variance * 10)

    def _compute_jerk_agitation(self) -> float:
        """Compute agitation from jerk (sudden movements)."""
        if len(self.jerk_history) < 5:
            return 0.0

        jerk = list(self.jerk_history)
        avg_jerk = np.mean(jerk)
        max_jerk = np.max(jerk)

        # Sudden jerky movements indicate agitation
        return min(1.0, avg_jerk * 0.7 + max_jerk * 0.3)

    def _compute_emotion_agitation(self, emotions: Optional[Dict]) -> float:
        """Compute agitation from emotion detection.

        Emotion alone is a WEAK signal for clinical agitation.
        Weights are intentionally low -- physical motion signals
        (motion_score, jerk_score) should dominate.  Sadness and
        mild negative affect should NOT drive agitation alerts.
        """
        if not emotions:
            return 0.0

        dominant = emotions.get("dominant_emotion", "neutral")
        emotion_scores = emotions.get("emotions", {})

        # Reduced weights: emotion is supplementary, not primary
        agitation_emotions = {
            "angry": 0.3,
            "fear": 0.2,
            "disgust": 0.15,
            "surprise": 0.1
        }

        if dominant in agitation_emotions:
            return agitation_emotions[dominant]

        # Weighted aggregate from raw scores
        score = 0.0
        for emotion, weight in agitation_emotions.items():
            score += emotion_scores.get(emotion, 0) * weight

        return min(1.0, score)

    def _detect_tube_pulling(self, keypoints: np.ndarray,
                             self_contact: Optional[Dict],
                             motion_energy: float) -> Dict:
        """
        Detect tube-pulling behavior.

        Indicators:
        - Hand near face/neck with motion
        - Self-contact with face detected
        - Repeated reaching motions toward face/tubes
        """
        result = {
            "detected": False,
            "risk_score": 0.0,
            "type": None,
            "indicators": []
        }

        # Check self-contact for face touching
        if self_contact:
            if self_contact.get("face_touch_detected", False):
                result["risk_score"] += 0.4
                result["indicators"].append("face_touch")

            contact_events = self_contact.get("contact_events", [])
            for event in contact_events:
                if "face" in event.get("type", "").lower():
                    result["risk_score"] += 0.2

        # Check hand proximity with motion
        hand_proximity = self._calculate_hand_proximity(keypoints)
        if hand_proximity > 0.5 and motion_energy > 0.1:
            result["risk_score"] += 0.3
            result["indicators"].append("hand_near_face_with_motion")

        # Check for sustained high proximity
        if len(self.hand_proximity_history) >= 5:
            sustained = np.mean(list(self.hand_proximity_history)[-5:])
            if sustained > 0.6:
                result["risk_score"] += 0.2
                result["indicators"].append("sustained_proximity")

        result["risk_score"] = min(1.0, result["risk_score"])
        result["detected"] = result["risk_score"] > 0.5

        if result["detected"]:
            if "face_touch" in result["indicators"]:
                result["type"] = "oxygen_mask_interference"
            elif hand_proximity > 0.7:
                result["type"] = "iv_line_interference"
            else:
                result["type"] = "general_tube_pulling"

        return result


class MedicalInterventionDetector:
    """
    Detects medical interventions based on multiple person detection,
    interaction patterns, and activity changes.
    """

    def __init__(self):
        self.person_count_history = deque(maxlen=30)
        self.interaction_history = deque(maxlen=60)
        self.last_intervention_time = 0.0
        self.intervention_cooldown = 60.0  # Minimum gap between interventions

    def detect(self, person_count: int, patient_keypoints: np.ndarray,
               other_keypoints: Optional[List[np.ndarray]] = None,
               activity: str = "unknown",
               vital_changes: Optional[Dict] = None,
               timestamp: Optional[float] = None) -> Dict:
        """
        Detect if medical intervention is occurring.

        Args:
            person_count: Number of people in frame
            patient_keypoints: Patient's keypoints
            other_keypoints: List of keypoints for other people
            activity: Current detected activity
            vital_changes: Changes in vital signs
            timestamp: Current timestamp

        Returns:
            Intervention detection result with care type classification
        """
        ts = timestamp or time.time()
        self.person_count_history.append(person_count)

        result = {
            "intervention_detected": False,
            "care_type": CareType.NO_INTERACTION,
            "confidence": 0.0,
            "indicators": [],
            "duration_seconds": 0.0,
            "timestamp": ts
        }

        # Multiple people = potential intervention
        if person_count > 1:
            result["indicators"].append("multiple_persons")
            result["confidence"] += 0.3

            # Analyze interaction patterns
            if other_keypoints:
                interaction = self._analyze_interaction(patient_keypoints, other_keypoints)
                if interaction["touching"]:
                    result["indicators"].append("physical_contact")
                    result["confidence"] += 0.3
                if interaction["close_proximity"]:
                    result["indicators"].append("close_proximity")
                    result["confidence"] += 0.2

        # Activity-based detection
        intervention_activities = [
            "receiving_care", "being_examined", "medication_administered",
            "vital_check", "repositioning_assisted"
        ]
        if activity in intervention_activities:
            result["indicators"].append(f"activity_{activity}")
            result["confidence"] += 0.4

        # Vital sign changes during potential intervention
        if vital_changes:
            if vital_changes.get("significant_change", False):
                result["indicators"].append("vital_sign_change")
                result["confidence"] += 0.2

        result["confidence"] = min(1.0, result["confidence"])
        result["intervention_detected"] = result["confidence"] > 0.5

        # Classify care type
        if result["intervention_detected"]:
            result["care_type"] = self._classify_care_type(
                person_count, result["indicators"], activity
            )

            if ts - self.last_intervention_time > self.intervention_cooldown:
                self.last_intervention_time = ts

        return result

    def _analyze_interaction(self, patient_kps: np.ndarray,
                             others_kps: List[np.ndarray]) -> Dict:
        """Analyze interaction between patient and others."""
        result = {"touching": False, "close_proximity": False, "distance": float("inf")}

        if patient_kps is None or not others_kps:
            return result

        try:
            # Patient center (hip midpoint)
            patient_center = (patient_kps[11, :2] + patient_kps[12, :2]) / 2

            for other_kps in others_kps:
                if other_kps is None or len(other_kps) < 17:
                    continue

                other_center = (other_kps[11, :2] + other_kps[12, :2]) / 2
                distance = np.linalg.norm(patient_center - other_center)

                result["distance"] = min(result["distance"], distance)

                # Close proximity check (normalized coordinates)
                if distance < 0.2:
                    result["close_proximity"] = True

                # Touch detection - check if any limbs overlap
                for limb_idx in [9, 10, 15, 16]:  # Wrists and ankles
                    if other_kps[limb_idx, 2] > 0.3:  # Confidence check
                        limb_pos = other_kps[limb_idx, :2]
                        # Check distance to patient body
                        for patient_joint in [5, 6, 11, 12]:  # Shoulders and hips
                            if patient_kps[patient_joint, 2] > 0.3:
                                joint_pos = patient_kps[patient_joint, :2]
                                if np.linalg.norm(limb_pos - joint_pos) < 0.1:
                                    result["touching"] = True
                                    break

        except Exception as e:
            log.debug(f"Interaction analysis error: {e}")

        return result

    def _classify_care_type(self, person_count: int,
                            indicators: List[str], activity: str) -> CareType:
        """Classify the type of care being provided."""
        # Simple heuristics - could be enhanced with ML
        if "vital_sign_change" in indicators and person_count >= 3:
            return CareType.MEDICAL_EMERGENCY

        if person_count >= 3:
            return CareType.DOCTOR_CARE

        if "physical_contact" in indicators:
            return CareType.NURSE_CARE

        if "close_proximity" in indicators:
            return CareType.NURSE_CARE

        if activity in ["grooming", "eating", "drinking"]:
            if person_count > 1:
                return CareType.NURSE_CARE
            return CareType.SELF_CARE

        return CareType.NO_INTERACTION


class ClinicalEventTracker:
    """
    Main clinical event tracker integrating all detection systems.
    Provides unified interface for monitoring, alerting, and reporting.

    Supports database persistence for activity, posture, clinical events,
    vitals, sleep sessions, and interventions.
    """

    def __init__(self, patient_id: str, fps: float = 15.0, enable_db: bool = True):
        """
        Initialize clinical event tracker.

        Args:
            patient_id: Patient identifier
            fps: Frame rate for temporal analysis
            enable_db: Enable database persistence (default True)
        """
        self.patient_id = patient_id
        self.fps = fps
        self.enable_db = enable_db
        self.db = None

        # Initialize database connection
        if enable_db:
            try:
                from storage.db import LocalDB
                self.db = LocalDB()
                log.info("Database persistence enabled for patient %s", patient_id)
            except Exception as e:
                log.warning("Failed to initialize database: %s. Continuing without persistence.", e)
                self.enable_db = False

        # Sub-trackers
        self.sleep_tracker = SleepTracker(fps)
        self.agitation_detector = AgitationDetector(fps)
        self.intervention_detector = MedicalInterventionDetector()

        # Event storage (in-memory for fast access)
        self.events = deque(maxlen=10000)           # Store last 10k events
        self.alerts = deque(maxlen=1000)             # Store last 1k alerts
        self.vitals_history = deque(maxlen=int(fps * 3600))  # 1 hour of vitals

        # State tracking
        self.current_state = {
            "sleep": None,
            "agitation": None,
            "intervention": None,
            "last_update": 0.0
        }

        # Progress tracking
        self.hourly_stats = {}
        self.daily_stats = {}

        # Activity mapping for detection types
        self._load_activity_mapping()

        # Track last posture for change detection
        self._last_posture = None
        self._last_activity = None
        self._posture_persist_interval = 5.0     # Persist posture every N seconds
        self._last_posture_persist_time = 0.0
        self._activity_persist_interval = 1.0    # Persist activity every N seconds
        self._last_activity_persist_time = 0.0

    def _load_activity_mapping(self):
        """Load activity feature mapping for detection types."""
        try:
            from analytics.clinical_activity_mapping import ACTIVITY_FEATURE_MAPPING
            self.activity_mapping = ACTIVITY_FEATURE_MAPPING
        except ImportError:
            self.activity_mapping = {}
            log.warning("Could not load activity mapping")

    def update(self, keypoints: np.ndarray, posture: str, activity: str,
               motion_energy: float, jerk_index: float,
               person_count: int = 1,
               other_keypoints: Optional[List[np.ndarray]] = None,
               emotions: Optional[Dict] = None,
               self_contact: Optional[Dict] = None,
               vitals: Optional[VitalSignReading] = None,
               timestamp: Optional[float] = None,
               confidence: float = 0.8,
               features: Optional[np.ndarray] = None) -> Dict:
        """
        Main update method - processes all tracking systems.

        Args:
            keypoints: Current pose keypoints
            posture: Current posture classification
            activity: Current activity classification
            motion_energy: Motion energy value
            jerk_index: Movement jerkiness
            person_count: Number of people in frame
            other_keypoints: Keypoints of other people
            emotions: Emotion detection results
            self_contact: Self-contact detection results
            vitals: Vital sign reading
            timestamp: Current timestamp
            confidence: Classification confidence
            features: Feature vector for storage

        Returns:
            Comprehensive tracking result with alerts
        """
        ts = timestamp or time.time()
        self.current_state["last_update"] = ts

        # Store vitals
        if vitals:
            self.vitals_history.append(vitals)
            self._persist_vitals(vitals, ts)

        # Sleep tracking
        sleep_result = self.sleep_tracker.update(
            motion_energy, posture,
            vitals.breath_rate if vitals else None,
            ts
        )
        self.current_state["sleep"] = sleep_result

        # Agitation detection
        agitation_result = self.agitation_detector.detect(
            keypoints, motion_energy, jerk_index,
            emotions, self_contact, ts
        )
        self.current_state["agitation"] = agitation_result

        # Intervention detection
        vital_changes = self._analyze_vital_changes() if vitals else None
        intervention_result = self.intervention_detector.detect(
            person_count, keypoints, other_keypoints,
            activity, vital_changes, ts
        )
        self.current_state["intervention"] = intervention_result

        # Generate alerts
        alerts = self._generate_alerts(
            sleep_result, agitation_result, intervention_result,
            activity, vitals, ts
        )

        # Store events
        for alert in alerts:
            self.alerts.append(alert)

        # Create clinical event
        severity = self._determine_severity(agitation_result, alerts)
        event = ClinicalEvent(
            timestamp=ts,
            event_type=self._determine_event_type(activity, agitation_result),
            severity=severity,
            description=self._generate_description(activity, agitation_result, sleep_result),
            activity=activity,
            confidence=confidence,
            care_type=intervention_result["care_type"],
            vitals=vitals,
            intervention_detected=intervention_result["intervention_detected"],
            requires_acknowledgment=any(a["severity"] in [AlertSeverity.CRITICAL, AlertSeverity.HIGH] for a in alerts),
            metadata={
                "sleep": sleep_result,
                "agitation": agitation_result,
                "intervention": intervention_result
            }
        )
        self.events.append(event)

        # Update statistics
        self._update_statistics(event, ts)

        # Database persistence
        self._persist_activity(activity, confidence, posture, motion_energy,
                               jerk_index, features, keypoints, ts)
        self._persist_posture(posture, keypoints, ts)
        self._persist_clinical_event(event, sleep_result, agitation_result,
                                     intervention_result, vitals, ts)

        # Persist intervention if detected
        if intervention_result["intervention_detected"]:
            self._persist_intervention(intervention_result, activity, vital_changes, ts)

        return {
            "patient_id": self.patient_id,
            "timestamp": ts,
            "sleep": sleep_result,
            "agitation": agitation_result,
            "intervention": intervention_result,
            "alerts": [self._alert_to_dict(a) for a in alerts],
            "current_activity": activity,
            "care_type": intervention_result["care_type"].value,
            "requires_attention": len(alerts) > 0 and any(
                a["severity"] in [AlertSeverity.CRITICAL, AlertSeverity.HIGH] for a in alerts
            )
        }

    def _persist_activity(self, activity: str, confidence: float, posture: str,
                          motion_energy: float, jerk_index: float,
                          features: Optional[np.ndarray], keypoints: np.ndarray,
                          timestamp: float):
        """Persist activity detection to database."""
        if not self.enable_db or not self.db:
            return

        # Rate limit persistence
        if timestamp - self._last_activity_persist_time < self._activity_persist_interval:
            # Only persist if activity changed
            if activity == self._last_activity:
                return

        try:
            # Get activity metadata
            activity_info = self.activity_mapping.get(activity, {})
            detection_type = activity_info.get("detection_type", "unknown")
            risk_type = activity_info.get("risk_type", "unknown")

            # Prepare keypoints for storage
            kps_list = keypoints.tolist() if keypoints is not None else None
            feat_list = features.tolist() if features is not None else None

            self.db.insert_activity_detection(
                patient_id=self.patient_id,
                activity=activity,
                confidence=confidence,
                ts=timestamp,
                detection_type=detection_type,
                risk_type=risk_type,
                posture=posture,
                motion_energy=motion_energy,
                jerk_index=jerk_index,
                features=feat_list,
                keypoints=kps_list,
                context={"activity_info": activity_info.get("clinical_meaning", "")}
            )

            self._last_activity = activity
            self._last_activity_persist_time = timestamp
        except Exception as e:
            log.debug("Failed to persist activity: %s", e)

    def _persist_posture(self, posture: str, keypoints: np.ndarray, timestamp: float):
        """Persist posture to database."""
        if not self.enable_db or not self.db:
            return

        # Rate limit posture persistence
        posture_changed = posture != self._last_posture
        time_elapsed = timestamp - self._last_posture_persist_time

        if not posture_changed and time_elapsed < self._posture_persist_interval:
            return

        try:
            # Determine if bed posture
            bed_postures = ['supine', 'prone', 'left_lateral', 'right_lateral', 'fetal', 'side']
            is_bed_posture = posture in bed_postures

            # Calculate completeness from keypoint confidence
            completeness = 0.0
            if keypoints is not None and len(keypoints) > 0:
                confidences = keypoints[:, 2] if keypoints.shape[1] > 2 else np.ones(len(keypoints))
                completeness = float(np.mean(confidences > 0.3))

            # Convert keypoints to 3D list if needed
            kps_3d = keypoints.tolist() if keypoints is not None else None

            self.db.insert_posture(
                patient_id=self.patient_id,
                posture=posture,
                ts=timestamp,
                posture_confidence=completeness,
                is_bed_posture=is_bed_posture,
                completeness_score=completeness,
                keypoints_3d=kps_3d
            )

            self._last_posture = posture
            self._last_posture_persist_time = timestamp
        except Exception as e:
            log.debug("Failed to persist posture: %s", e)

    def _persist_clinical_event(self, event: ClinicalEvent, sleep: Dict,
                                agitation: Dict, intervention: Dict,
                                vitals: Optional[VitalSignReading], timestamp: float):
        """Persist clinical event to database."""
        if not self.enable_db or not self.db:
            return

        # Only persist significant events to avoid DB bloat
        is_significant = (
            event.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM] or
            agitation.get("is_agitated", False) or
            agitation.get("tube_pull_detected", False) or
            intervention.get("intervention_detected", False) or
            sleep.get("is_restless", False)
        )

        if not is_significant:
            return

        try:
            # Collect indicators
            indicators = []
            if agitation.get("tube_pull_detected"):
                indicators.append("tube_pull_detected")
            if agitation.get("is_agitated"):
                indicators.append(f"agitation_{agitation.get('agitation_level', 'unknown')}")
            if sleep.get("is_restless"):
                indicators.append("restless_sleep")

            self.db.insert_clinical_event(
                patient_id=self.patient_id,
                event_type=event.event_type,
                ts=timestamp,
                severity=event.severity.value,
                activity=event.activity,
                posture=sleep.get("sleep_stage", "unknown") if sleep.get("in_bed") else "active",
                care_type=event.care_type.value if event.care_type else None,
                intervention_detected=event.intervention_detected,
                sleep_stage=sleep.get("sleep_stage"),
                agitation_level=agitation.get("agitation_level"),
                agitation_score=agitation.get("agitation_score"),
                tube_pull_risk=agitation.get("tube_pull_risk"),
                distress_level="high" if agitation.get("is_agitated") else "low",
                breath_rate=vitals.breath_rate if vitals else None,
                motion_energy=agitation.get("components", {}).get("motion"),
                description=event.description,
                indicators=indicators,
                requires_acknowledgment=event.requires_acknowledgment
            )
        except Exception as e:
            log.debug("Failed to persist clinical event: %s", e)

    def _persist_vitals(self, vitals: VitalSignReading, timestamp: float):
        """Persist vitals to database."""
        if not self.enable_db or not self.db:
            return

        try:
            self.db.insert_vitals(
                patient_id=self.patient_id,
                ts=timestamp,
                breath_rate=vitals.breath_rate,
                breath_confidence=vitals.breath_confidence,
                hrv_proxy=vitals.hrv_proxy,
                motion_energy=vitals.motion_energy
            )
        except Exception as e:
            log.debug("Failed to persist vitals: %s", e)

    def _persist_intervention(self, intervention: Dict, activity: str,
                              vital_changes: Optional[Dict], timestamp: float):
        """Persist intervention record to database."""
        if not self.enable_db or not self.db:
            return

        try:
            self.db.insert_intervention_record(
                patient_id=self.patient_id,
                care_type=intervention["care_type"].value,
                ts=timestamp,
                person_count=len(intervention.get("indicators", [])),
                activity_during=activity,
                vital_changes=vital_changes
            )
        except Exception as e:
            log.debug("Failed to persist intervention: %s", e)

    def save_sleep_session(self):
        """
        Save current sleep session summary to database.
        Call this when sleep session ends or at regular intervals.
        """
        if not self.enable_db or not self.db:
            return

        sleep_state = self.current_state.get("sleep", {})
        if not sleep_state:
            return

        try:
            sleep_start = self.sleep_tracker.sleep_start_time
            if not sleep_start:
                return

            current_time = time.time()
            duration_minutes = (current_time - sleep_start) / 60

            summary = self.sleep_tracker.get_sleep_summary()

            self.db.insert_sleep_session(
                patient_id=self.patient_id,
                session_start=sleep_start,
                session_end=current_time,
                total_duration_minutes=duration_minutes,
                sleep_quality_score=summary.get("quality_score", 0),
                disturbance_count=summary.get("disturbances", 0),
                avg_motion_energy=np.mean(list(self.sleep_tracker.motion_history))
                if self.sleep_tracker.motion_history else 0
            )
            log.info("Sleep session saved for patient %s", self.patient_id)
        except Exception as e:
            log.debug("Failed to save sleep session: %s", e)

    def get_db_summary(self, hours: int = 24) -> Dict:
        """
        Get comprehensive summary from database.

        Args:
            hours: Number of hours to analyze

        Returns:
            Dict with activity, posture, clinical events, vitals summary
        """
        if not self.enable_db or not self.db:
            return {"error": "Database not enabled"}

        return self.db.get_patient_clinical_summary(self.patient_id, hours)

    def get_activity_timeline_from_db(self, hours: int = 24, bucket_minutes: int = 15) -> list:
        """
        Get activity timeline from database for graphing.

        Args:
            hours: Hours of data to retrieve
            bucket_minutes: Time bucket size

        Returns:
            List of time-bucketed activity data
        """
        if not self.enable_db or not self.db:
            return []

        start_ts = time.time() - (hours * 3600)
        return self.db.get_activity_timeline(self.patient_id, start_ts, bucket_minutes=bucket_minutes)

    def query_activities(self, activity: str = None, start_ts: float = None,
                         end_ts: float = None, limit: int = 1000) -> list:
        """
        Query activity detections from database.

        Args:
            activity: Filter by activity name
            start_ts: Start timestamp
            end_ts: End timestamp
            limit: Maximum results

        Returns:
            List of activity detection records
        """
        if not self.enable_db or not self.db:
            return []

        return self.db.query_activity_detections(
            patient_id=self.patient_id,
            activity=activity,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit
        )

    def query_clinical_events(self, event_type: str = None, severity: str = None,
                              start_ts: float = None, end_ts: float = None,
                              unacknowledged_only: bool = False, limit: int = 1000) -> list:
        """
        Query clinical events from database.

        Args:
            event_type: Filter by event type
            severity: Filter by severity
            start_ts: Start timestamp
            end_ts: End timestamp
            unacknowledged_only: Only return unacknowledged alerts
            limit: Maximum results

        Returns:
            List of clinical event records
        """
        if not self.enable_db or not self.db:
            return []

        return self.db.query_clinical_events(
            patient_id=self.patient_id,
            event_type=event_type,
            severity=severity,
            start_ts=start_ts,
            end_ts=end_ts,
            unacknowledged_only=unacknowledged_only,
            limit=limit
        )

    def _analyze_vital_changes(self) -> Dict:
        """Analyze recent vital sign changes."""
        if len(self.vitals_history) < 10:
            return {"significant_change": False}

        recent = list(self.vitals_history)[-10:]
        breath_rates = [v.breath_rate for v in recent if v.breath_rate > 0]

        if len(breath_rates) < 5:
            return {"significant_change": False}

        variance = np.var(breath_rates)
        mean_rate = np.mean(breath_rates)

        # Significant if variance is high or rate is abnormal
        significant = variance > 4 or mean_rate < 10 or mean_rate > 25

        return {
            "significant_change": significant,
            "breath_rate_variance": variance,
            "mean_breath_rate": mean_rate
        }

    def _generate_alerts(self, sleep: Dict, agitation: Dict,
                         intervention: Dict, activity: str,
                         vitals: Optional[VitalSignReading],
                         timestamp: float) -> List[Dict]:
        """Generate clinical alerts based on current state."""
        alerts = []

        # Agitation alerts
        if agitation["agitation_level"] == "critical":
            alerts.append({
                "type": "agitation_critical",
                "severity": AlertSeverity.CRITICAL,
                "message": "CRITICAL: Patient showing severe agitation",
                "action": "Immediate intervention required",
                "timestamp": timestamp
            })
        elif agitation["agitation_level"] == "severe":
            alerts.append({
                "type": "agitation_severe",
                "severity": AlertSeverity.HIGH,
                "message": "Patient showing severe agitation",
                "action": "Assess patient, consider PRN medication",
                "timestamp": timestamp
            })

        # Tube pulling alerts
        if agitation["tube_pull_detected"]:
            alerts.append({
                "type": "tube_pulling",
                "severity": AlertSeverity.CRITICAL,
                "message": f"ALERT: Tube/device interference detected - {agitation['tube_pull_type']}",
                "action": "Check all lines and devices immediately",
                "timestamp": timestamp
            })

        # Sleep disturbance alerts
        if sleep["is_restless"] and sleep.get("disturbance_count", 0) > 5:
            alerts.append({
                "type": "sleep_disturbance",
                "severity": AlertSeverity.MEDIUM,
                "message": "Patient showing restless sleep with multiple disturbances",
                "action": "Assess for pain, discomfort, or environmental factors",
                "timestamp": timestamp
            })

        # Critical activity alerts
        critical_activities = ["falling", "fallen", "seizure", "convulsion", "gasping", "unresponsive"]
        if activity in critical_activities:
            alerts.append({
                "type": f"activity_{activity}",
                "severity": AlertSeverity.CRITICAL,
                "message": f"EMERGENCY: {activity.replace('_', ' ').title()} detected",
                "action": "Immediate response required",
                "timestamp": timestamp
            })

        # Tremor detection
        if activity == "tremor":
            alerts.append({
                "type": "tremor_detected",
                "severity": AlertSeverity.HIGH,
                "message": "Tremor activity detected",
                "action": "Assess neurological status and medication effects",
                "timestamp": timestamp
            })

        # Vital sign alerts
        if vitals:
            if vitals.breath_rate < 8 or vitals.breath_rate > 30:
                alerts.append({
                    "type": "respiratory_abnormal",
                    "severity": AlertSeverity.HIGH,
                    "message": f"Abnormal respiratory rate: {vitals.breath_rate:.1f} breaths/min",
                    "action": "Assess respiratory status",
                    "timestamp": timestamp
                })

        return alerts

    def _determine_event_type(self, activity: str, agitation: Dict) -> str:
        """Determine the primary event type."""
        if agitation["tube_pull_detected"]:
            return "tube_pulling"
        if agitation["agitation_level"] in ["severe", "critical"]:
            return "agitation"
        return activity

    def _determine_severity(self, agitation: Dict, alerts: List[Dict]) -> AlertSeverity:
        """Determine overall severity."""
        if alerts:
            severities = [a["severity"] for a in alerts]
            if AlertSeverity.CRITICAL in severities:
                return AlertSeverity.CRITICAL
            if AlertSeverity.HIGH in severities:
                return AlertSeverity.HIGH
            if AlertSeverity.MEDIUM in severities:
                return AlertSeverity.MEDIUM

        if agitation["agitation_level"] == "critical":
            return AlertSeverity.CRITICAL
        if agitation["agitation_level"] == "severe":
            return AlertSeverity.HIGH

        return AlertSeverity.LOW

    def _generate_description(self, activity: str, agitation: Dict, sleep: Dict) -> str:
        """Generate human-readable event description."""
        parts = []

        if agitation["tube_pull_detected"]:
            parts.append(f"Tube pulling detected ({agitation['tube_pull_type']})")

        if agitation["is_agitated"]:
            parts.append(f"Agitation level: {agitation['agitation_level']}")

        if sleep["is_sleeping"]:
            parts.append(f"Sleep stage: {sleep['sleep_stage']}")
        elif sleep["is_restless"]:
            parts.append("Restless sleep")

        parts.append(f"Activity: {activity}")

        return "; ".join(parts)

    def _alert_to_dict(self, alert: Dict) -> Dict:
        """Convert alert to serializable dict."""
        return {
            "type": alert["type"],
            "severity": alert["severity"].value,
            "message": alert["message"],
            "action": alert["action"],
            "timestamp": alert["timestamp"]
        }

    def _update_statistics(self, event: ClinicalEvent, timestamp: float):
        """Update hourly and daily statistics."""
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        hour_key = dt.strftime("%Y-%m-%d-%H")
        day_key = dt.strftime("%Y-%m-%d")

        # Hourly stats
        if hour_key not in self.hourly_stats:
            self.hourly_stats[hour_key] = {
                "events": 0,
                "critical_alerts": 0,
                "agitation_episodes": 0,
                "tube_pull_attempts": 0,
                "interventions": 0,
                "sleep_disturbances": 0
            }

        stats = self.hourly_stats[hour_key]
        stats["events"] += 1

        if event.severity == AlertSeverity.CRITICAL:
            stats["critical_alerts"] += 1
        if event.metadata.get("agitation", {}).get("is_agitated"):
            stats["agitation_episodes"] += 1
        if event.metadata.get("agitation", {}).get("tube_pull_detected"):
            stats["tube_pull_attempts"] += 1
        if event.intervention_detected:
            stats["interventions"] += 1
        if event.metadata.get("sleep", {}).get("is_restless"):
            stats["sleep_disturbances"] += 1

    def get_patient_progress(self, hours: int = 24) -> PatientProgress:
        """
        Generate patient progress report.

        Args:
            hours: Number of hours to analyze

        Returns:
            PatientProgress summary
        """
        current_time = time.time()
        window_start = current_time - (hours * 3600)

        # Filter events to window
        recent_events = [e for e in self.events if e.timestamp >= window_start]

        if not recent_events:
            return PatientProgress(
                patient_id=self.patient_id,
                period_start=window_start,
                period_end=current_time
            )

        # Analyze trends
        first_half = [e for e in recent_events if e.timestamp < (window_start + current_time) / 2]
        second_half = [e for e in recent_events if e.timestamp >= (window_start + current_time) / 2]

        # Agitation trend
        first_agitation = np.mean([
            e.metadata.get("agitation", {}).get("agitation_score", 0)
            for e in first_half
        ]) if first_half else 0
        second_agitation = np.mean([
            e.metadata.get("agitation", {}).get("agitation_score", 0)
            for e in second_half
        ]) if second_half else 0

        if second_agitation < first_agitation * 0.8:
            agitation_trend = "improving"
        elif second_agitation > first_agitation * 1.2:
            agitation_trend = "worsening"
        else:
            agitation_trend = "stable"

        # Sleep quality
        sleep_events = [e for e in recent_events if e.metadata.get("sleep", {}).get("in_bed")]
        sleep_quality = np.mean([
            e.metadata.get("sleep", {}).get("sleep_quality_score", 50)
            for e in sleep_events
        ]) if sleep_events else 50.0

        # Count critical events and interventions
        critical_count = sum(1 for e in recent_events if e.severity == AlertSeverity.CRITICAL)
        intervention_count = sum(1 for e in recent_events if e.intervention_detected)

        # Improvement/concern indicators
        improvements = []
        concerns = []

        if agitation_trend == "improving":
            improvements.append("Agitation levels decreasing")
        elif agitation_trend == "worsening":
            concerns.append("Agitation levels increasing")

        if sleep_quality > 70:
            improvements.append("Good sleep quality")
        elif sleep_quality < 40:
            concerns.append("Poor sleep quality")

        tube_pulls = sum(1 for e in recent_events
                         if e.metadata.get("agitation", {}).get("tube_pull_detected"))
        if tube_pulls > 0:
            concerns.append(f"{tube_pulls} tube-pulling attempts detected")

        if critical_count == 0:
            improvements.append("No critical events")
        else:
            concerns.append(f"{critical_count} critical events occurred")

        return PatientProgress(
            patient_id=self.patient_id,
            period_start=window_start,
            period_end=current_time,
            sleep_quality_score=round(sleep_quality, 1),
            agitation_trend=agitation_trend,
            mobility_trend="stable",  # Would need mobility tracking
            intervention_count=intervention_count,
            critical_events=critical_count,
            improvement_indicators=improvements,
            concern_indicators=concerns
        )

    def get_live_dashboard_data(self) -> Dict:
        """
        Get data for live dashboard visualization.

        Returns:
            Dict with current state and recent history for graphing
        """
        current_time = time.time()

        # Get recent events for graphs
        hour_ago = current_time - 3600
        recent_events = [e for e in self.events if e.timestamp >= hour_ago]

        # Prepare time series data
        agitation_series = []
        sleep_series = []
        activity_distribution = {}
        intervention_timeline = []

        for event in recent_events:
            ts = event.timestamp

            # Agitation over time
            agitation_score = event.metadata.get("agitation", {}).get("agitation_score", 0)
            agitation_series.append({"time": ts, "value": agitation_score})

            # Sleep quality over time
            sleep_score = event.metadata.get("sleep", {}).get("sleep_quality_score", 0)
            if event.metadata.get("sleep", {}).get("in_bed"):
                sleep_series.append({"time": ts, "value": sleep_score})

            # Activity distribution
            activity_distribution[event.activity] = activity_distribution.get(event.activity, 0) + 1

            # Interventions
            if event.intervention_detected:
                intervention_timeline.append({
                    "time": ts,
                    "type": event.care_type.value,
                    "duration": 0  # Would need duration tracking
                })

        # Current alerts
        recent_alerts = [self._alert_to_dict(a) for a in list(self.alerts)[-10:]]

        return {
            "patient_id": self.patient_id,
            "current_time": current_time,
            "current_state": {
                "sleep": self.current_state.get("sleep"),
                "agitation": self.current_state.get("agitation"),
                "intervention": {
                    "detected": self.current_state.get("intervention", {}).get("intervention_detected", False),
                    "care_type": self.current_state.get("intervention", {}).get("care_type", CareType.NO_INTERACTION).value
                }
            },
            "charts": {
                "agitation_timeline": agitation_series[-100:],  # Last 100 points
                "sleep_quality_timeline": sleep_series[-100:],
                "activity_distribution": activity_distribution,
                "intervention_events": intervention_timeline
            },
            "alerts": recent_alerts,
            "progress": self.get_patient_progress(24).__dict__,
            "statistics": {
                "hourly": dict(list(self.hourly_stats.items())[-24:])  # Last 24 hours
            }
        }

    def generate_shift_report(self) -> Dict:
        """Generate end-of-shift nursing report."""
        progress = self.get_patient_progress(8)  # 8-hour shift

        # Get all events in shift
        shift_start = progress.period_start
        shift_events = [e for e in self.events if e.timestamp >= shift_start]

        # Summarize by category
        event_summary = {}
        for event in shift_events:
            etype = event.event_type
            event_summary[etype] = event_summary.get(etype, 0) + 1

        # Get peak agitation times
        agitation_peaks = []
        for event in shift_events:
            ag_score = event.metadata.get("agitation", {}).get("agitation_score", 0)
            if ag_score > 0.5:
                agitation_peaks.append({
                    "time": event.timestamp,
                    "level": event.metadata.get("agitation", {}).get("agitation_level"),
                    "score": ag_score
                })

        return {
            "patient_id": self.patient_id,
            "shift_start": shift_start,
            "shift_end": progress.period_end,
            "summary": {
                "total_events": len(shift_events),
                "critical_events": progress.critical_events,
                "interventions": progress.intervention_count,
                "tube_pull_attempts": sum(1 for e in shift_events
                                          if e.metadata.get("agitation", {}).get("tube_pull_detected"))
            },
            "sleep_summary": {
                "quality_score": progress.sleep_quality_score,
                "disturbances": sum(1 for e in shift_events
                                    if e.metadata.get("sleep", {}).get("is_restless"))
            },
            "agitation_summary": {
                "trend": progress.agitation_trend,
                "peak_episodes": agitation_peaks[:5]  # Top 5 peaks
            },
            "event_breakdown": event_summary,
            "improvement_indicators": progress.improvement_indicators,
            "concerns": progress.concern_indicators,
            "recommendations": self._generate_recommendations(progress)
        }

    def _generate_recommendations(self, progress: PatientProgress) -> List[str]:
        """Generate clinical recommendations based on progress."""
        recommendations = []

        if progress.agitation_trend == "worsening":
            recommendations.append("Consider pain assessment and PRN medication review")

        if progress.sleep_quality_score < 50:
            recommendations.append("Assess sleep environment and consider sleep hygiene interventions")

        if progress.critical_events > 2:
            recommendations.append("Review safety protocols and consider 1:1 observation")

        if "tube-pulling" in " ".join(progress.concern_indicators).lower():
            recommendations.append("Evaluate need for mittens or increased monitoring")

        return recommendations
