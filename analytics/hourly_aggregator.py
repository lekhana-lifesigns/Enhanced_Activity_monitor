# analytics/hourly_aggregator.py
"""
Hourly Aggregation System for Enhanced Activity Monitor.

Architecture:
    Live Stream (seconds) → HourlyAggregator (in-memory) → DB (hourly) → Reports

Design Principles:
    1. Live stream data is authoritative but ephemeral (seconds lifetime)
    2. Hourly aggregates are derived and persistent (no raw frames stored)
    3. Reports generated from hourly aggregates only (no live dependency)
    4. Minimal DB writes (once per hour, not per frame)

Metrics Aggregated:
    - Posture distribution (% time in each posture)
    - Support surface usage (% time on bed/chair/etc.)
    - Clinical events (bed exits, falls, immobility periods)
    - Risk scores (max/avg agitation, delirium, respiratory)
    - Activity patterns (dominant activities)
"""

import time
import logging
import threading
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

log = logging.getLogger("hourly_aggregator")


class PostureType(Enum):
    """Valid posture types."""
    LYING = "lying"
    SITTING = "sitting"
    STANDING = "standing"
    KNEELING = "kneeling"
    SQUATTING = "squatting"
    TRANSITIONING = "transitioning"
    UNKNOWN = "unknown"


class SupportSurfaceType(Enum):
    """Valid support surface types."""
    BED = "bed"
    CHAIR = "chair"
    COUCH = "couch"
    WHEELCHAIR = "wheelchair"
    FLOOR = "floor"
    UNKNOWN = "unknown"


@dataclass
class HourlyMetrics:
    """
    Aggregated metrics for a single hour.

    All time values are in seconds.
    All percentages are 0.0-1.0.
    """
    # Time window
    hour_start: float = 0.0
    hour_end: float = 0.0
    device_id: str = ""
    patient_id: str = ""

    # Observation stats
    total_frames: int = 0
    frames_with_person: int = 0
    total_observation_seconds: float = 0.0

    # Posture distribution (seconds in each posture)
    posture_seconds: Dict[str, float] = field(default_factory=dict)
    dominant_posture: str = "unknown"

    # Support surface distribution (seconds on each surface)
    support_surface_seconds: Dict[str, float] = field(default_factory=dict)
    time_in_bed_seconds: float = 0.0
    time_in_bed_pct: float = 0.0

    # Clinical events
    bed_exit_count: int = 0
    bed_exit_attempts: int = 0  # Includes failed attempts
    fall_events: int = 0
    immobility_periods: int = 0
    immobility_total_seconds: float = 0.0
    distress_periods: int = 0
    distress_total_seconds: float = 0.0

    # Risk scores (max and avg over the hour)
    max_agitation_score: float = 0.0
    avg_agitation_score: float = 0.0
    max_delirium_risk: float = 0.0
    avg_delirium_risk: float = 0.0
    max_respiratory_distress: float = 0.0
    avg_respiratory_distress: float = 0.0
    max_fall_risk: float = 0.0
    avg_fall_risk: float = 0.0

    # Activity patterns
    activity_seconds: Dict[str, float] = field(default_factory=dict)
    dominant_activity: str = "unknown"

    # Alerts summary
    alert_counts: Dict[str, int] = field(default_factory=dict)
    critical_alerts: int = 0
    high_risk_alerts: int = 0

    # Quality metrics
    avg_pose_confidence: float = 0.0
    keypoint_visibility_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HourlyMetrics":
        """Create from dictionary."""
        return cls(**data)


class HourlyAggregator:
    """
    Collects live stream data and produces hourly aggregates.

    Usage:
        aggregator = HourlyAggregator(device_id="bed_01", patient_id="patient_01")

        # Per-frame updates
        aggregator.update(
            posture="sitting",
            support_surface="chair",
            confidence=0.85,
            clinical_decision={"agitation_score": 0.3, ...}
        )

        # Called automatically or manually every hour
        hourly_metrics = aggregator.flush_hour()
        db.save_hourly_aggregate(hourly_metrics)
    """

    def __init__(
        self,
        device_id: str,
        patient_id: str = "unknown",
        auto_flush: bool = True,
        flush_callback: Optional[callable] = None,
    ):
        """
        Args:
            device_id: Unique device identifier
            patient_id: Patient identifier (can be updated)
            auto_flush: Automatically flush at hour boundaries
            flush_callback: Callback(HourlyMetrics) when hour flushes
        """
        self.device_id = device_id
        self.patient_id = patient_id
        self.auto_flush = auto_flush
        self.flush_callback = flush_callback

        self._lock = threading.Lock()
        self._current_hour_start = self._get_hour_start(time.time())

        # Live accumulators (reset each hour)
        self._reset_accumulators()

        # Track state transitions
        self._last_posture: Optional[str] = None
        self._last_support_surface: Optional[str] = None
        self._last_update_time: float = 0.0

        # Track clinical state
        self._immobility_start: Optional[float] = None
        self._distress_start: Optional[float] = None
        self._in_bed: bool = False

        log.info(
            "HourlyAggregator initialized (device=%s, patient=%s, auto_flush=%s)",
            device_id, patient_id, auto_flush
        )

    def _get_hour_start(self, ts: float) -> float:
        """Get the start of the hour containing timestamp."""
        dt = datetime.fromtimestamp(ts)
        hour_start = dt.replace(minute=0, second=0, microsecond=0)
        return hour_start.timestamp()

    def _reset_accumulators(self):
        """Reset all accumulators for a new hour."""
        self._frame_count = 0
        self._frames_with_person = 0

        # Posture time (seconds)
        self._posture_time: Dict[str, float] = defaultdict(float)

        # Support surface time (seconds)
        self._surface_time: Dict[str, float] = defaultdict(float)

        # Activity time (seconds)
        self._activity_time: Dict[str, float] = defaultdict(float)

        # Clinical events
        self._bed_exits = 0
        self._bed_exit_attempts = 0
        self._falls = 0
        self._immobility_periods = 0
        self._immobility_seconds = 0.0
        self._distress_periods = 0
        self._distress_seconds = 0.0

        # Risk score accumulators
        self._agitation_scores: List[float] = []
        self._delirium_scores: List[float] = []
        self._respiratory_scores: List[float] = []
        self._fall_risk_scores: List[float] = []

        # Alert counts
        self._alert_counts: Dict[str, int] = defaultdict(int)

        # Quality metrics
        self._confidence_sum = 0.0
        self._visibility_sum = 0.0

    def update(
        self,
        posture: str = "unknown",
        support_surface: str = "unknown",
        confidence: float = 0.0,
        person_present: bool = True,
        activity: str = "unknown",
        clinical_decision: Optional[Dict[str, Any]] = None,
        fall_detected: bool = False,
        immobility_detected: bool = False,
        distress_detected: bool = False,
        keypoint_visibility: float = 1.0,
        timestamp: Optional[float] = None,
    ):
        """
        Update aggregator with a frame's data.

        Args:
            posture: Current posture (lying|sitting|standing|...)
            support_surface: Current support surface (bed|chair|...)
            confidence: Pose confidence 0-1
            person_present: Whether person is visible
            activity: Current activity label
            clinical_decision: Clinical monitor output dict
            fall_detected: Fall event detected this frame
            immobility_detected: Currently in immobility state
            distress_detected: Currently in distress state
            keypoint_visibility: % of keypoints visible (0-1)
            timestamp: Frame timestamp (defaults to now)
        """
        now = timestamp or time.time()

        with self._lock:
            # Check for hour boundary
            current_hour = self._get_hour_start(now)
            if current_hour > self._current_hour_start and self.auto_flush:
                self._do_flush()
                self._current_hour_start = current_hour

            # Calculate delta time since last update
            delta_t = now - self._last_update_time if self._last_update_time > 0 else 0.0
            delta_t = min(delta_t, 5.0)  # Cap at 5 seconds to handle gaps

            self._frame_count += 1

            if person_present:
                self._frames_with_person += 1

                # Accumulate posture time
                posture_lower = posture.lower() if posture else "unknown"
                self._posture_time[posture_lower] += delta_t

                # Accumulate surface time
                surface_lower = support_surface.lower() if support_surface else "unknown"
                self._surface_time[surface_lower] += delta_t

                # Accumulate activity time
                activity_lower = activity.lower() if activity else "unknown"
                self._activity_time[activity_lower] += delta_t

                # Track bed state transitions
                is_on_bed = surface_lower == "bed"
                if self._in_bed and not is_on_bed:
                    # Bed exit detected
                    self._bed_exits += 1
                    self._bed_exit_attempts += 1
                elif not self._in_bed and is_on_bed:
                    # Returned to bed (might indicate failed exit attempt)
                    pass
                self._in_bed = is_on_bed

                # Track clinical events
                if fall_detected:
                    self._falls += 1

                # Track immobility periods
                if immobility_detected:
                    if self._immobility_start is None:
                        self._immobility_start = now
                        self._immobility_periods += 1
                    self._immobility_seconds += delta_t
                else:
                    self._immobility_start = None

                # Track distress periods
                if distress_detected:
                    if self._distress_start is None:
                        self._distress_start = now
                        self._distress_periods += 1
                    self._distress_seconds += delta_t
                else:
                    self._distress_start = None

                # Accumulate clinical scores
                if clinical_decision:
                    if "agitation_score" in clinical_decision:
                        self._agitation_scores.append(clinical_decision["agitation_score"])
                    if "delirium_risk" in clinical_decision:
                        self._delirium_scores.append(clinical_decision["delirium_risk"])
                    if "respiratory_distress" in clinical_decision:
                        self._respiratory_scores.append(clinical_decision["respiratory_distress"])
                    if "fall_risk" in clinical_decision:
                        self._fall_risk_scores.append(clinical_decision["fall_risk"])

                    # Track alerts
                    alert_level = clinical_decision.get("alert", "LOW_RISK")
                    self._alert_counts[alert_level] += 1

                # Quality metrics
                self._confidence_sum += confidence
                self._visibility_sum += keypoint_visibility

            # Update tracking state
            self._last_posture = posture
            self._last_support_surface = support_surface
            self._last_update_time = now

    def record_alert(self, alert_level: str):
        """Record an alert event."""
        with self._lock:
            self._alert_counts[alert_level] += 1

    def record_bed_exit_attempt(self):
        """Record a bed exit attempt (even if not completed)."""
        with self._lock:
            self._bed_exit_attempts += 1

    def _do_flush(self) -> HourlyMetrics:
        """
        Flush current hour's data and create metrics.
        Internal method - call flush_hour() from outside.
        """
        hour_end = self._current_hour_start + 3600

        # Calculate derived metrics
        total_time = sum(self._posture_time.values())
        bed_time = self._surface_time.get("bed", 0.0)

        # Find dominant posture
        dominant_posture = "unknown"
        max_posture_time = 0.0
        for posture, ptime in self._posture_time.items():
            if ptime > max_posture_time:
                max_posture_time = ptime
                dominant_posture = posture

        # Find dominant activity
        dominant_activity = "unknown"
        max_activity_time = 0.0
        for activity, atime in self._activity_time.items():
            if atime > max_activity_time:
                max_activity_time = atime
                dominant_activity = activity

        # Calculate averages
        avg_agitation = sum(self._agitation_scores) / len(self._agitation_scores) if self._agitation_scores else 0.0
        avg_delirium = sum(self._delirium_scores) / len(self._delirium_scores) if self._delirium_scores else 0.0
        avg_respiratory = sum(self._respiratory_scores) / len(self._respiratory_scores) if self._respiratory_scores else 0.0
        avg_fall_risk = sum(self._fall_risk_scores) / len(self._fall_risk_scores) if self._fall_risk_scores else 0.0

        avg_confidence = self._confidence_sum / self._frames_with_person if self._frames_with_person > 0 else 0.0
        avg_visibility = self._visibility_sum / self._frames_with_person if self._frames_with_person > 0 else 0.0

        metrics = HourlyMetrics(
            hour_start=self._current_hour_start,
            hour_end=hour_end,
            device_id=self.device_id,
            patient_id=self.patient_id,

            total_frames=self._frame_count,
            frames_with_person=self._frames_with_person,
            total_observation_seconds=total_time,

            posture_seconds=dict(self._posture_time),
            dominant_posture=dominant_posture,

            support_surface_seconds=dict(self._surface_time),
            time_in_bed_seconds=bed_time,
            time_in_bed_pct=bed_time / total_time if total_time > 0 else 0.0,

            bed_exit_count=self._bed_exits,
            bed_exit_attempts=self._bed_exit_attempts,
            fall_events=self._falls,
            immobility_periods=self._immobility_periods,
            immobility_total_seconds=self._immobility_seconds,
            distress_periods=self._distress_periods,
            distress_total_seconds=self._distress_seconds,

            max_agitation_score=max(self._agitation_scores) if self._agitation_scores else 0.0,
            avg_agitation_score=round(avg_agitation, 3),
            max_delirium_risk=max(self._delirium_scores) if self._delirium_scores else 0.0,
            avg_delirium_risk=round(avg_delirium, 3),
            max_respiratory_distress=max(self._respiratory_scores) if self._respiratory_scores else 0.0,
            avg_respiratory_distress=round(avg_respiratory, 3),
            max_fall_risk=max(self._fall_risk_scores) if self._fall_risk_scores else 0.0,
            avg_fall_risk=round(avg_fall_risk, 3),

            activity_seconds=dict(self._activity_time),
            dominant_activity=dominant_activity,

            alert_counts=dict(self._alert_counts),
            critical_alerts=self._alert_counts.get("CRITICAL", 0),
            high_risk_alerts=self._alert_counts.get("HIGH_RISK", 0),

            avg_pose_confidence=round(avg_confidence, 3),
            keypoint_visibility_pct=round(avg_visibility, 3),
        )

        # Call flush callback if provided
        if self.flush_callback:
            try:
                self.flush_callback(metrics)
            except Exception as e:
                log.exception("Flush callback failed: %s", e)

        # Reset for next hour
        self._reset_accumulators()

        log.info(
            "Hour flushed: %s | frames=%d | bed_pct=%.1f%% | exits=%d | falls=%d",
            datetime.fromtimestamp(self._current_hour_start).strftime("%Y-%m-%d %H:00"),
            metrics.total_frames,
            metrics.time_in_bed_pct * 100,
            metrics.bed_exit_count,
            metrics.fall_events
        )

        return metrics

    def flush_hour(self) -> HourlyMetrics:
        """
        Manually flush current hour's data.

        Returns:
            HourlyMetrics for the current hour
        """
        with self._lock:
            metrics = self._do_flush()
            self._current_hour_start = self._get_hour_start(time.time())
            return metrics

    def get_current_stats(self) -> Dict[str, Any]:
        """
        Get current (incomplete) hour statistics.
        Useful for real-time dashboards.
        """
        with self._lock:
            total_time = sum(self._posture_time.values())
            return {
                "hour_start": datetime.fromtimestamp(self._current_hour_start).isoformat(),
                "frames_processed": self._frame_count,
                "frames_with_person": self._frames_with_person,
                "observation_seconds": round(total_time, 1),
                "posture_distribution": {
                    k: round(v / total_time * 100, 1) if total_time > 0 else 0
                    for k, v in self._posture_time.items()
                },
                "bed_exits": self._bed_exits,
                "falls": self._falls,
                "current_alerts": dict(self._alert_counts),
            }

    def set_patient_id(self, patient_id: str):
        """Update patient ID (e.g., after face recognition)."""
        with self._lock:
            self.patient_id = patient_id


class HourlyAggregatorManager:
    """
    Manages multiple HourlyAggregators for multi-camera/multi-patient setups.

    Usage:
        manager = HourlyAggregatorManager(db=local_db)

        # Get or create aggregator for a device
        aggregator = manager.get_aggregator("bed_01", "patient_123")
        aggregator.update(...)

        # Manual flush all (e.g., on shutdown)
        manager.flush_all()
    """

    def __init__(self, db=None):
        """
        Args:
            db: LocalDB instance for persisting hourly aggregates
        """
        self.db = db
        self._aggregators: Dict[str, HourlyAggregator] = {}
        self._lock = threading.Lock()

        log.info("HourlyAggregatorManager initialized")

    def _on_flush(self, metrics: HourlyMetrics):
        """Callback when an aggregator flushes its hour."""
        if self.db:
            try:
                self.db.insert_hourly_aggregate(metrics)
            except Exception as e:
                log.exception("Failed to persist hourly aggregate: %s", e)

    def get_aggregator(
        self,
        device_id: str,
        patient_id: str = "unknown"
    ) -> HourlyAggregator:
        """
        Get or create an aggregator for a device.

        Args:
            device_id: Device identifier
            patient_id: Patient identifier

        Returns:
            HourlyAggregator instance
        """
        with self._lock:
            if device_id not in self._aggregators:
                self._aggregators[device_id] = HourlyAggregator(
                    device_id=device_id,
                    patient_id=patient_id,
                    auto_flush=True,
                    flush_callback=self._on_flush,
                )
            else:
                # Update patient ID if changed
                self._aggregators[device_id].set_patient_id(patient_id)

            return self._aggregators[device_id]

    def flush_all(self) -> List[HourlyMetrics]:
        """
        Flush all aggregators.
        Call on shutdown or for manual report generation.

        Returns:
            List of HourlyMetrics from all aggregators
        """
        results = []
        with self._lock:
            for device_id, aggregator in self._aggregators.items():
                try:
                    metrics = aggregator.flush_hour()
                    results.append(metrics)
                except Exception as e:
                    log.exception("Failed to flush aggregator %s: %s", device_id, e)
        return results

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current stats from all aggregators."""
        with self._lock:
            return {
                device_id: agg.get_current_stats()
                for device_id, agg in self._aggregators.items()
            }


# Module-level singleton
_manager: Optional[HourlyAggregatorManager] = None
_manager_lock = threading.Lock()


def get_aggregator_manager(db=None) -> HourlyAggregatorManager:
    """Get or create the global aggregator manager."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = HourlyAggregatorManager(db=db)
    return _manager


def get_aggregator(device_id: str, patient_id: str = "unknown") -> HourlyAggregator:
    """Convenience function to get an aggregator for a device."""
    return get_aggregator_manager().get_aggregator(device_id, patient_id)
