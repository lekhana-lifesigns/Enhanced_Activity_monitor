# analytics/clinical_monitor.py
"""
Clinical Monitor for EAM System (Enterprise-Grade)
Implements clinical outcomes: fall risk, bed exit, immobility, distress detection.

CRITICAL DESIGN PRINCIPLES:
1. Posture and Support Surface are SEPARATE concepts
   - Posture: lying | sitting | standing | kneeling | squatting | unknown
   - Support Surface: bed | chair | couch | wheelchair | floor | unknown

2. Bed exit is a SUPPORT SURFACE transition, not a posture change
   - Correct: support_surface: bed -> not bed
   - Wrong: posture not in BED_POSTURES

3. State authority order:
   1. Temporal state machine (authoritative)
   2. Support surface
   3. Posture
   4. Activity
   5. Raw pose

4. Confidence must be properly composed, not trusted from detector
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger("clinical_monitor")


@dataclass
class PostureHysteresis:
    """Posture state with temporal hysteresis."""
    current: str = "unknown"
    since: float = field(default_factory=time.time)
    confidence: float = 0.0
    support_surface: str = "unknown"
    support_confidence: float = 0.0

    def update(self, posture: str, confidence: float,
               support_surface: str, support_confidence: float,
               min_stability_time: float = 0.8) -> bool:
        """
        Update posture with hysteresis.
        Returns True if posture change is confirmed.
        """
        now = time.time()

        # Same posture - reinforce
        if posture == self.current:
            self.confidence = 0.7 * confidence + 0.3 * self.confidence
            self.support_surface = support_surface
            self.support_confidence = support_confidence
            return False

        # Different posture - check stability
        time_in_new = now - self.since if posture != self.current else 0

        # Only confirm change after stability period
        if time_in_new >= min_stability_time and confidence > 0.5:
            old_posture = self.current
            self.current = posture
            self.since = now
            self.confidence = confidence
            self.support_surface = support_surface
            self.support_confidence = support_confidence
            log.debug("Posture confirmed: %s -> %s (after %.1fs)",
                     old_posture, posture, time_in_new)
            return True

        return False


class ClinicalMonitor:
    """
    Clinical monitoring system for patient safety outcomes.
    Focused on clinically actionable signals (not generic HAR).

    CORRECT SEMANTICS:
    - Posture: what the body is doing (biomechanical)
    - Support Surface: what supports the body (environmental)
    - These are INDEPENDENT dimensions
    """

    # Valid postures (biomechanical only - NO location)
    VALID_POSTURES = {'lying', 'sitting', 'standing', 'kneeling', 'squatting', 'unknown'}

    # Valid support surfaces (environmental only - NO posture)
    VALID_SUPPORT_SURFACES = {'bed', 'chair', 'couch', 'wheelchair', 'floor', 'unknown'}

    # Support surfaces considered "in bed" for bed exit detection
    BED_SURFACES = {'bed'}

    def __init__(self, window_size: int = 300, alert_cooldown: int = 60):
        """
        Args:
            window_size: seconds of history to maintain
            alert_cooldown: minimum seconds between similar alerts
        """
        self.window_size = window_size
        self.alert_cooldown = alert_cooldown

        # Patient state with CORRECT semantics
        self.patient_state = {
            'posture': 'unknown',
            'support_surface': 'unknown',
            'support_confidence': 0.0,
            'on_bed': False,  # Derived from support_surface, NOT posture
            'last_bed_exit': None,
            'immobility_start': None,
            'distress_start': None,
            'fall_risk_level': 'low',
            'baseline_state': 'collecting',
        }

        # Posture hysteresis (temporal authority)
        self.posture_hysteresis = PostureHysteresis()

        # Alert throttling
        self.last_alerts = {
            'bed_exit': 0.0,
            'fall_risk': 0.0,
            'immobility': 0.0,
            'distress': 0.0,
            'sensor_drift': 0.0,
        }

        # Rolling buffers
        self.pose_history = deque(maxlen=window_size)
        self.activity_history = deque(maxlen=window_size)
        self.baseline_history = deque(maxlen=window_size * 10)

        # Baseline anomaly score history (for trend detection)
        self.anomaly_score_history = deque(maxlen=window_size)

        # Pelvis tracking for bed exit (velocity-based)
        self.pelvis_height_history = deque(maxlen=30)

        # Clinical thresholds
        self.immobility_threshold = 600.0   # seconds (10 min)
        self.distress_threshold = 120.0     # seconds
        self.fall_risk_threshold = 0.5      # anomaly_score threshold
        self.sensor_drift_threshold = 2.0   # PCA reconstruction error threshold
        self.bed_exit_velocity_threshold = 0.05  # normalized units/sec
        self.support_confidence_threshold = 0.7

        log.info("ClinicalMonitor initialized (posture/support surface separated)")

    def update_clinical_state(
        self,
        pose_result: Dict,
        activity_result: Dict,
        baseline_info: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Update clinical state and generate alerts.

        Expected pose_result schema:
        {
            "posture": "sitting",           # biomechanical only
            "support_surface": "couch",     # environmental only
            "support_confidence": 0.82,
            "posture_confidence": 0.71,
            "pelvis_height_norm": 0.42,     # optional
            "vertical_velocity": 0.01,      # optional
            "temporal_stability": 0.85,     # from PostureStateMachine
        }
        """
        alerts = []
        current_time = time.time()

        # Extract posture (biomechanical)
        posture = pose_result.get('posture', 'unknown')
        if posture and posture.upper() in ['SITTING', 'STANDING', 'LYING']:
            posture = posture.lower()
        posture_confidence = pose_result.get('posture_confidence',
                                             pose_result.get('confidence', 0.0))

        # Extract support surface (environmental) - CRITICAL
        support_surface = pose_result.get('support_surface', {})
        if isinstance(support_surface, dict):
            surface_type = support_surface.get('surface_type', 'unknown')
            support_conf = pose_result.get('support_confidence',
                                          support_surface.get('confidence', 0.0))
        else:
            surface_type = str(support_surface) if support_surface else 'unknown'
            support_conf = pose_result.get('support_confidence', 0.0)

        # Normalize surface type
        if surface_type in ['elevated', 'ground']:
            # Map generic types to specific where possible
            if posture == 'lying':
                surface_type = 'bed'
            elif posture == 'sitting' and surface_type == 'elevated':
                surface_type = 'chair'  # Could be couch, chair, etc.

        # Extract pelvis tracking data
        pelvis_height = pose_result.get('pelvis_height_norm', 0.5)
        vertical_velocity = pose_result.get('vertical_velocity', 0.0)

        # Track pelvis for bed exit detection
        self.pelvis_height_history.append({
            'timestamp': current_time,
            'height': pelvis_height,
            'velocity': vertical_velocity,
        })

        # Activity
        activity = activity_result.get('activity', 'unknown')
        activity_confidence = activity_result.get('confidence', 0.0)

        # Apply posture hysteresis (temporal authority)
        posture_changed = self.posture_hysteresis.update(
            posture=posture,
            confidence=posture_confidence,
            support_surface=surface_type,
            support_confidence=support_conf
        )

        # Use hysteresis-confirmed values
        confirmed_posture = self.posture_hysteresis.current
        confirmed_surface = self.posture_hysteresis.support_surface
        confirmed_support_conf = self.posture_hysteresis.support_confidence

        # Update histories with confirmed values
        self.pose_history.append({
            'timestamp': current_time,
            'posture': confirmed_posture,
            'support_surface': confirmed_surface,
            'support_confidence': confirmed_support_conf,
            'activity': activity,
            'confidence': posture_confidence
        })
        self.activity_history.append({
            'timestamp': current_time,
            'activity': activity
        })

        # Track baseline anomaly scores over time
        if baseline_info:
            self.anomaly_score_history.append({
                'timestamp': current_time,
                'anomaly_score': baseline_info.get('anomaly_score', 0.0),
                'kl_score': baseline_info.get('kl_score', 0.0),
                'pca_error': baseline_info.get('pca_reconstruction_error', 0.0),
                'cluster_id': baseline_info.get('cluster_id', -1),
            })
            self.patient_state['baseline_state'] = baseline_info.get(
                'baseline_state', 'collecting'
            )

        # Determine if on bed (from SUPPORT SURFACE, not posture)
        was_on_bed = self.patient_state['on_bed']
        now_on_bed = (
            confirmed_surface in self.BED_SURFACES and
            confirmed_support_conf > self.support_confidence_threshold
        )

        # 1. Bed exit detection (CORRECT: support surface transition)
        alert = self._detect_bed_exit(
            was_on_bed=was_on_bed,
            now_on_bed=now_on_bed,
            posture=confirmed_posture,
            activity=activity,
            pelvis_velocity=vertical_velocity,
            current_time=current_time
        )
        if alert:
            alerts.append(alert)

        # 2. Fall risk
        alert = self._detect_fall_risk(
            pose_result, activity_result, baseline_info, current_time
        )
        if alert:
            alerts.append(alert)

        # 3. Immobility (CORRECT: any supported posture, not just bed)
        alert = self._detect_immobility(current_time)
        if alert:
            alerts.append(alert)

        # 4. Distress
        alert = self._detect_distress(
            activity_result, baseline_info, current_time
        )
        if alert:
            alerts.append(alert)

        # 5. Sensor drift
        alert = self._detect_sensor_drift(baseline_info, current_time)
        if alert:
            alerts.append(alert)

        # Update state with CORRECT semantics
        self.patient_state['posture'] = confirmed_posture
        self.patient_state['support_surface'] = confirmed_surface
        self.patient_state['support_confidence'] = confirmed_support_conf
        self.patient_state['on_bed'] = now_on_bed

        return alerts

    def _detect_bed_exit(
        self,
        was_on_bed: bool,
        now_on_bed: bool,
        posture: str,
        activity: str,
        pelvis_velocity: float,
        current_time: float
    ) -> Optional[Dict]:
        """
        Detect bed exit using SUPPORT SURFACE transition (correct approach).

        Bed exit is detected when:
        1. Support surface changes from bed to non-bed
        2. AND vertical velocity indicates rising movement
        3. AND state persists > 0.8s (handled by hysteresis)
        """
        if was_on_bed and not now_on_bed:
            # Verify with pelvis movement (rising motion)
            rising = pelvis_velocity > self.bed_exit_velocity_threshold

            if current_time - self.last_alerts['bed_exit'] > self.alert_cooldown:
                self.last_alerts['bed_exit'] = current_time
                self.patient_state['last_bed_exit'] = current_time

                return {
                    'type': 'bed_exit',
                    'severity': 'high',
                    'message': 'Patient has exited bed',
                    'timestamp': current_time,
                    'posture': posture,
                    'activity': activity,
                    'rising_motion': rising,
                    'support_surface_before': 'bed',
                    'support_surface_after': self.patient_state.get('support_surface', 'unknown'),
                    'requires_response': True
                }

        return None

    def _detect_fall_risk(
        self,
        pose_result: Dict,
        activity_result: Dict,
        baseline_info: Optional[Dict],
        current_time: float
    ) -> Optional[Dict]:
        """Detect fall risk based on posture, activity, and baseline anomalies."""
        posture = pose_result.get('posture', 'unknown')
        if isinstance(posture, str):
            posture = posture.lower()
        activity = activity_result.get('activity', 'unknown')

        risk_factors = []

        # Unstable postures
        if posture in {'kneeling', 'squatting', 'transitioning'}:
            risk_factors.append('unstable_posture')

        # High-risk activities
        if activity in {'falling', 'tripping', 'stumbling'}:
            risk_factors.append('high_risk_activity')

        # Low support confidence with non-lying posture
        support_conf = pose_result.get('support_confidence', 0.0)
        if posture not in {'lying', 'unknown'} and support_conf < 0.5:
            risk_factors.append('low_support_confidence')

        # Behavioral anomaly from baseline
        if baseline_info and baseline_info.get('is_anomaly'):
            anomaly_score = baseline_info.get('anomaly_score', 0.0)
            if anomaly_score > self.fall_risk_threshold:
                risk_factors.append('behavioral_anomaly')

        # Rising anomaly trend
        if self._is_anomaly_trend_rising():
            risk_factors.append('anomaly_trend_rising')

        # Determine risk level
        if len(risk_factors) >= 2:
            risk_level = 'high'
        elif len(risk_factors) == 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        if risk_level != self.patient_state['fall_risk_level']:
            if risk_level in {'medium', 'high'}:
                if current_time - self.last_alerts['fall_risk'] > self.alert_cooldown:
                    self.last_alerts['fall_risk'] = current_time
                    self.patient_state['fall_risk_level'] = risk_level

                    alert = {
                        'type': 'fall_risk',
                        'severity': risk_level,
                        'message': f'Fall risk increased to {risk_level}',
                        'timestamp': current_time,
                        'risk_factors': risk_factors,
                        'posture': posture,
                        'activity': activity,
                        'support_surface': self.patient_state.get('support_surface'),
                        'requires_response': risk_level == 'high'
                    }
                    if baseline_info:
                        alert['anomaly_score'] = baseline_info.get('anomaly_score', 0.0)
                        alert['cluster_id'] = baseline_info.get('cluster_id', -1)
                    return alert
            else:
                self.patient_state['fall_risk_level'] = risk_level

        return None

    def _detect_immobility(self, current_time: float) -> Optional[Dict]:
        """
        Detect prolonged immobility.
        CORRECT: Works with ANY supported posture, not just bed.
        """
        if len(self.pose_history) < 60:
            return None

        recent = list(self.pose_history)[-300:]

        # Immobile = still activity with high support confidence
        # NOT limited to bed postures
        immobile = sum(
            1 for p in recent
            if p['activity'] == 'still' and p['support_confidence'] > 0.7
        )

        ratio = immobile / len(recent)

        if ratio > 0.8:
            if self.patient_state['immobility_start'] is None:
                self.patient_state['immobility_start'] = current_time
            elif current_time - self.patient_state['immobility_start'] > self.immobility_threshold:
                if current_time - self.last_alerts['immobility'] > self.alert_cooldown:
                    self.last_alerts['immobility'] = current_time

                    # Get dominant support surface during immobility
                    surfaces = [p['support_surface'] for p in recent]
                    dominant_surface = max(set(surfaces), key=surfaces.count)

                    return {
                        'type': 'immobility',
                        'severity': 'medium',
                        'message': 'Prolonged immobility detected',
                        'timestamp': current_time,
                        'duration_minutes': (current_time - self.patient_state['immobility_start']) / 60.0,
                        'immobility_ratio': ratio,
                        'support_surface': dominant_surface,
                        'requires_response': True
                    }
        else:
            self.patient_state['immobility_start'] = None

        return None

    def _detect_distress(
        self,
        activity_result: Dict,
        baseline_info: Optional[Dict],
        current_time: float
    ) -> Optional[Dict]:
        """Detect patient distress signals."""
        activity = activity_result.get('activity', 'unknown')
        indicators = []

        if activity in {'agitated', 'thrashing', 'restless', 'distress_signs'}:
            indicators.append('agitation')

        if baseline_info and baseline_info.get('is_anomaly'):
            indicators.append('behavioral_deviation')

        if activity in {'pulling_at_tubes', 'rubbing_face'}:
            indicators.append('self_soothing')

        if indicators:
            if self.patient_state['distress_start'] is None:
                self.patient_state['distress_start'] = current_time
            elif current_time - self.patient_state['distress_start'] > self.distress_threshold:
                if current_time - self.last_alerts['distress'] > self.alert_cooldown:
                    self.last_alerts['distress'] = current_time

                    alert = {
                        'type': 'distress',
                        'severity': 'medium',
                        'message': 'Patient distress signals detected',
                        'timestamp': current_time,
                        'duration_minutes': (current_time - self.patient_state['distress_start']) / 60.0,
                        'indicators': indicators,
                        'activity': activity,
                        'requires_response': True
                    }
                    if baseline_info:
                        alert['anomaly_score'] = baseline_info.get('anomaly_score', 0.0)
                    return alert
        else:
            self.patient_state['distress_start'] = None

        return None

    def _detect_sensor_drift(
        self,
        baseline_info: Optional[Dict],
        current_time: float
    ) -> Optional[Dict]:
        """
        Detect sensor/camera drift via PCA reconstruction error.
        High error = features outside learned subspace.
        """
        if not baseline_info or baseline_info.get('baseline_state') != 'ready':
            return None

        pca_error = baseline_info.get('pca_reconstruction_error', 0.0)
        if pca_error < self.sensor_drift_threshold:
            return None

        if current_time - self.last_alerts['sensor_drift'] > self.alert_cooldown * 5:
            self.last_alerts['sensor_drift'] = current_time
            return {
                'type': 'sensor_drift',
                'severity': 'low',
                'message': 'Feature instability detected (possible camera/lighting change)',
                'timestamp': current_time,
                'pca_reconstruction_error': pca_error,
                'requires_response': False
            }

        return None

    def _is_anomaly_trend_rising(self) -> bool:
        """Check if anomaly scores are trending upward."""
        if len(self.anomaly_score_history) < 30:
            return False

        recent = list(self.anomaly_score_history)
        scores = [r['anomaly_score'] for r in recent]

        n = len(scores)
        split = max(1, n // 3)
        early_mean = np.mean(scores[:split])
        late_mean = np.mean(scores[-split:])

        return late_mean > early_mean + 0.15

    def get_clinical_summary(self) -> Dict:
        """Get summary of clinical state."""
        baseline_ready = self.patient_state.get('baseline_state') == 'ready'

        avg_anomaly = 0.0
        if self.anomaly_score_history:
            avg_anomaly = np.mean([r['anomaly_score'] for r in self.anomaly_score_history])

        return {
            'patient_state': {
                'posture': self.patient_state['posture'],
                'support_surface': self.patient_state['support_surface'],
                'on_bed': self.patient_state['on_bed'],
                'fall_risk_level': self.patient_state['fall_risk_level'],
            },
            'current_metrics': {
                'time_on_bed': self._calculate_time_on_bed(),
                'activity_diversity': self._calculate_activity_diversity(),
                'posture_stability': self._calculate_posture_stability(),
                'avg_anomaly_score': float(avg_anomaly),
            },
            'system_health': {
                'buffer_size': len(self.pose_history),
                'baseline_ready': baseline_ready,
                'anomaly_history_size': len(self.anomaly_score_history),
            }
        }

    def _calculate_time_on_bed(self) -> float:
        """Calculate fraction of time on bed support surface."""
        if not self.pose_history:
            return 0.0
        on_bed = sum(
            1 for p in self.pose_history
            if p.get('support_surface') in self.BED_SURFACES
            and p.get('support_confidence', 0) > self.support_confidence_threshold
        )
        return on_bed / len(self.pose_history)

    def _calculate_activity_diversity(self) -> float:
        """Calculate activity diversity (entropy)."""
        if not self.activity_history:
            return 0.0
        acts = [p['activity'] for p in self.activity_history]
        entropy = 0.0
        for a in set(acts):
            p = acts.count(a) / len(acts)
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _calculate_posture_stability(self) -> float:
        """Calculate posture stability (fewer transitions = more stable)."""
        if len(self.pose_history) < 2:
            return 1.0
        history = list(self.pose_history)
        transitions = sum(
            1 for i in range(1, len(history))
            if history[i]['posture'] != history[i - 1]['posture']
        )
        return max(0.0, 1.0 - transitions / len(history))
