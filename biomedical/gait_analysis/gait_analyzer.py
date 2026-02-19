# biomedical/gait_analysis/gait_analyzer.py
"""
Gait Analysis using HPE
Analyzes walking patterns for clinical assessment.

Implements methods from:
- Shin et al. (2021): Temporospatial parameters from monocular video
- Kanko et al. (2021): Comparison of HPE methods for gait analysis
- Washabaugh et al. (2022): DeepLabCut vs marker-based for gait
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
import math
import time

log = logging.getLogger("gait_analyzer")

class GaitAnalyzer:
    """
    Analyzes gait patterns using HPE for clinical assessment.
    Computes spatiotemporal parameters and detects abnormalities.
    """

    def __init__(self, fps=30, analysis_window=300):  # 10 seconds
        self.fps = fps
        self.analysis_window = analysis_window

        self.pose_history = deque(maxlen=analysis_window)
        self.step_events = deque(maxlen=1000)  # Bounded to prevent memory growth
        self.gait_cycles = deque(maxlen=500)  # Bounded to prevent memory growth

        # Gait parameters
        self.parameters = {
            'step_length': [],
            'stride_length': [],
            'cadence': [],
            'walking_speed': [],
            'step_time': [],
            'stance_time': [],
            'swing_time': []
        }

        # Clinical thresholds (approximate, should be calibrated)
        self.thresholds = {
            'normal_cadence': (100, 120),  # steps/min
            'normal_speed': (1.0, 1.5),    # m/s
            'normal_step_length': (0.5, 0.8)  # meters
        }

    def analyze_gait(self, keypoints_2d: np.ndarray,
                    keypoints_3d: Optional[np.ndarray] = None,
                    timestamp: Optional[float] = None) -> Dict:
        """
        Analyze gait from pose keypoints.

        Args:
            keypoints_2d: 2D keypoints (17, 3)
            keypoints_3d: Optional 3D keypoints (17, 4)
            timestamp: Frame timestamp

        Returns:
            Dict with gait parameters and clinical assessment
        """
        if timestamp is None:
            timestamp = time.time()

        self.pose_history.append({
            'keypoints_2d': keypoints_2d.copy(),
            'keypoints_3d': keypoints_3d.copy() if keypoints_3d is not None else None,
            'timestamp': timestamp
        })

        if len(self.pose_history) < 30:  # Need minimum frames for analysis
            return {"status": "collecting_data", "frames_collected": len(self.pose_history)}

        # Detect step events
        step_detected = self._detect_step_events()

        # Compute gait parameters
        if len(self.step_events) >= 2:
            self._compute_gait_parameters()

        # Assess gait quality
        assessment = self._assess_gait_quality()

        return {
            "gait_parameters": self.parameters.copy(),
            "step_events": len(self.step_events),
            "gait_cycles": len(self.gait_cycles),
            "assessment": assessment,
            "step_detected": step_detected,
            "frames_analyzed": len(self.pose_history)
        }

    def _detect_step_events(self) -> bool:
        """Detect heel strikes and toe-offs using ankle positions."""
        if len(self.pose_history) < 5:
            return False

        current = self.pose_history[-1]
        prev = self.pose_history[-2]

        kps_curr = current['keypoints_2d']
        kps_prev = prev['keypoints_2d']

        # Use ankle positions for step detection
        left_ankle_curr = kps_curr[15][:2] if kps_curr[15][2] > 0.5 else None
        right_ankle_curr = kps_curr[16][:2] if kps_curr[16][2] > 0.5 else None
        left_ankle_prev = kps_prev[15][:2] if kps_prev[15][2] > 0.5 else None
        right_ankle_prev = kps_prev[16][:2] if kps_prev[16][2] > 0.5 else None

        step_detected = False

        # Detect left heel strike (ankle moves forward and down)
        if (left_ankle_curr is not None and left_ankle_prev is not None and
            left_ankle_curr[0] > left_ankle_prev[0] and  # Forward movement
            left_ankle_curr[1] > left_ankle_prev[1]):   # Downward movement
            self.step_events.append({
                'timestamp': time.time(),  # Normalize to wall-clock time
                'foot': 'left',
                'type': 'heel_strike',
                'position': left_ankle_curr
            })
            step_detected = True

        # Detect right heel strike
        if (right_ankle_curr is not None and right_ankle_prev is not None and
            right_ankle_curr[0] > right_ankle_prev[0] and
            right_ankle_curr[1] > right_ankle_prev[1]):
            self.step_events.append({
                'timestamp': time.time(),  # Normalize to wall-clock time
                'foot': 'right',
                'type': 'heel_strike',
                'position': right_ankle_curr
            })
            step_detected = True

        return step_detected

    def _compute_gait_parameters(self):
        """Compute spatiotemporal gait parameters."""
        if len(self.step_events) < 2:
            return

        # Get recent step events (last 10 seconds)
        recent_steps = [s for s in self.step_events
                       if time.time() - s['timestamp'] < 10.0]

        if len(recent_steps) < 2:
            return

        # Compute step times
        step_times = []
        for i in range(1, len(recent_steps)):
            dt = recent_steps[i]['timestamp'] - recent_steps[i-1]['timestamp']
            if dt > 0.1 and dt < 2.0:  # Reasonable step time range
                step_times.append(dt)

        if step_times:
            mean_step_time = np.mean(step_times)
            self.parameters['step_time'].append(mean_step_time)

            # Cadence (steps per minute)
            cadence = 60.0 / mean_step_time
            self.parameters['cadence'].append(cadence)

        # Estimate step length (simplified - would need calibration)
        # Using ankle positions and assuming known camera setup
        step_lengths = []
        for i in range(1, len(recent_steps), 2):  # Alternate feet
            if i < len(recent_steps):
                pos1 = recent_steps[i-1]['position']
                pos2 = recent_steps[i]['position']

                # Simple distance calculation (would need scale calibration)
                distance = np.linalg.norm(pos2 - pos1)
                step_lengths.append(distance)

        if step_lengths:
            self.parameters['step_length'].append(np.mean(step_lengths))

        # Stride length (two steps)
        if len(step_lengths) >= 2:
            stride_length = np.mean(step_lengths) * 2
            self.parameters['stride_length'].append(stride_length)

        # Walking speed (simplified)
        if step_lengths and step_times:
            speed = np.mean(step_lengths) / np.mean(step_times)  # pixels/sec
            self.parameters['walking_speed'].append(speed)

        # Keep only recent parameters (last 50)
        for key in self.parameters:
            if len(self.parameters[key]) > 50:
                self.parameters[key] = self.parameters[key][-50:]

    def _assess_gait_quality(self) -> Dict:
        """Assess gait quality against clinical norms."""
        assessment = {
            "overall_quality": "unknown",
            "abnormalities": [],
            "recommendations": [],
            "confidence": 0.0
        }

        if not self.parameters['cadence']:
            return assessment

        # Assess cadence
        recent_cadence = np.mean(self.parameters['cadence'][-10:]) if len(self.parameters['cadence']) >= 10 else np.mean(self.parameters['cadence'])
        cadence_normal = self.thresholds['normal_cadence'][0] <= recent_cadence <= self.thresholds['normal_cadence'][1]

        if not cadence_normal:
            if recent_cadence < self.thresholds['normal_cadence'][0]:
                assessment["abnormalities"].append("Low cadence (slow walking)")
                assessment["recommendations"].append("Consider increasing walking speed")
            else:
                assessment["abnormalities"].append("High cadence (fast walking)")
                assessment["recommendations"].append("Consider slowing down")

        # Assess step length
        if self.parameters['step_length']:
            recent_step_length = np.mean(self.parameters['step_length'][-10:]) if len(self.parameters['step_length']) >= 10 else np.mean(self.parameters['step_length'])
            step_normal = self.thresholds['normal_step_length'][0] <= recent_step_length <= self.thresholds['normal_step_length'][1]

            if not step_normal:
                if recent_step_length < self.thresholds['normal_step_length'][0]:
                    assessment["abnormalities"].append("Short step length")
                    assessment["recommendations"].append("Try taking longer steps")
                else:
                    assessment["abnormalities"].append("Long step length")
                    assessment["recommendations"].append("Reduce step length for stability")

        # Assess walking speed
        if self.parameters['walking_speed']:
            recent_speed = np.mean(self.parameters['walking_speed'][-10:]) if len(self.parameters['walking_speed']) >= 10 else np.mean(self.parameters['walking_speed'])
            speed_normal = self.thresholds['normal_speed'][0] <= recent_speed <= self.thresholds['normal_speed'][1]

            if not speed_normal:
                if recent_speed < self.thresholds['normal_speed'][0]:
                    assessment["abnormalities"].append("Slow walking speed")
                    assessment["recommendations"].append("Increase walking pace gradually")
                else:
                    assessment["abnormalities"].append("Fast walking speed")
                    assessment["recommendations"].append("Slow down for better control")

        # Overall assessment
        abnormality_count = len(assessment["abnormalities"])
        if abnormality_count == 0:
            assessment["overall_quality"] = "normal"
            assessment["confidence"] = 0.8
        elif abnormality_count == 1:
            assessment["overall_quality"] = "mildly_abnormal"
            assessment["confidence"] = 0.6
        else:
            assessment["overall_quality"] = "abnormal"
            assessment["confidence"] = 0.7

        return assessment

    def get_gait_metrics(self) -> Dict:
        """Get comprehensive gait metrics summary."""
        metrics = {}

        for param in self.parameters:
            if self.parameters[param]:
                values = np.array(self.parameters[param])
                metrics[f"{param}_mean"] = np.mean(values)
                metrics[f"{param}_std"] = np.std(values)
                metrics[f"{param}_min"] = np.min(values)
                metrics[f"{param}_max"] = np.max(values)

        # Additional derived metrics
        if 'cadence' in metrics and 'step_length' in metrics:
            # Walking speed in more interpretable units
            metrics['estimated_speed_m_s'] = metrics['step_length_mean'] * metrics['cadence_mean'] / 60.0

        # Asymmetry metrics
        if len(self.step_events) >= 4:
            left_steps = [s for s in self.step_events[-20:] if s['foot'] == 'left']
            right_steps = [s for s in self.step_events[-20:] if s['foot'] == 'right']

            if left_steps and right_steps:
                left_times = [s['timestamp'] for s in left_steps]
                right_times = [s['timestamp'] for s in right_steps]

                # Step time asymmetry
                if len(left_times) >= 2 and len(right_times) >= 2:
                    left_step_times = np.diff(left_times)
                    right_step_times = np.diff(right_times)

                    left_mean = np.mean(left_step_times)
                    right_mean = np.mean(right_step_times)

                    metrics['step_time_asymmetry'] = abs(left_mean - right_mean) / max(left_mean, right_mean)

        return metrics

    def reset_analysis(self):
        """Reset gait analysis for new session."""
        self.pose_history.clear()
        self.step_events.clear()
        self.gait_cycles.clear()

        for key in self.parameters:
            self.parameters[key].clear()