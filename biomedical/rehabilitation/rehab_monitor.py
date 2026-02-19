# biomedical/rehabilitation/rehab_monitor.py
"""
Rehabilitation Monitoring using HPE
Tracks patient progress in therapeutic exercises.

Implements methods from:
- Xu W. et al. (2022): 3D pose estimation for rehabilitation
- Li Y. et al. (2020): Lightweight models for telerehabilitation
- Tao et al. (2020): Robot-assisted rehabilitation with HPE
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
import math
import time

log = logging.getLogger("rehab_monitor")

class RehabilitationMonitor:
    """
    Monitors patient rehabilitation exercises using HPE.
    Provides real-time feedback and progress tracking.
    """

    def __init__(self, exercise_type="upper_limb", target_reps=10, session_duration=300):
        self.exercise_type = exercise_type
        self.target_reps = target_reps
        self.session_duration = session_duration  # seconds

        self.pose_history = deque(maxlen=100)  # Last 100 frames
        self.rep_count = 0
        self.session_start = None  # Lazy initialization when session actually begins
        self.last_rep_time = 0

        # Exercise-specific parameters
        self.exercise_params = self._get_exercise_parameters(exercise_type)

        # Performance metrics
        self.metrics = {
            'reps_completed': 0,
            'accuracy_score': 0.0,
            'range_of_motion': 0.0,
            'movement_smoothness': 0.0,
            'fatigue_indicators': []
        }

    def monitor_exercise(self, keypoints_2d: np.ndarray,
                        keypoints_3d: Optional[np.ndarray] = None) -> Dict:
        """
        Monitor exercise execution in real-time.

        Args:
            keypoints_2d: 2D keypoints (17, 3)
            keypoints_3d: Optional 3D keypoints (17, 4)

        Returns:
            Dict with exercise feedback and metrics
        """
        # Initialize session_start on first frame if not already set
        if self.session_start is None:
            self.session_start = time.time()

        self.pose_history.append(keypoints_2d.copy())

        if len(self.pose_history) < 5:  # Need minimum history
            return {"status": "initializing", "frames_collected": len(self.pose_history)}

        # Analyze current pose
        pose_analysis = self._analyze_pose(keypoints_2d, keypoints_3d)

        # Detect repetitions
        rep_detected = self._detect_repetition(pose_analysis)

        # Update metrics
        self._update_metrics(pose_analysis, rep_detected)

        # Generate feedback
        feedback = self._generate_feedback(pose_analysis, rep_detected)

        # Check session completion
        session_complete = self._check_session_completion()

        return {
            "exercise_type": self.exercise_type,
            "rep_count": self.rep_count,
            "target_reps": self.target_reps,
            "pose_analysis": pose_analysis,
            "rep_detected": rep_detected,
            "feedback": feedback,
            "metrics": self.metrics.copy(),
            "session_complete": session_complete,
            "time_remaining": max(0, self.session_duration - (time.time() - self.session_start)) if self.session_start is not None else self.session_duration
        }

    def _get_exercise_parameters(self, exercise_type: str) -> Dict:
        """Get exercise-specific parameters."""
        params = {
            "upper_limb": {
                "key_joints": [5, 6, 7, 8, 9, 10],  # shoulders, elbows, wrists
                "range_thresholds": {"shoulder_flexion": (30, 150), "elbow_flexion": (10, 140)},
                "rep_phases": ["start", "mid", "end"],
                "critical_angles": {"shoulder": 90, "elbow": 90}
            },
            "lower_limb": {
                "key_joints": [11, 12, 13, 14, 15, 16],  # hips, knees, ankles
                "range_thresholds": {"knee_flexion": (10, 120), "hip_flexion": (20, 100)},
                "rep_phases": ["extension", "flexion"],
                "critical_angles": {"knee": 90, "hip": 45}
            },
            "gait_training": {
                "key_joints": [11, 12, 13, 14, 15, 16],  # hips, knees, ankles
                "range_thresholds": {"knee_flexion": (0, 60), "step_length": (0.3, 0.8)},
                "rep_phases": ["stance", "swing"],
                "critical_angles": {"knee": 30}
            }
        }
        return params.get(exercise_type, params["upper_limb"])

    def _analyze_pose(self, kps_2d: np.ndarray, kps_3d: Optional[np.ndarray]) -> Dict:
        """Analyze current pose for exercise execution."""
        analysis = {}

        # Compute joint angles
        angles = self._compute_joint_angles(kps_2d)
        analysis["joint_angles"] = angles

        # Assess range of motion
        rom = self._assess_range_of_motion(angles)
        analysis["range_of_motion"] = rom

        # Check form accuracy
        form_score = self._assess_form(angles, kps_2d)
        analysis["form_score"] = form_score

        # Detect movement phase
        phase = self._detect_phase(angles)
        analysis["phase"] = phase

        # Compute movement velocity
        if len(self.pose_history) >= 2:
            velocity = self._compute_velocity(kps_2d)
            analysis["velocity"] = velocity

        return analysis

    def _compute_joint_angles(self, kps: np.ndarray) -> Dict:
        """Compute relevant joint angles for the exercise."""
        angles = {}

        if self.exercise_type == "upper_limb":
            # Shoulder angles
            angles['left_shoulder'] = self._compute_angle(kps, 7, 5, 11)  # elbow-shoulder-hip
            angles['right_shoulder'] = self._compute_angle(kps, 8, 6, 12)

            # Elbow angles
            angles['left_elbow'] = self._compute_angle(kps, 9, 7, 5)   # wrist-elbow-shoulder
            angles['right_elbow'] = self._compute_angle(kps, 10, 8, 6)

        elif self.exercise_type in ["lower_limb", "gait_training"]:
            # Hip angles
            angles['left_hip'] = self._compute_angle(kps, 13, 11, 5)   # knee-hip-shoulder
            angles['right_hip'] = self._compute_angle(kps, 14, 12, 6)

            # Knee angles
            angles['left_knee'] = self._compute_angle(kps, 15, 13, 11) # ankle-knee-hip
            angles['right_knee'] = self._compute_angle(kps, 16, 14, 12)

        return angles

    def _assess_range_of_motion(self, angles: Dict) -> float:
        """Assess if movement covers adequate range of motion."""
        params = self.exercise_params

        if self.exercise_type == "upper_limb":
            shoulder_range = max(angles.get('left_shoulder', 0), angles.get('right_shoulder', 0))
            elbow_range = max(angles.get('left_elbow', 0), angles.get('right_elbow', 0))

            # Check if ranges meet thresholds
            shoulder_ok = params["range_thresholds"]["shoulder_flexion"][0] <= shoulder_range <= params["range_thresholds"]["shoulder_flexion"][1]
            elbow_ok = params["range_thresholds"]["elbow_flexion"][0] <= elbow_range <= params["range_thresholds"]["elbow_flexion"][1]

            return (shoulder_ok + elbow_ok) / 2.0

        elif self.exercise_type in ["lower_limb", "gait_training"]:
            knee_range = max(angles.get('left_knee', 0), angles.get('right_knee', 0))
            hip_range = max(angles.get('left_hip', 0), angles.get('right_hip', 0))

            knee_ok = params["range_thresholds"]["knee_flexion"][0] <= knee_range <= params["range_thresholds"]["knee_flexion"][1]
            hip_ok = params["range_thresholds"]["hip_flexion"][0] <= hip_range <= params["range_thresholds"]["hip_flexion"][1]

            return (knee_ok + hip_ok) / 2.0

        return 0.5  # Default

    def _assess_form(self, angles: Dict, kps: np.ndarray) -> float:
        """Assess exercise form accuracy."""
        score = 1.0
        penalties = 0

        if self.exercise_type == "upper_limb":
            # Check for proper shoulder positioning
            left_shoulder = angles.get('left_shoulder', 0)
            right_shoulder = angles.get('right_shoulder', 0)

            if abs(left_shoulder - right_shoulder) > 30:  # Asymmetrical movement
                penalties += 0.2

            # Check elbow positioning
            left_elbow = angles.get('left_elbow', 0)
            right_elbow = angles.get('right_elbow', 0)

            if left_elbow < 10 or right_elbow < 10:  # Over-extension
                penalties += 0.3

        elif self.exercise_type in ["lower_limb", "gait_training"]:
            # Check knee alignment
            left_knee = angles.get('left_knee', 0)
            right_knee = angles.get('right_knee', 0)

            if abs(left_knee - right_knee) > 20:  # Asymmetrical
                penalties += 0.2

        score = max(0.0, score - penalties)
        return score

    def _detect_phase(self, angles: Dict) -> str:
        """Detect current phase of exercise."""
        params = self.exercise_params

        if self.exercise_type == "upper_limb":
            elbow_angle = max(angles.get('left_elbow', 0), angles.get('right_elbow', 0))

            if elbow_angle < params["critical_angles"]["elbow"] * 0.3:
                return "extended"
            elif elbow_angle > params["critical_angles"]["elbow"] * 0.7:
                return "flexed"
            else:
                return "mid"

        elif self.exercise_type in ["lower_limb", "gait_training"]:
            knee_angle = max(angles.get('left_knee', 0), angles.get('right_knee', 0))

            if knee_angle < params["critical_angles"]["knee"] * 0.5:
                return "extended"
            else:
                return "flexed"

        return "unknown"

    def _detect_repetition(self, pose_analysis: Dict) -> bool:
        """Detect if a repetition has been completed."""
        current_phase = pose_analysis.get("phase", "unknown")

        # Simple repetition detection based on phase transitions
        if len(self.pose_history) < 10:
            return False

        # Look for phase transitions in recent history
        recent_phases = []
        for i in range(max(0, len(self.pose_history) - 10), len(self.pose_history)):
            # Re-analyze each pose (simplified)
            angles = self._compute_joint_angles(self.pose_history[i])
            phase = self._detect_phase(angles)
            recent_phases.append(phase)

        # Detect full cycle (e.g., extended -> flexed -> extended)
        if len(recent_phases) >= 5:
            transitions = []
            for i in range(1, len(recent_phases)):
                if recent_phases[i] != recent_phases[i-1]:
                    transitions.append((recent_phases[i-1], recent_phases[i]))

            # Check for complete rep (flexed -> extended)
            if ("flexed", "extended") in transitions and time.time() - self.last_rep_time > 1.0:
                self.rep_count += 1
                self.last_rep_time = time.time()
                return True

        return False

    def _compute_velocity(self, current_kps: np.ndarray) -> float:
        """Compute average velocity of key joints."""
        if len(self.pose_history) < 2:
            return 0.0

        prev_kps = self.pose_history[-2]
        velocities = []

        for joint_idx in self.exercise_params["key_joints"]:
            if (current_kps[joint_idx][2] > 0.5 and prev_kps[joint_idx][2] > 0.5):
                vel = np.linalg.norm(current_kps[joint_idx][:2] - prev_kps[joint_idx][:2])
                velocities.append(vel)

        return np.mean(velocities) if velocities else 0.0

    def _update_metrics(self, pose_analysis: Dict, rep_detected: bool):
        """Update performance metrics."""
        self.metrics['reps_completed'] = self.rep_count

        # Update accuracy score (weighted average)
        current_accuracy = pose_analysis.get('form_score', 0.5)
        self.metrics['accuracy_score'] = 0.9 * self.metrics['accuracy_score'] + 0.1 * current_accuracy

        # Update range of motion
        current_rom = pose_analysis.get('range_of_motion', 0.5)
        self.metrics['range_of_motion'] = 0.95 * self.metrics['range_of_motion'] + 0.05 * current_rom

        # Update movement smoothness (lower velocity variation = smoother)
        if 'velocity' in pose_analysis:
            velocity = pose_analysis['velocity']
            self.metrics['movement_smoothness'] = 0.9 * self.metrics['movement_smoothness'] + 0.1 * (1.0 / (1.0 + velocity))

    def _generate_feedback(self, pose_analysis: Dict, rep_detected: bool) -> Dict:
        """Generate real-time feedback for the patient."""
        feedback = {
            "message": "",
            "type": "info",  # "success", "warning", "error", "info"
            "suggestions": []
        }

        form_score = pose_analysis.get('form_score', 0.5)
        rom = pose_analysis.get('range_of_motion', 0.5)

        if rep_detected:
            feedback["message"] = f"Great! Repetition {self.rep_count} completed."
            feedback["type"] = "success"
        elif form_score < 0.7:
            feedback["message"] = "Adjust your form."
            feedback["type"] = "warning"
            if self.exercise_type == "upper_limb":
                feedback["suggestions"].append("Keep elbows close to your body")
                feedback["suggestions"].append("Move both arms symmetrically")
            elif self.exercise_type in ["lower_limb", "gait_training"]:
                feedback["suggestions"].append("Keep knees aligned")
                feedback["suggestions"].append("Maintain balanced posture")
        elif rom < 0.8:
            feedback["message"] = "Increase your range of motion."
            feedback["type"] = "info"
            feedback["suggestions"].append("Try to move further in each direction")
        else:
            feedback["message"] = "Good form! Keep going."
            feedback["type"] = "info"

        return feedback

    def _check_session_completion(self) -> bool:
        """Check if session goals have been met."""
        if self.session_start is None:
            return False  # Session hasn't started yet
        
        time_elapsed = time.time() - self.session_start

        if (self.rep_count >= self.target_reps or time_elapsed >= self.session_duration):
            return True

        return False

    def _compute_angle(self, kps: np.ndarray, joint1: int, joint2: int, joint3: int) -> float:
        """Compute angle at joint2 formed by joint1-joint2-joint3."""
        if (kps[joint1][2] < 0.5 or kps[joint2][2] < 0.5 or kps[joint3][2] < 0.5):
            return 0.0

        v1 = kps[joint1][:2] - kps[joint2][:2]
        v2 = kps[joint3][:2] - kps[joint2][:2]

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        epsilon = np.finfo(float).eps
        if n1 < epsilon or n2 < epsilon:
            return np.nan  # Return NaN for zero-length vectors
        
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))