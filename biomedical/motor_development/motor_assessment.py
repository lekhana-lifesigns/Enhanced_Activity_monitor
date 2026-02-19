# biomedical/motor_development/motor_assessment.py
"""
Motor Development Assessment for Infants
Based on HPE for early detection of neurological disorders.

Implements methods from:
- Doroniewicz et al. (2020): Automatic detection of writhing movements
- McCay et al. (2020): HOJO2D/HOJD2D features
- Chambers et al. (2020): NGBS for atypical development detection
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
import math

log = logging.getLogger("motor_assessment")

class MotorDevelopmentAssessor:
    """
    Assesses infant motor development using HPE.
    Detects atypical movements indicative of neurological disorders.
    """

    def __init__(self, window_size=300, fps=30):  # 10 seconds at 30fps
        self.window_size = window_size
        self.fps = fps
        self.pose_history = deque(maxlen=window_size)
        self.feature_history = deque(maxlen=window_size)

        # Movement quality thresholds (from literature)
        self.writhing_threshold = 0.7  # Movement quality score
        self.fidgety_threshold = 0.6   # Fidgety movement score

    def assess_movement_quality(self, keypoints_2d: np.ndarray,
                              keypoints_3d: Optional[np.ndarray] = None) -> Dict:
        """
        Assess movement quality using multiple metrics.

        Args:
            keypoints_2d: 2D keypoints (17, 3) - x, y, confidence
            keypoints_3d: Optional 3D keypoints (17, 4) - x, y, z, confidence

        Returns:
            Dict with movement quality scores and classifications
        """
        if len(self.pose_history) < 10:  # Need minimum history
            self.pose_history.append(keypoints_2d.copy())
            return {"status": "collecting_data", "frames_collected": len(self.pose_history)}

        self.pose_history.append(keypoints_2d.copy())

        # Extract features
        features = self._extract_motor_features(keypoints_2d, keypoints_3d)
        self.feature_history.append(features)

        # Compute movement quality scores
        quality_scores = self._compute_quality_scores()

        # Classify movements
        classification = self._classify_movements(quality_scores)

        return {
            "movement_quality": quality_scores,
            "classification": classification,
            "features": features,
            "frames_analyzed": len(self.pose_history)
        }

    def _extract_motor_features(self, kps_2d: np.ndarray,
                               kps_3d: Optional[np.ndarray]) -> Dict:
        """Extract motor development features from pose."""
        features = {}

        # Basic pose features
        features.update(self._compute_basic_pose_features(kps_2d))

        # Movement features (require history)
        if len(self.pose_history) > 1:
            features.update(self._compute_movement_features(kps_2d))

        # HOJO2D/HOJD2D features (McCay et al., 2020)
        if len(self.pose_history) >= 5:
            features.update(self._compute_hoj_features())

        return features

    def _compute_basic_pose_features(self, kps: np.ndarray) -> Dict:
        """Compute basic pose features."""
        features = {}

        # Joint angles
        features['shoulder_angle'] = self._compute_angle(kps, 5, 6, 11)  # Left shoulder-hip
        features['elbow_angle_left'] = self._compute_angle(kps, 7, 5, 9)   # Left elbow
        features['elbow_angle_right'] = self._compute_angle(kps, 8, 6, 10) # Right elbow
        features['knee_angle_left'] = self._compute_angle(kps, 13, 11, 15) # Left knee
        features['knee_angle_right'] = self._compute_angle(kps, 14, 12, 16) # Right knee

        # Limb positions relative to torso
        torso_center = (kps[5] + kps[6] + kps[11] + kps[12]) / 4  # Average of shoulders and hips
        features['left_arm_position'] = np.linalg.norm(kps[9][:2] - torso_center[:2])  # Wrist to torso
        features['right_arm_position'] = np.linalg.norm(kps[10][:2] - torso_center[:2])
        features['left_leg_position'] = np.linalg.norm(kps[15][:2] - torso_center[:2])  # Ankle to torso
        features['right_leg_position'] = np.linalg.norm(kps[16][:2] - torso_center[:2])

        return features

    def _compute_movement_features(self, current_kps: np.ndarray) -> Dict:
        """Compute movement-based features."""
        features = {}
        prev_kps = self.pose_history[-2]

        # Velocity features
        velocities = []
        for i in range(len(current_kps)):
            if current_kps[i][2] > 0.5 and prev_kps[i][2] > 0.5:  # Both keypoints confident
                vel = np.linalg.norm(current_kps[i][:2] - prev_kps[i][:2]) * self.fps
                velocities.append(vel)

        features['mean_velocity'] = np.mean(velocities) if velocities else 0
        features['max_velocity'] = np.max(velocities) if velocities else 0
        features['velocity_variance'] = np.var(velocities) if velocities else 0

        # Movement smoothness (jerk)
        if len(self.pose_history) >= 3:
            jerk = self._compute_jerk()
            features['movement_jerk'] = jerk

        return features

    def _compute_hoj_features(self) -> Dict:
        """Compute HOJO2D and HOJD2D features (McCay et al., 2020)."""
        features = {}

        # Get recent pose history
        recent_poses = list(self.pose_history)[-10:]  # Last 10 frames

        # HOJO2D: Histograms of Joint Orientations
        orientations = []
        for pose in recent_poses:
            # Compute joint orientations relative to torso
            torso_vector = pose[6][:2] - pose[5][:2]  # Right shoulder to left shoulder
            for joint_idx in [7, 8, 9, 10, 13, 14, 15, 16]:  # Limbs
                if pose[joint_idx][2] > 0.5:
                    joint_vector = pose[joint_idx][:2] - pose[5][:2]  # From left shoulder
                    angle = self._compute_vector_angle(torso_vector, joint_vector)
                    orientations.append(angle)

        if orientations:
            features['hojo2d_mean'] = np.mean(orientations)
            features['hojo2d_std'] = np.std(orientations)

        # HOJD2D: Histograms of Joint Displacements
        displacements = []
        for i in range(1, len(recent_poses)):
            for joint_idx in range(17):
                if (recent_poses[i][joint_idx][2] > 0.5 and
                    recent_poses[i-1][joint_idx][2] > 0.5):
                    disp = np.linalg.norm(recent_poses[i][joint_idx][:2] -
                                        recent_poses[i-1][joint_idx][:2])
                    displacements.append(disp)

        if displacements:
            features['hojd2d_mean'] = np.mean(displacements)
            features['hojd2d_std'] = np.std(displacements)

        return features

    def _compute_quality_scores(self) -> Dict:
        """Compute movement quality scores."""
        scores = {}

        # Writhing movement quality (Doroniewicz et al., 2020)
        # Based on smooth, continuous movements
        if len(self.feature_history) >= 10:
            recent_features = list(self.feature_history)[-10:]
            velocities = [f['mean_velocity'] for f in recent_features if 'mean_velocity' in f]

            if velocities:
                # High quality writhing: moderate, consistent velocity
                mean_vel = np.mean(velocities)
                if mean_vel > 1e-8:  # Guard against zero or near-zero mean velocity
                    vel_consistency = 1.0 / (1.0 + np.std(velocities) / mean_vel)  # Lower variance = higher consistency
                else:
                    vel_consistency = 0.0  # Safe default when mean velocity is zero
                vel_moderation = 1.0 - abs(mean_vel - 50) / 50  # Optimal velocity around 50 pixels/sec

                scores['writhing_quality'] = (vel_consistency + vel_moderation) / 2

        # Fidgety movement detection (Reich et al., 2021)
        # Small, fast movements
        if len(self.feature_history) >= 5:
            recent_features = list(self.feature_history)[-5:]
            jerks = [f.get('movement_jerk', 0) for f in recent_features]

            if jerks:
                scores['fidgety_score'] = np.mean(jerks) / 100  # Normalize

        return scores

    def _classify_movements(self, scores: Dict) -> Dict:
        """Classify movements as typical/atypical."""
        classification = {
            "overall_risk": "unknown",
            "movement_type": "unknown",
            "confidence": 0.0
        }

        if 'writhing_quality' in scores:
            if scores['writhing_quality'] > self.writhing_threshold:
                classification["movement_type"] = "normal_writhing"
                classification["overall_risk"] = "low"
                classification["confidence"] = scores['writhing_quality']
            else:
                classification["movement_type"] = "poor_writhing"
                classification["overall_risk"] = "high"
                classification["confidence"] = 1.0 - scores['writhing_quality']

        if 'fidgety_score' in scores:
            if scores['fidgety_score'] > self.fidgety_threshold:
                classification["movement_type"] = "fidgety"
                classification["overall_risk"] = "medium"

        return classification

    def _compute_angle(self, kps: np.ndarray, joint1: int, joint2: int, joint3: int) -> float:
        """Compute angle at joint2 formed by joint1-joint2-joint3."""
        if (kps[joint1][2] < 0.5 or kps[joint2][2] < 0.5 or kps[joint3][2] < 0.5):
            return 0.0

        v1 = kps[joint1][:2] - kps[joint2][:2]
        v2 = kps[joint3][:2] - kps[joint2][:2]

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0  # Return safe default for zero-length vectors
        
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))

    def _compute_vector_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute angle between two vectors."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0  # Return safe default for zero-length vectors
        
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))

    def _compute_jerk(self) -> float:
        """Compute movement jerk (rate of change of acceleration)."""
        if len(self.pose_history) < 3:
            return 0.0

        # Simple jerk computation from recent positions
        recent = list(self.pose_history)[-3:]
        positions = []

        for pose in recent:
            valid_kps = [pose[i][:2] for i in range(17) if pose[i][2] > 0.5]
            if valid_kps:
                com = np.mean(valid_kps, axis=0)
                positions.append(com)
            else:
                # No valid keypoints, append placeholder
                positions.append(np.array([np.nan, np.nan]))

        # Compute accelerations
        accels = []
        for i in range(1, len(positions)):
            vel = (positions[i] - positions[i-1]) * self.fps
            accels.append(vel)

        if len(accels) < 2:
            return 0.0

        # Compute jerk
        jerks = []
        for i in range(1, len(accels)):
            jerk = (accels[i] - accels[i-1]) * self.fps
            jerks.append(np.linalg.norm(jerk))

        return np.mean(jerks) if jerks else 0.0