# pipeline/pose/multi_scale_pose.py
"""
Multi-Scale Pose Estimation (TODO-062).
Handles varying person sizes by estimating pose at multiple scales.
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional

log = logging.getLogger("multi_scale_pose")


class MultiScalePoseEstimator:
    """
    Multi-scale pose estimator for varying person sizes.
    TODO-062: Multi-Scale Pose Estimation
    """
    
    def __init__(self, base_estimator, scales=[0.8, 1.0, 1.2]):
        """
        Initialize multi-scale pose estimator.
        
        Args:
            base_estimator: Base pose estimator (MoveNet/MediaPipe)
            scales: List of scales to try
        """
        self.base_estimator = base_estimator
        self.scales = scales
        log.info("Multi-scale pose estimator initialized with scales: %s", scales)
    
    def estimate(self, frame, bbox=None):
        """
        Estimate pose at multiple scales and merge results.
        
        Args:
            frame: Input frame
            bbox: Optional bounding box
        
        Returns:
            Merged pose keypoints
        """
        if bbox:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame
        
        poses = []
        confidences = []
        
        for scale in self.scales:
            try:
                # Scale the crop
                h, w = crop.shape[:2]
                scaled_crop = cv2.resize(crop, (int(w * scale), int(h * scale)))
                
                # Estimate pose at this scale
                pose = self.base_estimator.infer(scaled_crop)
                
                if pose:
                    # Scale keypoints back to original size
                    scaled_pose = [
                        (kp[0] / scale, kp[1] / scale, kp[2])
                        for kp in pose
                    ]
                    poses.append(scaled_pose)
                    
                    # Average confidence
                    avg_conf = np.mean([kp[2] for kp in pose if len(kp) > 2])
                    confidences.append(avg_conf)
            except Exception as e:
                log.debug("Multi-scale pose estimation failed at scale %.1f: %s", scale, e)
        
        if not poses:
            return None
        
        # Merge poses (weighted by confidence)
        return self._merge_poses(poses, confidences)
    
    def _merge_poses(self, poses: List[List], confidences: List[float]) -> List[Tuple[float, float, float]]:
        """Merge poses from multiple scales, weighted by confidence."""
        if not poses:
            return None
        
        if len(poses) == 1:
            return poses[0]
        
        # Normalize confidences
        total_conf = sum(confidences)
        if total_conf < 1e-6:
            # Use first pose if all confidences are zero
            return poses[0]
        
        weights = [c / total_conf for c in confidences]
        
        # Weighted average of keypoints
        merged = []
        num_joints = len(poses[0])
        
        for i in range(num_joints):
            x_sum = 0.0
            y_sum = 0.0
            conf_sum = 0.0
            
            for pose, weight in zip(poses, weights):
                if i < len(pose) and len(pose[i]) >= 3:
                    x_sum += pose[i][0] * weight
                    y_sum += pose[i][1] * weight
                    conf_sum += pose[i][2] * weight
            
            merged.append((x_sum, y_sum, conf_sum))
        
        return merged

