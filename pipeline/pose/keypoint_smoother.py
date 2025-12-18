# pipeline/pose/keypoint_smoother.py
"""
Keypoint Smoothing with Self-Contact Constraints (SC3D-enhanced).
Reduces jitter in keypoint positions using EMA and SC3D self-contact awareness.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from collections import deque

log = logging.getLogger("keypoint_smoother")

try:
    from pipeline.pose.self_contact_detector import SelfContactDetector
    SC3D_AVAILABLE = True
except ImportError:
    SC3D_AVAILABLE = False
    log.warning("SelfContactDetector not available, using basic smoothing")


class KeypointSmoother:
    """
    Keypoint smoother with SC3D self-contact constraints.
    Uses EMA (Exponential Moving Average) with adaptive smoothing based on contact regions.
    """
    
    def __init__(self, alpha=0.7, use_self_contact=True, window_size=5):
        """
        Initialize keypoint smoother.
        
        Args:
            alpha: EMA smoothing factor (0-1), higher = more weight to current
            use_self_contact: Use SC3D self-contact constraints
            window_size: Window size for temporal smoothing
        """
        self.alpha = alpha
        self.use_self_contact = use_self_contact and SC3D_AVAILABLE
        self.smoothed_kps = None
        self.contact_history = deque(maxlen=window_size)
        
        if self.use_self_contact:
            try:
                self.self_contact_detector = SelfContactDetector()
                log.info("Keypoint smoother initialized with SC3D self-contact support")
            except Exception as e:
                log.warning("Failed to initialize SC3D detector: %s, using basic smoothing", e)
                self.use_self_contact = False
                self.self_contact_detector = None
        else:
            self.self_contact_detector = None
            log.info("Keypoint smoother initialized (basic mode)")
    
    def smooth(self, kps: List[Tuple[float, float, float]], 
               kps_3d: Optional[List[Tuple[float, float, float, float]]] = None) -> List[Tuple[float, float, float]]:
        """
        Smooth keypoints using EMA with optional SC3D contact-aware adjustment.
        
        Args:
            kps: Current 2D keypoints (x, y, confidence)
            kps_3d: Optional 3D keypoints for contact detection
        
        Returns:
            Smoothed 2D keypoints
        """
        if not kps or len(kps) < 17:
            return kps
        
        # Initialize with first frame
        if self.smoothed_kps is None:
            self.smoothed_kps = list(kps)
            return self.smoothed_kps
        
        # Basic EMA smoothing
        smoothed = [
            (
                self.alpha * kp[0] + (1 - self.alpha) * prev[0],
                self.alpha * kp[1] + (1 - self.alpha) * prev[1],
                kp[2]  # Keep confidence
            )
            for kp, prev in zip(kps, self.smoothed_kps)
        ]
        
        # Apply self-contact constraints if available
        if self.use_self_contact and self.self_contact_detector and kps_3d:
            try:
                # Detect self-contact regions
                contact_signature = self.self_contact_detector.detect(kps_3d)
                self.contact_history.append(contact_signature)
                
                # Adjust smoothing for contact regions
                for i, (kp, prev_kp) in enumerate(zip(kps, self.smoothed_kps)):
                    contact_info = contact_signature.get(i, {})
                    if contact_info.get("in_contact", False):
                        # In contact: use higher weight for current (more responsive)
                        contact_alpha = min(0.9, self.alpha + 0.2)
                        smoothed[i] = (
                            contact_alpha * kp[0] + (1 - contact_alpha) * prev_kp[0],
                            contact_alpha * kp[1] + (1 - contact_alpha) * prev_kp[1],
                            kp[2]
                        )
            except Exception as e:
                log.debug("SC3D contact detection failed: %s", e)
        
        self.smoothed_kps = smoothed
        return self.smoothed_kps
    
    def reset(self):
        """Reset smoother state."""
        self.smoothed_kps = None
        self.contact_history.clear()
    
    def get_smoothed(self):
        """Get current smoothed keypoints."""
        return self.smoothed_kps
