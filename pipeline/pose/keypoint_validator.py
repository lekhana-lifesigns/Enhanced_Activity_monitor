# pipeline/pose/keypoint_validator.py
"""
Keypoint Validation Module
Validates and filters keypoints before processing for clinical-grade robustness
"""
import numpy as np
import logging
from typing import List, Optional, Tuple

log = logging.getLogger("keypoint_validator")


class KeypointValidator:
    """
    Validates keypoints for clinical-grade processing.
    Filters invalid, NaN, or out-of-range values.
    """
    
    def __init__(self, 
                 min_keypoints: int = 5,
                 min_confidence: float = 0.3,
                 clamp_to_range: bool = True):
        """
        Args:
            min_keypoints: Minimum number of valid keypoints required
            min_confidence: Minimum confidence threshold
            clamp_to_range: Whether to clamp values to [0, 1] range
        """
        self.min_keypoints = min_keypoints
        self.min_confidence = min_confidence
        self.clamp_to_range = clamp_to_range
    
    def validate(self, kps: Optional[List], kps_3d: Optional[List] = None, 
                 use_self_contact: bool = False) -> Optional[List]:
        """
        Validate and filter keypoints.
        TODO-061: Enhanced with SC3D self-contact validation.
        
        Args:
            kps: List of keypoints [(x, y, conf), ...]
            kps_3d: Optional 3D keypoints for self-contact validation
            use_self_contact: Use SC3D self-contact constraints
        
        Returns:
            Validated keypoints or None if insufficient valid keypoints
        """
        if not kps or len(kps) == 0:
            return None
        
        valid_kps = []
        invalid_count = 0
        
        for kp in kps:
            if not kp or len(kp) < 3:
                invalid_count += 1
                continue
            
            try:
                x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
                
                # Check for NaN/Inf
                if np.isnan(x) or np.isnan(y) or np.isnan(conf):
                    invalid_count += 1
                    continue
                if np.isinf(x) or np.isinf(y) or np.isinf(conf):
                    invalid_count += 1
                    continue
                
                # Check confidence (TODO-061: SC3D-aware validation)
                self_contact_detector = None
                if use_self_contact and kps_3d:
                    try:
                        from pipeline.pose.self_contact_detector import SelfContactDetector
                        self_contact_detector = SelfContactDetector()
                        contact_signature = self_contact_detector.detect(kps_3d)
                        # In contact regions, accept lower confidence (often occluded)
                        contact_info = contact_signature.get(len(valid_kps), {})
                        if contact_info.get("in_contact", False):
                            min_conf = self.min_confidence * 0.5  # Lower threshold for contact
                        else:
                            min_conf = self.min_confidence
                    except Exception:
                        min_conf = self.min_confidence
                else:
                    min_conf = self.min_confidence
                
                if conf < min_conf:
                    invalid_count += 1
                    continue
                
                # Clamp to valid range
                if self.clamp_to_range:
                    x_clamped = np.clip(x, 0.0, 1.0)
                    y_clamped = np.clip(y, 0.0, 1.0)
                    # If clamping changed values significantly, mark as suspicious
                    if abs(x - x_clamped) > 0.1 or abs(y - y_clamped) > 0.1:
                        invalid_count += 1
                        continue
                    x, y = x_clamped, y_clamped
                    conf = np.clip(conf, 0.0, 1.0)
                
                valid_kps.append((x, y, conf))
                
            except (ValueError, TypeError, IndexError) as e:
                log.debug("Invalid keypoint format: %s", e)
                invalid_count += 1
                continue
        
        # Check minimum keypoints - require at least min_keypoints valid keypoints
        if len(valid_kps) < self.min_keypoints:
            log.debug("Insufficient valid keypoints: %d < %d (invalid: %d)", 
                     len(valid_kps), self.min_keypoints, invalid_count)
            return None
        
        # Additional check: if too many keypoints are invalid, reject even if we have minimum
        total_keypoints = len(kps)
        invalid_ratio = invalid_count / total_keypoints if total_keypoints > 0 else 1.0
        if invalid_ratio > 0.7:  # More than 70% invalid
            log.debug("Too many invalid keypoints: %.1f%% invalid", invalid_ratio * 100)
            return None
        
        return valid_kps
    
    def validate_batch(self, kps_list: List[List]) -> List[Optional[List]]:
        """
        Validate batch of keypoint lists.
        
        Args:
            kps_list: List of keypoint lists
        
        Returns:
            List of validated keypoints (None for invalid)
        """
        return [self.validate(kps) for kps in kps_list]
    
    def get_validity_stats(self, kps: Optional[List]) -> dict:
        """
        Get statistics about keypoint validity.
        
        Args:
            kps: List of keypoints
        
        Returns:
            Statistics dict
        """
        if not kps:
            return {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "valid_ratio": 0.0,
                "avg_confidence": 0.0
            }
        
        valid_count = 0
        invalid_count = 0
        confidences = []
        
        for kp in kps:
            if not kp or len(kp) < 3:
                invalid_count += 1
                continue
            
            try:
                x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
                
                if (np.isnan(x) or np.isnan(y) or np.isnan(conf) or
                    np.isinf(x) or np.isinf(y) or np.isinf(conf) or
                    conf < self.min_confidence):
                    invalid_count += 1
                else:
                    valid_count += 1
                    confidences.append(conf)
            except:
                invalid_count += 1
        
        total = len(kps)
        valid_ratio = valid_count / total if total > 0 else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            "total": total,
            "valid": valid_count,
            "invalid": invalid_count,
            "valid_ratio": valid_ratio,
            "avg_confidence": avg_confidence
        }


def create_keypoint_validator(config: dict = None) -> KeypointValidator:
    """Factory function to create keypoint validator."""
    if config is None:
        config = {}
    
    return KeypointValidator(
        min_keypoints=config.get("min_keypoints", 5),
        min_confidence=config.get("min_confidence", 0.3),
        clamp_to_range=config.get("clamp_to_range", True)
    )

