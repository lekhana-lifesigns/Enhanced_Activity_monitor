# analytics/activity.py
# Activity Classification Module
# Classifies patient activity: sitting, standing, lying, walking

import numpy as np
import logging

from analytics.keypoint_utils import (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    MIN_CONFIDENCE, get_keypoint, compute_vertical_extent,
    compute_horizontal_extent_full as compute_horizontal_extent,
    compute_knee_angle,
)

log = logging.getLogger("activity")


def classify_activity(kps, kps_history=None):
    """
    Classify patient activity.
    
    Args:
        kps: Current keypoints
        kps_history: Optional history for temporal analysis
    
    Returns:
        dict with:
        - activity: "sitting" | "standing" | "lying" | "walking" | "unknown"
        - confidence: 0-1 confidence score
        - details: Additional activity metrics
    """
    if not kps or len(kps) < 13:
        return {
            "activity": "unknown",
            "confidence": 0.0,
            "details": {}
        }
    
    try:
        # Check minimum keypoint confidence before proceeding
        key_indices = [0, 5, 6, 11, 12, 15, 16]  # Nose, shoulders, hips, ankles
        valid_keypoints = sum(1 for idx in key_indices 
                            if idx < len(kps) and kps[idx][2] > MIN_CONFIDENCE)
        
        if valid_keypoints < 4:  # Need at least 4 of 7 critical keypoints
            return {
                "activity": "unknown",
                "confidence": 0.0,
                "details": {"reason": "insufficient_keypoints", "valid_count": valid_keypoints}
            }
        
        # Compute body metrics
        vertical_extent = compute_vertical_extent(kps)
        horizontal_extent = compute_horizontal_extent(kps)
        
        # Check if extents are valid (non-zero)
        if vertical_extent < 1e-6 or horizontal_extent < 1e-6:
            return {
                "activity": "unknown",
                "confidence": 0.0,
                "details": {"reason": "invalid_extents"}
            }
        
        aspect_ratio = vertical_extent / (horizontal_extent + 1e-6)
        
        # Get key body points
        nose = get_keypoint(kps, NOSE)
        lshoulder = get_keypoint(kps, LEFT_SHOULDER)
        rshoulder = get_keypoint(kps, RIGHT_SHOULDER)
        lhip = get_keypoint(kps, LEFT_HIP)
        rhip = get_keypoint(kps, RIGHT_HIP)
        lankle = get_keypoint(kps, LEFT_ANKLE)
        rankle = get_keypoint(kps, RIGHT_ANKLE)
        
        # Compute torso angle
        shoulder_y = (lshoulder[1] + rshoulder[1]) / 2.0
        hip_y = (lhip[1] + rhip[1]) / 2.0
        torso_vertical = abs(shoulder_y - hip_y)
        
        # Compute leg angles
        left_knee_angle = compute_knee_angle(kps, 'left')
        right_knee_angle = compute_knee_angle(kps, 'right')
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0
        
        # Activity classification logic
        
        # 1. LYING: Low vertical extent, high horizontal extent
        if aspect_ratio < 0.8 and horizontal_extent > 0.3:
            return {
                "activity": "lying",
                "confidence": 0.8,
                "details": {
                    "aspect_ratio": aspect_ratio,
                    "vertical_extent": vertical_extent,
                    "horizontal_extent": horizontal_extent
                }
            }
        
        # 2. WALKING: Knee angles change, temporal motion
        if kps_history and len(kps_history) >= 3:
            # Check for leg movement
            prev_kps = kps_history[-2] if len(kps_history) >= 2 else None
            if prev_kps:
                prev_left_angle = compute_knee_angle(prev_kps, 'left')
                prev_right_angle = compute_knee_angle(prev_kps, 'right')
                
                angle_change_left = abs(left_knee_angle - prev_left_angle)
                angle_change_right = abs(right_knee_angle - prev_right_angle)
                
                if (angle_change_left > 10.0 or angle_change_right > 10.0) and avg_knee_angle < 160.0:
                    return {
                        "activity": "walking",
                        "confidence": 0.7,
                        "details": {
                            "knee_angle": avg_knee_angle,
                            "angle_change": max(angle_change_left, angle_change_right)
                        }
                    }
        
        # 3. SITTING: Moderate vertical extent, knees bent
        if aspect_ratio > 0.8 and aspect_ratio < 1.5 and avg_knee_angle < 140.0:
            # Check if hips are lower than shoulders (sitting posture)
            if hip_y > shoulder_y + 0.1:
                return {
                    "activity": "sitting",
                    "confidence": 0.75,
                    "details": {
                        "aspect_ratio": aspect_ratio,
                        "knee_angle": avg_knee_angle,
                        "torso_vertical": torso_vertical
                    }
                }
        
        # 4. STANDING: High vertical extent, straight legs
        if aspect_ratio > 1.2 and avg_knee_angle > 160.0:
            return {
                "activity": "standing",
                "confidence": 0.8,
                "details": {
                    "aspect_ratio": aspect_ratio,
                    "knee_angle": avg_knee_angle,
                    "vertical_extent": vertical_extent
                }
            }
        
        # Default: unknown (with confidence based on keypoint quality)
        activity_confidence = compute_activity_confidence(kps)
        return {
            "activity": "unknown",
            "confidence": max(0.3, activity_confidence * 0.5),  # Use computed confidence
            "details": {
                "aspect_ratio": aspect_ratio,
                "knee_angle": avg_knee_angle,
                "vertical_extent": vertical_extent,
                "keypoint_confidence": activity_confidence
            }
        }
        
    except Exception as e:
        log.exception("Error in activity classification: %s", e)
        return {
            "activity": "unknown",
            "confidence": 0.0,
            "details": {}
        }


def compute_activity_confidence(kps):
    """
    Compute confidence in activity classification.
    Based on keypoint visibility and body pose quality.
    """
    if not kps or len(kps) < 13:
        return 0.0
    
    try:
        # Check keypoint visibility
        key_indices = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, 
                      LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE]
        
        visible_count = sum(1 for idx in key_indices 
                           if idx < len(kps) and kps[idx][2] > MIN_CONFIDENCE)
        
        confidence = visible_count / len(key_indices)
        return float(confidence)
        
    except Exception:
        return 0.0

