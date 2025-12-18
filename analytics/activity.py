# analytics/activity.py
# Activity Classification Module
# Classifies patient activity: sitting, standing, lying, walking

import numpy as np
import math
import logging

log = logging.getLogger("activity")

# Keypoint indices (MoveNet/COCO)
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

MIN_CONFIDENCE = 0.3


def get_keypoint(kps, idx, default=(0.0, 0.0, 0.0)):
    """Safely get keypoint with confidence check."""
    if idx < len(kps) and kps[idx][2] > MIN_CONFIDENCE:
        return kps[idx]
    return default


def compute_vertical_extent(kps):
    """Compute vertical extent of body (head to feet)."""
    try:
        nose = get_keypoint(kps, NOSE)
        lankle = get_keypoint(kps, LEFT_ANKLE)
        rankle = get_keypoint(kps, RIGHT_ANKLE)
        
        top_y = nose[1]
        bottom_y = max(lankle[1], rankle[1])
        
        return abs(bottom_y - top_y)
    except Exception:
        return 0.0


def compute_horizontal_extent(kps):
    """Compute horizontal extent of body."""
    try:
        lshoulder = get_keypoint(kps, LEFT_SHOULDER)
        rshoulder = get_keypoint(kps, RIGHT_SHOULDER)
        lhip = get_keypoint(kps, LEFT_HIP)
        rhip = get_keypoint(kps, RIGHT_HIP)
        
        left_x = min(lshoulder[0], lhip[0])
        right_x = max(rshoulder[0], rhip[0])
        
        return abs(right_x - left_x)
    except Exception:
        return 0.0


def compute_knee_angle(kps, side='left'):
    """Compute knee angle (for walking detection)."""
    try:
        if side == 'left':
            hip = get_keypoint(kps, LEFT_HIP)
            knee = get_keypoint(kps, LEFT_KNEE)
            ankle = get_keypoint(kps, LEFT_ANKLE)
        else:
            hip = get_keypoint(kps, RIGHT_HIP)
            knee = get_keypoint(kps, RIGHT_KNEE)
            ankle = get_keypoint(kps, RIGHT_ANKLE)
        
        # Vector from hip to knee
        v1_x = knee[0] - hip[0]
        v1_y = knee[1] - hip[1]
        
        # Vector from knee to ankle
        v2_x = ankle[0] - knee[0]
        v2_y = ankle[1] - knee[1]
        
        # Angle between vectors
        dot = v1_x * v2_x + v1_y * v2_y
        mag1 = math.sqrt(v1_x * v1_x + v1_y * v1_y)
        mag2 = math.sqrt(v2_x * v2_x + v2_y * v2_y)
        
        if mag1 < 1e-6 or mag2 < 1e-6:
            return 180.0
        
        cos_angle = dot / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    except Exception:
        return 180.0


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

