# analytics/frame_visibility.py
"""
Frame visibility analysis - detect if person is fully visible or partially visible.
"""
import numpy as np
import logging
import math

log = logging.getLogger("frame_visibility")

# COCO keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
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


def analyze_frame_visibility(kps_2d, bbox=None, frame_shape=None):
    """
    Analyze if person is fully visible in frame or only partially visible.
    
    Args:
        kps_2d: 2D keypoints (normalized coordinates)
        bbox: Optional bounding box [x, y, w, h]
        frame_shape: Optional (height, width) of frame
    
    Returns:
        dict with:
        - visibility_type: "full_body" | "upper_body" | "head_only" | "partial"
        - visible_parts: List of visible body parts
        - completeness_score: 0-1 (1.0 = fully visible)
        - distance_estimate: "close" | "medium" | "far" | "unknown"
    """
    if not kps_2d or len(kps_2d) < 5:
        return {
            "visibility_type": "unknown",
            "visible_parts": [],
            "completeness_score": 0.0,
            "distance_estimate": "unknown"
        }
    
    try:
        # Check which body parts are visible
        head_visible = any([
            get_keypoint(kps_2d, NOSE)[2] > MIN_CONFIDENCE,
            get_keypoint(kps_2d, LEFT_EYE)[2] > MIN_CONFIDENCE,
            get_keypoint(kps_2d, RIGHT_EYE)[2] > MIN_CONFIDENCE
        ])
        
        upper_body_visible = any([
            get_keypoint(kps_2d, LEFT_SHOULDER)[2] > MIN_CONFIDENCE,
            get_keypoint(kps_2d, RIGHT_SHOULDER)[2] > MIN_CONFIDENCE,
            get_keypoint(kps_2d, LEFT_ELBOW)[2] > MIN_CONFIDENCE,
            get_keypoint(kps_2d, RIGHT_ELBOW)[2] > MIN_CONFIDENCE
        ])
        
        torso_visible = any([
            get_keypoint(kps_2d, LEFT_HIP)[2] > MIN_CONFIDENCE,
            get_keypoint(kps_2d, RIGHT_HIP)[2] > MIN_CONFIDENCE
        ])
        
        lower_body_visible = any([
            get_keypoint(kps_2d, LEFT_KNEE)[2] > MIN_CONFIDENCE,
            get_keypoint(kps_2d, RIGHT_KNEE)[2] > MIN_CONFIDENCE,
            get_keypoint(kps_2d, LEFT_ANKLE)[2] > MIN_CONFIDENCE,
            get_keypoint(kps_2d, RIGHT_ANKLE)[2] > MIN_CONFIDENCE
        ])
        
        # Determine visibility type
        visible_parts = []
        if head_visible:
            visible_parts.append("head")
        if upper_body_visible:
            visible_parts.append("upper_body")
        if torso_visible:
            visible_parts.append("torso")
        if lower_body_visible:
            visible_parts.append("lower_body")
        
        # Classify visibility type
        if lower_body_visible and torso_visible and upper_body_visible and head_visible:
            visibility_type = "full_body"
            completeness_score = 1.0
        elif torso_visible and upper_body_visible and head_visible:
            visibility_type = "upper_body"
            completeness_score = 0.7
        elif upper_body_visible and head_visible:
            visibility_type = "upper_body"
            completeness_score = 0.5
        elif head_visible:
            visibility_type = "head_only"
            completeness_score = 0.3
        else:
            visibility_type = "partial"
            completeness_score = 0.2
        
        # Estimate distance based on keypoint spread and bbox size
        distance_estimate = estimate_distance(kps_2d, bbox)
        
        return {
            "visibility_type": visibility_type,
            "visible_parts": visible_parts,
            "completeness_score": completeness_score,
            "distance_estimate": distance_estimate,
            "head_visible": head_visible,
            "upper_body_visible": upper_body_visible,
            "torso_visible": torso_visible,
            "lower_body_visible": lower_body_visible
        }
        
    except Exception as e:
        log.exception("Error in frame visibility analysis: %s", e)
        return {
            "visibility_type": "unknown",
            "visible_parts": [],
            "completeness_score": 0.0,
            "distance_estimate": "unknown"
        }


def estimate_distance(kps_2d, bbox=None):
    """
    Estimate distance category based on keypoint spread and bbox size.
    
    Returns:
        "close" | "medium" | "far" | "unknown"
    """
    if not kps_2d:
        return "unknown"
    
    try:
        # Get visible keypoints
        visible_kps = [kp for kp in kps_2d if kp[2] > MIN_CONFIDENCE]
        if len(visible_kps) < 3:
            return "unknown"
        
        # Calculate spread of keypoints
        x_coords = [kp[0] for kp in visible_kps]
        y_coords = [kp[1] for kp in visible_kps]
        
        x_spread = max(x_coords) - min(x_coords)
        y_spread = max(y_coords) - min(y_coords)
        total_spread = math.sqrt(x_spread**2 + y_spread**2)
        
        # Use bbox size if available
        if bbox and len(bbox) >= 4:
            bbox_area = bbox[2] * bbox[3]  # width * height
            # Normalize by frame size (assuming normalized bbox)
            if bbox_area > 0.3:  # Large bbox = close
                return "close"
            elif bbox_area > 0.1:  # Medium bbox = medium distance
                return "medium"
            else:  # Small bbox = far
                return "far"
        
        # Use keypoint spread as fallback
        if total_spread > 0.5:
            return "close"
        elif total_spread > 0.2:
            return "medium"
        else:
            return "far"
            
    except Exception as e:
        log.exception("Error estimating distance: %s", e)
        return "unknown"

