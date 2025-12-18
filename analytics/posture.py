# analytics/posture.py
# Posture Analysis Module
# Analyzes patient posture: spine curvature, bed angle, symmetry

import numpy as np
import math
import logging

log = logging.getLogger("posture")

# Keypoint indices (MoveNet/COCO)
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14

MIN_CONFIDENCE = 0.3


def get_keypoint(kps, idx, default=(0.0, 0.0, 0.0)):
    """Safely get keypoint with confidence check."""
    if idx < len(kps) and kps[idx][2] > MIN_CONFIDENCE:
        return kps[idx]
    return default


def analyze_spine_curvature(kps):
    """
    Analyze spine curvature.
    
    Returns:
        dict with:
        - curvature_angle: Spine angle from vertical (degrees)
        - curvature_type: "normal" | "forward" | "backward" | "lateral"
        - severity: 0-1 severity score
    """
    if not kps or len(kps) < 13:
        return {
            "curvature_angle": 0.0,
            "curvature_type": "normal",
            "severity": 0.0
        }
    
    try:
        # Get key points
        nose = get_keypoint(kps, NOSE)
        lshoulder = get_keypoint(kps, LEFT_SHOULDER)
        rshoulder = get_keypoint(kps, RIGHT_SHOULDER)
        lhip = get_keypoint(kps, LEFT_HIP)
        rhip = get_keypoint(kps, RIGHT_HIP)
        
        # Mid-shoulder point
        shoulder_x = (lshoulder[0] + rshoulder[0]) / 2.0
        shoulder_y = (lshoulder[1] + rshoulder[1]) / 2.0
        
        # Mid-hip point
        hip_x = (lhip[0] + rhip[0]) / 2.0
        hip_y = (lhip[1] + rhip[1]) / 2.0
        
        # Spine vector
        dx = hip_x - shoulder_x
        dy = hip_y - shoulder_y
        
        # Angle from vertical
        angle = math.degrees(math.atan2(dx, dy))
        abs_angle = abs(angle)
        
        # Determine curvature type
        if abs_angle < 10.0:
            curvature_type = "normal"
        elif angle > 0:
            curvature_type = "forward"  # Leaning forward
        else:
            curvature_type = "backward"  # Leaning backward
        
        # Severity (0-1)
        severity = min(1.0, abs_angle / 45.0)
        
        return {
            "curvature_angle": float(abs_angle),
            "curvature_type": curvature_type,
            "severity": float(severity)
        }
        
    except Exception as e:
        log.exception("Error in spine curvature analysis: %s", e)
        return {
            "curvature_angle": 0.0,
            "curvature_type": "normal",
            "severity": 0.0
        }


def compute_bed_angle(kps):
    """
    Compute angle relative to bed plane (for supine patients).
    Improved thresholds to reduce false positives.
    
    Returns:
        dict with:
        - bed_angle: Angle from horizontal (degrees)
        - orientation: "supine" | "prone" | "side" | "upright"
    """
    if not kps or len(kps) < 13:
        return {
            "bed_angle": 0.0,
            "orientation": "unknown"
        }
    
    try:
        # Get torso vector
        lshoulder = get_keypoint(kps, LEFT_SHOULDER)
        rshoulder = get_keypoint(kps, RIGHT_SHOULDER)
        lhip = get_keypoint(kps, LEFT_HIP)
        rhip = get_keypoint(kps, RIGHT_HIP)
        
        # Check confidence - need reliable keypoints
        if (lshoulder[2] < MIN_CONFIDENCE or rshoulder[2] < MIN_CONFIDENCE or
            lhip[2] < MIN_CONFIDENCE or rhip[2] < MIN_CONFIDENCE):
            return {
                "bed_angle": 0.0,
                "orientation": "unknown"
            }
        
        shoulder_x = (lshoulder[0] + rshoulder[0]) / 2.0
        shoulder_y = (lshoulder[1] + rshoulder[1]) / 2.0
        hip_x = (lhip[0] + rhip[0]) / 2.0
        hip_y = (lhip[1] + rhip[1]) / 2.0
        
        # Torso vector
        dx = hip_x - shoulder_x
        dy = hip_y - shoulder_y
        
        # Angle from horizontal (0 = horizontal, 90 = vertical)
        angle_rad = math.atan2(abs(dy), abs(dx))
        angle = math.degrees(angle_rad)
        
        # Compute vertical extent to distinguish lying vs sitting
        # When lying, vertical extent is small; when sitting, it's large
        nose = get_keypoint(kps, NOSE)
        vertical_extent = abs(nose[1] - hip_y) if nose[2] > MIN_CONFIDENCE else 0.0
        
        # Improved thresholds with vertical extent check
        # Supine: torso nearly horizontal AND low vertical extent
        if angle < 25.0 and vertical_extent < 0.4:
            orientation = "supine"
        # Upright/sitting: torso nearly vertical AND high vertical extent
        elif angle > 75.0 and vertical_extent > 0.3:
            orientation = "upright"
        # Side: intermediate angle OR low vertical extent (lying on side)
        elif 25.0 <= angle <= 75.0 or vertical_extent < 0.35:
            orientation = "side"
        # Prone: similar to supine but check if face is visible (nose position relative to shoulders)
        elif angle < 30.0 and vertical_extent < 0.4:
            # Check if nose is below shoulders (prone indicator)
            if nose[2] > MIN_CONFIDENCE and nose[1] > shoulder_y + 0.1:
                orientation = "prone"
            else:
                orientation = "supine"
        else:
            # Fallback: use angle only
            if angle < 30.0:
                orientation = "supine"
            elif angle > 80.0:
                orientation = "upright"
            else:
                orientation = "side"
        
        return {
            "bed_angle": float(angle),
            "orientation": orientation
        }
        
    except Exception as e:
        log.exception("Error in bed angle computation: %s", e)
        return {
            "bed_angle": 0.0,
            "orientation": "unknown"
        }


def compute_posture_symmetry(kps):
    """
    Compute left-right posture symmetry.
    
    Returns:
        dict with:
        - symmetry_index: 0-1 (1.0 = perfect symmetry)
        - asymmetry_type: "left" | "right" | "none"
        - asymmetry_score: 0-1 severity
    """
    if not kps or len(kps) < 13:
        return {
            "symmetry_index": 0.5,
            "asymmetry_type": "none",
            "asymmetry_score": 0.0
        }
    
    try:
        # Compare left vs right side keypoints
        left_points = []
        right_points = []
        
        # Shoulders
        ls = get_keypoint(kps, LEFT_SHOULDER)
        rs = get_keypoint(kps, RIGHT_SHOULDER)
        if ls[2] > MIN_CONFIDENCE and rs[2] > MIN_CONFIDENCE:
            left_points.append((ls[0], ls[1]))
            right_points.append((rs[0], rs[1]))
        
        # Hips
        lh = get_keypoint(kps, LEFT_HIP)
        rh = get_keypoint(kps, RIGHT_HIP)
        if lh[2] > MIN_CONFIDENCE and rh[2] > MIN_CONFIDENCE:
            left_points.append((lh[0], lh[1]))
            right_points.append((rh[0], rh[1]))
        
        # Knees
        lk = get_keypoint(kps, LEFT_KNEE)
        rk = get_keypoint(kps, RIGHT_KNEE)
        if lk[2] > MIN_CONFIDENCE and rk[2] > MIN_CONFIDENCE:
            left_points.append((lk[0], lk[1]))
            right_points.append((rk[0], rk[1]))
        
        if len(left_points) < 2:
            return {
                "symmetry_index": 0.5,
                "asymmetry_type": "none",
                "asymmetry_score": 0.0
            }
        
        # Compute midline
        mid_x = (ls[0] + rs[0] + lh[0] + rh[0]) / 4.0
        
        # Compute distances from midline
        left_distances = [abs(p[0] - mid_x) for p in left_points]
        right_distances = [abs(p[0] - mid_x) for p in right_points]
        
        avg_left = np.mean(left_distances) if left_distances else 0.0
        avg_right = np.mean(right_distances) if right_distances else 0.0
        
        # Symmetry index
        if avg_left + avg_right < 1e-6:
            symmetry_index = 1.0
        else:
            diff = abs(avg_left - avg_right) / (avg_left + avg_right)
            symmetry_index = 1.0 - diff
        
        # Asymmetry type
        if avg_left > avg_right * 1.2:
            asymmetry_type = "left"
        elif avg_right > avg_left * 1.2:
            asymmetry_type = "right"
        else:
            asymmetry_type = "none"
        
        asymmetry_score = 1.0 - symmetry_index
        
        return {
            "symmetry_index": float(np.clip(symmetry_index, 0.0, 1.0)),
            "asymmetry_type": asymmetry_type,
            "asymmetry_score": float(np.clip(asymmetry_score, 0.0, 1.0))
        }
        
    except Exception as e:
        log.exception("Error in posture symmetry computation: %s", e)
        return {
            "symmetry_index": 0.5,
            "asymmetry_type": "none",
            "asymmetry_score": 0.0
        }


def analyze_posture(kps, features=None):
    """
    Comprehensive posture analysis.
    
    Args:
        kps: Keypoints
        features: Optional feature vector (for extended analysis)
    
    Returns:
        dict with all posture metrics including posture_state
    """
    if not kps:
        return {
            "spine_curvature": {},
            "bed_angle": {},
            "symmetry": {},
            "overall_score": 0.0,
            "posture_state": "unknown"
        }
    
    try:
        spine = analyze_spine_curvature(kps)
        bed = compute_bed_angle(kps)
        symmetry = compute_posture_symmetry(kps)
        
        # Overall posture score (0-1, higher = better)
        overall_score = (
            0.4 * (1.0 - spine["severity"]) +
            0.3 * symmetry["symmetry_index"] +
            0.3 * (1.0 if bed["orientation"] != "unknown" else 0.5)
        )
        
        # Classify discrete posture state
        posture_state = classify_posture_state(kps)
        
        return {
            "spine_curvature": spine,
            "bed_angle": bed,
            "symmetry": symmetry,
            "overall_score": float(np.clip(overall_score, 0.0, 1.0)),
            "posture_state": posture_state
        }
        
    except Exception as e:
        log.exception("Error in posture analysis: %s", e)
        return {
            "spine_curvature": {},
            "bed_angle": {},
            "symmetry": {},
            "overall_score": 0.0,
            "posture_state": "unknown"
        }


def classify_posture_state(kps, use_strict_thresholds=True):
    """
    Classify discrete posture state from keypoints.
    Improved logic with stricter thresholds to reduce false positives.
    
    Args:
        kps: Keypoints
        use_strict_thresholds: Use stricter thresholds for better accuracy
    
    Returns:
        str: One of "supine", "prone", "left_lateral", "right_lateral", "side", "sitting", "unknown"
    """
    if not kps or len(kps) < 13:
        return "unknown"
    
    try:
        # Check minimum keypoint confidence before proceeding
        key_indices = [5, 6, 11, 12]  # Shoulders and hips (critical for posture)
        valid_keypoints = sum(1 for idx in key_indices 
                            if idx < len(kps) and kps[idx][2] > MIN_CONFIDENCE)
        
        if valid_keypoints < 3:  # Need at least 3 of 4 critical keypoints
            return "unknown"
        
        bed_analysis = compute_bed_angle(kps)
        symmetry_analysis = compute_posture_symmetry(kps)
        
        orientation = bed_analysis.get("orientation", "unknown")
        asymmetry_type = symmetry_analysis.get("asymmetry_type", "none")
        symmetry_index = symmetry_analysis.get("symmetry_index", 0.5)
        bed_angle = bed_analysis.get("bed_angle", 0.0)
        
        # If orientation is unknown, return unknown
        if orientation == "unknown":
            return "unknown"
        
        # Map orientation + asymmetry to discrete states with improved logic
        # Use stricter thresholds for better accuracy
        if use_strict_thresholds:
            if orientation == "supine":
                # Verify it's truly supine (low angle, good symmetry)
                if bed_angle < 25.0 and symmetry_index > 0.7:
                    return "supine"
                else:
                    # Might be transitioning - return side
                    return "side"
            elif orientation == "prone":
                # Prone: similar to supine but face down
                if bed_angle < 25.0:
                    return "prone"
                else:
                    return "side"
            elif orientation == "upright":
                # Only classify as sitting if we're very confident (high angle)
                if bed_angle > 75.0:
                    return "sitting"
                else:
                    # Likely false positive - return side
                    return "side"
            elif orientation == "side":
                # Use asymmetry to determine left vs right lateral
                # Require significant asymmetry (not just noise)
                if asymmetry_type == "left" and symmetry_index < 0.85:
                    return "left_lateral"
                elif asymmetry_type == "right" and symmetry_index < 0.85:
                    return "right_lateral"
                else:
                    # Generic side position (symmetric or unclear)
                    return "side"
            else:
                return "unknown"
        else:
            # Less strict thresholds (for backward compatibility)
            if orientation == "supine":
                return "supine"
            elif orientation == "upright":
                return "sitting"
            elif orientation == "side":
                if asymmetry_type == "left":
                    return "left_lateral"
                elif asymmetry_type == "right":
                    return "right_lateral"
                else:
                    return "side"
            else:
                return "unknown"
            
    except Exception as e:
        log.exception("Error in posture state classification: %s", e)
        return "unknown"

