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
    Assumes bed is horizontal.
    
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
        
        shoulder_x = (lshoulder[0] + rshoulder[0]) / 2.0
        shoulder_y = (lshoulder[1] + rshoulder[1]) / 2.0
        hip_x = (lhip[0] + rhip[0]) / 2.0
        hip_y = (lhip[1] + rhip[1]) / 2.0
        
        # Torso vector
        dx = hip_x - shoulder_x
        dy = hip_y - shoulder_y
        
        # Angle from horizontal
        angle = math.degrees(math.atan2(dy, dx))
        abs_angle = abs(angle)
        
        # Determine orientation
        if abs_angle < 20.0:
            orientation = "supine"  # Lying flat
        elif abs_angle > 70.0:
            orientation = "upright"  # Sitting/standing
        elif 20.0 <= abs_angle <= 70.0:
            orientation = "side"  # On side
        else:
            orientation = "unknown"
        
        return {
            "bed_angle": float(abs_angle),
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


def analyze_posture(kps):
    """
    Comprehensive posture analysis.
    
    Returns:
        dict with all posture metrics
    """
    if not kps:
        return {
            "spine_curvature": {},
            "bed_angle": {},
            "symmetry": {},
            "overall_score": 0.0
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
        
        return {
            "spine_curvature": spine,
            "bed_angle": bed,
            "symmetry": symmetry,
            "overall_score": float(np.clip(overall_score, 0.0, 1.0))
        }
        
    except Exception as e:
        log.exception("Error in posture analysis: %s", e)
        return {
            "spine_curvature": {},
            "bed_angle": {},
            "symmetry": {},
            "overall_score": 0.0
        }

