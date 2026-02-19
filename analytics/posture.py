# analytics/posture.py
# Posture Analysis Module
# Analyzes patient posture: spine curvature, bed angle, symmetry

import numpy as np
import math
import logging

from analytics.keypoint_utils import (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    MIN_CONFIDENCE, get_keypoint,
)

log = logging.getLogger("posture")


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


def compute_hip_knee_angle(kps):
    """
    Compute hip-knee angle to help distinguish sitting postures.
    Key for detecting relaxed sitting (couch) vs upright sitting.

    Returns:
        dict with left_angle, right_angle, avg_angle (degrees)
    """
    try:
        lhip = get_keypoint(kps, LEFT_HIP)
        rhip = get_keypoint(kps, RIGHT_HIP)
        lknee = get_keypoint(kps, LEFT_KNEE)
        rknee = get_keypoint(kps, RIGHT_KNEE)

        angles = {}

        # Left hip-knee angle (thigh angle from vertical)
        if lhip[2] > MIN_CONFIDENCE and lknee[2] > MIN_CONFIDENCE:
            dx = lknee[0] - lhip[0]
            dy = lknee[1] - lhip[1]
            angles["left_angle"] = abs(math.degrees(math.atan2(dx, dy)))
        else:
            angles["left_angle"] = 0.0

        # Right hip-knee angle
        if rhip[2] > MIN_CONFIDENCE and rknee[2] > MIN_CONFIDENCE:
            dx = rknee[0] - rhip[0]
            dy = rknee[1] - rhip[1]
            angles["right_angle"] = abs(math.degrees(math.atan2(dx, dy)))
        else:
            angles["right_angle"] = 0.0

        angles["avg_angle"] = (angles["left_angle"] + angles["right_angle"]) / 2.0
        return angles
    except Exception:
        return {"left_angle": 0.0, "right_angle": 0.0, "avg_angle": 0.0}


def compute_bed_angle(kps):
    """
    Compute angle relative to bed plane (for supine patients).
    Improved thresholds to reduce false positives.
    Enhanced to better detect sitting on couch/chair vs lying.

    Returns:
        dict with:
        - bed_angle: Angle from horizontal (degrees)
        - orientation: "supine" | "prone" | "side" | "upright" | "sitting_relaxed"
        - hip_knee_angle: Thigh angle (helps distinguish sitting types)
    """
    if not kps or len(kps) < 13:
        return {
            "bed_angle": 0.0,
            "orientation": "unknown",
            "hip_knee_angle": 0.0
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
                "orientation": "unknown",
                "hip_knee_angle": 0.0
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

        # Compute hip-knee angle for sitting detection
        # When sitting on couch: thighs are more horizontal (angle < 45)
        # When standing: thighs are vertical (angle > 70)
        hip_knee = compute_hip_knee_angle(kps)
        hip_knee_angle = hip_knee["avg_angle"]

        # Enhanced sitting detection using multiple cues:
        # 1. Torso angle (upright = higher angle)
        # 2. Vertical extent (sitting = moderate)
        # 3. Hip-knee angle (sitting = bent legs, < 60 degrees)

        # Sitting indicators (including relaxed couch sitting)
        is_sitting_candidate = (
            # Torso is somewhat upright (> 40 degrees from horizontal)
            angle > 40.0 and
            # Legs are bent (hip-knee angle indicates sitting)
            hip_knee_angle < 70.0 and hip_knee_angle > 10.0
        )

        # Relaxed sitting (couch): torso leaning back, legs extended forward
        is_relaxed_sitting = (
            45.0 < angle < 75.0 and  # Reclined torso
            hip_knee_angle < 50.0 and hip_knee_angle > 5.0 and  # Legs somewhat extended
            vertical_extent > 0.15  # Some vertical component
        )

        # Upright sitting: torso vertical, legs bent at 90 degrees
        is_upright_sitting = (
            angle > 70.0 and
            vertical_extent > 0.25 and
            hip_knee_angle > 30.0 and hip_knee_angle < 70.0
        )

        # Improved thresholds with sitting detection
        if is_relaxed_sitting:
            orientation = "sitting_relaxed"
        elif is_upright_sitting or (angle > 75.0 and vertical_extent > 0.3):
            orientation = "upright"
        elif is_sitting_candidate:
            orientation = "upright"  # Generic sitting
        # Supine: torso nearly horizontal AND low vertical extent
        elif angle < 25.0 and vertical_extent < 0.4:
            orientation = "supine"
        # Side: intermediate angle OR low vertical extent (lying on side)
        elif 25.0 <= angle <= 50.0 and vertical_extent < 0.35:
            orientation = "side"
        # Prone: similar to supine but check if face is visible
        elif angle < 30.0 and vertical_extent < 0.4:
            if nose[2] > MIN_CONFIDENCE and nose[1] > shoulder_y + 0.1:
                orientation = "prone"
            else:
                orientation = "supine"
        else:
            # Fallback: use angle and hip-knee together
            if angle < 30.0:
                orientation = "supine"
            elif angle > 50.0 or hip_knee_angle < 60.0:
                orientation = "upright"
            else:
                orientation = "side"

        return {
            "bed_angle": float(angle),
            "orientation": orientation,
            "hip_knee_angle": float(hip_knee_angle)
        }

    except Exception as e:
        log.exception("Error in bed angle computation: %s", e)
        return {
            "bed_angle": 0.0,
            "orientation": "unknown",
            "hip_knee_angle": 0.0
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
    Improved logic with better sitting detection (including couch/chair sitting).

    Args:
        kps: Keypoints
        use_strict_thresholds: Use stricter thresholds for better accuracy

    Returns:
        str: One of "supine", "prone", "left_lateral", "right_lateral", "side",
             "sitting", "sitting_relaxed", "standing", "unknown"
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
        hip_knee_angle = bed_analysis.get("hip_knee_angle", 0.0)

        # If orientation is unknown, return unknown
        if orientation == "unknown":
            return "unknown"

        # Map orientation + asymmetry to discrete states with improved logic
        if use_strict_thresholds:
            # Handle sitting states (including relaxed couch sitting)
            if orientation == "sitting_relaxed":
                # Relaxed sitting on couch/recliner
                return "sitting"  # Or "sitting_relaxed" if you want to distinguish

            elif orientation == "upright":
                # Upright sitting or standing
                # Distinguish sitting from standing using hip-knee angle
                if hip_knee_angle > 5.0 and hip_knee_angle < 70.0:
                    # Legs are bent - sitting
                    return "sitting"
                elif hip_knee_angle >= 70.0 or bed_angle > 80.0:
                    # Legs are straight and vertical - standing
                    return "standing"
                else:
                    # Default to sitting if angle is moderate
                    return "sitting"

            elif orientation == "supine":
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
            if orientation in ("supine",):
                return "supine"
            elif orientation in ("upright", "sitting_relaxed"):
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

