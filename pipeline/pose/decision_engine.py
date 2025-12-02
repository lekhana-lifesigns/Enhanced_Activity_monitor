# pipeline/pose/decision_engine.py
# Enhanced Clinical Decision Engine
# Multi-dimensional clinical scoring for ICU-grade patient monitoring

import math
import time
import logging
import numpy as np

log = logging.getLogger("decision")

# Clinical thresholds (tunable per deployment)
FALL_ANGLE_THRESHOLD = 45.0  # degrees
AGITATION_MOTION_ENERGY_THRESHOLD = 0.3
AGITATION_JERK_THRESHOLD = 0.2
DELIRIUM_ENTROPY_THRESHOLD = 0.6
RESPIRATORY_DISTRESS_BREATH_RATE_MIN = 8.0  # breaths/min
RESPIRATORY_DISTRESS_BREATH_RATE_MAX = 30.0  # breaths/min
HAND_PROXIMITY_RISK_THRESHOLD = 0.5


def compute_torso_angle(kps):
    """Compute torso angle from vertical (for fall detection)."""
    try:
        if len(kps) < 13:
            return 0.0
        nose = kps[0]
        lhip = kps[11]
        rhip = kps[12]
        hip_x = (lhip[0] + rhip[0]) / 2.0
        hip_y = (lhip[1] + rhip[1]) / 2.0
        dx = nose[0] - hip_x
        dy = nose[1] - hip_y
        angle = abs(math.degrees(math.atan2(dx, dy)))
        return angle
    except Exception:
        return 0.0


def compute_agitation_score(features, probs, label):
    """
    Compute agitation severity score (0-1).
    Higher score = more severe agitation.
    
    Uses:
    - Motion energy
    - Jerk index
    - Motor entropy
    - ML model prediction
    """
    if features is None or len(features) < 7:
        return 0.0
    
    try:
        motion_energy = features[0]
        jerk_index = features[1]
        motor_entropy = features[6]
        
        # Base score from motion characteristics
        motion_score = min(1.0, (motion_energy * 2.0 + jerk_index * 2.0) / 2.0)
        entropy_score = motor_entropy
        
        # Combine with ML prediction
        ml_score = 0.0
        if label in ["agitation", "restlessness", "delirium"]:
            ml_score = probs[0] if probs else 0.5
        
        # Weighted combination
        agitation_score = (
            0.4 * motion_score +
            0.3 * entropy_score +
            0.3 * ml_score
        )
        
        return float(np.clip(agitation_score, 0.0, 1.0))
    except Exception:
        return 0.0


def compute_delirium_risk(features, probs, label):
    """
    Compute delirium risk score (0-1).
    Higher score = higher risk of delirium.
    
    Uses:
    - Motion entropy (chaotic movement)
    - Inter-joint incoherence
    - Motion variability
    - ML model prediction
    """
    if features is None or len(features) < 9:
        return 0.0
    
    try:
        motor_entropy = features[6]
        motion_variability = features[8]
        
        # Delirium indicators: high entropy + high variability
        entropy_score = motor_entropy
        variability_score = min(1.0, motion_variability)
        
        # ML prediction
        ml_score = 0.0
        if label == "delirium":
            ml_score = probs[0] if probs else 0.0
        
        # Weighted combination
        delirium_risk = (
            0.4 * entropy_score +
            0.3 * variability_score +
            0.3 * ml_score
        )
        
        return float(np.clip(delirium_risk, 0.0, 1.0))
    except Exception:
        return 0.0


def compute_respiratory_distress(features):
    """
    Compute respiratory distress score (0-1).
    Higher score = more respiratory distress.
    
    Uses:
    - Breath rate proxy (too low or too high)
    - Breathing variability
    - Thorax expansion
    """
    if features is None or len(features) < 5:
        return 0.0
    
    try:
        breath_rate_proxy = features[4] * 40.0  # Denormalize (was normalized to 0-1)
        
        # Check if breath rate is abnormal
        if breath_rate_proxy < RESPIRATORY_DISTRESS_BREATH_RATE_MIN:
            # Too slow (bradypnea)
            rate_score = 1.0 - (breath_rate_proxy / RESPIRATORY_DISTRESS_BREATH_RATE_MIN)
        elif breath_rate_proxy > RESPIRATORY_DISTRESS_BREATH_RATE_MAX:
            # Too fast (tachypnea)
            rate_score = min(1.0, (breath_rate_proxy - RESPIRATORY_DISTRESS_BREATH_RATE_MAX) / 10.0)
        else:
            rate_score = 0.0
        
        # Breathing variability (from feature history - approximated)
        # Higher variability = more distress
        variability_score = min(1.0, features[8] * 0.5)  # Use motion variability as proxy
        
        # Combined score
        respiratory_distress = (
            0.7 * rate_score +
            0.3 * variability_score
        )
        
        return float(np.clip(respiratory_distress, 0.0, 1.0))
    except Exception:
        return 0.0


def compute_motor_asymmetry(kps, features):
    """
    Compute left vs right hemibody motor scores.
    
    Returns:
        (lhs_motor, rhs_motor) - both 0-1 scores
    """
    if kps is None or len(kps) < 13:
        return (0.5, 0.5)
    
    try:
        # Get left and right side keypoints
        left_points = []
        right_points = []
        
        # Shoulders
        if len(kps) > 6:
            ls = kps[5]  # LEFT_SHOULDER
            rs = kps[6]  # RIGHT_SHOULDER
            if ls[2] > 0.3 and rs[2] > 0.3:
                left_points.append((ls[0], ls[1]))
                right_points.append((rs[0], rs[1]))
        
        # Elbows
        if len(kps) > 8:
            le = kps[7]  # LEFT_ELBOW
            re = kps[8]  # RIGHT_ELBOW
            if le[2] > 0.3 and re[2] > 0.3:
                left_points.append((le[0], le[1]))
                right_points.append((re[0], re[1]))
        
        # Wrists
        if len(kps) > 10:
            lw = kps[9]  # LEFT_WRIST
            rw = kps[10]  # RIGHT_WRIST
            if lw[2] > 0.3 and rw[2] > 0.3:
                left_points.append((lw[0], lw[1]))
                right_points.append((rw[0], rw[1]))
        
        # Compute motor activity as distance from body center
        if len(left_points) == 0 or len(right_points) == 0:
            return (0.5, 0.5)
        
        # Body center (midpoint of shoulders/hips)
        if len(kps) > 12:
            ls = kps[5]
            rs = kps[6]
            lh = kps[11]
            rh = kps[12]
            center_x = (ls[0] + rs[0] + lh[0] + rh[0]) / 4.0
            center_y = (ls[1] + rs[1] + lh[1] + rh[1]) / 4.0
        else:
            center_x, center_y = 0.5, 0.5
        
        # Compute average distance from center for each side
        left_distances = [math.hypot(p[0] - center_x, p[1] - center_y) for p in left_points]
        right_distances = [math.hypot(p[0] - center_x, p[1] - center_y) for p in right_points]
        
        avg_left = np.mean(left_distances) if left_distances else 0.0
        avg_right = np.mean(right_distances) if right_distances else 0.0
        
        # Normalize to 0-1 (assuming max distance ~0.5 in normalized coordinates)
        lhs_motor = min(1.0, avg_left * 2.0)
        rhs_motor = min(1.0, avg_right * 2.0)
        
        return (float(lhs_motor), float(rhs_motor))
        
    except Exception:
        return (0.5, 0.5)


def compute_clinical_alert(agitation_score, delirium_risk, respiratory_distress, 
                          hand_proximity_risk, features):
    """
    Compute clinical alert level.
    
    Returns:
        "HIGH_RISK" | "MEDIUM_RISK" | "LOW_RISK"
    """
    try:
        # High risk conditions
        if hand_proximity_risk > HAND_PROXIMITY_RISK_THRESHOLD:
            return "HIGH_RISK"
        
        if respiratory_distress > 0.7:
            return "HIGH_RISK"
        
        if agitation_score > 0.8 or delirium_risk > 0.8:
            return "HIGH_RISK"
        
        # Medium risk conditions
        if agitation_score > 0.5 or delirium_risk > 0.5:
            return "MEDIUM_RISK"
        
        if respiratory_distress > 0.4:
            return "MEDIUM_RISK"
        
        # Low risk (default)
        return "LOW_RISK"
        
    except Exception:
        return "LOW_RISK"


def apply_rules(label, probs, kps, features=None):
    """
    Enhanced clinical decision engine.
    Combines ML predictions with clinical feature analysis.
    
    Args:
        label: ML model prediction label
        probs: ML model probability distribution
        kps: Current keypoints
        features: ICU feature vector (9-dim) from feature encoder
    
    Returns:
        dict with clinical scores and alert level
    """
    out = {
        "label": label,
        "confidence": float(probs[0]) if probs and len(probs) > 0 else 1.0,
        "ts": time.time()
    }
    
    try:
        # Compute torso angle (for fall detection)
        angle = compute_torso_angle(kps)
        out["torso_angle"] = angle
        
        # Clinical scoring (if features available)
        if features is not None and len(features) >= 9:
            # Agitation severity
            agitation_score = compute_agitation_score(features, probs, label)
            out["agitation_score"] = agitation_score
            
            # Delirium risk
            delirium_risk = compute_delirium_risk(features, probs, label)
            out["delirium_risk"] = delirium_risk
            
            # Respiratory distress
            respiratory_distress = compute_respiratory_distress(features)
            out["respiratory_distress"] = respiratory_distress
            
            # Motor asymmetry
            lhs_motor, rhs_motor = compute_motor_asymmetry(kps, features)
            out["lhs_motor"] = lhs_motor
            out["rhs_motor"] = rhs_motor
            
            # Hand proximity risk
            hand_proximity_risk = features[5] if len(features) > 5 else 0.0
            out["hand_proximity_risk"] = float(hand_proximity_risk)
            
            # Breath rate proxy
            breath_rate_proxy = features[4] * 40.0 if len(features) > 4 else 0.0
            out["breath_rate_proxy"] = float(breath_rate_proxy)
            
            # Motion entropy
            motion_entropy = features[6] if len(features) > 6 else 0.0
            out["motion_entropy"] = float(motion_entropy)
            
            # Clinical alert level
            alert = compute_clinical_alert(
                agitation_score, delirium_risk, respiratory_distress,
                hand_proximity_risk, features
            )
            out["alert"] = alert
            
            # Overall confidence (weighted by clinical scores)
            clinical_confidence = (
                0.3 * agitation_score +
                0.3 * delirium_risk +
                0.2 * respiratory_distress +
                0.2 * out["confidence"]
            )
            out["clinical_confidence"] = float(np.clip(clinical_confidence, 0.0, 1.0))
        else:
            # Fallback to basic scoring if features not available
            out["agitation_score"] = 0.0
            out["delirium_risk"] = 0.0
            out["respiratory_distress"] = 0.0
            out["lhs_motor"] = 0.5
            out["rhs_motor"] = 0.5
            out["hand_proximity_risk"] = 0.0
            out["breath_rate_proxy"] = 0.0
            out["motion_entropy"] = 0.0
            out["alert"] = "LOW_RISK"
            out["clinical_confidence"] = out["confidence"]
        
        # Legacy urgency mapping (for backward compatibility)
        if out.get("alert") == "HIGH_RISK":
            out["urgency"] = "critical"
        elif out.get("alert") == "MEDIUM_RISK":
            out["urgency"] = "medium"
        else:
            out["urgency"] = "low"
        
        # Fall detection (legacy)
        if angle > FALL_ANGLE_THRESHOLD and label == "fall":
            out["urgency"] = "critical"
            out["alert"] = "HIGH_RISK"
        elif label == "fall" and out["confidence"] > 0.8:
            out["urgency"] = "critical"
            out["alert"] = "HIGH_RISK"
        
    except Exception as e:
        log.exception("Error in decision engine: %s", e)
        out["urgency"] = "low"
        out["alert"] = "LOW_RISK"
        out["agitation_score"] = 0.0
        out["delirium_risk"] = 0.0
        out["respiratory_distress"] = 0.0
    
    return out
