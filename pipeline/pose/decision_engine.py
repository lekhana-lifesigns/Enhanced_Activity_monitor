import math
import time
import logging
import numpy as np

log = logging.getLogger("decision")

FALL_ANGLE_THRESHOLD = 45.0
HAND_PROXIMITY_RISK_THRESHOLD = 0.5


def compute_torso_angle(kps):
    try:
        if not kps or len(kps) < 13:
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

# (Other helper functions retained but simplified for brevity)

def compute_agitation_score(features, probs, label):
    try:
        if features is None or len(features) < 7:
            return 0.0
        motion_energy = float(features[0])
        jerk_index = float(features[1])
        motor_entropy = float(features[6])
        motion_score = min(1.0, (motion_energy * 2.0 + jerk_index * 2.0) / 2.0)
        ml_score = float(probs[0]) if probs else 0.0
        agitation_score = 0.4 * motion_score + 0.3 * motor_entropy + 0.3 * ml_score
        return float(np.clip(agitation_score, 0.0, 1.0))
    except Exception:
        return 0.0


def compute_delirium_risk(features, probs, label):
    try:
        if features is None or len(features) < 9:
            return 0.0
        motor_entropy = float(features[6])
        motion_variability = float(features[8])
        ml_score = float(probs[0]) if label == "delirium" and probs else 0.0
        delirium_risk = 0.4 * motor_entropy + 0.3 * min(1.0, motion_variability) + 0.3 * ml_score
        return float(np.clip(delirium_risk, 0.0, 1.0))
    except Exception:
        return 0.0


def compute_respiratory_distress(features):
    try:
        if features is None or len(features) < 5:
            return 0.0
        breath_rate_proxy = float(features[4]) * 40.0
        if breath_rate_proxy < 8.0:
            rate_score = 1.0 - (breath_rate_proxy / 8.0)
        elif breath_rate_proxy > 30.0:
            rate_score = min(1.0, (breath_rate_proxy - 30.0) / 10.0)
        else:
            rate_score = 0.0
        variability_score = min(1.0, float(features[8]) * 0.5) if len(features) > 8 else 0.0
        respiratory_distress = 0.7 * rate_score + 0.3 * variability_score
        return float(np.clip(respiratory_distress, 0.0, 1.0))
    except Exception:
        return 0.0


def compute_motor_asymmetry(kps, features):
    try:
        return (0.5, 0.5) if not kps or len(kps) < 13 else (0.5, 0.5)
    except Exception:
        return (0.5, 0.5)


def compute_clinical_alert(agitation_score, delirium_risk, respiratory_distress, hand_proximity_risk, features):
    try:
        if hand_proximity_risk > HAND_PROXIMITY_RISK_THRESHOLD:
            return "HIGH_RISK"
        if respiratory_distress > 0.7:
            return "HIGH_RISK"
        if agitation_score > 0.8 or delirium_risk > 0.8:
            return "HIGH_RISK"
        if agitation_score > 0.5 or delirium_risk > 0.5:
            return "MEDIUM_RISK"
        if respiratory_distress > 0.4:
            return "MEDIUM_RISK"
        return "LOW_RISK"
    except Exception:
        return "LOW_RISK"


def apply_rules(label, probs, kps, features=None, posture_state=None, patient_cfg=None, person_present=True):
    out = {"label": label, "confidence": float(probs[0]) if probs else 1.0, "ts": time.time()}
    try:
        out["torso_angle"] = compute_torso_angle(kps)
        if features is not None and len(features) >= 9:
            agitation_score = compute_agitation_score(features, probs, label)
            delirium_risk = compute_delirium_risk(features, probs, label)
            respiratory_distress = compute_respiratory_distress(features)
            lhs_motor, rhs_motor = compute_motor_asymmetry(kps, features)
            hand_proximity_risk = float(features[5]) if len(features) > 5 else 0.0
            out.update({"agitation_score": agitation_score, "delirium_risk": delirium_risk, "respiratory_distress": respiratory_distress, "lhs_motor": lhs_motor, "rhs_motor": rhs_motor, "hand_proximity_risk": hand_proximity_risk})
            out["alert"] = compute_clinical_alert(agitation_score, delirium_risk, respiratory_distress, hand_proximity_risk, features)
            clinical_confidence = 0.3 * agitation_score + 0.3 * delirium_risk + 0.2 * respiratory_distress + 0.2 * out["confidence"]
            out["clinical_confidence"] = float(np.clip(clinical_confidence, 0.0, 1.0))
        else:
            out.update({"agitation_score": 0.0, "delirium_risk": 0.0, "respiratory_distress": 0.0, "lhs_motor": 0.5, "rhs_motor": 0.5, "hand_proximity_risk": 0.0, "breath_rate_proxy": 0.0, "motion_entropy": 0.0, "alert": "LOW_RISK", "clinical_confidence": out["confidence"]})

        if out.get("alert") == "HIGH_RISK":
            out["urgency"] = "critical"
        elif out.get("alert") == "MEDIUM_RISK":
            out["urgency"] = "medium"
        else:
            out["urgency"] = "low"

        if out.get("torso_angle", 0.0) > FALL_ANGLE_THRESHOLD and label == "fall":
            out["urgency"] = "critical"
            out["alert"] = "HIGH_RISK"

        # patient policy checks (if provided)
        if patient_cfg and not person_present and patient_cfg.get("must_be_in_bed", False):
            out["policy_violation"] = True
            out["violation_type"] = "out_of_bed"
            out["alert"] = "HIGH_RISK"
            out["urgency"] = "critical"

    except Exception as e:
        log.exception("Error in decision engine: %s", e)
        out.update({"urgency": "low", "alert": "LOW_RISK", "agitation_score": 0.0, "delirium_risk": 0.0, "respiratory_distress": 0.0})
    return out