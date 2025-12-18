# pipeline/kinematics/skeleton.py
"""
Skeletal Structure Definition for COCO 17 Keypoints
Based on kinematic structure preservation approach (AAAI-20)
"""

import numpy as np
import logging

log = logging.getLogger("skeleton")

# COCO 17 Keypoint indices
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

# Skeletal joint connectivity (parent-child relationships)
# Format: {child_joint: parent_joint}
SKELETON_TREE = {
    # Head/face (nose is root for head)
    LEFT_EYE: NOSE,
    RIGHT_EYE: NOSE,
    LEFT_EAR: LEFT_EYE,
    RIGHT_EAR: RIGHT_EYE,
    
    # Upper body (shoulders are children of neck/center)
    # Note: In COCO, we use midpoint of shoulders as "neck"
    LEFT_SHOULDER: NOSE,  # Simplified: nose as root for upper body
    RIGHT_SHOULDER: NOSE,
    LEFT_ELBOW: LEFT_SHOULDER,
    RIGHT_ELBOW: RIGHT_SHOULDER,
    LEFT_WRIST: LEFT_ELBOW,
    RIGHT_WRIST: RIGHT_ELBOW,
    
    # Lower body (hips are root for lower body)
    LEFT_HIP: LEFT_SHOULDER,  # Simplified connection
    RIGHT_HIP: RIGHT_SHOULDER,
    LEFT_KNEE: LEFT_HIP,
    RIGHT_KNEE: RIGHT_HIP,
    LEFT_ANKLE: LEFT_KNEE,
    RIGHT_ANKLE: RIGHT_KNEE,
}

# Root joints (no parent)
ROOT_JOINTS = [NOSE, LEFT_HIP, RIGHT_HIP]

# Canonical bone length ratios (normalized to pelvis-to-neck = 1.0)
# Based on average human proportions
BONE_LENGTH_RATIOS = {
    # Head
    (NOSE, LEFT_EYE): 0.08,
    (NOSE, RIGHT_EYE): 0.08,
    (LEFT_EYE, LEFT_EAR): 0.06,
    (RIGHT_EYE, RIGHT_EAR): 0.06,
    
    # Upper body
    (NOSE, LEFT_SHOULDER): 0.25,  # Approximate neck to shoulder
    (NOSE, RIGHT_SHOULDER): 0.25,
    (LEFT_SHOULDER, LEFT_ELBOW): 0.30,
    (RIGHT_SHOULDER, RIGHT_ELBOW): 0.30,
    (LEFT_ELBOW, LEFT_WRIST): 0.25,
    (RIGHT_ELBOW, RIGHT_WRIST): 0.25,
    
    # Torso
    (LEFT_SHOULDER, LEFT_HIP): 0.50,  # Torso length
    (RIGHT_SHOULDER, RIGHT_HIP): 0.50,
    
    # Lower body
    (LEFT_HIP, LEFT_KNEE): 0.50,
    (RIGHT_HIP, RIGHT_KNEE): 0.50,
    (LEFT_KNEE, LEFT_ANKLE): 0.45,
    (RIGHT_KNEE, RIGHT_ANKLE): 0.45,
    
    # Hip connection (pelvis width)
    (LEFT_HIP, RIGHT_HIP): 0.20,
}

# Alternative: bone lengths relative to torso (shoulder-to-hip distance)
# This is more stable for different body sizes
def get_bone_length_ratio(joint_a, joint_b):
    """
    Get bone length ratio for a joint pair.
    Returns ratio relative to torso length (shoulder-to-hip).
    """
    key = (joint_a, joint_b)
    reverse_key = (joint_b, joint_a)
    
    if key in BONE_LENGTH_RATIOS:
        return BONE_LENGTH_RATIOS[key]
    elif reverse_key in BONE_LENGTH_RATIOS:
        return BONE_LENGTH_RATIOS[reverse_key]
    else:
        log.warning(f"No bone length ratio defined for joints {joint_a}-{joint_b}")
        return None


def get_parent(joint_idx):
    """Get parent joint index for a given joint."""
    return SKELETON_TREE.get(joint_idx, None)


def get_children(joint_idx):
    """Get all child joint indices for a given joint."""
    return [child for child, parent in SKELETON_TREE.items() if parent == joint_idx]


def is_root_joint(joint_idx):
    """Check if a joint is a root joint (no parent)."""
    return joint_idx in ROOT_JOINTS or joint_idx not in SKELETON_TREE


def get_joint_name(joint_idx):
    """Get human-readable name for a joint."""
    names = {
        NOSE: "nose",
        LEFT_EYE: "left_eye",
        RIGHT_EYE: "right_eye",
        LEFT_EAR: "left_ear",
        RIGHT_EAR: "right_ear",
        LEFT_SHOULDER: "left_shoulder",
        RIGHT_SHOULDER: "right_shoulder",
        LEFT_ELBOW: "left_elbow",
        RIGHT_ELBOW: "right_elbow",
        LEFT_WRIST: "left_wrist",
        RIGHT_WRIST: "right_wrist",
        LEFT_HIP: "left_hip",
        RIGHT_HIP: "right_hip",
        LEFT_KNEE: "left_knee",
        RIGHT_KNEE: "right_knee",
        LEFT_ANKLE: "left_ankle",
        RIGHT_ANKLE: "right_ankle",
    }
    return names.get(joint_idx, f"joint_{joint_idx}")


def compute_torso_length(kps_3d):
    """
    Compute torso length (shoulder-to-hip distance) for normalization.
    This is used as the reference scale for bone length ratios.
    """
    if not kps_3d or len(kps_3d) < 13:
        return None
    
    # Get shoulder midpoint
    left_shoulder = kps_3d[LEFT_SHOULDER] if len(kps_3d) > LEFT_SHOULDER else None
    right_shoulder = kps_3d[RIGHT_SHOULDER] if len(kps_3d) > RIGHT_SHOULDER else None
    
    # Get hip midpoint
    left_hip = kps_3d[LEFT_HIP] if len(kps_3d) > LEFT_HIP else None
    right_hip = kps_3d[RIGHT_HIP] if len(kps_3d) > RIGHT_HIP else None
    
    if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
        return None
    
    # Compute midpoints
    shoulder_mid = np.array([
        (left_shoulder[0] + right_shoulder[0]) / 2.0,
        (left_shoulder[1] + right_shoulder[1]) / 2.0,
        (left_shoulder[2] + right_shoulder[2]) / 2.0,
    ])
    
    hip_mid = np.array([
        (left_hip[0] + right_hip[0]) / 2.0,
        (left_hip[1] + right_hip[1]) / 2.0,
        (left_hip[2] + right_hip[2]) / 2.0,
    ])
    
    # Compute distance
    torso_length = np.linalg.norm(shoulder_mid - hip_mid)
    return float(torso_length)



