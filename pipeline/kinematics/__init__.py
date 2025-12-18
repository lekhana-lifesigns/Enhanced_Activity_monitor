# pipeline/kinematics/__init__.py
"""
Kinematic Structure Preservation Module
Based on AAAI-20: "Kinematic-Structure-Preserved Representation for Unsupervised 3D Human Pose Estimation"

This module provides:
- Forward kinematics transformation (Tfk)
- Camera projection transformation (Tc)
- Bone-length constraints and validation
- Skeletal structure definitions
"""

from .skeleton import (
    SKELETON_TREE, ROOT_JOINTS, BONE_LENGTH_RATIOS,
    get_parent, get_children, get_bone_length_ratio,
    compute_torso_length, get_joint_name,
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP
)

from .forward_kinematics import ForwardKinematics
from .camera_projection import CameraProjection

__all__ = [
    'ForwardKinematics',
    'CameraProjection',
    'SKELETON_TREE',
    'ROOT_JOINTS',
    'BONE_LENGTH_RATIOS',
    'get_parent',
    'get_children',
    'get_bone_length_ratio',
    'compute_torso_length',
    'get_joint_name',
]



