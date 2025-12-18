# pipeline/kinematics/forward_kinematics.py
"""
Forward Kinematics Transformation (Tfk)
Based on AAAI-20: "Kinematic-Structure-Preserved Representation for Unsupervised 3D Human Pose Estimation"

This module implements forward kinematics to compute 3D joint positions from local kinematic parameters,
ensuring anatomically valid poses with bone-length constraints.
"""

import numpy as np
import math
import logging
from typing import List, Tuple, Optional, Dict
from .skeleton import (
    SKELETON_TREE, ROOT_JOINTS, get_parent, get_children,
    get_bone_length_ratio, compute_torso_length,
    LEFT_HIP, RIGHT_HIP, NOSE, LEFT_SHOULDER, RIGHT_SHOULDER
)

log = logging.getLogger("forward_kinematics")

# Number of joints in COCO format
NUM_JOINTS = 17


class ForwardKinematics:
    """
    Forward Kinematics Transformer.
    Converts local kinematic parameters to 3D joint coordinates.
    
    The approach:
    1. Regress local kinematic parameters (unit vectors in parent-relative coordinates)
    2. Apply forward kinematics recursively to compute 3D joint positions
    3. Enforce bone-length constraints using canonical ratios
    """
    
    def __init__(self, use_bone_constraints=True, canonical_scale=1.0):
        """
        Initialize forward kinematics transformer.
        
        Args:
            use_bone_constraints: Whether to enforce bone-length constraints
            canonical_scale: Reference scale for bone lengths (torso length)
        """
        self.use_bone_constraints = use_bone_constraints
        self.canonical_scale = canonical_scale
        
        # Cache for computed joint positions
        self.joint_positions = {}
        
        log.info("Forward Kinematics initialized (bone constraints: %s)", use_bone_constraints)
    
    def forward(self, local_kinematic_params: np.ndarray, 
                root_positions: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
        """
        Forward kinematics transformation: local params → 3D joint positions.
        
        Args:
            local_kinematic_params: Array of shape (N, 3) where N is number of non-root joints
                                   Each row is a unit vector in parent-relative coordinates
            root_positions: Optional dict of root joint positions {joint_idx: [x, y, z]}
                          If None, uses canonical root positions
        
        Returns:
            3D joint positions as array of shape (17, 3) - (x, y, z) for each joint
        """
        # Initialize joint positions
        joint_positions = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        
        # Set root joint positions
        if root_positions is None:
            # Canonical root positions (pelvis at origin, neck above)
            joint_positions[LEFT_HIP] = np.array([-0.1, 0.0, 0.0])  # Left hip
            joint_positions[RIGHT_HIP] = np.array([0.1, 0.0, 0.0])   # Right hip
            
            # Estimate neck position (midpoint of shoulders, above hips)
            # This will be refined by forward kinematics
            neck_y = 0.5  # Torso length above pelvis
            joint_positions[NOSE] = np.array([0.0, neck_y, 0.0])  # Nose/neck
        else:
            for joint_idx, pos in root_positions.items():
                if 0 <= joint_idx < NUM_JOINTS:
                    joint_positions[joint_idx] = np.array(pos)
        
        # Process joints in topological order (parents before children)
        processed = set(ROOT_JOINTS)
        param_idx = 0
        
        # Get all joints in topological order
        joint_order = self._get_topological_order()
        
        for joint_idx in joint_order:
            if joint_idx in processed:
                continue
            
            parent_idx = get_parent(joint_idx)
            if parent_idx is None:
                # This is a root joint, already set
                processed.add(joint_idx)
                continue
            
            if parent_idx not in processed:
                # Parent not processed yet, skip for now (will process in next iteration)
                continue
            
            # Get local kinematic parameter (unit vector)
            if param_idx < len(local_kinematic_params):
                local_vector = local_kinematic_params[param_idx]
                param_idx += 1
            else:
                # Fallback: use default direction
                log.warning(f"Insufficient kinematic parameters for joint {joint_idx}")
                local_vector = np.array([0.0, 1.0, 0.0])  # Default: upward
            
            # Normalize to unit vector
            local_vector = local_vector / (np.linalg.norm(local_vector) + 1e-8)
            
            # Get bone length
            bone_length = self._get_bone_length(joint_idx, parent_idx, joint_positions)
            
            # Compute joint position: parent_pos + bone_length * local_vector
            parent_pos = joint_positions[parent_idx]
            joint_positions[joint_idx] = parent_pos + bone_length * local_vector
            
            processed.add(joint_idx)
        
        return joint_positions
    
    def _get_topological_order(self) -> List[int]:
        """
        Get joints in topological order (parents before children).
        This ensures we process joints in the correct order for forward kinematics.
        """
        order = []
        processed = set(ROOT_JOINTS)
        
        # Start with root joints
        queue = list(ROOT_JOINTS)
        
        while queue:
            joint_idx = queue.pop(0)
            if joint_idx not in order:
                order.append(joint_idx)
            
            # Add children to queue
            children = get_children(joint_idx)
            for child in children:
                if child not in processed:
                    queue.append(child)
                    processed.add(child)
        
        # Add any remaining joints
        for joint_idx in range(NUM_JOINTS):
            if joint_idx not in order:
                order.append(joint_idx)
        
        return order
    
    def _get_bone_length(self, joint_idx: int, parent_idx: int, 
                        joint_positions: np.ndarray) -> float:
        """
        Get bone length for a joint pair.
        Uses bone-length ratios if constraints are enabled.
        """
        if not self.use_bone_constraints:
            # No constraints: use default length
            return 0.3  # Default bone length
        
        # Get bone length ratio
        ratio = get_bone_length_ratio(parent_idx, joint_idx)
        if ratio is None:
            # Fallback: use default
            return 0.3
        
        # Compute reference scale (torso length)
        torso_length = compute_torso_length(joint_positions)
        if torso_length is None or torso_length < 1e-6:
            # Use canonical scale
            reference_length = self.canonical_scale
        else:
            reference_length = torso_length
        
        # Bone length = ratio * reference_length
        bone_length = ratio * reference_length
        
        return float(bone_length)
    
    def estimate_local_params_from_3d(self, kps_3d: np.ndarray) -> np.ndarray:
        """
        Inverse operation: estimate local kinematic parameters from 3D joint positions.
        Useful for initialization or validation.
        
        Args:
            kps_3d: 3D joint positions (17, 3)
        
        Returns:
            Local kinematic parameters (unit vectors) for non-root joints
        """
        local_params = []
        
        joint_order = self._get_topological_order()
        
        for joint_idx in joint_order:
            parent_idx = get_parent(joint_idx)
            if parent_idx is None:
                continue  # Skip root joints
            
            # Compute direction vector from parent to joint
            parent_pos = kps_3d[parent_idx]
            joint_pos = kps_3d[joint_idx]
            direction = joint_pos - parent_pos
            
            # Normalize to unit vector
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                unit_vector = direction / norm
            else:
                unit_vector = np.array([0.0, 1.0, 0.0])  # Default: upward
            
            local_params.append(unit_vector)
        
        return np.array(local_params)
    
    def validate_bone_lengths(self, kps_3d: np.ndarray, 
                              tolerance: float = 0.3) -> Tuple[bool, Dict]:
        """
        Validate that bone lengths match expected ratios.
        
        Args:
            kps_3d: 3D joint positions (17, 3)
            tolerance: Allowed deviation from expected ratio (0.3 = 30%)
        
        Returns:
            (is_valid, violations_dict)
        """
        violations = {}
        is_valid = True
        
        # Compute reference scale (torso length)
        torso_length = compute_torso_length(kps_3d)
        if torso_length is None or torso_length < 1e-6:
            return False, {"error": "Cannot compute torso length"}
        
        # Check each bone
        from .skeleton import BONE_LENGTH_RATIOS
        for (joint_a, joint_b), expected_ratio in BONE_LENGTH_RATIOS.items():
            if joint_a >= len(kps_3d) or joint_b >= len(kps_3d):
                continue
            
            # Compute actual bone length
            pos_a = kps_3d[joint_a]
            pos_b = kps_3d[joint_b]
            actual_length = np.linalg.norm(pos_b - pos_a)
            
            # Expected length
            expected_length = expected_ratio * torso_length
            
            # Check deviation
            if expected_length > 1e-6:
                deviation = abs(actual_length - expected_length) / expected_length
                if deviation > tolerance:
                    violations[(joint_a, joint_b)] = {
                        "actual": float(actual_length),
                        "expected": float(expected_length),
                        "deviation": float(deviation)
                    }
                    is_valid = False
        
        return is_valid, violations

