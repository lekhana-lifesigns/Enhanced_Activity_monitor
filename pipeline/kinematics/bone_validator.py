# pipeline/kinematics/bone_validator.py
"""
Bone-Length Constraint Validator
Validates and enforces anatomically plausible bone lengths.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from .skeleton import (
    BONE_LENGTH_RATIOS, get_bone_length_ratio, compute_torso_length,
    SKELETON_TREE
)

log = logging.getLogger("bone_validator")


class BoneLengthValidator:
    """
    Validates and enforces bone-length constraints.
    Ensures predicted poses are anatomically plausible.
    """
    
    def __init__(self, tolerance: float = 0.3, enforce_constraints: bool = True):
        """
        Initialize bone-length validator.
        
        Args:
            tolerance: Allowed deviation from expected bone length ratio (0.3 = 30%)
            enforce_constraints: Whether to actively correct violations (vs. just detect)
        """
        self.tolerance = tolerance
        self.enforce_constraints = enforce_constraints
        log.info("Bone-length validator initialized (tolerance=%.2f, enforce=%s)",
                 tolerance, enforce_constraints)
    
    def validate(self, kps_3d: np.ndarray) -> Tuple[bool, Dict, np.ndarray]:
        """
        Validate bone lengths and optionally correct violations.
        
        Args:
            kps_3d: 3D joint positions (17, 3) - (x, y, z)
        
        Returns:
            (is_valid, violations_dict, corrected_kps_3d)
            - is_valid: True if all bones are within tolerance
            - violations_dict: Dict of violations with details
            - corrected_kps_3d: Corrected 3D positions (if enforce_constraints=True)
        """
        if kps_3d is None or len(kps_3d) < 17:
            return False, {"error": "Insufficient keypoints"}, kps_3d
        
        violations = {}
        is_valid = True
        
        # Compute reference scale (torso length)
        torso_length = compute_torso_length(kps_3d)
        if torso_length is None or torso_length < 1e-6:
            return False, {"error": "Cannot compute torso length"}, kps_3d
        
        # Check each bone in skeleton tree
        for child_joint, parent_joint in SKELETON_TREE.items():
            if child_joint >= len(kps_3d) or parent_joint >= len(kps_3d):
                continue
            
            # Get expected bone length ratio
            ratio = get_bone_length_ratio(parent_joint, child_joint)
            if ratio is None:
                continue
            
            # Compute actual bone length
            parent_pos = kps_3d[parent_joint]
            child_pos = kps_3d[child_joint]
            actual_length = float(np.linalg.norm(child_pos - parent_pos))
            
            # Expected length
            expected_length = ratio * torso_length
            
            # Check deviation
            if expected_length > 1e-6:
                deviation = abs(actual_length - expected_length) / expected_length
                
                if deviation > self.tolerance:
                    violations[(parent_joint, child_joint)] = {
                        "actual_length": actual_length,
                        "expected_length": expected_length,
                        "deviation": deviation,
                        "ratio": ratio
                    }
                    is_valid = False
        
        # Correct violations if enabled
        corrected_kps_3d = kps_3d.copy()
        if self.enforce_constraints and not is_valid:
            corrected_kps_3d = self._correct_violations(corrected_kps_3d, violations, torso_length)
        
        return is_valid, violations, corrected_kps_3d
    
    def _correct_violations(self, kps_3d: np.ndarray, violations: Dict, 
                           torso_length: float) -> np.ndarray:
        """
        Correct bone-length violations by adjusting joint positions.
        Uses iterative refinement to minimize changes.
        """
        corrected = kps_3d.copy()
        max_iterations = 5
        
        for iteration in range(max_iterations):
            has_changes = False
            
            for (parent_joint, child_joint), violation_info in violations.items():
                if parent_joint >= len(corrected) or child_joint >= len(corrected):
                    continue
                
                parent_pos = corrected[parent_joint]
                child_pos = corrected[child_joint]
                
                # Current bone vector
                bone_vector = child_pos - parent_pos
                current_length = np.linalg.norm(bone_vector)
                
                # Expected length
                ratio = violation_info["ratio"]
                expected_length = ratio * torso_length
                
                # Adjust if deviation is significant
                if current_length > 1e-6:
                    scale_factor = expected_length / current_length
                    
                    # Only adjust if change is significant
                    if abs(scale_factor - 1.0) > 0.05:  # 5% threshold
                        # Adjust child position
                        corrected[child_joint] = parent_pos + bone_vector * scale_factor
                        has_changes = True
            
            if not has_changes:
                break
        
        return corrected
    
    def compute_bone_length_loss(self, kps_3d: np.ndarray) -> float:
        """
        Compute loss for bone-length violations (for training/optimization).
        
        Args:
            kps_3d: 3D joint positions (17, 3)
        
        Returns:
            Loss value (0 = perfect, higher = more violations)
        """
        if kps_3d is None or len(kps_3d) < 17:
            return 1.0  # High loss for invalid input
        
        torso_length = compute_torso_length(kps_3d)
        if torso_length is None or torso_length < 1e-6:
            return 1.0
        
        total_loss = 0.0
        num_bones = 0
        
        for child_joint, parent_joint in SKELETON_TREE.items():
            if child_joint >= len(kps_3d) or parent_joint >= len(kps_3d):
                continue
            
            ratio = get_bone_length_ratio(parent_joint, child_joint)
            if ratio is None:
                continue
            
            # Compute actual and expected lengths
            parent_pos = kps_3d[parent_joint]
            child_pos = kps_3d[child_joint]
            actual_length = np.linalg.norm(child_pos - parent_pos)
            expected_length = ratio * torso_length
            
            if expected_length > 1e-6:
                # Squared error normalized by expected length
                error = (actual_length - expected_length) / expected_length
                total_loss += error ** 2
                num_bones += 1
        
        if num_bones > 0:
            return float(total_loss / num_bones)
        else:
            return 1.0



