# pipeline/pose/self_contact_detector.py
"""
Self-Contact Detection based on SC3D (Self-Contact 3D).
Detects body surface signatures of self-contact and enforces 3D constraints.

Reference: Fieraru et al. "Learning Complex 3D Human Self-Contact" (AAAI 2021)
Website: https://sc3d.imar.ro/
Dataset: HumanSC3D (1,032 sequences, 4,128 contact events, 1,246,487 3D skeletons)
Dataset: FlickrSC3D (3,415 images, 3,969 contact events, 25,297 surface correspondences)
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque

log = logging.getLogger("self_contact")

# COCO 17 Keypoint Indices
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

# Common self-contact pairs (from SC3D dataset analysis)
# Format: (joint_a, joint_b, max_distance_threshold)
CONTACT_PAIRS = [
    # Hand-face contacts (face-touch detection)
    (LEFT_WRIST, NOSE, 0.08),
    (LEFT_WRIST, LEFT_EYE, 0.08),
    (LEFT_WRIST, RIGHT_EYE, 0.08),
    (LEFT_WRIST, LEFT_EAR, 0.08),
    (RIGHT_WRIST, NOSE, 0.08),
    (RIGHT_WRIST, LEFT_EYE, 0.08),
    (RIGHT_WRIST, RIGHT_EYE, 0.08),
    (RIGHT_WRIST, RIGHT_EAR, 0.08),
    
    # Hand-body contacts
    (LEFT_WRIST, LEFT_SHOULDER, 0.10),
    (LEFT_WRIST, RIGHT_SHOULDER, 0.10),
    (RIGHT_WRIST, LEFT_SHOULDER, 0.10),
    (RIGHT_WRIST, RIGHT_SHOULDER, 0.10),
    (LEFT_WRIST, LEFT_HIP, 0.12),
    (RIGHT_WRIST, RIGHT_HIP, 0.12),
    
    # Arm crossing
    (LEFT_ELBOW, RIGHT_ELBOW, 0.15),
    (LEFT_WRIST, RIGHT_WRIST, 0.12),
    (LEFT_WRIST, RIGHT_ELBOW, 0.12),
    (RIGHT_WRIST, LEFT_ELBOW, 0.12),
    
    # Leg crossing
    (LEFT_KNEE, RIGHT_KNEE, 0.15),
    (LEFT_ANKLE, RIGHT_ANKLE, 0.12),
    (LEFT_ANKLE, RIGHT_KNEE, 0.12),
    (RIGHT_ANKLE, LEFT_KNEE, 0.12),
    
    # Hand-elbow contacts
    (LEFT_WRIST, LEFT_ELBOW, 0.10),
    (RIGHT_WRIST, RIGHT_ELBOW, 0.10),
]

# Impossible pairs (should never intersect)
IMPOSSIBLE_PAIRS = [
    (LEFT_HIP, RIGHT_HIP),  # Hips shouldn't intersect
    (LEFT_SHOULDER, RIGHT_SHOULDER),  # Shoulders shouldn't intersect
    (LEFT_KNEE, RIGHT_KNEE),  # Knees shouldn't intersect (unless crossing)
    (LEFT_ANKLE, RIGHT_ANKLE),  # Ankles shouldn't intersect (unless crossing)
]


class SelfContactDetector:
    """
    Self-Contact Detection based on SC3D (Self-Contact 3D).
    Detects body surface signatures of self-contact and enforces 3D constraints.
    
    Based on: Fieraru et al. "Learning Complex 3D Human Self-Contact" (AAAI 2021)
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 geometric_threshold: float = 0.05,
                 use_temporal_smoothing: bool = True):
        """
        Initialize self-contact detector.
        
        Args:
            model_path: Path to SC3D model (optional, uses geometric fallback if None)
            geometric_threshold: Distance threshold for geometric contact detection (meters)
            use_temporal_smoothing: Use temporal smoothing for contact detection
        """
        self.model_path = model_path
        self.geometric_threshold = geometric_threshold
        self.use_temporal_smoothing = use_temporal_smoothing
        self.contact_model = None
        self.contact_history = deque(maxlen=5)  # Last 5 frames
        
        # Load model if available
        if model_path:
            self._load_model()
        else:
            log.info("Using geometric self-contact detection (SC3D model not loaded)")
    
    def _load_model(self):
        """Load SC3D model (placeholder for future implementation)."""
        # TODO: Load SC3D SCP (Self-Contact Prediction) model when available
        # For now, use geometric fallback
        log.warning("SC3D model loading not yet implemented, using geometric fallback")
        self.contact_model = None
    
    def detect(self, kps_3d: List[Tuple[float, float, float, float]]) -> Dict:
        """
        Detect self-contact regions in 3D pose.
        
        Args:
            kps_3d: 3D keypoints (17, 4) - (x, y, z, confidence)
        
        Returns:
            contact_signature: Dict mapping joint indices to contact info
            {
                joint_idx: {
                    "in_contact": bool,
                    "contacts": {other_joint_idx: distance},
                    "contact_confidence": float,
                    "contact_type": str  # "hand_face", "arm_crossing", "leg_crossing", etc.
                }
            }
        """
        if not kps_3d or len(kps_3d) < 17:
            return {}
        
        if self.contact_model is None:
            # Use geometric distance-based detection
            contact_signature = self._geometric_contact_detection(kps_3d)
        else:
            # Use learned model (SC3D SCP - Self-Contact Prediction)
            contact_signature = self._model_based_detection(kps_3d)
        
        # Apply temporal smoothing if enabled
        if self.use_temporal_smoothing:
            contact_signature = self._apply_temporal_smoothing(contact_signature)
        
        return contact_signature
    
    def _geometric_contact_detection(self, kps_3d: List[Tuple[float, float, float, float]]) -> Dict:
        """
        Geometric fallback: detect contacts based on 3D distances.
        Based on common contact patterns from SC3D dataset.
        """
        contact_signature = {}
        kps_array = np.array([[kp[0], kp[1], kp[2]] for kp in kps_3d])
        confidences = [kp[3] if len(kp) > 3 else 1.0 for kp in kps_3d]
        
        # Check each joint for contacts
        for i in range(len(kps_3d)):
            if confidences[i] < 0.3:  # Low confidence
                contact_signature[i] = {
                    "in_contact": False,
                    "contacts": {},
                    "contact_confidence": 0.0,
                    "contact_type": None
                }
                continue
            
            contacts = {}
            contact_types = []
            
            # Check against known contact pairs
            for j in range(len(kps_3d)):
                if i == j or confidences[j] < 0.3:
                    continue
                
                # Compute 3D distance
                dist = np.linalg.norm(kps_array[i] - kps_array[j])
                
                # Check if this is a known contact pair
                for pair_i, pair_j, threshold in CONTACT_PAIRS:
                    if (i == pair_i and j == pair_j) or (i == pair_j and j == pair_i):
                        if dist < threshold:
                            contacts[j] = float(dist)
                            
                            # Determine contact type
                            if (i in [LEFT_WRIST, RIGHT_WRIST] and j in [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]):
                                contact_types.append("hand_face")
                            elif (i in [LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST] and 
                                  j in [LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]):
                                contact_types.append("arm_crossing")
                            elif (i in [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE] and 
                                  j in [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]):
                                contact_types.append("leg_crossing")
                            elif (i in [LEFT_WRIST, RIGHT_WRIST] and j in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]):
                                contact_types.append("hand_body")
                            break
            
            # Also check generic threshold for other potential contacts
            for j in range(len(kps_3d)):
                if i == j or j in contacts or confidences[j] < 0.3:
                    continue
                
                dist = np.linalg.norm(kps_array[i] - kps_array[j])
                if dist < self.geometric_threshold:
                    # Only add if not in impossible pairs
                    is_impossible = any((i == p1 and j == p2) or (i == p2 and j == p1) 
                                       for p1, p2 in IMPOSSIBLE_PAIRS)
                    if not is_impossible:
                        contacts[j] = float(dist)
            
            contact_signature[i] = {
                "in_contact": len(contacts) > 0,
                "contacts": contacts,
                "contact_confidence": min(1.0, len(contacts) * 0.3) if contacts else 0.0,
                "contact_type": contact_types[0] if contact_types else None
            }
        
        return contact_signature
    
    def _model_based_detection(self, kps_3d: List[Tuple[float, float, float, float]]) -> Dict:
        """
        Use learned SC3D model for contact detection.
        Placeholder for future implementation.
        """
        # TODO: Implement SC3D SCP model inference
        # For now, fallback to geometric
        return self._geometric_contact_detection(kps_3d)
    
    def _apply_temporal_smoothing(self, contact_signature: Dict) -> Dict:
        """Apply temporal smoothing to contact detection."""
        self.contact_history.append(contact_signature)
        
        if len(self.contact_history) < 2:
            return contact_signature
        
        # Smooth by majority vote over last N frames
        smoothed = {}
        for joint_idx in contact_signature.keys():
            in_contact_count = sum(1 for hist in self.contact_history 
                                  if hist.get(joint_idx, {}).get("in_contact", False))
            
            # Require at least 2/3 frames to confirm contact
            threshold = len(self.contact_history) * 0.67
            smoothed[joint_idx] = contact_signature[joint_idx].copy()
            smoothed[joint_idx]["in_contact"] = in_contact_count >= threshold
        
        return smoothed
    
    def validate_3d_pose(self, kps_3d: List[Tuple[float, float, float, float]]) -> Tuple[bool, List]:
        """
        Validate 3D pose doesn't have incorrect self-intersections.
        
        Returns:
            (is_valid, violations) where violations is list of (joint_i, joint_j, distance)
        """
        violations = []
        kps_array = np.array([[kp[0], kp[1], kp[2]] for kp in kps_3d])
        confidences = [kp[3] if len(kp) > 3 else 1.0 for kp in kps_3d]
        
        # Check for impossible intersections
        for i, j in IMPOSSIBLE_PAIRS:
            if i < len(kps_3d) and j < len(kps_3d):
                if confidences[i] < 0.3 or confidences[j] < 0.3:
                    continue
                
                dist = np.linalg.norm(kps_array[i] - kps_array[j])
                if dist < 0.01:  # Too close (intersecting)
                    violations.append((i, j, float(dist)))
        
        # Check for parts that should be in contact but are too far
        contact_signature = self.detect(kps_3d)
        for joint_idx, contact_info in contact_signature.items():
            if contact_info.get("in_contact", False):
                # If in contact, check if distance is reasonable
                for other_idx, distance in contact_info.get("contacts", {}).items():
                    if distance > 0.15:  # Too far for contact
                        violations.append((joint_idx, other_idx, distance))
        
        return len(violations) == 0, violations
    
    def correct_intersections(self, kps_3d: List[Tuple[float, float, float, float]], 
                             violations: List[Tuple[int, int, float]]) -> List[Tuple[float, float, float, float]]:
        """
        Correct self-intersections in 3D pose.
        
        Args:
            kps_3d: Original 3D keypoints
            violations: List of (joint_i, joint_j, distance) violations
        
        Returns:
            Corrected 3D keypoints
        """
        corrected = list(kps_3d)
        kps_array = np.array([[kp[0], kp[1], kp[2]] for kp in kps_3d])
        confidences = [kp[3] if len(kp) > 3 else 1.0 for kp in kps_3d]
        
        for i, j, dist in violations:
            if i >= len(corrected) or j >= len(corrected):
                continue
            
            # Move joints apart slightly
            vec = kps_array[i] - kps_array[j]
            if np.linalg.norm(vec) < 1e-6:
                vec = np.array([0.05, 0, 0])  # Default separation
            
            vec = vec / np.linalg.norm(vec) * 0.05  # 5cm separation
            
            # Adjust the joint with lower confidence
            if confidences[i] < confidences[j]:
                corrected[i] = (
                    kps_3d[i][0] + vec[0],
                    kps_3d[i][1] + vec[1],
                    kps_3d[i][2] + vec[2],
                    kps_3d[i][3] if len(kps_3d[i]) > 3 else 1.0
                )
            else:
                corrected[j] = (
                    kps_3d[j][0] - vec[0],
                    kps_3d[j][1] - vec[1],
                    kps_3d[j][2] - vec[2],
                    kps_3d[j][3] if len(kps_3d[j]) > 3 else 1.0
                )
        
        return corrected
    
    def detect_face_touch(self, contact_signature: Dict, kps_3d: Optional[List] = None) -> bool:
        """
        Detect face-touch using SC3D contact signature.
        Based on SC3D paper: monocular detection of face-touch.
        """
        hand_indices = [LEFT_WRIST, RIGHT_WRIST]
        face_indices = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]
        
        for hand_idx in hand_indices:
            hand_contact = contact_signature.get(hand_idx, {})
            if hand_contact.get("in_contact", False):
                contacts = hand_contact.get("contacts", {})
                for face_idx in face_indices:
                    if face_idx in contacts:
                        return True
        return False
    
    def detect_arm_crossing(self, contact_signature: Dict) -> bool:
        """Detect arm crossing from contact signature."""
        arm_indices = [LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
        
        for i in arm_indices:
            contact_info = contact_signature.get(i, {})
            if contact_info.get("in_contact", False):
                contacts = contact_info.get("contacts", {})
                # Check if contacting opposite side
                if i in [LEFT_ELBOW, LEFT_WRIST]:
                    if RIGHT_ELBOW in contacts or RIGHT_WRIST in contacts:
                        return True
                else:
                    if LEFT_ELBOW in contacts or LEFT_WRIST in contacts:
                        return True
        return False
    
    def detect_leg_crossing(self, contact_signature: Dict) -> bool:
        """Detect leg crossing from contact signature."""
        leg_indices = [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
        
        for i in leg_indices:
            contact_info = contact_signature.get(i, {})
            if contact_info.get("in_contact", False):
                contacts = contact_info.get("contacts", {})
                # Check if contacting opposite side
                if i in [LEFT_KNEE, LEFT_ANKLE]:
                    if RIGHT_KNEE in contacts or RIGHT_ANKLE in contacts:
                        return True
                else:
                    if LEFT_KNEE in contacts or LEFT_ANKLE in contacts:
                        return True
        return False

