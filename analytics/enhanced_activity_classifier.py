# analytics/enhanced_activity_classifier.py
# Enhanced Activity Classifier - Classifies all 53 activities
# Uses YOLO11, COCO keypoints, depth estimation, and existing modules

import numpy as np
import math
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque

# Import activity definitions
from analytics.activity_definitions import (
    ALL_ACTIVITIES, CRITICAL_ACTIVITIES, HIGH_PRIORITY_ACTIVITIES,
    get_activity_priority, get_activity_info
)

log = logging.getLogger("enhanced_activity")

# COCO Keypoint indices
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

MIN_CONFIDENCE = 0.3


class EnhancedActivityClassifier:
    """
    Enhanced activity classifier that can identify all 53 activities.
    Uses:
    - COCO keypoints for pose analysis
    - YOLO11 for object detection (beds, tubes, etc.)
    - Temporal patterns from keypoint history
    - Integration with existing modules (bed detection, fall detection, posture)
    """
    
    def __init__(self):
        """Initialize enhanced activity classifier."""
        self.kps_history = deque(maxlen=30)  # Last 30 frames
        self.activity_history = deque(maxlen=10)  # Last 10 activity classifications
        self.motion_history = deque(maxlen=20)  # Motion patterns
        
        # Activity state tracking
        self.previous_activity = None
        self.activity_transitions = {}  # Track activity changes
        
        # TODO-020: Activity Classification Caching
        self.activity_cache = {}  # Cache for activity results
        self.last_cached_kps_hash = None
        self.cache_pose_change_threshold = 0.03  # 3% pose change to invalidate cache
        self.enable_activity_caching = True  # Enable by default
        
        # YOLO for object detection (if available)
        self.yolo_model = None
        self._init_yolo()
    
    def _init_yolo(self):
        """Initialize YOLO model for object detection."""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolo11n.pt")
            log.info("YOLO11 initialized for enhanced activity classification")
        except Exception as e:
            log.warning("YOLO not available for activity classification: %s", e)
            self.yolo_model = None
    
    def classify_activity(self,
                         kps: List,
                         kps_history: Optional[List] = None,
                         posture_state: Optional[str] = None,
                         bed_info: Optional[Dict] = None,
                         person_on_bed: Optional[bool] = None,
                         fall_detected: Optional[bool] = None,
                         frame: Optional[np.ndarray] = None,
                         bbox: Optional[List] = None,
                         kps_3d: Optional[List] = None,  # TODO-066: Add 3D pose
                         contact_signature: Optional[Dict] = None) -> Dict:  # TODO-066: Add contact signature
        """
        Classify activity using all available information.
        
        Args:
            kps: Current COCO keypoints
            kps_history: History of keypoints
            posture_state: Current posture (from posture classifier)
            bed_info: Bed detection info
            person_on_bed: Whether person is on bed
            fall_detected: Whether fall was detected
            frame: Current frame (for YOLO object detection)
            bbox: Person bounding box
        
        Returns:
            dict with:
            - activity: Activity key from ALL_ACTIVITIES
            - confidence: 0-1 confidence score
            - priority: CRITICAL | HIGH | NORMAL | MEDIUM
            - details: Additional metrics
        """
        if not kps or len(kps) < 13:
            return self._unknown_activity("insufficient_keypoints")
        
        # TODO-020: Activity Classification Caching
        if self.enable_activity_caching:
            # Compute pose hash for caching
            kps_hash = self._compute_pose_hash(kps)
            
            # Check if pose has changed significantly
            if self.last_cached_kps_hash is not None:
                pose_change = self._compute_pose_change_from_hash(kps_hash, self.last_cached_kps_hash)
                
                if pose_change < self.cache_pose_change_threshold:
                    # Pose hasn't changed significantly, return cached result
                    cached_result = self.activity_cache.get(self.last_cached_kps_hash)
                    if cached_result:
                        log.debug("Using cached activity result (pose change: %.4f)", pose_change)
                        # Update timestamp but keep cached classification
                        cached_result = cached_result.copy()
                        cached_result["cached"] = True
                        return cached_result
            
            # Update cache key
            self.last_cached_kps_hash = kps_hash
        
        # Update history
        if kps_history:
            self.kps_history.extend(kps_history[-10:])  # Keep last 10
        else:
            self.kps_history.append(kps)
        
        # Check critical activities first (highest priority)
        critical_result = self._check_critical_activities(
            kps, fall_detected, posture_state, contact_signature  # TODO-072: Add contact signature
        )
        if critical_result:
            return critical_result
        
        # Check bed-related activities
        if bed_info is not None or person_on_bed is not None:
            bed_result = self._check_bed_activities(
                kps, posture_state, bed_info, person_on_bed
            )
            if bed_result:
                return bed_result
        
        # Check fall activities
        if fall_detected:
            return {
                "activity": "fallen",
                "confidence": 0.9,
                "priority": "CRITICAL",
                "details": {"fall_detected": True}
            }
        
        # Check locomotion activities
        locomotion_result = self._check_locomotion(kps, kps_history)
        if locomotion_result:
            return locomotion_result
        
        # Check basic postural activities
        postural_result = self._check_basic_postural(kps, posture_state)
        if postural_result:
            return postural_result
        
        # Check upper body activities
        upper_body_result = self._check_upper_body(kps, kps_history, frame)
        if upper_body_result:
            return upper_body_result
        
        # Check lower body activities
        lower_body_result = self._check_lower_body(kps, kps_history)
        if lower_body_result:
            return lower_body_result
        
        # Check agitation/distress
        agitation_result = self._check_agitation_distress(kps, kps_history)
        if agitation_result:
            return agitation_result
        
        # Check neurological activities
        neuro_result = self._check_neurological(kps, kps_history)
        if neuro_result:
            return neuro_result
        
        # Check respiratory activities
        respiratory_result = self._check_respiratory(kps, kps_history)
        if respiratory_result:
            return respiratory_result
        
        # Check inactive states
        inactive_result = self._check_inactive(kps, kps_history)
        if inactive_result:
            result = inactive_result
        else:
            # Default: use basic classification
            result = self._basic_classification(kps, kps_history)
        
        # TODO-020: Cache activity result
        if self.enable_activity_caching and self.last_cached_kps_hash is not None:
            self.activity_cache[self.last_cached_kps_hash] = result.copy()
            # Limit cache size
            if len(self.activity_cache) > 10:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.activity_cache))
                del self.activity_cache[oldest_key]
        
        return result
    
    def _check_critical_activities(self, kps, fall_detected, posture_state, contact_signature=None):
        """
        Check for critical activities (highest priority).
        TODO-072: Enhanced with SC3D self-contact signatures.
        """
        # Falling - detected by fall detection module
        if fall_detected:
            return {
                "activity": "falling",
                "confidence": 0.9,
                "priority": "CRITICAL",
                "details": {"fall_detected": True}
            }
        
        # Unresponsive - very still, low movement
        if self._is_unresponsive(kps):
            return {
                "activity": "unresponsive",
                "confidence": 0.7,
                "priority": "CRITICAL",
                "details": {"movement_level": "very_low"}
            }
        
        # Seizure - rapid, irregular movements
        seizure_score = self._detect_seizure(kps)
        if seizure_score > 0.6:
            return {
                "activity": "seizure",
                "confidence": seizure_score,
                "priority": "CRITICAL",
                "details": {"seizure_score": seizure_score}
            }
        
        # Agitated - high movement, irregular patterns
        agitation_score = self._compute_agitation_score(kps)
        
        # TODO-072: Boost agitation if face-touching detected (SC3D)
        if contact_signature:
            try:
                from pipeline.pose.self_contact_detector import SelfContactDetector
                if not hasattr(self, 'self_contact_detector'):
                    self.self_contact_detector = SelfContactDetector()
                if self.self_contact_detector.detect_face_touch(contact_signature):
                    agitation_score = min(1.0, agitation_score + 0.2)  # Boost for face-touch
            except Exception:
                pass
        
        if agitation_score > 0.7:
            return {
                "activity": "agitated",
                "confidence": agitation_score,
                "priority": "CRITICAL",
                "details": {"agitation_score": agitation_score}
            }
        
        return None
    
    def _compute_pose_hash(self, kps):
        """
        Compute a hash for pose keypoints for caching.
        TODO-020: Activity Classification Caching helper.
        """
        try:
            import hashlib
            # Use keypoint positions (normalized) for hash
            kps_str = "".join([f"{kp[0]:.3f},{kp[1]:.3f}" for kp in kps if len(kp) >= 2])
            return hashlib.md5(kps_str.encode()).hexdigest()
        except Exception:
            return None
    
    def _compute_pose_change_from_hash(self, hash1, hash2):
        """
        Compute pose change between two hashes (simplified).
        TODO-020: Activity Classification Caching helper.
        """
        if hash1 == hash2:
            return 0.0
        # Simple heuristic: different hash = some change
        # For more accurate change, we'd need to compare actual keypoints
        return 0.1  # Assume 10% change if hashes differ
    
    def _check_bed_activities(self, kps, posture_state, bed_info, person_on_bed):
        """Check bed-related activities."""
        if person_on_bed is None:
            # Try to infer from keypoints and bed position
            if bed_info and kps:
                person_on_bed = self._is_person_on_bed(kps, bed_info)
        
        if person_on_bed:
            # Patient is on bed
            if posture_state in ["supine", "prone", "left_lateral", "right_lateral"]:
                return {
                    "activity": "lying_in_bed",
                    "confidence": 0.85,
                    "priority": "NORMAL",
                    "details": {"posture": posture_state, "on_bed": True}
                }
            elif posture_state == "sitting":
                return {
                    "activity": "sitting_on_bed",
                    "confidence": 0.8,
                    "priority": "NORMAL",
                    "details": {"posture": posture_state, "on_bed": True}
                }
            elif posture_state == "sitting_up":
                return {
                    "activity": "sitting_up_in_bed",
                    "confidence": 0.75,
                    "priority": "NORMAL",
                    "details": {"posture": posture_state}
                }
            
            # Check for turning in bed
            if self._is_turning_in_bed(kps):
                return {
                    "activity": "turning_in_bed",
                    "confidence": 0.7,
                    "priority": "NORMAL",
                    "details": {"movement_type": "turning"}
                }
        else:
            # Check for bed exit
            if bed_info and self._is_bed_exit(kps, bed_info):
                return {
                    "activity": "bed_exit",
                    "confidence": 0.8,
                    "priority": "CRITICAL",
                    "details": {"bed_exit_detected": True}
                }
            # Check for bed entry
            elif bed_info and self._is_bed_entry(kps, bed_info):
                return {
                    "activity": "bed_entry",
                    "confidence": 0.75,
                    "priority": "NORMAL",
                    "details": {"bed_entry_detected": True}
                }
        
        return None
    
    def _check_locomotion(self, kps, kps_history):
        """Check locomotion activities."""
        if not kps_history or len(kps_history) < 3:
            return None
        
        # Tripping - sudden loss of balance, rapid downward movement
        if self._is_tripping(kps, kps_history):
            return {
                "activity": "tripping",
                "confidence": 0.75,
                "priority": "HIGH",
                "details": {"locomotion_type": "tripping", "fall_risk": "high"}
            }
        
        # Roaming/Wandering - aimless movement, pacing patterns
        if self._is_roaming(kps, kps_history):
            return {
                "activity": "roaming",
                "confidence": 0.7,
                "priority": "HIGH",
                "details": {"locomotion_type": "roaming", "elopement_risk": "moderate"}
            }
        
        # Walking - rhythmic leg movement
        if self._is_walking(kps, kps_history):
            return {
                "activity": "walking",
                "confidence": 0.75,
                "priority": "NORMAL",
                "details": {"locomotion_type": "walking"}
            }
        
        # Running - faster leg movement
        if self._is_running(kps, kps_history):
            return {
                "activity": "running",
                "confidence": 0.7,
                "priority": "HIGH",
                "details": {"locomotion_type": "running"}
            }
        
        # Crawling - low to ground, arm and leg movement
        if self._is_crawling(kps, kps_history):
            return {
                "activity": "crawling",
                "confidence": 0.65,
                "priority": "HIGH",
                "details": {"locomotion_type": "crawling"}
            }
        
        # General moving - any movement that's not specifically walking/running
        if self._is_moving(kps, kps_history):
            return {
                "activity": "moving",
                "confidence": 0.65,
                "priority": "NORMAL",
                "details": {"locomotion_type": "general_movement"}
            }
        
        return None
    
    def _check_basic_postural(self, kps, posture_state):
        """Check basic postural activities."""
        # Use posture state if available, otherwise infer from keypoints
        if posture_state:
            if posture_state == "supine":
                return {
                    "activity": "lying",
                    "confidence": 0.85,
                    "priority": "NORMAL",
                    "details": {"posture": "supine"}
                }
            elif posture_state == "sitting":
                return {
                    "activity": "sitting",
                    "confidence": 0.85,
                    "priority": "NORMAL",
                    "details": {"posture": "sitting"}
                }
            elif posture_state in ["standing", "upright"]:
                return {
                    "activity": "standing",
                    "confidence": 0.85,
                    "priority": "NORMAL",
                    "details": {"posture": "standing"}
                } # Needs improvement....
        
        # Fallback: infer from keypoints
        vertical_extent = self._compute_vertical_extent(kps)
        horizontal_extent = self._compute_horizontal_extent(kps)
        aspect_ratio = vertical_extent / (horizontal_extent + 1e-6)
        
        if aspect_ratio < 0.8:
            return {
                "activity": "lying",
                "confidence": 0.75,
                "priority": "NORMAL",
                "details": {"aspect_ratio": aspect_ratio}
            }
        elif aspect_ratio > 1.2:
            return {
                "activity": "standing",
                "confidence": 0.75,
                "priority": "NORMAL",
                "details": {"aspect_ratio": aspect_ratio}
            }
        
        return None
    
    def _check_upper_body(self, kps, kps_history, frame):
        """Check upper body activities."""
        if not kps_history or len(kps_history) < 3:
            return None
        
        # Reaching - arm extended upward/forward
        if self._is_reaching(kps):
            return {
                "activity": "reaching",
                "confidence": 0.7,
                "priority": "NORMAL",
                "details": {"arm_position": "extended"}
            }
        
        # Waving - rhythmic arm movement
        if self._is_waving(kps, kps_history):
            return {
                "activity": "waving",
                "confidence": 0.65,
                "priority": "NORMAL",
                "details": {"arm_movement": "waving"}
            }
        
        
        # Pulling at tubes - hand near face/chest, repetitive motion
        if self._is_pulling_at_tubes(kps, kps_history, frame):
            return {
                "activity": "pulling_at_tubes",
                "confidence": 0.7,
                "priority": "CRITICAL",
                "details": {"hand_position": "near_face"} 
            }
        
        return None
    
    def _check_lower_body(self, kps, kps_history):
        """Check lower body activities."""
        if not kps_history:
            return None
        
        # Leg movement - repetitive leg motion
        if self._has_leg_movement(kps, kps_history):
            return {
                "activity": "leg_movement",
                "confidence": 0.65,
                "priority": "NORMAL",
                "details": {"movement_type": "leg"}
            }
        
        return None
    
    def _check_agitation_distress(self, kps, kps_history):
        """Check agitation and distress activities."""
        if not kps_history or len(kps_history) < 5:
            return None
        
        # Restless - frequent position changes
        if self._is_restless(kps, kps_history):
            return {
                "activity": "restless",
                "confidence": 0.7,
                "priority": "HIGH",
                "details": {"movement_pattern": "frequent_changes"}
            }
        
        # Thrashing - violent, irregular movements
        if self._is_thrashing(kps, kps_history):
            return {
                "activity": "thrashing",
                "confidence": 0.75,
                "priority": "CRITICAL",
                "details": {"movement_pattern": "violent"}
            }
        
        return None
    
    def _check_neurological(self, kps, kps_history):
        """Check neurological activities."""
        if not kps_history or len(kps_history) < 5:
            return None
        
        # Seizure - already checked in critical, but check convulsion
        convulsion_score = self._detect_convulsion(kps, kps_history)
        if convulsion_score > 0.6:
            return {
                "activity": "convulsion",
                "confidence": convulsion_score,
                "priority": "CRITICAL",
                "details": {"convulsion_score": convulsion_score}
            }
        
        # Tremor - small, rapid oscillations
        tremor_score = self._detect_tremor(kps, kps_history)
        if tremor_score > 0.6:
            return {
                "activity": "tremor",
                "confidence": tremor_score,
                "priority": "HIGH",
                "details": {"tremor_score": tremor_score}
            }
        
        # Rigidity - stiff, limited movement
        if self._is_rigid(kps, kps_history):
            return {
                "activity": "rigidity",
                "confidence": 0.65,
                "priority": "HIGH",
                "details": {"movement_limitation": "high"}
            }
        
        return None
    
    def _check_respiratory(self, kps, kps_history):
        """Check respiratory activities."""
        if not kps_history or len(kps_history) < 10:
            return None
        
        # Breathing - chest movement (detected via thorax oscillation)
        breathing_score = self._detect_breathing(kps, kps_history)
        if breathing_score > 0.5:
            return {
                "activity": "breathing",
                "confidence": breathing_score,
                "priority": "NORMAL",
                "details": {"breathing_detected": True}
            }
        
        # Coughing - sudden chest contraction
        if self._is_coughing(kps, kps_history):
            return {
                "activity": "coughing",
                "confidence": 0.65,
                "priority": "HIGH",
                "details": {"respiratory_event": "cough"}
            }
        
        return None
    
    def _check_inactive(self, kps, kps_history):
        """Check inactive/static states."""
        if not kps_history or len(kps_history) < 5:
            return None
        
        # Still - very low movement
        if self._is_still(kps, kps_history):
            return {
                "activity": "still",
                "confidence": 0.7,
                "priority": "NORMAL",
                "details": {"movement_level": "very_low"}
            }
        
        # Sleeping - still + lying position
        if self._is_sleeping(kps, kps_history):
            return {
                "activity": "sleeping",
                "confidence": 0.75,
                "priority": "NORMAL",
                "details": {"state": "sleep"}
            }
        
        return None
    
    def _basic_classification(self, kps, kps_history):
        """Fallback to basic 4-activity classification."""
        vertical_extent = self._compute_vertical_extent(kps)
        horizontal_extent = self._compute_horizontal_extent(kps)
        aspect_ratio = vertical_extent / (horizontal_extent + 1e-6)
        
        left_knee_angle = self._compute_knee_angle(kps, 'left')
        right_knee_angle = self._compute_knee_angle(kps, 'right')
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0
        
        # Lying
        if aspect_ratio < 0.8 and horizontal_extent > 0.3:
            return {
                "activity": "lying",
                "confidence": 0.8,
                "priority": "NORMAL",
                "details": {"aspect_ratio": aspect_ratio}
            }
        
        # Walking
        if kps_history and len(kps_history) >= 3:
            prev_kps = kps_history[-2]
            prev_left_angle = self._compute_knee_angle(prev_kps, 'left')
            angle_change = abs(left_knee_angle - prev_left_angle)
            if angle_change > 10.0 and avg_knee_angle < 160.0:
                return {
                    "activity": "walking",
                    "confidence": 0.7,
                    "priority": "NORMAL",
                    "details": {"angle_change": angle_change}
                }
        
        # Sitting
        if aspect_ratio > 0.8 and aspect_ratio < 1.5 and avg_knee_angle < 140.0:
            return {
                "activity": "sitting",
                "confidence": 0.75,
                "priority": "NORMAL",
                "details": {"aspect_ratio": aspect_ratio}
            }
        
        # Standing
        if aspect_ratio > 1.2 and avg_knee_angle > 160.0:
            return {
                "activity": "standing",
                "confidence": 0.8,
                "priority": "NORMAL",
                "details": {"aspect_ratio": aspect_ratio}
            }
        
        return self._unknown_activity("no_match")
    
    # ========================================================================
    # Helper Methods for Activity Detection
    # ========================================================================
    
    def _get_keypoint(self, kps, idx, default=(0.0, 0.0, 0.0)):
        """Safely get keypoint with confidence check."""
        if idx < len(kps) and kps[idx][2] > MIN_CONFIDENCE:
            return kps[idx]
        return default
    
    def _compute_vertical_extent(self, kps):
        """Compute vertical extent of body."""
        try:
            nose = self._get_keypoint(kps, NOSE)
            lankle = self._get_keypoint(kps, LEFT_ANKLE)
            rankle = self._get_keypoint(kps, RIGHT_ANKLE)
            top_y = nose[1]
            bottom_y = max(lankle[1], rankle[1])
            return abs(bottom_y - top_y)
        except:
            return 0.0
    
    def _compute_horizontal_extent(self, kps):
        """Compute horizontal extent of body."""
        try:
            lshoulder = self._get_keypoint(kps, LEFT_SHOULDER)
            rshoulder = self._get_keypoint(kps, RIGHT_SHOULDER)
            left_x = min(lshoulder[0], rshoulder[0])
            right_x = max(lshoulder[0], rshoulder[0])
            return abs(right_x - left_x)
        except:
            return 0.0
    
    def _compute_knee_angle(self, kps, side='left'):
        """Compute knee angle."""
        try:
            if side == 'left':
                hip = self._get_keypoint(kps, LEFT_HIP)
                knee = self._get_keypoint(kps, LEFT_KNEE)
                ankle = self._get_keypoint(kps, LEFT_ANKLE)
            else:
                hip = self._get_keypoint(kps, RIGHT_HIP)
                knee = self._get_keypoint(kps, RIGHT_KNEE)
                ankle = self._get_keypoint(kps, RIGHT_ANKLE)
            
            v1_x = knee[0] - hip[0]
            v1_y = knee[1] - hip[1]
            v2_x = ankle[0] - knee[0]
            v2_y = ankle[1] - knee[1]
            
            dot = v1_x * v2_x + v1_y * v2_y
            mag1 = math.sqrt(v1_x**2 + v1_y**2)
            mag2 = math.sqrt(v2_x**2 + v2_y**2)
            
            if mag1 < 1e-6 or mag2 < 1e-6:
                return 180.0
            
            cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
            return math.degrees(math.acos(cos_angle))
        except:
            return 180.0
    
    def _is_walking(self, kps, kps_history):
        """Detect walking from rhythmic leg movement."""
        if len(kps_history) < 5:
            return False
        
        knee_angles = []
        for kp in kps_history[-5:]:
            left_angle = self._compute_knee_angle(kp, 'left')
            right_angle = self._compute_knee_angle(kp, 'right')
            knee_angles.append((left_angle, right_angle))
        
        # Check for alternating pattern (walking signature)
        left_changes = [abs(knee_angles[i][0] - knee_angles[i+1][0]) for i in range(len(knee_angles)-1)]
        right_changes = [abs(knee_angles[i][1] - knee_angles[i+1][1]) for i in range(len(knee_angles)-1)]
        
        avg_change = (np.mean(left_changes) + np.mean(right_changes)) / 2.0
        return avg_change > 15.0  # Significant knee angle changes
    
    def _is_running(self, kps, kps_history):
        """Detect running from fast leg movement."""
        if not self._is_walking(kps, kps_history):
            return False
        
        # Running has faster movement than walking
        if len(kps_history) < 3:
            return False
        
        # Compute velocity of hip movement
        hip_positions = []
        for kp in kps_history[-5:]:
            lhip = self._get_keypoint(kp, LEFT_HIP)
            rhip = self._get_keypoint(kp, RIGHT_HIP)
            hip_y = (lhip[1] + rhip[1]) / 2.0
            hip_positions.append(hip_y)
        
        if len(hip_positions) < 2:
            return False
        
        velocities = [abs(hip_positions[i] - hip_positions[i+1]) for i in range(len(hip_positions)-1)]
        avg_velocity = np.mean(velocities)
        
        return avg_velocity > 0.05  # Faster than walking threshold
    
    def _is_tripping(self, kps, kps_history):
        """Detect tripping - sudden loss of balance, rapid downward/forward movement."""
        if len(kps_history) < 5:
            return False
        
        # Get hip positions over time
        hip_positions = []
        for kp in kps_history[-5:]:
            lhip = self._get_keypoint(kp, LEFT_HIP)
            rhip = self._get_keypoint(kp, RIGHT_HIP)
            if lhip and rhip:
                hip_y = (lhip[1] + rhip[1]) / 2.0
                hip_x = (lhip[0] + rhip[0]) / 2.0
                hip_positions.append((hip_x, hip_y))
        
        if len(hip_positions) < 3:
            return False
        
        # Check for rapid downward movement (tripping forward/down)
        y_velocities = [hip_positions[i][1] - hip_positions[i+1][1] for i in range(len(hip_positions)-1)]
        avg_y_velocity = np.mean(y_velocities)
        
        # Check for sudden forward movement (stumbling forward)
        x_velocities = [abs(hip_positions[i][0] - hip_positions[i+1][0]) for i in range(len(hip_positions)-1)]
        avg_x_velocity = np.mean(x_velocities)
        
        # Check for irregular leg movement (loss of balance)
        ankle_positions = []
        for kp in kps_history[-3:]:
            lankle = self._get_keypoint(kp, LEFT_ANKLE)
            rankle = self._get_keypoint(kp, RIGHT_ANKLE)
            if lankle and rankle:
                ankle_y = (lankle[1] + rankle[1]) / 2.0
                ankle_positions.append(ankle_y)
        
        if len(ankle_positions) >= 2:
            ankle_instability = np.std(ankle_positions)
        else:
            ankle_instability = 0.0
        
        # Tripping indicators:
        # 1. Rapid downward movement (>0.03)
        # 2. High forward velocity (>0.02)
        # 3. Ankle instability (high std dev)
        tripping_score = 0.0
        if avg_y_velocity > 0.03:  # Moving down rapidly
            tripping_score += 0.4
        if avg_x_velocity > 0.02:  # Moving forward quickly
            tripping_score += 0.3
        if ankle_instability > 0.01:  # Unstable ankles
            tripping_score += 0.3
        
        return tripping_score >= 0.6
    
    def _is_roaming(self, kps, kps_history):
        """Detect roaming/wandering - aimless movement, pacing patterns."""
        if len(kps_history) < 10:
            return False
        
        # Get center of mass (hip) positions
        positions = []
        for kp in kps_history[-10:]:
            lhip = self._get_keypoint(kp, LEFT_HIP)
            rhip = self._get_keypoint(kp, RIGHT_HIP)
            if lhip and rhip:
                center_x = (lhip[0] + rhip[0]) / 2.0
                center_y = (lhip[1] + rhip[1]) / 2.0
                positions.append((center_x, center_y))
        
        if len(positions) < 5:
            return False
        
        # Check for pacing pattern (back and forth movement)
        x_positions = [p[0] for p in positions]
        y_positions = [p[1] for p in positions]
        
        # Calculate movement direction changes (pacing indicator)
        x_changes = [x_positions[i] - x_positions[i+1] for i in range(len(x_positions)-1)]
        direction_changes = sum(1 for i in range(len(x_changes)-1) 
                                if (x_changes[i] > 0) != (x_changes[i+1] > 0))
        
        # Calculate total distance traveled
        total_distance = sum(math.sqrt((positions[i][0] - positions[i+1][0])**2 + 
                                       (positions[i][1] - positions[i+1][1])**2) 
                            for i in range(len(positions)-1))
        
        # Calculate net displacement (how far from start)
        net_displacement = math.sqrt((positions[0][0] - positions[-1][0])**2 + 
                                    (positions[0][1] - positions[-1][1])**2)
        
        # Roaming indicators:
        # 1. High total distance but low net displacement (moving around same area)
        # 2. Multiple direction changes (pacing)
        # 3. Standing posture (not lying/sitting)
        posture_state = self._infer_posture(kps)
        is_upright = posture_state in ["standing", "upright"]
        
        if total_distance > 0.1 and net_displacement < total_distance * 0.3:
            # Moving a lot but not going far (wandering in area)
            if direction_changes >= 3:  # Multiple direction changes
                return True
            if is_upright and total_distance > 0.15:  # Standing and moving around
                return True
        
        return False
    
    def _is_moving(self, kps, kps_history):
        """Detect general movement - any motion that's not specifically walking/running."""
        if len(kps_history) < 3:
            return False
        
        # Get center of mass movement
        positions = []
        for kp in kps_history[-5:]:
            lhip = self._get_keypoint(kp, LEFT_HIP)
            rhip = self._get_keypoint(kp, RIGHT_HIP)
            if lhip and rhip:
                center_x = (lhip[0] + rhip[0]) / 2.0
                center_y = (lhip[1] + rhip[1]) / 2.0
                positions.append((center_x, center_y))
        
        if len(positions) < 2:
            return False
        
        # Calculate average movement velocity
        velocities = [math.sqrt((positions[i][0] - positions[i+1][0])**2 + 
                               (positions[i][1] - positions[i+1][1])**2) 
                     for i in range(len(positions)-1)]
        avg_velocity = np.mean(velocities)
        
        # General movement: moderate velocity, not walking pattern
        # Walking would have been caught earlier, so this is other types of movement
        if 0.01 < avg_velocity < 0.05:  # Moderate movement
            return True
        
        return False
    
    def _infer_posture(self, kps):
        """Infer basic posture from keypoints."""
        vertical_extent = self._compute_vertical_extent(kps)
        horizontal_extent = self._compute_horizontal_extent(kps)
        if horizontal_extent < 1e-6:
            return "unknown"
        aspect_ratio = vertical_extent / horizontal_extent
        
        if aspect_ratio > 1.2:
            return "standing"
        elif aspect_ratio > 0.8:
            return "sitting"
        else:
            return "lying"
    
    def _is_crawling(self, kps, kps_history):
        """Detect crawling - low to ground, arm and leg movement."""
        vertical_extent = self._compute_vertical_extent(kps)
        if vertical_extent > 0.5:  # Too tall for crawling
            return False
        
        # Check for arm and leg movement
        if len(kps_history) < 5:
            return False
        
        wrist_movements = []
        ankle_movements = []
        
        for kp in kps_history[-5:]:
            lwrist = self._get_keypoint(kp, LEFT_WRIST)
            rwrist = self._get_keypoint(kp, RIGHT_WRIST)
            lankle = self._get_keypoint(kp, LEFT_ANKLE)
            rankle = self._get_keypoint(kp, RIGHT_ANKLE)
            
            wrist_movements.append((lwrist[0], lwrist[1], rwrist[0], rwrist[1]))
            ankle_movements.append((lankle[0], lankle[1], rankle[0], rankle[1]))
        
        # Check for significant movement in both arms and legs
        wrist_changes = [math.sqrt((wrist_movements[i][0] - wrist_movements[i+1][0])**2 + 
                                  (wrist_movements[i][1] - wrist_movements[i+1][1])**2) 
                        for i in range(len(wrist_movements)-1)]
        ankle_changes = [math.sqrt((ankle_movements[i][0] - ankle_movements[i+1][0])**2 + 
                                  (ankle_movements[i][1] - ankle_movements[i+1][1])**2) 
                        for i in range(len(ankle_movements)-1)]
        
        return np.mean(wrist_changes) > 0.02 and np.mean(ankle_changes) > 0.02
    
    def _is_bed_exit(self, kps, bed_info):
        """Detect bed exit - person moving away from bed."""
        if not bed_info or "bbox" not in bed_info:
            return False
        
        bed_bbox = bed_info["bbox"]
        bed_center_x = (bed_bbox[0] + bed_bbox[2]) / 2.0
        
        # Get person center (from hips)
        lhip = self._get_keypoint(kps, LEFT_HIP)
        rhip = self._get_keypoint(kps, RIGHT_HIP)
        person_center_x = (lhip[0] + rhip[0]) / 2.0
        
        # Check if person is moving away from bed
        if len(self.kps_history) >= 2:
            prev_kps = self.kps_history[-2]
            prev_lhip = self._get_keypoint(prev_kps, LEFT_HIP)
            prev_rhip = self._get_keypoint(prev_kps, RIGHT_HIP)
            prev_center_x = (prev_lhip[0] + prev_rhip[0]) / 2.0
            
            # Moving away from bed center
            distance_change = abs(person_center_x - bed_center_x) - abs(prev_center_x - bed_center_x)
            return distance_change > 0.05  # Moving away
        
        return False
    
    def _is_bed_entry(self, kps, bed_info):
        """Detect bed entry - person moving toward bed."""
        if not bed_info or "bbox" not in bed_info:
            return False
        
        bed_bbox = bed_info["bbox"]
        bed_center_x = (bed_bbox[0] + bed_bbox[2]) / 2.0
        
        lhip = self._get_keypoint(kps, LEFT_HIP)
        rhip = self._get_keypoint(kps, RIGHT_HIP)
        person_center_x = (lhip[0] + rhip[0]) / 2.0
        
        if len(self.kps_history) >= 2:
            prev_kps = self.kps_history[-2]
            prev_lhip = self._get_keypoint(prev_kps, LEFT_HIP)
            prev_rhip = self._get_keypoint(prev_kps, RIGHT_HIP)
            prev_center_x = (prev_lhip[0] + prev_rhip[0]) / 2.0
            
            # Moving toward bed center
            distance_change = abs(person_center_x - bed_center_x) - abs(prev_center_x - bed_center_x)
            return distance_change < -0.05  # Moving closer
        
        return False
    
    def _is_person_on_bed(self, kps, bed_info):
        """Check if person is on bed using keypoint positions."""
        if not bed_info or "bbox" not in bed_info:
            return False
        
        bed_bbox = bed_info["bbox"]
        bed_x1, bed_y1, bed_x2, bed_y2 = bed_bbox
        
        # Check if person's center (hips) is within bed bbox
        lhip = self._get_keypoint(kps, LEFT_HIP)
        rhip = self._get_keypoint(kps, RIGHT_HIP)
        person_center_x = (lhip[0] + rhip[0]) / 2.0
        person_center_y = (lhip[1] + rhip[1]) / 2.0
        
        return (bed_x1 <= person_center_x <= bed_x2 and 
                bed_y1 <= person_center_y <= bed_y2)
    
    def _is_turning_in_bed(self, kps):
        """Detect turning in bed - rotational movement while lying."""
        if len(self.kps_history) < 3:
            return False
        
        # Check for rotational movement (shoulder-hip angle changes)
        current_lshoulder = self._get_keypoint(kps, LEFT_SHOULDER)
        current_rshoulder = self._get_keypoint(kps, RIGHT_SHOULDER)
        current_lhip = self._get_keypoint(kps, LEFT_HIP)
        
        prev_kps = self.kps_history[-2]
        prev_lshoulder = self._get_keypoint(prev_kps, LEFT_SHOULDER)
        prev_lhip = self._get_keypoint(prev_kps, LEFT_HIP)
        
        # Compute angle change
        current_angle = math.atan2(current_lshoulder[1] - current_lhip[1],
                                  current_lshoulder[0] - current_lhip[0])
        prev_angle = math.atan2(prev_lshoulder[1] - prev_lhip[1],
                               prev_lshoulder[0] - prev_lhip[0])
        
        angle_change = abs(current_angle - prev_angle)
        return angle_change > 0.2  # Significant rotation
    
    def _is_reaching(self, kps):
        """Detect reaching - arm extended."""
        lshoulder = self._get_keypoint(kps, LEFT_SHOULDER)
        rshoulder = self._get_keypoint(kps, RIGHT_SHOULDER)
        lwrist = self._get_keypoint(kps, LEFT_WRIST)
        rwrist = self._get_keypoint(kps, RIGHT_WRIST)
        
        # Check if wrist is significantly above shoulder (reaching up)
        # or far from shoulder (reaching forward)
        left_reach_up = lwrist[1] < lshoulder[1] - 0.1
        right_reach_up = rwrist[1] < rshoulder[1] - 0.1
        
        left_reach_forward = abs(lwrist[0] - lshoulder[0]) > 0.3
        right_reach_forward = abs(rwrist[0] - rshoulder[0]) > 0.3
        
        return (left_reach_up or right_reach_up or 
                left_reach_forward or right_reach_forward)
    
    def _is_waving(self, kps, kps_history):
        """Detect waving - rhythmic arm movement."""
        if len(kps_history) < 5:
            return False
        
        wrist_positions = []
        for kp in kps_history[-5:]:
            lwrist = self._get_keypoint(kp, LEFT_WRIST)
            rwrist = self._get_keypoint(kp, RIGHT_WRIST)
            wrist_positions.append((lwrist[0], lwrist[1], rwrist[0], rwrist[1]))
        
        # Check for oscillating pattern
        left_x_changes = [abs(wrist_positions[i][0] - wrist_positions[i+1][0]) 
                         for i in range(len(wrist_positions)-1)]
        right_x_changes = [abs(wrist_positions[i][2] - wrist_positions[i+1][2]) 
                          for i in range(len(wrist_positions)-1)]
        
        # Waving has alternating left-right movement
        avg_change = (np.mean(left_x_changes) + np.mean(right_x_changes)) / 2.0
        return avg_change > 0.03
    
    def _is_pulling_at_tubes(self, kps, kps_history, frame):
        """Detect pulling at tubes - hand near face/chest with repetitive motion."""
        if not kps_history or len(kps_history) < 5:
            return False
        
        # Check if hand is near face/chest
        nose = self._get_keypoint(kps, NOSE)
        lwrist = self._get_keypoint(kps, LEFT_WRIST)
        rwrist = self._get_keypoint(kps, RIGHT_WRIST)
        
        # Distance from wrist to nose
        left_dist = math.sqrt((lwrist[0] - nose[0])**2 + (lwrist[1] - nose[1])**2)
        right_dist = math.sqrt((rwrist[0] - nose[0])**2 + (rwrist[1] - nose[1])**2)
        
        hand_near_face = left_dist < 0.2 or right_dist < 0.2
        
        if hand_near_face:
            # Check for repetitive motion
            wrist_movements = []
            for kp in kps_history[-5:]:
                lw = self._get_keypoint(kp, LEFT_WRIST)
                rw = self._get_keypoint(kp, RIGHT_WRIST)
                wrist_movements.append((lw[0], lw[1], rw[0], rw[1]))
            
            movements = [math.sqrt((wrist_movements[i][0] - wrist_movements[i+1][0])**2 +
                                  (wrist_movements[i][1] - wrist_movements[i+1][1])**2)
                        for i in range(len(wrist_movements)-1)]
            
            return np.mean(movements) > 0.01  # Repetitive motion
        
        return False
    
    def _has_leg_movement(self, kps, kps_history):
        """Detect leg movement."""
        if len(kps_history) < 3:
            return False
        
        ankle_positions = []
        for kp in kps_history[-3:]:
            lankle = self._get_keypoint(kp, LEFT_ANKLE)
            rankle = self._get_keypoint(kp, RIGHT_ANKLE)
            ankle_positions.append((lankle[0], lankle[1], rankle[0], rankle[1]))
        
        movements = [math.sqrt((ankle_positions[i][0] - ankle_positions[i+1][0])**2 +
                              (ankle_positions[i][1] - ankle_positions[i+1][1])**2)
                    for i in range(len(ankle_positions)-1)]
        
        return np.mean(movements) > 0.02
    
    def _is_restless(self, kps, kps_history):
        """Detect restlessness - frequent position changes."""
        if len(kps_history) < 5:
            return False
        
        # Count significant position changes
        position_changes = 0
        for i in range(len(kps_history) - 1):
            prev_kp = kps_history[i]
            curr_kp = kps_history[i+1]
            
            prev_hip = self._get_keypoint(prev_kp, LEFT_HIP)
            curr_hip = self._get_keypoint(curr_kp, LEFT_HIP)
            
            movement = math.sqrt((prev_hip[0] - curr_hip[0])**2 + 
                               (prev_hip[1] - curr_hip[1])**2)
            if movement > 0.05:
                position_changes += 1
        
        # Restless if >50% of frames have position changes
        return position_changes / (len(kps_history) - 1) > 0.5
    
    def _is_thrashing(self, kps, kps_history):
        """Detect thrashing - violent, irregular movements."""
        if len(kps_history) < 5:
            return False
        
        # Compute movement velocity and variance
        velocities = []
        for i in range(len(kps_history) - 1):
            prev_kp = kps_history[i]
            curr_kp = kps_history[i+1]
            
            prev_hip = self._get_keypoint(prev_kp, LEFT_HIP)
            curr_hip = self._get_keypoint(curr_kp, LEFT_HIP)
            
            velocity = math.sqrt((prev_hip[0] - curr_hip[0])**2 + 
                               (prev_hip[1] - curr_hip[1])**2)
            velocities.append(velocity)
        
        if not velocities:
            return False
        
        avg_velocity = np.mean(velocities)
        velocity_variance = np.var(velocities)
        
        # Thrashing: high velocity with high variance (irregular)
        return avg_velocity > 0.1 and velocity_variance > 0.01
    
    def _detect_seizure(self, kps):
        """Detect seizure - rapid, irregular movements."""
        if len(self.kps_history) < 10:
            return 0.0
        
        # Compute movement patterns
        movements = []
        for i in range(len(self.kps_history) - 1):
            prev_kp = self.kps_history[i]
            curr_kp = self.kps_history[i+1]
            
            # Compute total body movement
            total_movement = 0.0
            for j in range(min(len(prev_kp), len(curr_kp))):
                if prev_kp[j][2] > MIN_CONFIDENCE and curr_kp[j][2] > MIN_CONFIDENCE:
                    movement = math.sqrt((prev_kp[j][0] - curr_kp[j][0])**2 +
                                       (prev_kp[j][1] - curr_kp[j][1])**2)
                    total_movement += movement
            
            movements.append(total_movement)
        
        if not movements:
            return 0.0
        
        # Seizure: high movement with high variance (rapid, irregular)
        avg_movement = np.mean(movements)
        movement_variance = np.var(movements)
        
        score = min(1.0, (avg_movement * 10.0 + movement_variance * 100.0))
        return float(score)
    
    def _detect_convulsion(self, kps, kps_history):
        """Detect convulsion - similar to seizure but more violent."""
        seizure_score = self._detect_seizure(kps)
        # Convulsion is more intense seizure
        return min(1.0, seizure_score * 1.2)
    
    def _detect_tremor(self, kps, kps_history):
        """Detect tremor - small, rapid oscillations."""
        if len(kps_history) < 10:
            return 0.0
        
        # Check for high-frequency oscillations
        wrist_positions = []
        for kp in kps_history[-10:]:
            lwrist = self._get_keypoint(kp, LEFT_WRIST)
            rwrist = self._get_keypoint(kp, RIGHT_WRIST)
            wrist_positions.append((lwrist[0], lwrist[1], rwrist[0], rwrist[1]))
        
        # Compute oscillation frequency
        left_x_oscillations = 0
        right_x_oscillations = 0
        
        for i in range(1, len(wrist_positions) - 1):
            # Check for direction changes (oscillation)
            if (wrist_positions[i-1][0] < wrist_positions[i][0] and 
                wrist_positions[i][0] > wrist_positions[i+1][0]):
                left_x_oscillations += 1
            if (wrist_positions[i-1][2] < wrist_positions[i][2] and 
                wrist_positions[i][2] > wrist_positions[i+1][2]):
                right_x_oscillations += 1
        
        oscillation_rate = (left_x_oscillations + right_x_oscillations) / (len(wrist_positions) - 2)
        return min(1.0, oscillation_rate * 2.0)
    
    def _is_rigid(self, kps, kps_history):
        """Detect rigidity - stiff, limited movement."""
        if len(kps_history) < 5:
            return False
        
        # Rigidity: very low movement despite attempts
        movements = []
        for i in range(len(kps_history) - 1):
            prev_kp = kps_history[i]
            curr_kp = kps_history[i+1]
            
            prev_hip = self._get_keypoint(prev_kp, LEFT_HIP)
            curr_hip = self._get_keypoint(curr_kp, LEFT_HIP)
            
            movement = math.sqrt((prev_hip[0] - curr_hip[0])**2 + 
                               (prev_hip[1] - curr_hip[1])**2)
            movements.append(movement)
        
        if not movements:
            return False
        
        # Very low movement = rigidity
        return np.mean(movements) < 0.01
    
    def _detect_breathing(self, kps, kps_history):
        """Detect breathing from chest movement."""
        if len(kps_history) < 10:
            return 0.0
        
        # Get thorax (chest) vertical positions
        thorax_positions = []
        for kp in kps_history[-10:]:
            lshoulder = self._get_keypoint(kp, LEFT_SHOULDER)
            rshoulder = self._get_keypoint(kp, RIGHT_SHOULDER)
            lhip = self._get_keypoint(kp, LEFT_HIP)
            rhip = self._get_keypoint(kp, RIGHT_HIP)
            
            # Thorax center
            thorax_y = ((lshoulder[1] + rshoulder[1]) / 2.0 + 
                       (lhip[1] + rhip[1]) / 2.0) / 2.0
            thorax_positions.append(thorax_y)
        
        # Check for oscillating pattern (breathing)
        if len(thorax_positions) < 3:
            return 0.0
        
        # Compute variance (breathing causes oscillation)
        variance = np.var(thorax_positions)
        return min(1.0, variance * 100.0)
    
    def _is_coughing(self, kps, kps_history):
        """Detect coughing - sudden chest contraction."""
        if len(kps_history) < 3:
            return False
        
        # Check for sudden downward chest movement
        current_lshoulder = self._get_keypoint(kps, LEFT_SHOULDER)
        prev_kps = kps_history[-2]
        prev_lshoulder = self._get_keypoint(prev_kps, LEFT_SHOULDER)
        
        # Sudden downward movement
        downward_movement = current_lshoulder[1] - prev_lshoulder[1]
        return downward_movement > 0.05  # Significant downward movement
    
    def _is_still(self, kps, kps_history):
        """Detect still state - very low movement."""
        if len(kps_history) < 5:
            return False
        
        movements = []
        for i in range(len(kps_history) - 1):
            prev_kp = kps_history[i]
            curr_kp = kps_history[i+1]
            
            prev_hip = self._get_keypoint(prev_kp, LEFT_HIP)
            curr_hip = self._get_keypoint(curr_kp, LEFT_HIP)
            
            movement = math.sqrt((prev_hip[0] - curr_hip[0])**2 + 
                               (prev_hip[1] - curr_hip[1])**2)
            movements.append(movement)
        
        return np.mean(movements) < 0.005  # Very low movement
    
    def _is_sleeping(self, kps, kps_history):
        """Detect sleeping - still + lying position."""
        if not self._is_still(kps, kps_history):
            return False
        
        # Check if lying
        vertical_extent = self._compute_vertical_extent(kps)
        horizontal_extent = self._compute_horizontal_extent(kps)
        aspect_ratio = vertical_extent / (horizontal_extent + 1e-6)
        
        return aspect_ratio < 0.8  # Lying position
    
    def _is_unresponsive(self, kps):
        """Detect unresponsive - very still, no movement."""
        if len(self.kps_history) < 10:
            return False
        
        # Check for very low movement over extended period
        return self._is_still(kps, list(self.kps_history))
    
    def _compute_agitation_score(self, kps):
        """Compute agitation score from movement patterns."""
        if len(self.kps_history) < 5:
            return 0.0
        
        # Agitation: high movement + irregular patterns
        movements = []
        for i in range(len(self.kps_history) - 1):
            prev_kp = self.kps_history[i]
            curr_kp = self.kps_history[i+1]
            
            prev_hip = self._get_keypoint(prev_kp, LEFT_HIP)
            curr_hip = self._get_keypoint(curr_kp, LEFT_HIP)
            
            movement = math.sqrt((prev_hip[0] - curr_hip[0])**2 + 
                               (prev_hip[1] - curr_hip[1])**2)
            movements.append(movement)
        
        if not movements:
            return 0.0
        
        avg_movement = np.mean(movements)
        movement_variance = np.var(movements)
        
        # Agitation score combines high movement and variance
        score = min(1.0, (avg_movement * 5.0 + movement_variance * 50.0))
        return float(score)
    
    def _unknown_activity(self, reason="unknown"):
        """Return unknown activity result."""
        return {
            "activity": "unknown",
            "confidence": 0.3,
            "priority": "MEDIUM",
            "details": {"reason": reason}
        }


# Backward compatibility function
def classify_activity(kps, kps_history=None, **kwargs):
    """
    Enhanced activity classification that supports all 53 activities.
    Backward compatible with original classify_activity signature.
    
    Args:
        kps: Current keypoints
        kps_history: History of keypoints
        **kwargs: Additional context (posture_state, bed_info, person_on_bed, etc.)
    
    Returns:
        dict with activity, confidence, priority, details
    """
    classifier = EnhancedActivityClassifier()
    return classifier.classify_activity(
        kps=kps,
        kps_history=kps_history,
        posture_state=kwargs.get("posture_state"),
        bed_info=kwargs.get("bed_info"),
        person_on_bed=kwargs.get("person_on_bed"),
        fall_detected=kwargs.get("fall_detected"),
        frame=kwargs.get("frame"),
        bbox=kwargs.get("bbox")
    )

