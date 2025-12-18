# analytics/fall_detection.py
"""
Fall Detection Algorithm
Detects patient falls using pose estimation and movement analysis.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

log = logging.getLogger("fall_detection")


class FallDetector:
    """
    Detects patient falls using multiple indicators:
    1. Rapid downward movement
    2. Low vertical position (near ground)
    3. Sudden posture change
    4. Movement velocity analysis
    """
    
    def __init__(self, 
                 downward_velocity_threshold: float = 0.3,
                 ground_level_threshold: float = 0.8,
                 min_confidence: float = 0.7):
        """
        Initialize fall detector.
        
        Args:
            downward_velocity_threshold: Minimum downward velocity to consider (normalized)
            ground_level_threshold: Y position threshold for ground level (normalized, 0-1)
            min_confidence: Minimum confidence for fall detection
        """
        self.downward_velocity_threshold = downward_velocity_threshold
        self.ground_level_threshold = ground_level_threshold
        self.min_confidence = min_confidence
        
        # History for velocity calculation
        self.position_history = []
        self.max_history = 10
    
    def detect_fall(self, 
                   kps: List,
                   kps_history: List[List],
                   posture_state: str,
                   frame_shape: Tuple[int, int]) -> Dict:
        """
        Detect if patient has fallen.
        
        Args:
            kps: Current keypoints
            kps_history: History of keypoints (last N frames)
            posture_state: Current posture state
            frame_shape: (height, width) of frame
        
        Returns:
            Dictionary with fall detection result
        """
        if not kps or not kps_history:
            return {
                'fall_detected': False,
                'confidence': 0.0,
                'indicators': []
            }
        
        indicators = []
        confidence = 0.0
        
        # Indicator 1: Rapid downward movement
        downward_movement = self._check_downward_movement(kps, kps_history, frame_shape)
        if downward_movement['detected']:
            indicators.append({
                'type': 'rapid_downward_movement',
                'velocity': downward_movement['velocity'],
                'confidence': downward_movement['confidence']
            })
            confidence += 0.4
        
        # Indicator 2: Low vertical position (near ground)
        ground_position = self._check_ground_position(kps, frame_shape)
        if ground_position['detected']:
            indicators.append({
                'type': 'ground_level_position',
                'y_position': ground_position['y_position'],
                'confidence': ground_position['confidence']
            })
            confidence += 0.3
        
        # Indicator 3: Sudden posture change
        posture_change = self._check_posture_change(posture_state, kps_history)
        if posture_change['detected']:
            indicators.append({
                'type': 'sudden_posture_change',
                'from': posture_change['from'],
                'to': posture_change['to'],
                'confidence': posture_change['confidence']
            })
            confidence += 0.2
        
        # Indicator 4: Horizontal spread (lying on ground)
        horizontal_spread = self._check_horizontal_spread(kps, frame_shape)
        if horizontal_spread['detected']:
            indicators.append({
                'type': 'horizontal_spread',
                'spread_ratio': horizontal_spread['spread_ratio'],
                'confidence': horizontal_spread['confidence']
            })
            confidence += 0.1
        
        # Normalize confidence
        confidence = min(confidence, 1.0)
        
        fall_detected = confidence >= self.min_confidence and len(indicators) >= 2
        
        return {
            'fall_detected': fall_detected,
            'confidence': confidence,
            'indicators': indicators,
            'timestamp': None  # Will be set by caller
        }
    
    def _check_downward_movement(self, kps: List, kps_history: List[List], frame_shape: Tuple[int, int]) -> Dict:
        """Check for rapid downward movement."""
        if len(kps_history) < 2:
            return {'detected': False, 'velocity': 0.0, 'confidence': 0.0}
        
        # Calculate center of mass (average Y position of keypoints)
        def get_center_y(kps_list):
            valid_kps = [kp for kp in kps_list if kp and len(kp) >= 2]
            if not valid_kps:
                return None
            return np.mean([kp[1] for kp in valid_kps])
        
        current_y = get_center_y(kps)
        previous_y = get_center_y(kps_history[-1]) if kps_history else None
        
        if current_y is None or previous_y is None:
            return {'detected': False, 'velocity': 0.0, 'confidence': 0.0}
        
        # Normalize to frame height
        frame_height = frame_shape[0]
        current_y_norm = current_y / frame_height
        previous_y_norm = previous_y / frame_height
        
        # Calculate velocity (downward is positive)
        velocity = current_y_norm - previous_y_norm
        
        if velocity > self.downward_velocity_threshold:
            # Strong downward movement detected
            confidence = min(velocity / self.downward_velocity_threshold, 1.0)
            return {
                'detected': True,
                'velocity': velocity,
                'confidence': confidence
            }
        
        return {'detected': False, 'velocity': velocity, 'confidence': 0.0}
    
    def _check_ground_position(self, kps: List, frame_shape: Tuple[int, int]) -> Dict:
        """Check if patient is near ground level."""
        valid_kps = [kp for kp in kps if kp and len(kp) >= 2]
        if not valid_kps:
            return {'detected': False, 'y_position': 0.0, 'confidence': 0.0}
        
        # Get average Y position
        avg_y = np.mean([kp[1] for kp in valid_kps])
        y_position_norm = avg_y / frame_shape[0]
        
        if y_position_norm > self.ground_level_threshold:
            # Near ground level
            confidence = (y_position_norm - self.ground_level_threshold) / (1.0 - self.ground_level_threshold)
            confidence = min(confidence, 1.0)
            return {
                'detected': True,
                'y_position': y_position_norm,
                'confidence': confidence
            }
        
        return {'detected': False, 'y_position': y_position_norm, 'confidence': 0.0}
    
    def _check_posture_change(self, current_posture: str, kps_history: List[List]) -> Dict:
        """Check for sudden posture change (e.g., standing -> lying)."""
        # This would require posture history, simplified for now
        # In full implementation, would track posture over time
        
        # Check if current posture is "lying" or "unknown" after movement
        if current_posture in ['supine', 'prone', 'side', 'unknown']:
            # Could indicate fall if previous was upright
            return {
                'detected': True,
                'from': 'upright',
                'to': current_posture,
                'confidence': 0.5
            }
        
        return {'detected': False, 'from': None, 'to': None, 'confidence': 0.0}
    
    def _check_horizontal_spread(self, kps: List, frame_shape: Tuple[int, int]) -> Dict:
        """Check if patient is spread horizontally (lying on ground)."""
        valid_kps = [kp for kp in kps if kp and len(kp) >= 2]
        if len(valid_kps) < 5:
            return {'detected': False, 'spread_ratio': 0.0, 'confidence': 0.0}
        
        # Calculate bounding box
        xs = [kp[0] for kp in valid_kps]
        ys = [kp[1] for kp in valid_kps]
        
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        if height == 0:
            return {'detected': False, 'spread_ratio': 0.0, 'confidence': 0.0}
        
        # Horizontal spread ratio (width/height)
        spread_ratio = width / height
        
        # If spread ratio is high, patient is lying horizontally
        if spread_ratio > 1.5:  # Wider than tall
            confidence = min((spread_ratio - 1.5) / 1.0, 1.0)
            return {
                'detected': True,
                'spread_ratio': spread_ratio,
                'confidence': confidence
            }
        
        return {'detected': False, 'spread_ratio': spread_ratio, 'confidence': 0.0}


def detect_patient_fall(kps: List,
                       kps_history: List[List],
                       posture_state: str,
                       frame_shape: Tuple[int, int]) -> Dict:
    """
    Convenience function for fall detection.
    
    Args:
        kps: Current keypoints
        kps_history: History of keypoints
        posture_state: Current posture state
        frame_shape: (height, width) of frame
    
    Returns:
        Fall detection result dictionary
    """
    detector = FallDetector()
    result = detector.detect_fall(kps, kps_history, posture_state, frame_shape)
    return result

