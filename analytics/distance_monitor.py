# analytics/distance_monitor.py
# Distance Monitoring and Patient Feedback Module
# Monitors patient distance from camera and provides feedback for optimal positioning

import numpy as np
import logging
import math
from typing import Dict, Optional, Tuple, List

log = logging.getLogger("distance_monitor")

# Distance thresholds (in meters)
OPTIMAL_DISTANCE_MIN = 1.5  # 150cm minimum
OPTIMAL_DISTANCE_MAX = 3.0  # 300cm maximum
OPTIMAL_DISTANCE_TARGET = 2.0  # 200cm target
TOO_CLOSE_THRESHOLD = 1.0  # 100cm - too close
TOO_FAR_THRESHOLD = 4.0  # 400cm - too far


class DistanceMonitor:
    """
    Monitors patient distance from camera and provides feedback.
    Ensures optimal distance for accurate pose estimation and activity monitoring.
    """
    
    def __init__(self, 
                 optimal_min: float = OPTIMAL_DISTANCE_MIN,
                 optimal_max: float = OPTIMAL_DISTANCE_MAX,
                 target: float = OPTIMAL_DISTANCE_TARGET,
                 too_close: float = TOO_CLOSE_THRESHOLD,
                 too_far: float = TOO_FAR_THRESHOLD):
        """
        Initialize distance monitor.
        
        Args:
            optimal_min: Minimum optimal distance in meters
            optimal_max: Maximum optimal distance in meters
            target: Target distance in meters
            too_close: Threshold for "too close" in meters
            too_far: Threshold for "too far" in meters
        """
        self.optimal_min = optimal_min
        self.optimal_max = optimal_max
        self.target = target
        self.too_close = too_close
        self.too_far = too_far
        
        # Distance history for smoothing
        self.distance_history = []
        self.history_size = 10
        
        # Feedback state
        self.last_feedback_time = 0
        self.feedback_interval = 5.0  # Don't spam feedback (every 5 seconds max)
        self.feedback_count = 0
    
    def estimate_distance_from_keypoints(self, kps: List, bbox: Optional[List] = None, 
                                        frame_shape: Optional[Tuple] = None) -> float:
        """
        Estimate distance from camera using keypoint size and bbox.
        
        Args:
            kps: 2D keypoints
            bbox: Bounding box [x1, y1, x2, y2]
            frame_shape: (height, width) of frame
        
        Returns:
            Estimated distance in meters
        """
        if not kps or len(kps) < 5:
            return 0.0
        
        try:
            # Method 1: Use bbox size (most reliable)
            if bbox and len(bbox) >= 4 and frame_shape:
                h, w = frame_shape[:2]
                x1, y1, x2, y2 = bbox[:4]
                
                # Calculate bbox dimensions
                bbox_width = abs(x2 - x1)
                bbox_height = abs(y2 - y1)
                bbox_area = bbox_width * bbox_height
                frame_area = w * h
                
                # Normalize bbox area
                area_ratio = bbox_area / frame_area if frame_area > 0 else 0
                
                # Estimate distance based on area ratio
                # Larger area = closer, smaller area = farther
                # Calibrated for typical person size (1.7m tall, 0.5m wide)
                # At 2m distance, person should occupy ~30-40% of frame
                if area_ratio > 0.5:  # Very large = very close
                    distance = 1.0 + (0.5 - area_ratio) * 2.0  # 0.5-1.5m range
                elif area_ratio > 0.2:  # Medium = optimal range
                    distance = 1.5 + (0.2 - area_ratio) * 5.0  # 1.5-3.0m range
                else:  # Small = far
                    distance = 3.0 + (0.2 - area_ratio) * 10.0  # 3.0-5.0m range
                
                return float(np.clip(distance, 0.5, 6.0))
            
            # Method 2: Use keypoint spread (fallback)
            visible_kps = [kp for kp in kps if len(kp) >= 3 and kp[2] > 0.3]
            if len(visible_kps) < 5:
                return 0.0
            
            # Calculate keypoint spread
            x_coords = [kp[0] for kp in visible_kps]
            y_coords = [kp[1] for kp in visible_kps]
            
            x_spread = max(x_coords) - min(x_coords)
            y_spread = max(y_coords) - min(y_coords)
            total_spread = math.sqrt(x_spread**2 + y_spread**2)
            
            # Estimate distance from spread
            # Larger spread = closer
            if total_spread > 0.6:
                distance = 1.0 + (0.6 - total_spread) * 2.0
            elif total_spread > 0.3:
                distance = 1.5 + (0.3 - total_spread) * 5.0
            else:
                distance = 3.0 + (0.3 - total_spread) * 10.0
            
            return float(np.clip(distance, 0.5, 6.0))
            
        except Exception as e:
            log.debug("Distance estimation error: %s", e)
            return 0.0
    
    def estimate_distance_from_3d(self, kps_3d: List) -> float:
        """
        Estimate distance from 3D keypoints (most accurate).
        
        Args:
            kps_3d: 3D keypoints with depth (z coordinate)
        
        Returns:
            Estimated distance in meters
        """
        if not kps_3d:
            return 0.0
        
        try:
            # Use torso keypoints (shoulders, hips) for distance
            # These are most reliable for distance measurement
            torso_depths = []
            for kp in kps_3d:
                if len(kp) >= 3:
                    z = kp[2]  # Depth (z coordinate)
                    conf = kp[3] if len(kp) > 3 else 1.0
                    if conf > 0.3:  # Only use confident keypoints
                        torso_depths.append(z)
            
            if torso_depths:
                distance = float(np.median(torso_depths))  # Use median for robustness
                return distance
            return 0.0
            
        except Exception as e:
            log.debug("3D distance estimation error: %s", e)
            return 0.0
    
    def check_distance(self, distance: float) -> Dict:
        """
        Check if distance is optimal and provide feedback.
        
        Args:
            distance: Current distance in meters
        
        Returns:
            dict with:
            - status: "optimal" | "too_close" | "too_far" | "close_to_optimal" | "far_from_optimal"
            - distance: Current distance in meters
            - distance_cm: Distance in centimeters
            - feedback_message: Human-readable feedback
            - needs_adjustment: bool
            - adjustment_direction: "move_back" | "move_forward" | "optimal"
        """
        if distance <= 0:
            return {
                "status": "unknown",
                "distance": 0.0,
                "distance_cm": 0,
                "feedback_message": "Distance cannot be determined",
                "needs_adjustment": False,
                "adjustment_direction": "optimal"
            }
        
        distance_cm = int(distance * 100)
        
        # Update history for smoothing
        self.distance_history.append(distance)
        if len(self.distance_history) > self.history_size:
            self.distance_history.pop(0)
        
        # Use smoothed distance (median of history)
        if len(self.distance_history) >= 3:
            smoothed_distance = float(np.median(self.distance_history))
        else:
            smoothed_distance = distance
        
        # Determine status
        if smoothed_distance < self.too_close:
            status = "too_close"
            feedback = f"Too close to camera ({distance_cm}cm). Please move back to approximately {int(self.target * 100)}cm."
            needs_adjustment = True
            direction = "move_back"
        elif smoothed_distance < self.optimal_min:
            status = "close_to_optimal"
            feedback = f"Close to optimal range ({distance_cm}cm). Move back slightly to {int(self.target * 100)}cm for best results."
            needs_adjustment = True
            direction = "move_back"
        elif smoothed_distance <= self.optimal_max:
            status = "optimal"
            feedback = f"Distance is optimal ({distance_cm}cm). Perfect for monitoring."
            needs_adjustment = False
            direction = "optimal"
        elif smoothed_distance < self.too_far:
            status = "far_from_optimal"
            feedback = f"Far from optimal range ({distance_cm}cm). Move forward to {int(self.target * 100)}cm for best results."
            needs_adjustment = True
            direction = "move_forward"
        else:
            status = "too_far"
            feedback = f"Too far from camera ({distance_cm}cm). Please move forward to approximately {int(self.target * 100)}cm."
            needs_adjustment = True
            direction = "move_forward"
        
        return {
            "status": status,
            "distance": float(smoothed_distance),
            "distance_cm": int(smoothed_distance * 100),
            "feedback_message": feedback,
            "needs_adjustment": needs_adjustment,
            "adjustment_direction": direction,
            "target_distance_cm": int(self.target * 100)
        }
    
    def get_feedback(self, distance: float, force: bool = False) -> Optional[Dict]:
        """
        Get feedback message if adjustment is needed.
        Throttles feedback to avoid spamming.
        
        Args:
            distance: Current distance in meters
            force: Force feedback even if recently sent
        
        Returns:
            Feedback dict or None if throttled
        """
        import time
        
        check_result = self.check_distance(distance)
        
        if not check_result["needs_adjustment"]:
            return None
        
        # Throttle feedback
        current_time = time.time()
        if not force and (current_time - self.last_feedback_time) < self.feedback_interval:
            return None
        
        self.last_feedback_time = current_time
        self.feedback_count += 1
        
        return {
            "message": check_result["feedback_message"],
            "status": check_result["status"],
            "distance_cm": check_result["distance_cm"],
            "target_cm": check_result["target_distance_cm"],
            "direction": check_result["adjustment_direction"],
            "timestamp": current_time
        }

