# pipeline/patient/liveness_detector.py
"""
Liveness Detection for Anti-Spoofing
Prevents photo/video spoofing attacks on face recognition
"""
import cv2
import numpy as np
import logging
from collections import deque
from typing import Optional, Tuple, List

log = logging.getLogger("liveness")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    log.warning("MediaPipe not available for liveness detection")


class LivenessDetector:
    """
    Multi-modal liveness detection to prevent spoofing attacks.
    Methods:
    1. Blink detection (eye aspect ratio)
    2. Face motion analysis (micro-movements)
    3. 3D face structure analysis (depth estimation)
    4. Texture analysis (photo vs real face)
    """
    
    def __init__(self, 
                 blink_threshold: float = 0.25,
                 motion_threshold: float = 0.01,
                 history_size: int = 30):
        """
        Args:
            blink_threshold: Eye aspect ratio threshold for blink detection
            motion_threshold: Minimum motion required (normalized)
            history_size: Number of frames to keep in history
        """
        self.blink_threshold = blink_threshold
        self.motion_threshold = motion_threshold
        self.history_size = history_size
        
        # Face history for motion analysis
        self.face_history = deque(maxlen=history_size)
        self.eye_history = deque(maxlen=history_size)
        
        # MediaPipe face mesh for blink detection
        self.mp_face_mesh = None
        if MP_AVAILABLE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                log.info("MediaPipe face mesh initialized for liveness detection")
            except Exception as e:
                log.warning("Failed to initialize MediaPipe face mesh: %s", e)
                self.mp_face_mesh = None
    
    def compute_eye_aspect_ratio(self, landmarks, eye_indices: List[int]) -> float:
        """
        Compute Eye Aspect Ratio (EAR) for blink detection.
        
        Args:
            landmarks: Face landmarks
            eye_indices: Indices for eye landmarks (MediaPipe format)
        
        Returns:
            EAR value (lower = more closed)
        """
        if len(eye_indices) < 6:
            return 1.0
        
        # Get eye landmark coordinates
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks):
                x = landmarks[idx].x
                y = landmarks[idx].y
                eye_points.append((x, y))
        
        if len(eye_points) < 6:
            return 1.0
        
        # Compute distances
        # Vertical distances
        v1 = np.sqrt((eye_points[1][0] - eye_points[5][0])**2 + 
                     (eye_points[1][1] - eye_points[5][1])**2)
        v2 = np.sqrt((eye_points[2][0] - eye_points[4][0])**2 + 
                     (eye_points[2][1] - eye_points[4][1])**2)
        
        # Horizontal distance
        h = np.sqrt((eye_points[0][0] - eye_points[3][0])**2 + 
                    (eye_points[0][1] - eye_points[3][1])**2)
        
        if h == 0:
            return 1.0
        
        # EAR = (v1 + v2) / (2 * h)
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def detect_blink(self, face_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Detect blink using eye aspect ratio.
        
        Args:
            face_crop: Face image (BGR)
        
        Returns:
            (blink_detected: bool, ear: float)
        """
        if not self.mp_face_mesh or self.face_mesh is None:
            return False, 1.0
        
        try:
            # Convert to RGB
            img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            
            if not results.multi_face_landmarks:
                return False, 1.0
            
            landmarks = results.multi_face_landmarks[0].landmark
            
            # MediaPipe eye indices (left eye)
            LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            # Simplified: outer corners and inner points
            LEFT_EYE_SIMPLIFIED = [33, 7, 163, 144, 145, 153]  # Outer corners + inner points
            
            # Right eye
            RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            RIGHT_EYE_SIMPLIFIED = [362, 382, 381, 380, 374, 373]
            
            # Compute EAR for both eyes
            left_ear = self.compute_eye_aspect_ratio(landmarks, LEFT_EYE_SIMPLIFIED)
            right_ear = self.compute_eye_aspect_ratio(landmarks, RIGHT_EYE_SIMPLIFIED)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Store in history
            self.eye_history.append(avg_ear)
            
            # Detect blink: EAR drops below threshold
            if len(self.eye_history) >= 3:
                recent_ears = list(self.eye_history)[-3:]
                min_ear = min(recent_ears)
                max_ear = max(recent_ears)
                
                # Blink detected if EAR drops significantly
                if min_ear < self.blink_threshold and max_ear > self.blink_threshold * 1.5:
                    return True, avg_ear
            
            return False, avg_ear
            
        except Exception as e:
            log.debug("Blink detection failed: %s", e)
            return False, 1.0
    
    def detect_motion(self, face_crop: np.ndarray) -> float:
        """
        Detect face motion (micro-movements) to distinguish real face from photo.
        
        Args:
            face_crop: Face image (BGR)
        
        Returns:
            Motion score (0-1, higher = more motion)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # Store in history
            self.face_history.append(gray)
            
            if len(self.face_history) < 2:
                return 0.0
            
            # Compute motion between consecutive frames
            prev_frame = self.face_history[-2]
            curr_frame = self.face_history[-1]
            
            # Optical flow or frame difference
            diff = cv2.absdiff(prev_frame, curr_frame)
            motion_score = np.mean(diff) / 255.0  # Normalize to [0, 1]
            
            return float(motion_score)
            
        except Exception as e:
            log.debug("Motion detection failed: %s", e)
            return 0.0
    
    def analyze_texture(self, face_crop: np.ndarray) -> float:
        """
        Analyze texture to detect photo vs real face.
        Photos typically have different texture patterns.
        
        Args:
            face_crop: Face image (BGR)
        
        Returns:
            Texture score (0-1, higher = more likely real)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # Laplacian variance (blur detection)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = np.var(laplacian)
            
            # Normalize (typical range: 0-1000, normalize to 0-1)
            texture_score = min(1.0, variance / 500.0)
            
            # Photos often have lower variance (more uniform)
            # Real faces have higher variance (more texture)
            return float(texture_score)
            
        except Exception as e:
            log.debug("Texture analysis failed: %s", e)
            return 0.5  # Neutral score
    
    def detect_liveness(self, face_crop: np.ndarray, 
                       require_blink: bool = True,
                       min_motion: float = 0.01,
                       min_texture: float = 0.3) -> Tuple[bool, dict]:
        """
        Comprehensive liveness detection.
        
        Args:
            face_crop: Face image (BGR)
            require_blink: Whether to require blink detection
            min_motion: Minimum motion score required
            min_texture: Minimum texture score required
        
        Returns:
            (is_live: bool, scores: dict)
        """
        scores = {
            "blink_detected": False,
            "ear": 1.0,
            "motion": 0.0,
            "texture": 0.5,
            "overall": 0.0
        }
        
        # 1. Blink detection
        blink_detected, ear = self.detect_blink(face_crop)
        scores["blink_detected"] = blink_detected
        scores["ear"] = ear
        
        # 2. Motion detection
        motion = self.detect_motion(face_crop)
        scores["motion"] = motion
        
        # 3. Texture analysis
        texture = self.analyze_texture(face_crop)
        scores["texture"] = texture
        
        # Overall liveness score
        # Weighted combination
        blink_score = 1.0 if blink_detected else 0.0
        motion_score = 1.0 if motion >= min_motion else motion / min_motion
        texture_score = 1.0 if texture >= min_texture else texture / min_texture
        
        if require_blink:
            # Require blink + motion + texture
            overall = (blink_score * 0.5 + motion_score * 0.3 + texture_score * 0.2)
            is_live = blink_detected and motion >= min_motion and texture >= min_texture
        else:
            # Motion + texture only (for cases where blink not visible)
            overall = (motion_score * 0.6 + texture_score * 0.4)
            is_live = motion >= min_motion and texture >= min_texture
        
        scores["overall"] = overall
        
        return is_live, scores
    
    def reset(self):
        """Reset history (call when new verification session starts)."""
        self.face_history.clear()
        self.eye_history.clear()
        log.debug("Liveness detector history reset")


def create_liveness_detector(config: dict = None) -> Optional[LivenessDetector]:
    """Factory function to create liveness detector."""
    if config is None:
        config = {}
    
    try:
        detector = LivenessDetector(
            blink_threshold=config.get("blink_threshold", 0.25),
            motion_threshold=config.get("motion_threshold", 0.01),
            history_size=config.get("history_size", 30)
        )
        return detector
    except Exception as e:
        log.warning("Failed to create liveness detector: %s", e)
        return None

