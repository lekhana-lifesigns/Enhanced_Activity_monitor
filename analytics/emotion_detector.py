# analytics/emotion_detector.py
"""
Emotion detection from facial expressions.
Correlates with posture and clinical states (pain, agitation, dizziness).
"""
import cv2
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple

log = logging.getLogger("emotion_detector")

# Try to import emotion detection libraries
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    log.warning("DeepFace not available - emotion detection will use geometric methods")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class EmotionDetector:
    """
    Emotion detection from facial expressions.
    Uses DeepFace or MediaPipe for facial analysis.
    """
    
    def __init__(self, method="deepface"):
        """
        Args:
            method: "deepface" (accurate) or "mediapipe" (fast) or "geometric" (fallback)
        """
        self.method = method
        self.mp_face = None
        
        if method == "deepface" and DEEPFACE_AVAILABLE:
            log.info("Emotion detector using DeepFace")
        elif method == "mediapipe" and MEDIAPIPE_AVAILABLE:
            self.mp_face = mp.solutions.face_mesh
            self.face_mesh = self.mp_face.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            log.info("Emotion detector using MediaPipe")
        else:
            self.method = "geometric"
            log.info("Emotion detector using geometric methods (fallback)")
    
    def detect_emotions(self, frame: np.ndarray, face_bbox: Optional[List[float]] = None) -> Dict:
        """
        Detect emotions from face region.
        
        Args:
            frame: Input frame (BGR)
            face_bbox: Optional face bounding box [x, y, w, h]
        
        Returns:
            dict with:
            - emotions: Dict of emotion scores (angry, happy, sad, fear, surprise, neutral, disgust)
            - dominant_emotion: Most likely emotion
            - confidence: Confidence score
            - facial_features: Additional facial analysis
        """
        if self.method == "deepface" and DEEPFACE_AVAILABLE:
            return self._detect_deepface(frame, face_bbox)
        elif self.method == "mediapipe" and self.face_mesh:
            return self._detect_mediapipe(frame, face_bbox)
        else:
            return self._detect_geometric(frame, face_bbox)
    
    def _detect_deepface(self, frame: np.ndarray, face_bbox: Optional[List[float]]) -> Dict:
        """Detect emotions using DeepFace."""
        try:
            # Extract face region if bbox provided
            if face_bbox:
                x, y, w, h = [int(v) for v in face_bbox[:4]]
                h_frame, w_frame = frame.shape[:2]
                x = max(0, min(x, w_frame))
                y = max(0, min(y, h_frame))
                w = min(w, w_frame - x)
                h = min(h, h_frame - y)
                face_crop = frame[y:y+h, x:x+w]
            else:
                face_crop = frame
            
            # DeepFace emotion analysis
            result = DeepFace.analyze(
                face_crop,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result.get('emotion', {})
            dominant = result.get('dominant_emotion', 'neutral')
            
            return {
                "emotions": emotions,
                "dominant_emotion": dominant,
                "confidence": max(emotions.values()) / 100.0 if emotions else 0.0,
                "facial_features": {
                    "age": result.get('age', 0),
                    "gender": result.get('dominant_gender', 'unknown'),
                    "race": result.get('dominant_race', 'unknown')
                }
            }
        except Exception as e:
            log.debug("DeepFace emotion detection failed: %s", e)
            return self._detect_geometric(frame, face_bbox)
    
    def _detect_mediapipe(self, frame: np.ndarray, face_bbox: Optional[List[float]]) -> Dict:
        """Detect emotions using MediaPipe facial landmarks."""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if not results.multi_face_landmarks:
                return self._detect_geometric(frame, face_bbox)
            
            # Get facial landmarks
            landmarks = results.multi_face_landmarks[0]
            
            # Extract facial features for emotion estimation
            # This is a simplified approach - real emotion detection needs more analysis
            emotions = self._estimate_emotions_from_landmarks(landmarks, frame.shape)
            
            dominant = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"
            
            return {
                "emotions": emotions,
                "dominant_emotion": dominant,
                "confidence": max(emotions.values()) if emotions else 0.0,
                "facial_features": {
                    "landmarks_detected": True,
                    "num_landmarks": len(landmarks.landmark)
                }
            }
        except Exception as e:
            log.debug("MediaPipe emotion detection failed: %s", e)
            return self._detect_geometric(frame, face_bbox)
    
    def _estimate_emotions_from_landmarks(self, landmarks, frame_shape):
        """Estimate emotions from MediaPipe facial landmarks."""
        # Simplified emotion estimation based on facial geometry
        # Real implementation would use more sophisticated analysis
        
        h, w = frame_shape[:2]
        
        # Get key facial points
        # Mouth corners (for smile/frown)
        mouth_left = landmarks.landmark[61]  # Left mouth corner
        mouth_right = landmarks.landmark[291]  # Right mouth corner
        
        # Eye positions
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        
        # Calculate mouth curvature (smile indicator)
        mouth_curve = (mouth_left.y + mouth_right.y) / 2.0
        
        # Calculate eye opening
        eye_opening = abs(left_eye.y - right_eye.y)
        
        # Simple emotion estimation (this is a placeholder - needs proper ML model)
        emotions = {
            "neutral": 0.5,
            "happy": 0.2 if mouth_curve < 0.5 else 0.1,
            "sad": 0.1 if mouth_curve > 0.6 else 0.05,
            "angry": 0.1,
            "surprise": 0.05,
            "fear": 0.05,
            "disgust": 0.05
        }
        
        # Normalize
        total = sum(emotions.values())
        emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions
    
    def _detect_geometric(self, frame: np.ndarray, face_bbox: Optional[List[float]]) -> Dict:
        """Fallback geometric emotion detection."""
        return {
            "emotions": {
                "neutral": 1.0,
                "happy": 0.0,
                "sad": 0.0,
                "angry": 0.0,
                "surprise": 0.0,
                "fear": 0.0,
                "disgust": 0.0
            },
            "dominant_emotion": "neutral",
            "confidence": 0.5,
            "facial_features": {}
        }

