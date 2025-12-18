# pipeline/patient/face_recognition.py
"""
DeepFace-based patient face recognition for identity verification.
"""
import os
import cv2
import numpy as np
import logging
import json
from typing import Optional, Dict, Tuple

log = logging.getLogger("face_recognition")

# Try to import DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    log.warning("DeepFace not available. Install with: pip install deepface")


class PatientFaceRecognizer:
    """
    Patient face recognition using DeepFace.
    Verifies patient identity using facial features.
    """
    
    def __init__(self, reference_faces_dir="storage/patient_faces", model_name="VGG-Face"):
        """
        Args:
            reference_faces_dir: Directory storing reference face images
            model_name: DeepFace model ("VGG-Face", "Facenet", "ArcFace", etc.)
            enable_liveness: Enable liveness detection (anti-spoofing)
            enable_rate_limiting: Enable rate limiting (security)
        """
        if not DEEPFACE_AVAILABLE:
            log.error("DeepFace not available. Face recognition disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.reference_faces_dir = reference_faces_dir
        self.model_name = model_name
        self.reference_faces = {}
        
        # Create directory if it doesn't exist
        os.makedirs(reference_faces_dir, exist_ok=True)
        
        # Load existing reference faces
        self._load_reference_faces()
        
        # Initialize liveness detector
        self.liveness_detector = None
        if enable_liveness:
            try:
                from .liveness_detector import create_liveness_detector
                self.liveness_detector = create_liveness_detector()
                if self.liveness_detector:
                    log.info("Liveness detection enabled")
            except Exception as e:
                log.warning("Failed to initialize liveness detector: %s", e)
        
        # Initialize rate limiter
        self.rate_limiter = None
        if enable_rate_limiting:
            try:
                from .rate_limiter import create_rate_limiter
                self.rate_limiter = create_rate_limiter()
                log.info("Rate limiting enabled")
            except Exception as e:
                log.warning("Failed to initialize rate limiter: %s", e)
        
        log.info("PatientFaceRecognizer initialized (model: %s, liveness: %s, rate_limit: %s)", 
                model_name, enable_liveness, enable_rate_limiting)
    
    def _load_reference_faces(self):
        """Load reference faces from storage directory."""
        if not os.path.exists(self.reference_faces_dir):
            return
        
        # Look for reference images
        for filename in os.listdir(self.reference_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                patient_id = os.path.splitext(filename)[0]
                face_path = os.path.join(self.reference_faces_dir, filename)
                self.reference_faces[patient_id] = face_path
                log.debug("Loaded reference face for patient: %s", patient_id)
    
    def extract_face(self, frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        """
        Extract face region from frame using bounding box.
        
        Args:
            frame: Input frame (BGR)
            bbox: Bounding box [x, y, w, h] or [x1, y1, x2, y2]
        
        Returns:
            Face crop (BGR) or None if extraction fails
        """
        try:
            h, w = frame.shape[:2]
            
            # Parse bbox format
            if len(bbox) >= 4:
                if len(bbox) == 4:
                    # Assume [x, y, w, h]
                    x, y, bw, bh = bbox
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + bw), int(y + bh)
                else:
                    x1, y1, x2, y2 = bbox[:4]
                
                # Clamp to frame bounds
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                if x2 > x1 and y2 > y1:
                    # Extract face region with padding
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # Resize if too small
                    if face_crop.shape[0] < 64 or face_crop.shape[1] < 64:
                        face_crop = cv2.resize(face_crop, (128, 128))
                    
                    return face_crop
        except Exception as e:
            log.exception("Failed to extract face: %s", e)
        
        return None
    
    def verify_patient(self, frame: np.ndarray, bbox: list, patient_id: str, 
                      threshold: float = 0.6,
                      liveness_detector=None,
                      rate_limiter=None) -> Tuple[bool, float, dict]:
        """
        Verify if detected person matches patient ID with anti-spoofing.
        
        Args:
            frame: Input frame (BGR)
            bbox: Person bounding box
            patient_id: Expected patient ID
            threshold: Similarity threshold (0-1, higher = stricter)
            liveness_detector: Optional liveness detector for anti-spoofing
            rate_limiter: Optional rate limiter for security
        
        Returns:
            (verified: bool, confidence: float, metadata: dict)
        """
        metadata = {
            "liveness_passed": False,
            "rate_limit_passed": True,
            "liveness_scores": {},
            "rate_limit_reason": None
        }
        
        if not self.enabled:
            return False, 0.0, metadata
        
        if patient_id not in self.reference_faces:
            log.warning("No reference face found for patient: %s", patient_id)
            return False, 0.0, metadata
        
        # Rate limiting check
        if rate_limiter:
            allowed, reason = rate_limiter.check(patient_id)
            metadata["rate_limit_passed"] = allowed
            metadata["rate_limit_reason"] = reason
            
            if not allowed:
                log.warning("Rate limit exceeded for patient %s: %s", patient_id, reason)
                return False, 0.0, metadata
        
        try:
            # Extract face from current frame
            face_crop = self.extract_face(frame, bbox)
            if face_crop is None:
                return False, 0.0, metadata
            
            # Liveness detection (anti-spoofing) (use instance liveness_detector if not provided)
            active_liveness_detector = liveness_detector or self.liveness_detector
            if active_liveness_detector:
                is_live, liveness_scores = active_liveness_detector.detect_liveness(face_crop)
                metadata["liveness_passed"] = is_live
                metadata["liveness_scores"] = liveness_scores
                
                if not is_live:
                    log.warning("Liveness detection failed for patient %s. Possible spoofing attempt.", patient_id)
                    return False, 0.0, metadata
            
            # Compare with reference face
            reference_path = self.reference_faces[patient_id]
            
            result = DeepFace.verify(
                face_crop,
                reference_path,
                model_name=self.model_name,
                enforce_detection=False,
                distance_metric="cosine"
            )
            
            verified = result["verified"]
            distance = result.get("distance", 1.0)
            confidence = 1.0 - min(1.0, distance)  # Convert distance to confidence
            
            if verified and confidence >= threshold:
                # Record success (reset rate limiter)
                active_rate_limiter = rate_limiter or self.rate_limiter
                if active_rate_limiter:
                    active_rate_limiter.record_success(patient_id)
                
                log.debug("Patient %s verified (confidence: %.2f, liveness: %s)", 
                         patient_id, confidence, metadata.get("liveness_passed", "N/A"))
                return True, confidence, metadata
            else:
                log.debug("Patient %s verification failed (confidence: %.2f)", patient_id, confidence)
                return False, confidence, metadata
                
        except Exception as e:
            log.exception("Face verification failed for patient %s: %s", patient_id, e)
            return False, 0.0, metadata
    
    def enroll_patient(self, frame: np.ndarray, bbox: list, patient_id: str) -> bool:
        """
        Enroll a new patient by saving reference face.
        
        Args:
            frame: Input frame (BGR)
            bbox: Person bounding box
            patient_id: Patient ID
        
        Returns:
            True if enrollment successful
        """
        if not self.enabled:
            return False
        
        try:
            # Extract face
            face_crop = self.extract_face(frame, bbox)
            if face_crop is None:
                log.error("Failed to extract face for enrollment")
                return False
            
            # Save reference face
            reference_path = os.path.join(self.reference_faces_dir, f"{patient_id}.jpg")
            cv2.imwrite(reference_path, face_crop)
            
            # Update cache
            self.reference_faces[patient_id] = reference_path
            
            log.info("Patient %s enrolled successfully", patient_id)
            return True
            
        except Exception as e:
            log.exception("Failed to enroll patient %s: %s", patient_id, e)
            return False
    
    def find_patient(self, frame: np.ndarray, bbox: list, threshold: float = 0.6) -> Optional[str]:
        """
        Find patient ID by comparing with all reference faces.
        
        Args:
            frame: Input frame (BGR)
            bbox: Person bounding box
            threshold: Similarity threshold
        
        Returns:
            Patient ID if match found, None otherwise
        """
        if not self.enabled or not self.reference_faces:
            return None
        
        try:
            face_crop = self.extract_face(frame, bbox)
            if face_crop is None:
                return None
            
            best_match = None
            best_confidence = 0.0
            
            # Compare with all reference faces
            for patient_id, reference_path in self.reference_faces.items():
                try:
                    result = DeepFace.verify(
                        face_crop,
                        reference_path,
                        model_name=self.model_name,
                        enforce_detection=False,
                        distance_metric="cosine"
                    )
                    
                    if result["verified"]:
                        distance = result.get("distance", 1.0)
                        confidence = 1.0 - min(1.0, distance)
                        
                        if confidence > best_confidence and confidence >= threshold:
                            best_confidence = confidence
                            best_match = patient_id
                except Exception:
                    continue
            
            if best_match:
                log.debug("Found patient match: %s (confidence: %.2f)", best_match, best_confidence)
                return best_match
            
            return None
            
        except Exception as e:
            log.exception("Failed to find patient: %s", e)
            return None
    
    def has_reference(self, patient_id: str) -> bool:
        """Check if reference face exists for patient."""
        return patient_id in self.reference_faces

