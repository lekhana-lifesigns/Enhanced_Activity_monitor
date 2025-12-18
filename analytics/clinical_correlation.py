# analytics/clinical_correlation.py
"""
Clinical correlation engine.
Correlates posture, emotion, and movement to understand:
- Pain levels
- Agitation states
- Dizziness/vertigo
- Overall patient distress
"""
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple

log = logging.getLogger("clinical_correlation")


class ClinicalCorrelationEngine:
    """
    Correlates multiple signals to understand patient clinical state.
    """
    
    def __init__(self):
        """Initialize clinical correlation engine."""
        self.pain_history = []
        self.agitation_history = []
        self.dizziness_history = []
        self.history_size = 30  # ~2 seconds at 15 FPS
    
    def correlate_clinical_state(self, 
                                posture_state: str,
                                posture_3d: Optional[List[Tuple[float, float, float, float]]],
                                emotions: Dict,
                                movement_features: Optional[Dict],
                                frame_visibility: Dict,
                                distance: float) -> Dict:
        """
        Correlate multiple signals to determine clinical state.
        
        Args:
            posture_state: Current posture (supine, prone, lateral, etc.)
            posture_3d: 3D keypoints (x, y, z, confidence)
            emotions: Emotion detection results
            movement_features: Movement features (motion energy, jerk, etc.)
            frame_visibility: Frame visibility analysis
            distance: Distance from camera (meters)
        
        Returns:
            dict with:
            - pain_score: 0-1 pain level
            - agitation_score: 0-1 agitation level
            - dizziness_score: 0-1 dizziness/vertigo level
            - distress_level: "low" | "medium" | "high" | "critical"
            - clinical_indicators: List of detected indicators
            - confidence: Overall confidence in assessment
        """
        try:
            # Initialize scores
            pain_score = 0.0
            agitation_score = 0.0
            dizziness_score = 0.0
            indicators = []
            
            # 1. PAIN DETECTION
            pain_score, pain_indicators = self._assess_pain(
                posture_state, posture_3d, emotions, movement_features
            )
            indicators.extend(pain_indicators)
            
            # 2. AGITATION DETECTION
            agitation_score, agitation_indicators = self._assess_agitation(
                posture_state, emotions, movement_features, frame_visibility
            )
            indicators.extend(agitation_indicators)
            
            # 3. DIZZINESS DETECTION
            dizziness_score, dizziness_indicators = self._assess_dizziness(
                posture_state, posture_3d, emotions, movement_features, distance
            )
            indicators.extend(dizziness_indicators)
            
            # Update history
            self.pain_history.append(pain_score)
            self.agitation_history.append(agitation_score)
            self.dizziness_history.append(dizziness_score)
            
            if len(self.pain_history) > self.history_size:
                self.pain_history.pop(0)
            if len(self.agitation_history) > self.history_size:
                self.agitation_history.pop(0)
            if len(self.dizziness_history) > self.history_size:
                self.dizziness_history.pop(0)
            
            # Temporal smoothing
            pain_smoothed = np.mean(self.pain_history) if self.pain_history else pain_score
            agitation_smoothed = np.mean(self.agitation_history) if self.agitation_history else agitation_score
            dizziness_smoothed = np.mean(self.dizziness_history) if self.dizziness_history else dizziness_score
            
            # Overall distress level
            max_score = max(pain_smoothed, agitation_smoothed, dizziness_smoothed)
            if max_score >= 0.8:
                distress_level = "critical"
            elif max_score >= 0.6:
                distress_level = "high"
            elif max_score >= 0.4:
                distress_level = "medium"
            else:
                distress_level = "low"
            
            # Confidence based on data quality
            confidence = self._compute_confidence(
                posture_3d, emotions, movement_features, frame_visibility
            )
            
            return {
                "pain_score": float(pain_smoothed),
                "agitation_score": float(agitation_smoothed),
                "dizziness_score": float(dizziness_smoothed),
                "distress_level": distress_level,
                "clinical_indicators": indicators,
                "confidence": float(confidence),
                "distance_meters": float(distance),
                "frame_visibility": frame_visibility.get("visibility_type", "unknown"),
                "completeness": frame_visibility.get("completeness_score", 0.0)
            }
            
        except Exception as e:
            log.exception("Error in clinical correlation: %s", e)
            return {
                "pain_score": 0.0,
                "agitation_score": 0.0,
                "dizziness_score": 0.0,
                "distress_level": "low",
                "clinical_indicators": [],
                "confidence": 0.0,
                "distance_meters": 0.0,
                "frame_visibility": "unknown",
                "completeness": 0.0
            }
    
    def _assess_pain(self, posture_state, posture_3d, emotions, movement_features) -> Tuple[float, List[str]]:
        """Assess pain level from multiple signals."""
        pain_score = 0.0
        indicators = []
        
        # Pain indicators from posture
        if posture_state in ["prone", "side"]:
            # Uncomfortable postures may indicate pain
            pain_score += 0.2
            indicators.append("uncomfortable_posture")
        
        # Pain indicators from emotions
        if emotions:
            dominant = emotions.get("dominant_emotion", "neutral")
            if dominant in ["sad", "angry", "fear"]:
                pain_score += 0.3
                indicators.append(f"negative_emotion_{dominant}")
            
            # High negative emotion scores
            negative_emotions = emotions.get("emotions", {})
            if negative_emotions:
                negative_sum = sum([
                    negative_emotions.get("sad", 0),
                    negative_emotions.get("angry", 0),
                    negative_emotions.get("fear", 0),
                    negative_emotions.get("disgust", 0)
                ])
                if negative_sum > 0.5:
                    pain_score += 0.2
                    indicators.append("high_negative_emotions")
        
        # Pain indicators from movement
        if movement_features:
            # Restricted movement may indicate pain
            motion_energy = movement_features.get("motion_energy", 0.0)
            if motion_energy < 0.1:
                pain_score += 0.1
                indicators.append("restricted_movement")
            
            # Guarding behavior (protecting painful area)
            # This would need more sophisticated analysis
        
        # Pain indicators from 3D posture
        if posture_3d:
            # Asymmetric posture may indicate pain
            pain_score += self._check_posture_asymmetry(posture_3d) * 0.2
            if self._check_posture_asymmetry(posture_3d) > 0.5:
                indicators.append("asymmetric_posture")
        
        return min(1.0, pain_score), indicators
    
    def _assess_agitation(self, posture_state, emotions, movement_features, frame_visibility) -> Tuple[float, List[str]]:
        """Assess agitation level."""
        agitation_score = 0.0
        indicators = []
        
        # Agitation from movement
        if movement_features:
            motion_energy = movement_features.get("motion_energy", 0.0)
            jerk_index = movement_features.get("jerk_index", 0.0)
            
            # High, erratic movement indicates agitation
            if motion_energy > 0.7:
                agitation_score += 0.3
                indicators.append("high_movement")
            
            if jerk_index > 0.6:
                agitation_score += 0.3
                indicators.append("erratic_movement")
        
        # Agitation from emotions
        if emotions:
            dominant = emotions.get("dominant_emotion", "neutral")
            if dominant == "angry":
                agitation_score += 0.4
                indicators.append("angry_expression")
            
            if dominant == "fear":
                agitation_score += 0.3
                indicators.append("fearful_expression")
        
        # Agitation from posture
        if posture_state == "sitting":
            # Restless sitting may indicate agitation
            if movement_features and movement_features.get("motion_energy", 0.0) > 0.5:
                agitation_score += 0.2
                indicators.append("restless_sitting")
        
        # Agitation from frame visibility (frequent position changes)
        if frame_visibility.get("completeness_score", 1.0) < 0.7:
            # Partial visibility may indicate frequent movement
            agitation_score += 0.1
            indicators.append("frequent_position_changes")
        
        return min(1.0, agitation_score), indicators
    
    def _assess_dizziness(self, posture_state, posture_3d, emotions, movement_features, distance) -> Tuple[float, List[str]]:
        """Assess dizziness/vertigo level."""
        dizziness_score = 0.0
        indicators = []
        
        # Dizziness from posture
        if posture_state in ["sitting", "upright"]:
            # Dizziness often causes difficulty maintaining upright posture
            if movement_features:
                # Unsteady movement
                jerk_index = movement_features.get("jerk_index", 0.0)
                if jerk_index > 0.5:
                    dizziness_score += 0.3
                    indicators.append("unsteady_upright_posture")
        
        # Dizziness from 3D analysis
        if posture_3d:
            # Swaying or tilting may indicate dizziness
            sway = self._check_posture_sway(posture_3d)
            if sway > 0.4:
                dizziness_score += 0.4
                indicators.append("postural_sway")
        
        # Dizziness from emotions
        if emotions:
            dominant = emotions.get("dominant_emotion", "neutral")
            if dominant == "fear" or dominant == "surprise":
                # Dizziness can cause fear/surprise
                dizziness_score += 0.2
                indicators.append("fearful_expression")
        
        # Dizziness from distance (if person is very close, may be trying to stabilize)
        if distance > 0 and distance < 1.5:  # Very close to camera
            if posture_state in ["sitting", "upright"]:
                dizziness_score += 0.1
                indicators.append("close_proximity_seeking_support")
        
        return min(1.0, dizziness_score), indicators
    
    def _check_posture_asymmetry(self, posture_3d: List[Tuple[float, float, float, float]]) -> float:
        """Check posture asymmetry (indicator of pain or discomfort)."""
        if not posture_3d or len(posture_3d) < 6:
            return 0.0
        
        try:
            # Compare left vs right side keypoints
            # Shoulders (indices 5, 6), Hips (indices 11, 12)
            left_shoulder = posture_3d[5] if len(posture_3d) > 5 else None
            right_shoulder = posture_3d[6] if len(posture_3d) > 6 else None
            left_hip = posture_3d[11] if len(posture_3d) > 11 else None
            right_hip = posture_3d[12] if len(posture_3d) > 12 else None
            
            if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
                return 0.0
            
            # Calculate asymmetry in Y (vertical) and Z (depth) coordinates
            shoulder_y_diff = abs(left_shoulder[1] - right_shoulder[1])
            hip_y_diff = abs(left_hip[1] - right_hip[1])
            
            # Normalize (typical asymmetry is < 0.1m, significant is > 0.2m)
            asymmetry = (shoulder_y_diff + hip_y_diff) / 2.0
            asymmetry_score = min(1.0, asymmetry / 0.2)
            
            return asymmetry_score
        except Exception:
            return 0.0
    
    def _check_posture_sway(self, posture_3d: List[Tuple[float, float, float, float]]) -> float:
        """Check for postural sway (indicator of dizziness/balance issues)."""
        if not posture_3d or len(posture_3d) < 5:
            return 0.0
        
        try:
            # Get head/torso position
            nose = posture_3d[0] if len(posture_3d) > 0 else None
            if not nose:
                return 0.0
            
            # Calculate variance in X and Y positions (sway)
            # This is simplified - real sway detection needs temporal analysis
            x_positions = [kp[0] for kp in posture_3d if len(kp) >= 3]
            y_positions = [kp[1] for kp in posture_3d if len(kp) >= 3]
            
            if len(x_positions) < 3:
                return 0.0
            
            x_variance = np.var(x_positions)
            y_variance = np.var(y_positions)
            
            # Normalize (typical variance is < 0.01, significant is > 0.05)
            sway_score = min(1.0, (x_variance + y_variance) / 0.1)
            
            return sway_score
        except Exception:
            return 0.0
    
    def _compute_confidence(self, posture_3d, emotions, movement_features, frame_visibility) -> float:
        """Compute overall confidence in clinical assessment."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence with 3D data
        if posture_3d and len(posture_3d) >= 10:
            confidence += 0.2
        
        # Higher confidence with emotion data
        if emotions and emotions.get("confidence", 0.0) > 0.5:
            confidence += 0.15
        
        # Higher confidence with movement features
        if movement_features:
            confidence += 0.1
        
        # Higher confidence with full body visibility
        if frame_visibility.get("completeness_score", 0.0) > 0.8:
            confidence += 0.05
        
        return min(1.0, confidence)

