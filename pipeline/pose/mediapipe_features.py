# pipeline/pose/mediapipe_features.py
# MediaPipe-derived features for ICU monitoring
# Converts MediaPipe heuristics into GRU-compatible features

import numpy as np
import math
import logging

log = logging.getLogger("mediapipe_features")

"""
This module extracts useful features from MediaPipe FaceMesh and Pose
that can be integrated into the ICU Feature Encoder.

These features complement the existing 4-layer medical signal architecture
by adding facial/gestural indicators of agitation, discomfort, and behavior.
"""


def compute_mouth_opening(face_landmarks):
    """
    Compute mouth opening ratio from FaceMesh landmarks.
    
    MediaPipe FaceMesh indices:
    - Upper lip: 13, 14, 15, 16, 17
    - Lower lip: 18, 19, 20, 21, 22
    
    Returns:
        mouth_opening_ratio: 0-1 (0 = closed, 1 = fully open)
    """
    try:
        if not face_landmarks or len(face_landmarks) < 23:
            return 0.0
        
        # Upper lip points
        upper_y = np.mean([face_landmarks[i].y for i in [13, 14, 15, 16, 17]])
        # Lower lip points
        lower_y = np.mean([face_landmarks[i].y for i in [18, 19, 20, 21, 22]])
        
        # Mouth opening = vertical distance
        mouth_opening = abs(lower_y - upper_y)
        
        # Normalize (assuming max opening ~0.1 in normalized coordinates)
        mouth_opening_ratio = min(1.0, mouth_opening * 10.0)
        
        return float(mouth_opening_ratio)
    except Exception:
        return 0.0


def compute_hand_to_mouth_distance(hand_landmarks, face_landmarks):
    """
    Compute distance from hand to mouth.
    
    Useful for:
    - Eating detection
    - Self-soothing behaviors
    - Agitation indicators
    
    Returns:
        distance: Normalized distance (0-1, lower = closer)
        risk_score: 0-1 (higher = hand very close to mouth)
    """
    try:
        if not hand_landmarks or not face_landmarks or len(face_landmarks) < 13:
            return (1.0, 0.0)
        
        # Hand center (wrist or palm)
        hand_x = hand_landmarks[0].x  # Wrist
        hand_y = hand_landmarks[0].y
        
        # Mouth center (between upper and lower lip)
        mouth_x = face_landmarks[13].x  # Upper lip center
        mouth_y = (face_landmarks[13].y + face_landmarks[18].y) / 2.0
        
        # Distance
        dx = hand_x - mouth_x
        dy = hand_y - mouth_y
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Risk score (inverse distance, normalized)
        risk_score = max(0.0, 1.0 - distance * 5.0)
        
        return (float(distance), float(risk_score))
    except Exception:
        return (1.0, 0.0)


def compute_lip_velocity(face_landmarks_history, fps=15.0):
    """
    Compute lip movement velocity.
    
    High velocity = rapid lip movements (talking, coughing, etc.)
    
    Returns:
        lip_velocity: Normalized velocity (0-1)
    """
    try:
        if len(face_landmarks_history) < 2:
            return 0.0
        
        # Get mouth center positions
        mouth_positions = []
        for landmarks in face_landmarks_history[-5:]:  # Last 5 frames
            if landmarks and len(landmarks) > 18:
                mouth_y = (landmarks[13].y + landmarks[18].y) / 2.0
                mouth_positions.append(mouth_y)
        
        if len(mouth_positions) < 2:
            return 0.0
        
        # Compute velocity as change in position
        velocities = []
        for i in range(1, len(mouth_positions)):
            dy = abs(mouth_positions[i] - mouth_positions[i-1])
            velocity = dy * fps  # pixels per second
            velocities.append(velocity)
        
        avg_velocity = np.mean(velocities) if velocities else 0.0
        
        # Normalize (assuming max ~0.05 per frame at 15fps = 0.75)
        normalized_velocity = min(1.0, avg_velocity / 0.75)
        
        return float(normalized_velocity)
    except Exception:
        return 0.0


def compute_nose_forward_velocity(face_landmarks_history, fps=15.0):
    """
    Compute nose forward movement velocity.
    
    Forward movement can indicate:
    - Leaning forward (discomfort, agitation)
    - Coughing (forward thrust)
    
    Returns:
        nose_forward_velocity: Normalized velocity (0-1)
    """
    try:
        if len(face_landmarks_history) < 2:
            return 0.0
        
        # Get nose tip positions (x-coordinate = forward/backward)
        nose_positions = []
        for landmarks in face_landmarks_history[-5:]:
            if landmarks and len(landmarks) > 1:
                nose_x = landmarks[1].x  # Nose tip
                nose_positions.append(nose_x)
        
        if len(nose_positions) < 2:
            return 0.0
        
        # Compute forward velocity (positive x change)
        velocities = []
        for i in range(1, len(nose_positions)):
            dx = nose_positions[i] - nose_positions[i-1]
            if dx > 0:  # Forward movement
                velocity = dx * fps
                velocities.append(velocity)
        
        avg_velocity = np.mean(velocities) if velocities else 0.0
        
        # Normalize
        normalized_velocity = min(1.0, avg_velocity * 20.0)
        
        return float(normalized_velocity)
    except Exception:
        return 0.0


def compute_head_bobbing_rate(face_landmarks_history, fps=15.0):
    """
    Compute head bobbing frequency.
    
    Head bobbing can indicate:
    - Agitation
    - Discomfort
    - Restlessness
    
    Returns:
        bobbing_rate: Frequency in Hz (0-2 Hz typical)
    """
    try:
        if len(face_landmarks_history) < 10:
            return 0.0
        
        # Get head position (forehead or nose)
        head_y_positions = []
        for landmarks in face_landmarks_history:
            if landmarks and len(landmarks) > 10:
                # Use forehead point (index 10) or nose
                head_y = landmarks[10].y if len(landmarks) > 10 else landmarks[1].y
                head_y_positions.append(head_y)
        
        if len(head_y_positions) < 10:
            return 0.0
        
        # FFT to find dominant frequency
        from scipy import signal
        
        y_signal = np.array(head_y_positions)
        y_detrended = signal.detrend(y_signal)
        
        fft = np.fft.fft(y_detrended)
        freqs = np.fft.fftfreq(len(y_detrended), 1.0 / fps)
        
        # Find peak in bobbing range (0.5-2 Hz)
        valid_idx = (freqs > 0.5) & (freqs < 2.0)
        if np.any(valid_idx):
            power = np.abs(fft[valid_idx])
            peak_idx = np.argmax(power)
            peak_freq = freqs[valid_idx][peak_idx]
            return float(np.clip(peak_freq, 0.0, 2.0))
        
        return 0.0
    except Exception:
        return 0.0


def compute_facial_asymmetry(face_landmarks):
    """
    Compute facial asymmetry score.
    
    Asymmetry can indicate:
    - Pain (grimacing)
    - Neurological issues
    - Agitation expressions
    
    Returns:
        asymmetry_score: 0-1 (higher = more asymmetric)
    """
    try:
        if not face_landmarks or len(face_landmarks) < 20:
            return 0.0
        
        # Compare left vs right facial features
        # Left eye vs right eye
        left_eye_y = face_landmarks[33].y if len(face_landmarks) > 33 else 0.0
        right_eye_y = face_landmarks[263].y if len(face_landmarks) > 263 else 0.0
        
        # Left mouth corner vs right mouth corner
        left_mouth_x = face_landmarks[61].x if len(face_landmarks) > 61 else 0.0
        right_mouth_x = face_landmarks[291].x if len(face_landmarks) > 291 else 0.0
        
        # Compute asymmetry
        eye_asymmetry = abs(left_eye_y - right_eye_y)
        mouth_asymmetry = abs(left_mouth_x - right_mouth_x)
        
        # Combined asymmetry score
        asymmetry = (eye_asymmetry + mouth_asymmetry) / 2.0
        
        # Normalize
        asymmetry_score = min(1.0, asymmetry * 10.0)
        
        return float(asymmetry_score)
    except Exception:
        return 0.0


def extract_mediapipe_features(face_landmarks, hand_landmarks, face_history=None):
    """
    Extract all MediaPipe-derived features for GRU input.
    
    These features complement the existing ICU feature encoder by adding
    facial/gestural indicators.
    
    Args:
        face_landmarks: Current MediaPipe FaceMesh landmarks
        hand_landmarks: Current MediaPipe Hand landmarks (list)
        face_history: List of previous face landmark frames
    
    Returns:
        np.array of shape (N,) with MediaPipe features:
        [0] mouth_opening_ratio
        [1] hand_to_mouth_distance
        [2] hand_to_mouth_risk
        [3] lip_velocity
        [4] nose_forward_velocity
        [5] head_bobbing_rate (normalized)
        [6] facial_asymmetry
    """
    features = []
    
    # Mouth opening
    mouth_opening = compute_mouth_opening(face_landmarks)
    features.append(mouth_opening)
    
    # Hand to mouth (use closest hand)
    hand_to_mouth_dist = 1.0
    hand_to_mouth_risk = 0.0
    if hand_landmarks:
        for hand in hand_landmarks:
            dist, risk = compute_hand_to_mouth_distance(hand, face_landmarks)
            if dist < hand_to_mouth_dist:
                hand_to_mouth_dist = dist
                hand_to_mouth_risk = risk
    
    features.append(hand_to_mouth_dist)
    features.append(hand_to_mouth_risk)
    
    # Lip velocity
    lip_vel = compute_lip_velocity(face_history or [], fps=15.0)
    features.append(lip_vel)
    
    # Nose forward velocity
    nose_vel = compute_nose_forward_velocity(face_history or [], fps=15.0)
    features.append(nose_vel)
    
    # Head bobbing rate (normalize to 0-1)
    bobbing_rate = compute_head_bobbing_rate(face_history or [], fps=15.0)
    features.append(bobbing_rate / 2.0)  # Normalize (max 2 Hz)
    
    # Facial asymmetry
    asymmetry = compute_facial_asymmetry(face_landmarks)
    features.append(asymmetry)
    
    return np.array(features, dtype=np.float32)


def integrate_with_icu_features(icu_features, mediapipe_features):
    """
    Integrate MediaPipe features with ICU feature vector.
    
    Args:
        icu_features: 9-dim ICU feature vector
        mediapipe_features: 7-dim MediaPipe feature vector
    
    Returns:
        Combined feature vector (16-dim)
    """
    if icu_features is None:
        icu_features = np.zeros(9, dtype=np.float32)
    
    if mediapipe_features is None:
        mediapipe_features = np.zeros(7, dtype=np.float32)
    
    combined = np.concatenate([icu_features, mediapipe_features])
    return combined

