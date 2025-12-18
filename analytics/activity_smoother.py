# analytics/activity_smoother.py
"""
Temporal smoothing for activity classification.
Similar to posture smoother but for activity states.
"""
import logging
from collections import deque

log = logging.getLogger("activity_smoother")


class ActivityStateMachine:
    """
    State machine for activity classification with temporal smoothing.
    Uses hysteresis to prevent rapid activity changes.
    """
    
    def __init__(self, transition_threshold=8, history_size=15):
        """
        Args:
            transition_threshold: Number of consecutive frames required to change state
            history_size: Size of activity history buffer
        """
        self.current_state = "unknown"
        self.candidate_state = None
        self.candidate_count = 0
        self.transition_threshold = transition_threshold
        self.history = deque(maxlen=history_size)
        self.stable_count = 0
        
    def update(self, detected_activity, confidence=0.5):
        """
        Update state machine with new activity detection.
        
        Args:
            detected_activity: Detected activity state (from classifier)
            confidence: Confidence score (0-1)
        
        Returns:
            Smoothed activity state
        """
        # Weight by confidence - low confidence detections are less reliable
        if confidence < 0.5:
            # Low confidence - don't change state easily
            if detected_activity != self.current_state:
                # Require more frames for low-confidence transitions
                effective_threshold = self.transition_threshold * 2
            else:
                effective_threshold = self.transition_threshold
        else:
            effective_threshold = self.transition_threshold
        
        # Add to history
        self.history.append((detected_activity, confidence))
        
        # If same as current state, reset candidate
        if detected_activity == self.current_state:
            self.candidate_state = None
            self.candidate_count = 0
            self.stable_count += 1
            return self.current_state
        
        # If same as candidate, increment count
        if detected_activity == self.candidate_state:
            self.candidate_count += 1
        else:
            # New candidate state
            self.candidate_state = detected_activity
            self.candidate_count = 1
        
        # Check if we've seen enough consecutive frames of candidate state
        if self.candidate_count >= effective_threshold:
            old_state = self.current_state
            self.current_state = self.candidate_state
            self.candidate_count = 0
            self.stable_count = 0
            
            if old_state != self.current_state:
                log.info("Activity transition: %s → %s (after %d frames, confidence: %.2f)", 
                        old_state, self.current_state, effective_threshold, confidence)
        
        return self.current_state
    
    def get_confidence(self):
        """
        Get confidence in current state based on stability.
        
        Returns:
            Confidence score (0-1)
        """
        if self.current_state == "unknown":
            return 0.0
        
        # Confidence increases with stability
        stability_score = min(1.0, self.stable_count / self.transition_threshold)
        
        # Also consider history consistency and average confidence
        if len(self.history) > 0:
            recent_states = [(s, c) for s, c in list(self.history)[-self.transition_threshold:] if s == self.current_state]
            if recent_states:
                consistency = len(recent_states) / min(self.transition_threshold, len(self.history))
                avg_conf = sum(c for _, c in recent_states) / len(recent_states)
                return (stability_score + consistency + avg_conf) / 3.0
        
        return stability_score
    
    def reset(self):
        """Reset state machine to initial state."""
        self.current_state = "unknown"
        self.candidate_state = None
        self.candidate_count = 0
        self.stable_count = 0
        self.history.clear()
    
    def get_state(self):
        """Get current smoothed state."""
        return self.current_state

