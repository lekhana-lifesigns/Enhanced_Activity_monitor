# analytics/posture_smoother.py
"""
Temporal smoothing for posture classification.
Implements state machine with hysteresis to prevent rapid posture changes.
"""
import logging
from collections import deque

log = logging.getLogger("posture_smoother")


class PostureStateMachine:
    """
    State machine for posture classification with temporal smoothing.
    Uses hysteresis to prevent rapid state changes.
    """
    
    def __init__(self, transition_threshold=5, history_size=10):
        """
        Args:
            transition_threshold: Number of consecutive frames required to change state
            history_size: Size of posture history buffer
        """
        self.current_state = "unknown"
        self.candidate_state = None
        self.candidate_count = 0
        self.transition_threshold = transition_threshold
        self.history = deque(maxlen=history_size)
        self.stable_count = 0
        
    def update(self, detected_posture):
        """
        Update state machine with new posture detection.
        
        Args:
            detected_posture: Detected posture state (from classifier)
        
        Returns:
            Smoothed posture state
        """
        # Add to history
        self.history.append(detected_posture)
        
        # If same as current state, reset candidate
        if detected_posture == self.current_state:
            self.candidate_state = None
            self.candidate_count = 0
            self.stable_count += 1
            return self.current_state
        
        # If same as candidate, increment count
        if detected_posture == self.candidate_state:
            self.candidate_count += 1
        else:
            # New candidate state
            self.candidate_state = detected_posture
            self.candidate_count = 1
        
        # Check if we've seen enough consecutive frames of candidate state
        if self.candidate_count >= self.transition_threshold:
            old_state = self.current_state
            self.current_state = self.candidate_state
            self.candidate_count = 0
            self.stable_count = 0
            
            if old_state != self.current_state:
                log.info("Posture transition: %s → %s (after %d frames)", 
                        old_state, self.current_state, self.transition_threshold)
        
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
        
        # Also consider history consistency
        if len(self.history) > 0:
            recent_states = list(self.history)[-self.transition_threshold:]
            consistency = sum(1 for s in recent_states if s == self.current_state) / len(recent_states)
            return (stability_score + consistency) / 2.0
        
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


class PostureSmoother:
    """
    High-level posture smoother with multiple smoothing strategies.
    """
    
    def __init__(self, transition_threshold=5, history_size=15, use_majority_vote=True):
        """
        Args:
            transition_threshold: Frames required for state transition
            history_size: Size of history buffer
            use_majority_vote: Use majority voting as additional smoothing
        """
        self.state_machine = PostureStateMachine(transition_threshold, history_size)
        self.use_majority_vote = use_majority_vote
        self.history = deque(maxlen=history_size)
    
    def smooth(self, detected_posture):
        """
        Smooth posture detection.
        
        Args:
            detected_posture: Raw posture from classifier
        
        Returns:
            Smoothed posture state
        """
        # Add to history
        self.history.append(detected_posture)
        
        # Update state machine
        state_machine_result = self.state_machine.update(detected_posture)
        
        # Optional: Majority vote smoothing
        if self.use_majority_vote and len(self.history) >= 3:
            # Get most common state in recent history
            recent = list(self.history)[-5:]  # Last 5 frames
            from collections import Counter
            most_common = Counter(recent).most_common(1)[0]
            
            # If majority vote differs from state machine, use it if consistent
            if most_common[1] >= 3 and most_common[0] != state_machine_result:
                # Only use if it's been consistent
                if self.state_machine.candidate_state == most_common[0]:
                    return most_common[0]
        
        return state_machine_result
    
    def get_confidence(self):
        """Get confidence in smoothed posture."""
        return self.state_machine.get_confidence()
    
    def get_state(self):
        """Get current smoothed state."""
        return self.state_machine.get_state()
    
    def reset(self):
        """Reset smoother."""
        self.state_machine.reset()
        self.history.clear()

