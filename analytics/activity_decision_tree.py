# analytics/activity_decision_tree.py
# Early-Exit Decision Tree for Activity Classification

import numpy as np
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

log = logging.getLogger("activity_decision_tree")

# COCO Keypoint indices
NOSE = 0
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


class ActivityDecisionTree:
 """
 Priority-based decision tree with early exit and shared computations.

 - Decision trees enable O(log n) classification
 - Early exit reduces computation for common cases
 - Shared feature computation avoids redundancy

 """

 def __init__(self):
 # Pre-computed thresholds for fast binary decisions
 self.thresholds = {
 'critical_motion': 0.8, # High movement = seizure/agitation
 'stillness': 0.005, # Very low = sleeping/unresponsive
 'moderate_movement': 0.05, # Moderate = repositioning/restless
 'vertical_ratio_lying': 0.8, # Low ratio = lying
 'vertical_ratio_standing': 1.5, # High ratio = standing
 'leg_motion': 0.02, # Leg movement threshold
 'hand_near_face': 0.2, # Hand proximity to face
 'seizure_variance': 0.01, # Movement variance for seizure
 }

 # Shared computation cache (cleared each frame)
 self.frame_cache = {}

 def classify_with_early_exit(self, kps: np.ndarray, kps_history: Optional[List],
 context: Optional[Dict] = None) -> Dict:
 """
 Classify activity using decision tree with early exit.


 Args:
 kps: Current keypoints (17, 3)
 kps_history: History of keypoints
 context: Additional context (fall_detected, bed_info, etc.)

 Returns:
 Activity classification result
 """
 # Clear frame cache
 self.frame_cache.clear()

 context = context or {}

 # LEVEL 0: Pre-compute shared features (done once, reused many times)
 shared = self._precompute_shared_features(kps, kps_history)

 # LEVEL 1: Critical activities (highest priority, immediate exit)
 # Fast motion-based critical checks first
 if context.get('fall_detected'):
 return self._create_result('falling', 0.9, 'CRITICAL',
 {'fall_detected': True})

 if shared['avg_movement'] > self.thresholds['critical_motion']:
 # High movement - check for seizure/agitation
 if shared['movement_variance'] > self.thresholds['seizure_variance']:
 return self._create_result('seizure', 0.85, 'CRITICAL',
 {'seizure_score': 0.85})
 else:
 return self._create_result('agitated', 0.8, 'CRITICAL',
 {'agitation_score': 0.8})

 # LEVEL 2: Stillness-based activities (second fastest)
 if shared['avg_movement'] < self.thresholds['stillness']:
 if shared['lying_posture']:
 return self._create_result('sleeping', 0.85, 'NORMAL',
 {'state': 'sleep'})
 else:
 return self._create_result('still', 0.75, 'NORMAL',
 {'movement_level': 'very_low'})

 # LEVEL 3: Posture-based classification (medium complexity)
 # Binary decision on vertical aspect ratio
 if shared['vertical_ratio'] < self.thresholds['vertical_ratio_lying']:
 # Lying posture subtree
 return self._classify_lying_activities(kps, shared, context)

 elif shared['vertical_ratio'] > self.thresholds['vertical_ratio_standing']:
 # Standing/upright posture subtree
 return self._classify_standing_activities(kps, shared, kps_history)

 else:
 # Sitting posture (between lying and standing)
 return self._classify_sitting_activities(kps, shared, context)

 def _precompute_shared_features(self, kps: np.ndarray,
 kps_history: Optional[List]) -> Dict:
 """
 Compute features once, reuse across all classification checks.


 Returns:
 Dict of pre-computed features
 """
 cache = {}

 # Convert to numpy array
 kps_array = np.asarray(kps, dtype=np.float32)

 # Movement features (used by critical activity detection)
 if kps_history and len(kps_history) >= 3:
 history_array = np.array(list(kps_history)[-5:]) # Last 5 frames

 # Vectorized movement computation
 hip_positions = history_array[:, LEFT_HIP, :2] # (T, 2)
 movements = np.linalg.norm(np.diff(hip_positions, axis=0), axis=1) # (T-1,)

 cache['avg_movement'] = float(np.mean(movements))
 cache['movement_variance'] = float(np.var(movements))
 cache['max_movement'] = float(np.max(movements))

 # Leg motion indicator
 ankle_positions = history_array[:, LEFT_ANKLE, :2]
 ankle_movements = np.linalg.norm(np.diff(ankle_positions, axis=0), axis=1)
 cache['has_leg_motion'] = np.mean(ankle_movements) > self.thresholds['leg_motion']
 else:
 cache['avg_movement'] = 0.0
 cache['movement_variance'] = 0.0
 cache['max_movement'] = 0.0
 cache['has_leg_motion'] = False

 # Posture features (vectorized)
 valid_mask = kps_array[:, 2] > MIN_CONFIDENCE
 if np.sum(valid_mask) > 0:
 y_coords = kps_array[valid_mask, 1]
 x_coords = kps_array[valid_mask, 0]

 cache['vertical_extent'] = float(np.ptp(y_coords)) # Range (max - min)
 cache['horizontal_extent'] = float(np.ptp(x_coords))
 cache['vertical_ratio'] = cache['vertical_extent'] / (cache['horizontal_extent'] + 1e-6)
 cache['lying_posture'] = cache['vertical_ratio'] < self.thresholds['vertical_ratio_lying']
 else:
 cache['vertical_extent'] = 0.0
 cache['horizontal_extent'] = 0.0
 cache['vertical_ratio'] = 1.0
 cache['lying_posture'] = False

 # Hand proximity (for tube-pulling, agitation)
 nose = kps_array[NOSE, :2]
 left_wrist = kps_array[LEFT_WRIST, :2]
 right_wrist = kps_array[RIGHT_WRIST, :2]

 left_dist = np.linalg.norm(left_wrist - nose)
 right_dist = np.linalg.norm(right_wrist - nose)
 cache['hand_near_face'] = min(left_dist, right_dist) < self.thresholds['hand_near_face']

 return cache

 def _classify_lying_activities(self, kps: np.ndarray, shared: Dict,
 context: Dict) -> Dict:
 """
 Classify lying-posture activities (subtree).

 Binary decisions for lying activities.
 """
 # Check if on bed
 if context.get('person_on_bed'):
 # On bed - check movement level
 if shared['avg_movement'] > self.thresholds['moderate_movement']:
 return self._create_result('turning_in_bed', 0.75, 'NORMAL',
 {'movement_type': 'turning'})
 else:
 return self._create_result('lying_in_bed', 0.85, 'NORMAL',
 {'on_bed': True})
 else:
 # Not on bed
 if shared['avg_movement'] < self.thresholds['stillness']:
 return self._create_result('lying', 0.8, 'NORMAL',
 {'posture': 'lying'})
 else:
 return self._create_result('repositioning_in_bed', 0.7, 'NORMAL',
 {'movement_type': 'repositioning'})

 def _classify_standing_activities(self, kps: np.ndarray, shared: Dict,
 kps_history: Optional[List]) -> Dict:
 """
 Classify standing/upright activities (subtree).

 Binary tree for locomotion.
 """
 # Check leg motion
 if shared['has_leg_motion']:
 # Locomotion subtree
 if shared['avg_movement'] > 0.05:
 return self._create_result('running', 0.7, 'HIGH',
 {'locomotion_type': 'running'})
 else:
 return self._create_result('walking', 0.75, 'NORMAL',
 {'locomotion_type': 'walking'})
 else:
 # No leg motion - standing still or reaching
 if shared['hand_near_face']:
 return self._create_result('reaching', 0.7, 'NORMAL',
 {'arm_position': 'extended'})
 else:
 return self._create_result('standing', 0.85, 'NORMAL',
 {'posture': 'standing'})

 def _classify_sitting_activities(self, kps: np.ndarray, shared: Dict,
 context: Dict) -> Dict:
 """
 Classify sitting activities (subtree).
 """
 if context.get('person_on_bed'):
 return self._create_result('sitting_on_bed', 0.8, 'NORMAL',
 {'on_bed': True})
 else:
 return self._create_result('sitting', 0.85, 'NORMAL',
 {'posture': 'sitting'})

 def _create_result(self, activity: str, confidence: float,
 priority: str, details: Dict) -> Dict:
 """
 Create standardized classification result.

 Args:
 activity: Activity name
 confidence: Confidence score (0-1)
 priority: Priority level
 details: Additional details

 Returns:
 Classification result dict
 """
 return {
 'activity': activity,
 'confidence': confidence,
 'priority': priority,
 'details': details,
 'early_exit': True,
 }


# DYNAMIC PROGRAMMING FOR ACTIVITY SEQUENCE OPTIMIZATION

class ActivitySequenceOptimizer:
 """
 Dynamic programming for optimal activity sequence prediction.

 - Dynamic programming for sequence optimization
 - Viterbi-like algorithm for most probable sequence
 - Temporal consistency through transition costs

 """

 def __init__(self, activity_labels: List[str],
 transition_costs: Optional[Dict] = None):
 """
 Args:
 activity_labels: List of all possible activity names
 transition_costs: Dict of {(from, to): cost} (lower = more likely)
 """
 self.labels = activity_labels
 self.num_states = len(activity_labels)

 # Transition cost matrix
 if transition_costs is None:
 self.transition_costs = self._default_transition_costs()
 else:
 self.transition_costs = transition_costs

 def _default_transition_costs(self) -> Dict[Tuple[str, str], float]:
 """
 Define reasonable transition costs based on activity logic.


 Lower cost = more likely transition
 """
 costs = {}

 # Likely transitions (low cost = 0.1)
 likely = [
 ('lying', 'sitting'),
 ('sitting', 'standing'),
 ('standing', 'walking'),
 ('walking', 'standing'),
 ('sitting', 'lying'),
 ('lying_in_bed', 'turning_in_bed'),
 ('turning_in_bed', 'lying_in_bed'),
 ('standing', 'sitting'),
 ('walking', 'running'),
 ('running', 'walking'),
 ]
 for from_act, to_act in likely:
 costs[(from_act, to_act)] = 0.1

 # Unlikely transitions (high cost = 0.9)
 unlikely = [
 ('sleeping', 'running'),
 ('lying', 'running'),
 ('fallen', 'walking'),
 ('seizure', 'standing'),
 ]
 for from_act, to_act in unlikely:
 costs[(from_act, to_act)] = 0.9

 # Self-transitions (very low cost = 0.05)
 for act in self.labels:
 costs[(act, act)] = 0.05

 # Default for unspecified pairs (medium cost = 0.5)
 for from_act in self.labels:
 for to_act in self.labels:
 if (from_act, to_act) not in costs:
 costs[(from_act, to_act)] = 0.5

 return costs

 def optimize_sequence(self, activity_sequence: List[str],
 confidence_sequence: List[float]) -> List[str]:
 """
 Optimize activity sequence using dynamic programming (Viterbi algorithm).


 Args:
 activity_sequence: List of activity predictions (raw)
 confidence_sequence: List of confidence scores

 Returns:
 Optimized activity sequence with better temporal consistency
 """
 if len(activity_sequence) < 2:
 return activity_sequence

 T = len(activity_sequence)

 # DP table: dp[t][state] = (max_score, best_prev_state)
 dp = [{} for _ in range(T)]

 # Initialize first frame
 first_activity = activity_sequence[0]
 first_conf = confidence_sequence[0]
 dp[0][first_activity] = (first_conf, None)

 # Forward pass (DP recurrence)
 for t in range(1, T):
 current_activity = activity_sequence[t]
 current_conf = confidence_sequence[t]

 # Try all possible previous states
 for prev_state in dp[t-1]:
 prev_score, _ = dp[t-1][prev_state]

 # Transition cost
 transition_cost = self.transition_costs.get(
 (prev_state, current_activity),
 0.5 # Default cost
 )

 # Total score: accumulated score + current confidence - transition cost
 score = prev_score + current_conf - transition_cost

 # Update if better score
 if current_activity not in dp[t] or score > dp[t][current_activity][0]:
 dp[t][current_activity] = (score, prev_state)

 # Backward pass: reconstruct best path
 optimized_sequence = [None] * T

 # Find best final state
 final_activities = dp[T-1]
 best_final = max(final_activities.items(), key=lambda x: x[1][0])
 optimized_sequence[T-1] = best_final[0]

 # Backtrack
 for t in range(T-2, -1, -1):
 _, prev_state = dp[t+1][optimized_sequence[t+1]]
 optimized_sequence[t] = prev_state

 return optimized_sequence

 def learn_transitions(self, activity_sequences: List[List[str]]):
 """
 Learn transition costs from historical data.


 Args:
 activity_sequences: List of activity sequences for learning
 """
 # Count transitions
 transition_counts = defaultdict(int)
 total_transitions = 0

 for sequence in activity_sequences:
 for i in range(len(sequence) - 1):
 transition = (sequence[i], sequence[i+1])
 transition_counts[transition] += 1
 total_transitions += 1

 # Convert counts to costs (inverse probability)
 for transition, count in transition_counts.items():
 probability = count / total_transitions
 # Cost = -log(probability) normalized to [0, 1]
 cost = -np.log(probability + 1e-6)
 cost = np.clip(cost / 5.0, 0.0, 1.0) # Normalize
 self.transition_costs[transition] = float(cost)
