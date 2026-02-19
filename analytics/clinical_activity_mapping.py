# analytics/clinical_activity_mapping.py
"""
Clinical Activity Mapping for EAC System
Maps activities to relevant features, defines alert thresholds, and provides nurse-facing explanations.
Based on clinical requirements for timely detection, persistence, and explainability.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

log = logging.getLogger("clinical_mapping")

# ACTIVITY TO FEATURE MAPPING
# Complete mapping of all 56 activities for hospital-scale clinical analysis

# Feature indices from ICUFeatureEncoder:
# [0] motion_energy, [1] jerk_index, [2] posture_instability, [3] sway_score,
# [4] breath_rate_proxy (normalized 0-1), [5] hand_proximity_risk,
# [6] motor_entropy, [7] symmetry_index, [8] motion_variability (normalized 0-1)

# Activity detection types:
# - "static": Posture-based, detected from single frame or short window
# - "dynamic": Motion-based, requires temporal analysis
# - "transitional": State change detection
# - "behavioral": Pattern-based over longer windows

ACTIVITY_FEATURE_MAPPING = {
 # =========================================================================
 # BASIC POSTURAL ACTIVITIES (Static Detection)
 # =========================================================================
 "lying": {
 "name": "Lying Down",
 "detection_type": "static",
 "observation_window": 5,
 "relevant_joints": ["left_hip", "right_hip", "left_shoulder", "right_shoulder", "spine"],
 "feature_weights": {2: 0.4, 3: 0.3, 7: 0.3},
 "clinical_meaning": "Rest state, pressure ulcer risk if prolonged",
 "risk_type": "postural",
 "long_term_indicators": ["time_in_position", "position_changes_per_hour"]
 },
 "sitting": {
 "name": "Sitting",
 "detection_type": "static",
 "observation_window": 10,
 "relevant_joints": ["left_hip", "right_hip", "left_knee", "right_knee", "spine", "neck"],
 "feature_weights": {2: 0.4, 3: 0.3, 6: 0.2, 7: 0.1},
 "clinical_meaning": "Fatigue, recovery, readiness for mobility",
 "risk_type": "postural",
 "long_term_indicators": ["sitting_duration", "posture_quality"]
 },
 "standing": {
 "name": "Standing",
 "detection_type": "static",
 "observation_window": 5,
 "relevant_joints": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "spine"],
 "feature_weights": {2: 0.3, 3: 0.4, 6: 0.2, 7: 0.1},
 "clinical_meaning": "Orthostatic risk, mobility readiness",
 "risk_type": "balance",
 "long_term_indicators": ["standing_tolerance", "sway_patterns"]
 },
 "kneeling": {
 "name": "Kneeling",
 "detection_type": "static",
 "observation_window": 5,
 "relevant_joints": ["left_knee", "right_knee", "left_hip", "right_hip", "left_ankle", "right_ankle"],
 "feature_weights": {2: 0.5, 3: 0.3, 7: 0.2},
 "clinical_meaning": "Unusual position, may indicate fall recovery or distress",
 "risk_type": "safety",
 "long_term_indicators": ["kneeling_frequency", "context_analysis"]
 },
 "crouching": {
 "name": "Crouching",
 "detection_type": "static",
 "observation_window": 5,
 "relevant_joints": ["left_knee", "right_knee", "left_hip", "right_hip", "spine"],
 "feature_weights": {2: 0.4, 3: 0.3, 6: 0.3},
 "clinical_meaning": "Pain, distress, or fall recovery",
 "risk_type": "safety",
 "long_term_indicators": ["crouching_frequency", "duration_patterns"]
 },

 # =========================================================================
 # LOCOMOTION ACTIVITIES (Dynamic Detection)
 # =========================================================================
 "walking": {
 "name": "Walking",
 "detection_type": "dynamic",
 "observation_window": 60,
 "relevant_joints": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
 "feature_weights": {0: 0.4, 1: 0.3, 3: 0.2, 6: 0.1},
 "clinical_meaning": "Fall risk, mobility assessment, restlessness",
 "risk_type": "mobility",
 "long_term_indicators": ["gait_quality", "walking_distance", "walking_speed", "unsteadiness_index"]
 },
 "running": {
 "name": "Running",
 "detection_type": "dynamic",
 "observation_window": 30,
 "relevant_joints": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
 "feature_weights": {0: 0.5, 1: 0.3, 6: 0.2},
 "clinical_meaning": "Agitation, elopement risk, emergency",
 "risk_type": "safety",
 "long_term_indicators": ["running_episodes", "context_triggers"]
 },
 "crawling": {
 "name": "Crawling",
 "detection_type": "dynamic",
 "observation_window": 30,
 "relevant_joints": ["left_wrist", "right_wrist", "left_knee", "right_knee", "left_ankle", "right_ankle"],
 "feature_weights": {0: 0.4, 1: 0.3, 2: 0.3},
 "clinical_meaning": "Post-fall, weakness, confusion",
 "risk_type": "safety",
 "long_term_indicators": ["crawling_episodes", "associated_falls"]
 },
 "rolling": {
 "name": "Rolling",
 "detection_type": "dynamic",
 "observation_window": 30,
 "relevant_joints": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "spine"],
 "feature_weights": {0: 0.3, 6: 0.4, 7: 0.3},
 "clinical_meaning": "Restlessness, pain, seizure activity",
 "risk_type": "behavioral",
 "long_term_indicators": ["rolling_frequency", "time_of_day_patterns"]
 },
 "moving": {
 "name": "General Movement",
 "detection_type": "dynamic",
 "observation_window": 30,
 "relevant_joints": ["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
 "feature_weights": {0: 0.5, 6: 0.3, 8: 0.2},
 "clinical_meaning": "Activity level, restlessness",
 "risk_type": "mobility",
 "long_term_indicators": ["activity_level", "movement_patterns"]
 },
 "tripping": {
 "name": "Tripping",
 "detection_type": "dynamic",
 "observation_window": 5,
 "relevant_joints": ["left_ankle", "right_ankle", "left_hip", "right_hip", "spine"],
 "feature_weights": {0: 0.3, 1: 0.4, 3: 0.3},
 "clinical_meaning": "Immediate fall risk, gait instability",
 "risk_type": "safety",
 "long_term_indicators": ["tripping_frequency", "fall_correlation"]
 },
 "roaming": {
 "name": "Roaming/Wandering",
 "detection_type": "behavioral",
 "observation_window": 300,
 "relevant_joints": ["left_hip", "right_hip", "left_ankle", "right_ankle"],
 "feature_weights": {0: 0.3, 6: 0.4, 8: 0.3},
 "clinical_meaning": "Confusion, elopement risk, agitation",
 "risk_type": "safety",
 "long_term_indicators": ["wandering_episodes", "time_patterns", "elopement_attempts"]
 },

 # =========================================================================
 # BED-RELATED ACTIVITIES (Transitional/Static)
 # =========================================================================
 "lying_in_bed": {
 "name": "Lying in Bed",
 "detection_type": "static",
 "observation_window": 10,
 "relevant_joints": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "spine"],
 "feature_weights": {2: 0.3, 3: 0.3, 7: 0.4},
 "clinical_meaning": "Normal rest, pressure ulcer risk if prolonged",
 "risk_type": "postural",
 "long_term_indicators": ["time_in_bed", "position_distribution", "turn_frequency"]
 },
 "sitting_on_bed": {
 "name": "Sitting on Bed",
 "detection_type": "static",
 "observation_window": 10,
 "relevant_joints": ["left_hip", "right_hip", "left_knee", "right_knee", "spine"],
 "feature_weights": {2: 0.4, 3: 0.3, 6: 0.3},
 "clinical_meaning": "Transfer preparation, fall risk",
 "risk_type": "safety",
 "long_term_indicators": ["sitting_edge_frequency", "transfer_success_rate"]
 },
 "bed_exit": {
 "name": "Bed Exit",
 "detection_type": "transitional",
 "observation_window": 0,
 "relevant_joints": ["left_hip", "right_hip", "left_ankle", "right_ankle", "spine"],
 "feature_weights": {0: 0.4, 1: 0.3, 3: 0.3},
 "clinical_meaning": "Fall risk, unsupervised mobility",
 "risk_type": "safety",
 "long_term_indicators": ["bed_exit_times", "night_exits", "fall_events_after_exit"]
 },
 "bed_entry": {
 "name": "Bed Entry",
 "detection_type": "transitional",
 "observation_window": 0,
 "relevant_joints": ["left_hip", "right_hip", "left_ankle", "right_ankle", "spine"],
 "feature_weights": {0: 0.3, 1: 0.3, 3: 0.4},
 "clinical_meaning": "Fatigue, distress timing",
 "risk_type": "behavioral",
 "long_term_indicators": ["bed_entry_times", "sleep_pattern_analysis"]
 },
 "turning_in_bed": {
 "name": "Turning in Bed",
 "detection_type": "dynamic",
 "observation_window": 30,
 "relevant_joints": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "spine"],
 "feature_weights": {0: 0.3, 6: 0.4, 7: 0.3},
 "clinical_meaning": "Restlessness, pain, pressure relief",
 "risk_type": "behavioral",
 "long_term_indicators": ["turn_frequency", "time_distribution"]
 },
 "sitting_up_in_bed": {
 "name": "Sitting Up in Bed",
 "detection_type": "transitional",
 "observation_window": 10,
 "relevant_joints": ["left_hip", "right_hip", "spine", "neck", "left_shoulder", "right_shoulder"],
 "feature_weights": {0: 0.3, 2: 0.4, 3: 0.3},
 "clinical_meaning": "Mobility readiness, independence indicator",
 "risk_type": "mobility",
 "long_term_indicators": ["independence_score", "assistance_required"]
 },

 # =========================================================================
 # TRANSFER ACTIVITIES (Transitional)
 # =========================================================================
 "transferring": {
 "name": "Transferring",
 "detection_type": "transitional",
 "observation_window": 30,
 "relevant_joints": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
 "feature_weights": {0: 0.3, 1: 0.3, 3: 0.4},
 "clinical_meaning": "High fall risk period",
 "risk_type": "safety",
 "long_term_indicators": ["transfer_success_rate", "assistance_level"]
 },
 "pivoting": {
 "name": "Pivoting",
 "detection_type": "dynamic",
 "observation_window": 15,
 "relevant_joints": ["left_hip", "right_hip", "left_ankle", "right_ankle", "spine"],
 "feature_weights": {0: 0.2, 3: 0.5, 6: 0.3},
 "clinical_meaning": "Balance assessment during transfers",
 "risk_type": "balance",
 "long_term_indicators": ["pivot_stability", "balance_trends"]
 },

 # =========================================================================
 # UPPER BODY ACTIVITIES (Dynamic)
 # =========================================================================
 "reaching": {
 "name": "Reaching",
 "detection_type": "dynamic",
 "observation_window": 15,
 "relevant_joints": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
 "feature_weights": {0: 0.4, 1: 0.3, 7: 0.3},
 "clinical_meaning": "Assistance needs, overreaching risk",
 "risk_type": "safety",
 "long_term_indicators": ["reaching_frequency", "assistance_needs"]
 },
 "waving": {
 "name": "Waving",
 "detection_type": "dynamic",
 "observation_window": 10,
 "relevant_joints": ["left_wrist", "right_wrist", "left_elbow", "right_elbow"],
 "feature_weights": {0: 0.4, 6: 0.4, 8: 0.2},
 "clinical_meaning": "Communication attempt, attention seeking",
 "risk_type": "behavioral",
 "long_term_indicators": ["communication_attempts", "response_patterns"]
 },
 "pointing": {
 "name": "Pointing",
 "detection_type": "dynamic",
 "observation_window": 10,
 "relevant_joints": ["left_wrist", "right_wrist", "left_elbow", "right_elbow"],
 "feature_weights": {0: 0.3, 7: 0.4, 8: 0.3},
 "clinical_meaning": "Communication gesture",
 "risk_type": "behavioral",
 "long_term_indicators": ["communication_frequency"]
 },
 "arm_raising": {
 "name": "Arm Raising",
 "detection_type": "dynamic",
 "observation_window": 10,
 "relevant_joints": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
 "feature_weights": {0: 0.4, 7: 0.3, 8: 0.3},
 "clinical_meaning": "Exercise, distress signal, stroke symptom",
 "risk_type": "neurological",
 "long_term_indicators": ["arm_strength_symmetry", "range_of_motion"]
 },
 "hand_to_face": {
 "name": "Hand to Face",
 "detection_type": "dynamic",
 "observation_window": 15,
 "relevant_joints": ["left_wrist", "right_wrist", "nose", "left_eye", "right_eye"],
 "feature_weights": {5: 0.6, 0: 0.2, 6: 0.2},
 "clinical_meaning": "Device interference risk, self-soothing, fatigue",
 "risk_type": "device_safety",
 "long_term_indicators": ["face_touch_frequency", "device_interference_events"]
 },

 # =========================================================================
 # LOWER BODY ACTIVITIES (Dynamic)
 # =========================================================================
 "leg_movement": {
 "name": "Leg Movement",
 "detection_type": "dynamic",
 "observation_window": 30,
 "relevant_joints": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
 "feature_weights": {0: 0.4, 6: 0.3, 8: 0.3},
 "clinical_meaning": "Restlessness, exercise, discomfort",
 "risk_type": "behavioral",
 "long_term_indicators": ["leg_activity_patterns", "restless_leg_index"]
 },
 "foot_movement": {
 "name": "Foot Movement",
 "detection_type": "dynamic",
 "observation_window": 30,
 "relevant_joints": ["left_ankle", "right_ankle"],
 "feature_weights": {0: 0.3, 6: 0.4, 8: 0.3},
 "clinical_meaning": "Anxiety, restlessness, neurological activity",
 "risk_type": "behavioral",
 "long_term_indicators": ["foot_tapping_frequency", "anxiety_correlation"]
 },

 # =========================================================================
 # AGITATION AND DISTRESS ACTIVITIES (Behavioral)
 # =========================================================================
 "restless": {
 "name": "Restless",
 "detection_type": "behavioral",
 "observation_window": 120,
 "relevant_joints": ["left_hip", "right_hip", "left_shoulder", "right_shoulder", "spine"],
 "feature_weights": {0: 0.3, 6: 0.4, 8: 0.3},
 "clinical_meaning": "Pain, anxiety, delirium",
 "risk_type": "behavioral",
 "long_term_indicators": ["restlessness_score", "time_patterns", "pain_correlation"]
 },
 "agitated": {
 "name": "Agitated",
 "detection_type": "behavioral",
 "observation_window": 60,
 "relevant_joints": ["left_wrist", "right_wrist", "left_hip", "right_hip", "spine"],
 "feature_weights": {0: 0.4, 1: 0.3, 6: 0.3},
 "clinical_meaning": "Immediate intervention required",
 "risk_type": "safety",
 "long_term_indicators": ["agitation_episodes", "intervention_effectiveness", "trigger_patterns"]
 },
 "pulling_at_tubes": {
 "name": "Pulling at Tubes/Devices",
 "detection_type": "dynamic",
 "observation_window": 0,
 "relevant_joints": ["left_wrist", "right_wrist", "left_elbow", "right_elbow"],
 "feature_weights": {5: 0.7, 0: 0.2, 1: 0.1},
 "clinical_meaning": "Device safety, self-extubation risk",
 "risk_type": "critical_device",
 "long_term_indicators": ["device_interference_events", "restraint_needs"]
 },
 "thrashing": {
 "name": "Thrashing",
 "detection_type": "dynamic",
 "observation_window": 30,
 "relevant_joints": ["left_wrist", "right_wrist", "left_ankle", "right_ankle", "spine"],
 "feature_weights": {0: 0.4, 1: 0.4, 6: 0.2},
 "clinical_meaning": "Seizure, severe agitation, distress",
 "risk_type": "safety",
 "long_term_indicators": ["thrashing_episodes", "seizure_correlation"]
 },
 "rocking": {
 "name": "Rocking",
 "detection_type": "behavioral",
 "observation_window": 60,
 "relevant_joints": ["left_hip", "right_hip", "spine", "left_shoulder", "right_shoulder"],
 "feature_weights": {0: 0.3, 6: 0.5, 7: 0.2},
 "clinical_meaning": "Self-soothing, agitation, neurological condition",
 "risk_type": "behavioral",
 "long_term_indicators": ["rocking_frequency", "duration_patterns"]
 },

 # =========================================================================
 # FALL-RELATED ACTIVITIES (Transitional/Critical)
 # =========================================================================
 "falling": {
 "name": "Falling",
 "detection_type": "transitional",
 "observation_window": 0,
 "relevant_joints": ["left_hip", "right_hip", "left_shoulder", "right_shoulder", "spine"],
 "feature_weights": {0: 0.3, 1: 0.5, 3: 0.2},
 "clinical_meaning": "Emergency response required",
 "risk_type": "critical_safety",
 "long_term_indicators": ["fall_count", "fall_circumstances", "injury_severity"]
 },
 "fallen": {
 "name": "Fallen (On Ground)",
 "detection_type": "static",
 "observation_window": 0,
 "relevant_joints": ["left_hip", "right_hip", "left_shoulder", "right_shoulder", "spine"],
 "feature_weights": {2: 0.5, 3: 0.3, 7: 0.2},
 "clinical_meaning": "Immediate assistance required",
 "risk_type": "critical_safety",
 "long_term_indicators": ["time_on_ground", "injury_assessment"]
 },
 "getting_up_from_floor": {
 "name": "Getting Up from Floor",
 "detection_type": "transitional",
 "observation_window": 30,
 "relevant_joints": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
 "feature_weights": {0: 0.4, 1: 0.3, 3: 0.3},
 "clinical_meaning": "Fall recovery, assistance assessment",
 "risk_type": "safety",
 "long_term_indicators": ["recovery_time", "assistance_required"]
 },

 # =========================================================================
 # NEUROLOGICAL ACTIVITIES (Dynamic/Critical)
 # =========================================================================
 "seizure": {
 "name": "Seizure",
 "detection_type": "dynamic",
 "observation_window": 30,
 "relevant_joints": ["left_wrist", "right_wrist", "left_ankle", "right_ankle", "spine", "neck"],
 "feature_weights": {0: 0.3, 1: 0.4, 6: 0.3},
 "clinical_meaning": "Emergency medical intervention",
 "risk_type": "critical_neurological",
 "long_term_indicators": ["seizure_frequency", "seizure_duration", "seizure_type"]
 },
 "convulsion": {
 "name": "Convulsion",
 "detection_type": "dynamic",
 "observation_window": 30,
 "relevant_joints": ["left_wrist", "right_wrist", "left_ankle", "right_ankle", "spine"],
 "feature_weights": {0: 0.4, 1: 0.4, 6: 0.2},
 "clinical_meaning": "Emergency response required",
 "risk_type": "critical_neurological",
 "long_term_indicators": ["convulsion_count", "severity_trend"]
 },
 "tremor": {
 "name": "Tremor",
 "detection_type": "dynamic",
 "observation_window": 60,
 "relevant_joints": ["left_wrist", "right_wrist", "left_elbow", "right_elbow"],
 "feature_weights": {0: 0.3, 1: 0.3, 6: 0.4},
 "clinical_meaning": "Medication effects, neurological condition",
 "risk_type": "neurological",
 "long_term_indicators": ["tremor_frequency", "tremor_amplitude", "medication_correlation"]
 },
 "rigidity": {
 "name": "Rigidity",
 "detection_type": "static",
 "observation_window": 60,
 "relevant_joints": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "spine"],
 "feature_weights": {0: 0.2, 6: 0.4, 7: 0.4},
 "clinical_meaning": "Neurological condition, medication effect",
 "risk_type": "neurological",
 "long_term_indicators": ["rigidity_duration", "progression_trend"]
 },

 # =========================================================================
 # RESPIRATORY ACTIVITIES (Dynamic)
 # =========================================================================
 "breathing": {
 "name": "Breathing",
 "detection_type": "dynamic",
 "observation_window": 30,
 "relevant_joints": ["left_shoulder", "right_shoulder", "spine"],
 "feature_weights": {4: 0.6, 0: 0.2, 6: 0.2},
 "clinical_meaning": "Vital sign monitoring",
 "risk_type": "respiratory",
 "long_term_indicators": ["breathing_rate", "breathing_pattern", "apnea_events"]
 },
 "coughing": {
 "name": "Coughing",
 "detection_type": "dynamic",
 "observation_window": 120,
 "relevant_joints": ["left_shoulder", "right_shoulder", "spine", "neck"],
 "feature_weights": {0: 0.3, 1: 0.4, 4: 0.2, 6: 0.1},
 "clinical_meaning": "Respiratory distress, infection indicator",
 "risk_type": "respiratory",
 "long_term_indicators": ["cough_frequency", "cough_severity", "pattern_changes"]
 },
 "gasping": {
 "name": "Gasping",
 "detection_type": "dynamic",
 "observation_window": 0,
 "relevant_joints": ["left_shoulder", "right_shoulder", "spine", "neck"],
 "feature_weights": {0: 0.3, 1: 0.4, 4: 0.3},
 "clinical_meaning": "Respiratory emergency",
 "risk_type": "critical_respiratory",
 "long_term_indicators": ["gasping_episodes", "oxygen_correlation"]
 },

 # =========================================================================
 # EATING AND DRINKING ACTIVITIES (Behavioral)
 # =========================================================================
 "eating": {
 "name": "Eating",
 "detection_type": "behavioral",
 "observation_window": 300,
 "relevant_joints": ["left_wrist", "right_wrist", "left_elbow", "right_elbow", "nose"],
 "feature_weights": {5: 0.5, 0: 0.3, 6: 0.2},
 "clinical_meaning": "Nutritional intake, aspiration risk",
 "risk_type": "nutritional",
 "long_term_indicators": ["meal_duration", "eating_independence", "aspiration_events"]
 },
 "drinking": {
 "name": "Drinking",
 "detection_type": "behavioral",
 "observation_window": 60,
 "relevant_joints": ["left_wrist", "right_wrist", "nose", "neck"],
 "feature_weights": {5: 0.5, 0: 0.3, 7: 0.2},
 "clinical_meaning": "Hydration monitoring, aspiration risk",
 "risk_type": "nutritional",
 "long_term_indicators": ["hydration_frequency", "drinking_independence"]
 },
 "chewing": {
 "name": "Chewing",
 "detection_type": "dynamic",
 "observation_window": 60,
 "relevant_joints": ["nose", "left_eye", "right_eye", "neck"],
 "feature_weights": {0: 0.4, 6: 0.4, 8: 0.2},
 "clinical_meaning": "Eating activity, oral motor function",
 "risk_type": "nutritional",
 "long_term_indicators": ["chewing_duration", "meal_patterns"]
 },

 # =========================================================================
 # PERSONAL CARE ACTIVITIES (Behavioral)
 # =========================================================================
 "grooming": {
 "name": "Grooming",
 "detection_type": "behavioral",
 "observation_window": 120,
 "relevant_joints": ["left_wrist", "right_wrist", "nose", "left_eye", "right_eye"],
 "feature_weights": {5: 0.4, 0: 0.3, 7: 0.3},
 "clinical_meaning": "Self-care capability, independence indicator",
 "risk_type": "functional",
 "long_term_indicators": ["grooming_frequency", "independence_level"]
 },
 "dressing": {
 "name": "Dressing",
 "detection_type": "behavioral",
 "observation_window": 180,
 "relevant_joints": ["left_wrist", "right_wrist", "left_elbow", "right_elbow", "spine"],
 "feature_weights": {0: 0.4, 6: 0.3, 7: 0.3},
 "clinical_meaning": "Independence, mobility indicator",
 "risk_type": "functional",
 "long_term_indicators": ["dressing_time", "assistance_level"]
 },
 "toileting": {
 "name": "Toileting",
 "detection_type": "behavioral",
 "observation_window": 300,
 "relevant_joints": ["left_hip", "right_hip", "left_knee", "right_knee"],
 "feature_weights": {0: 0.3, 2: 0.4, 3: 0.3},
 "clinical_meaning": "Fall risk during transfers, independence",
 "risk_type": "safety",
 "long_term_indicators": ["toileting_frequency", "falls_during_toileting", "assistance_needs"]
 },

 # =========================================================================
 # SOCIAL AND COMMUNICATION ACTIVITIES (Behavioral)
 # =========================================================================
 "talking": {
 "name": "Talking",
 "detection_type": "behavioral",
 "observation_window": 60,
 "relevant_joints": ["nose", "neck", "left_shoulder", "right_shoulder"],
 "feature_weights": {0: 0.3, 6: 0.4, 8: 0.3},
 "clinical_meaning": "Communication, cognitive status",
 "risk_type": "cognitive",
 "long_term_indicators": ["communication_frequency", "social_engagement"]
 },
 "gesturing": {
 "name": "Gesturing",
 "detection_type": "dynamic",
 "observation_window": 30,
 "relevant_joints": ["left_wrist", "right_wrist", "left_elbow", "right_elbow"],
 "feature_weights": {0: 0.4, 6: 0.3, 8: 0.3},
 "clinical_meaning": "Communication method",
 "risk_type": "cognitive",
 "long_term_indicators": ["gesture_frequency", "communication_effectiveness"]
 },

 # =========================================================================
 # EXERCISE AND REHABILITATION ACTIVITIES (Behavioral)
 # =========================================================================
 "exercising": {
 "name": "Exercising",
 "detection_type": "behavioral",
 "observation_window": 300,
 "relevant_joints": ["left_wrist", "right_wrist", "left_ankle", "right_ankle", "left_hip", "right_hip"],
 "feature_weights": {0: 0.5, 6: 0.3, 8: 0.2},
 "clinical_meaning": "Rehabilitation progress, mobility improvement",
 "risk_type": "functional",
 "long_term_indicators": ["exercise_duration", "exercise_frequency", "intensity_progression"]
 },
 "stretching": {
 "name": "Stretching",
 "detection_type": "dynamic",
 "observation_window": 60,
 "relevant_joints": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "spine"],
 "feature_weights": {0: 0.3, 7: 0.4, 8: 0.3},
 "clinical_meaning": "Flexibility, mobility indicator",
 "risk_type": "functional",
 "long_term_indicators": ["stretching_frequency", "range_of_motion_trends"]
 },
 "range_of_motion": {
 "name": "Range of Motion Exercise",
 "detection_type": "behavioral",
 "observation_window": 120,
 "relevant_joints": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_knee", "right_knee"],
 "feature_weights": {0: 0.3, 7: 0.4, 8: 0.3},
 "clinical_meaning": "Rehabilitation assessment",
 "risk_type": "functional",
 "long_term_indicators": ["rom_improvement", "joint_specific_progress"]
 },

 # =========================================================================
 # INACTIVE/STATIC STATES (Static)
 # =========================================================================
 "still": {
 "name": "Still",
 "detection_type": "static",
 "observation_window": 30,
 "relevant_joints": ["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
 "feature_weights": {0: 0.2, 6: 0.4, 8: 0.4},
 "clinical_meaning": "Rest state, possible unresponsiveness",
 "risk_type": "behavioral",
 "long_term_indicators": ["stillness_duration", "responsiveness_checks"]
 },
 "sleeping": {
 "name": "Sleeping",
 "detection_type": "behavioral",
 "observation_window": 300,
 "relevant_joints": ["left_hip", "right_hip", "left_shoulder", "right_shoulder", "spine"],
 "feature_weights": {0: 0.2, 4: 0.4, 6: 0.4},
 "clinical_meaning": "Sleep pattern monitoring",
 "risk_type": "behavioral",
 "long_term_indicators": ["sleep_duration", "sleep_quality", "interruptions"]
 },
 "unresponsive": {
 "name": "Unresponsive",
 "detection_type": "static",
 "observation_window": 0,
 "relevant_joints": ["left_hip", "right_hip", "left_shoulder", "right_shoulder", "spine"],
 "feature_weights": {0: 0.3, 4: 0.4, 6: 0.3},
 "clinical_meaning": "Medical emergency assessment required",
 "risk_type": "critical_safety",
 "long_term_indicators": ["unresponsive_episodes", "duration_to_response"]
 },

 # =========================================================================
 # DEVICE-RELATED ACTIVITIES (Critical)
 # =========================================================================
 "moving_to_restricted_side": {
 "name": "Moving to Restricted Side",
 "detection_type": "dynamic",
 "observation_window": 120,
 "relevant_joints": ["left_hip", "right_hip", "left_shoulder", "right_shoulder", "spine"],
 "feature_weights": {0: 0.3, 2: 0.3, 3: 0.2, 6: 0.2},
 "clinical_meaning": "Bed-exit / fall risk",
 "risk_type": "safety",
 "long_term_indicators": ["side_preference", "exit_attempts"]
 },
 "removing_oxygen_mask": {
 "name": "Removing Oxygen Mask",
 "detection_type": "transitional",
 "observation_window": 0,
 "relevant_joints": ["left_wrist", "right_wrist", "left_elbow", "right_elbow", "nose"],
 "feature_weights": {5: 0.8, 0: 0.1, 1: 0.1},
 "clinical_meaning": "Critical device safety alert",
 "risk_type": "critical_device",
 "long_term_indicators": ["mask_removal_events", "oxygen_desaturation_correlation"]
 },
 "pulling_iv_devices": {
 "name": "Pulling IV / Devices",
 "detection_type": "transitional",
 "observation_window": 0,
 "relevant_joints": ["left_wrist", "right_wrist", "left_elbow", "right_elbow"],
 "feature_weights": {5: 0.7, 0: 0.2, 1: 0.1},
 "clinical_meaning": "Critical device safety alert",
 "risk_type": "critical_device",
 "long_term_indicators": ["device_pull_events", "restraint_assessment"]
 }
}

# CLINICAL ALERT THRESHOLDS

CLINICAL_THRESHOLDS = {
 "walking": {
 "persistence_threshold": 0.6, # Activity must persist >60% of window
 "risk_score_threshold": {
 "low": 0.3,
 "medium": 0.6,
 "high": 0.8
 },
 "temporal_requirements": {
 "min_duration": 60, # seconds
 "max_gap": 10 # seconds between detections
 }
 },

 "sitting": {
 "persistence_threshold": 0.7,
 "risk_score_threshold": {
 "low": 0.4,
 "medium": 0.7,
 "high": 0.9
 },
 "temporal_requirements": {
 "min_duration": 120,
 "max_gap": 15
 }
 },

 "standing": {
 "persistence_threshold": 0.5,
 "risk_score_threshold": {
 "low": 0.3,
 "medium": 0.6,
 "high": 0.8
 },
 "temporal_requirements": {
 "min_duration": 5, # event-based but with minimum
 "max_gap": 5
 }
 },

 "coughing": {
 "persistence_threshold": 0.6,
 "risk_score_threshold": {
 "low": 0.4,
 "medium": 0.7,
 "high": 0.9
 },
 "temporal_requirements": {
 "min_duration": 120,
 "max_gap": 20
 }
 },

 "moving_to_restricted_side": {
 "persistence_threshold": 0.7,
 "risk_score_threshold": {
 "low": 0.5,
 "medium": 0.8,
 "high": 0.95
 },
 "temporal_requirements": {
 "min_duration": 120,
 "max_gap": 10
 }
 },

 "removing_oxygen_mask": {
 "persistence_threshold": 0.8, # High confidence required
 "risk_score_threshold": {
 "low": 0.6,
 "medium": 0.8,
 "high": 0.95
 },
 "temporal_requirements": {
 "min_duration": 1, # Near instant
 "max_gap": 1
 }
 },

 "pulling_iv_devices": {
 "persistence_threshold": 0.8,
 "risk_score_threshold": {
 "low": 0.6,
 "medium": 0.8,
 "high": 0.95
 },
 "temporal_requirements": {
 "min_duration": 1,
 "max_gap": 1
 }
 }
}

# NURSE-FACING EXPLANATION TEMPLATES

EXPLANATION_TEMPLATES = {
 "walking": {
 "low": "Patient walking normally. Monitor for any unsteadiness.",
 "medium": "Patient walking with some instability. Check balance and fall risk.",
 "high": "Patient walking with significant instability. Immediate assessment needed for fall prevention."
 },

 "sitting": {
 "low": "Patient sitting comfortably. Continue monitoring.",
 "medium": "Patient sitting with postural changes. Assess for fatigue or discomfort.",
 "high": "Patient sitting with significant postural instability. Evaluate for pain or distress."
 },

 "standing": {
 "low": "Patient standing briefly. Monitor stability.",
 "medium": "Patient standing with balance concerns. Assist if needed.",
 "high": "Patient standing with orthostatic instability. Immediate support required."
 },

 "coughing": {
 "low": "Occasional coughing detected. Monitor respiratory status.",
 "medium": "Frequent coughing observed. Assess respiratory function.",
 "high": "Persistent coughing with distress signs. Urgent respiratory evaluation needed."
 },

 "moving_to_restricted_side": {
 "low": "Patient moving laterally. Monitor position.",
 "medium": "Patient approaching restricted area. Prepare to intervene.",
 "high": "Patient in restricted position with fall risk. Immediate assistance required."
 },

 "removing_oxygen_mask": {
 "low": "Possible mask adjustment. Verify positioning.",
 "medium": "Mask displacement detected. Reapply and assess oxygenation.",
 "high": "CRITICAL: Oxygen mask removed. Immediate intervention required."
 },

 "pulling_iv_devices": {
 "low": "Possible device contact. Verify IV integrity.",
 "medium": "Device manipulation detected. Check IV site and function.",
 "high": "CRITICAL: IV/device pulled. Immediate medical attention required."
 }
}

# REGULATORY VALIDATION PROTOCOL

VALIDATION_PROTOCOL = {
 "phases": [
 {
 "name": "Technical Validation",
 "duration": "2 weeks",
 "objectives": [
 "Verify feature extraction accuracy (>95% keypoint detection)",
 "Validate temporal windowing logic",
 "Test edge case handling (occlusion, lighting)",
 "Benchmark computational performance (<50ms/frame)"
 ],
 "metrics": [
 "Feature extraction precision/recall",
 "False positive/negative rates on synthetic data",
 "System latency and resource usage"
 ]
 },
 {
 "name": "Clinical Pilot",
 "duration": "4 weeks",
 "objectives": [
 "Deploy in low-acuity ward with nurse supervision",
 "Collect real patient data with ground truth annotations",
 "Validate alert accuracy against clinical observations",
 "Assess nurse workflow integration"
 ],
 "metrics": [
 "Alert sensitivity/specificity vs nurse assessments",
 "Time to nurse response",
 "Alert acceptance rate",
 "False alarm frequency"
 ]
 },
 {
 "name": "Regulatory Compliance",
 "duration": "6 weeks",
 "objectives": [
 "Conduct formal validation per IEC 62304",
 "Generate traceability matrix",
 "Perform risk management per ISO 14971",
 "Prepare FDA submission documentation"
 ],
 "metrics": [
 "Compliance with medical device standards",
 "Risk mitigation effectiveness",
 "Documentation completeness",
 "Validation test coverage"
 ]
 }
 ],

 "success_criteria": {
 "alert_accuracy": ">90% sensitivity, <5% false positive rate",
 "response_time": "<30 seconds average nurse response",
 "system_reliability": ">99.5% uptime",
 "clinical_acceptance": ">80% nurse satisfaction rating"
 },

 "data_requirements": {
 "diverse_patients": "100+ patients across age/gender/conditions",
 "environmental_variety": "Multiple wards, lighting conditions, camera angles",
 "temporal_coverage": "24/7 monitoring with peak activity periods",
 "ground_truth": "Expert clinical annotations for all alerts"
 }
}

# UTILITY FUNCTIONS

def get_activity_config(activity_name: str) -> Optional[Dict]:
 """Get configuration for a specific activity."""
 return ACTIVITY_FEATURE_MAPPING.get(activity_name)

def get_thresholds(activity_name: str) -> Optional[Dict]:
 """Get clinical thresholds for an activity."""
 return CLINICAL_THRESHOLDS.get(activity_name)

def get_explanation(activity_name: str, risk_level: str) -> str:
 """Get nurse-facing explanation for activity and risk level."""
 templates = EXPLANATION_TEMPLATES.get(activity_name, {})
 return templates.get(risk_level, "Activity detected. Clinical assessment recommended.")

def compute_activity_risk_score(activity_name: str, features: np.ndarray) -> float:
 """
 Compute weighted risk score for an activity based on relevant features.

 Args:
 activity_name: Name of the activity
 features: Feature vector from ICUFeatureEncoder

 Returns:
 Weighted risk score (0-1)
 """
 config = get_activity_config(activity_name)
 if not config:
 return 0.0

 weights = config["feature_weights"]
 score = 0.0
 total_weight = 0.0

 for idx, weight in weights.items():
 if idx < len(features):
 score += features[idx] * weight
 total_weight += weight

 return score / total_weight if total_weight > 0 else 0.0

def determine_risk_level(activity_name: str, risk_score: float) -> str:
 """Determine risk level based on score and thresholds."""
 thresholds = get_thresholds(activity_name)
 if not thresholds:
 return "low"

 if risk_score >= thresholds["risk_score_threshold"]["high"]:
 return "high"
 elif risk_score >= thresholds["risk_score_threshold"]["medium"]:
 return "medium"
 else:
 return "low"

def validate_activity_detection(activity_name: str, detection_duration: float,
 persistence_ratio: float) -> bool:
 """
 Validate if activity detection meets temporal requirements.

 Args:
 activity_name: Activity being detected
 detection_duration: How long activity has been detected (seconds)
 persistence_ratio: Ratio of time activity was present in window

 Returns:
 True if detection is valid for alerting
 """
 thresholds = get_thresholds(activity_name)
 if not thresholds:
 return False

 temp_req = thresholds["temporal_requirements"]

 # Check minimum duration
 if detection_duration < temp_req["min_duration"]:
 return False

 # Check persistence threshold
 if persistence_ratio < thresholds["persistence_threshold"]:
 return False

 return True


# =========================================================================
# LONG-TERM CLINICAL ANALYSIS FOR HOSPITAL SCALE
# =========================================================================

# Activity categories for hospital-scale reporting
ACTIVITY_CATEGORIES = {
 "critical": [
 "falling", "fallen", "seizure", "convulsion", "gasping",
 "unresponsive", "agitated", "thrashing", "pulling_at_tubes",
 "removing_oxygen_mask", "pulling_iv_devices"
 ],
 "high_priority": [
 "bed_exit", "tripping", "roaming", "running", "restless",
 "tremor", "coughing", "getting_up_from_floor"
 ],
 "mobility": [
 "walking", "standing", "sitting", "lying", "transferring",
 "pivoting", "sitting_up_in_bed", "bed_entry"
 ],
 "bed_related": [
 "lying_in_bed", "sitting_on_bed", "turning_in_bed",
 "moving_to_restricted_side"
 ],
 "behavioral": [
 "restless", "rocking", "waving", "pointing", "gesturing",
 "hand_to_face", "leg_movement", "foot_movement", "moving"
 ],
 "self_care": [
 "eating", "drinking", "chewing", "grooming", "dressing", "toileting"
 ],
 "rehabilitation": [
 "exercising", "stretching", "range_of_motion", "reaching", "arm_raising"
 ],
 "rest": [
 "still", "sleeping", "breathing"
 ],
 "neurological": [
 "seizure", "convulsion", "tremor", "rigidity"
 ],
 "respiratory": [
 "breathing", "coughing", "gasping"
 ]
}

# Risk type classifications for hospital reporting
RISK_TYPE_CATEGORIES = {
 "critical_safety": {"alert_level": "immediate", "response_time_sla": 30},
 "critical_device": {"alert_level": "immediate", "response_time_sla": 60},
 "critical_neurological": {"alert_level": "immediate", "response_time_sla": 30},
 "critical_respiratory": {"alert_level": "immediate", "response_time_sla": 30},
 "safety": {"alert_level": "high", "response_time_sla": 120},
 "balance": {"alert_level": "medium", "response_time_sla": 300},
 "mobility": {"alert_level": "medium", "response_time_sla": 300},
 "behavioral": {"alert_level": "low", "response_time_sla": 600},
 "postural": {"alert_level": "low", "response_time_sla": 900},
 "functional": {"alert_level": "info", "response_time_sla": None},
 "nutritional": {"alert_level": "info", "response_time_sla": None},
 "cognitive": {"alert_level": "info", "response_time_sla": None},
 "device_safety": {"alert_level": "high", "response_time_sla": 120},
 "neurological": {"alert_level": "high", "response_time_sla": 120},
 "respiratory": {"alert_level": "high", "response_time_sla": 120}
}


class LongTermClinicalAnalyzer:
 """
 Long-term clinical analysis for hospital-scale patient monitoring.
 Aggregates activity data for clinical decision support and reporting.
 """

 def __init__(self, patient_id: str, analysis_window_hours: int = 24):
 """
 Args:
 patient_id: Unique patient identifier
 analysis_window_hours: Hours of data to analyze
 """
 self.patient_id = patient_id
 self.analysis_window_hours = analysis_window_hours
 self.activity_log = []
 self.alert_log = []
 self.clinical_scores = {}

 def log_activity(self, activity: str, confidence: float, timestamp: float,
 features: Optional[np.ndarray] = None,
 context: Optional[Dict] = None):
 """Log an activity detection for long-term analysis."""
 entry = {
 "activity": activity,
 "confidence": confidence,
 "timestamp": timestamp,
 "detection_type": ACTIVITY_FEATURE_MAPPING.get(activity, {}).get("detection_type", "unknown"),
 "risk_type": ACTIVITY_FEATURE_MAPPING.get(activity, {}).get("risk_type", "unknown"),
 "features": features.tolist() if features is not None else None,
 "context": context or {}
 }
 self.activity_log.append(entry)

 def generate_clinical_summary(self) -> Dict:
 """
 Generate comprehensive clinical summary for nursing/physician review.

 Returns:
 Dict with clinical metrics, trends, and recommendations
 """
 import time
 current_time = time.time()
 window_start = current_time - (self.analysis_window_hours * 3600)

 # Filter to analysis window
 recent_activities = [a for a in self.activity_log if a["timestamp"] >= window_start]

 if not recent_activities:
 return {"status": "no_data", "patient_id": self.patient_id}

 # Activity frequency analysis
 activity_counts = {}
 for a in recent_activities:
 activity_counts[a["activity"]] = activity_counts.get(a["activity"], 0) + 1

 # Category analysis
 category_stats = {}
 for category, activities in ACTIVITY_CATEGORIES.items():
 category_count = sum(activity_counts.get(act, 0) for act in activities)
 category_stats[category] = {
 "total_count": category_count,
 "activities": {act: activity_counts.get(act, 0) for act in activities if activity_counts.get(act, 0) > 0}
 }

 # Risk analysis
 risk_events = {"critical": 0, "high": 0, "medium": 0, "low": 0}
 for a in recent_activities:
 risk_type = a.get("risk_type", "unknown")
 risk_cat = RISK_TYPE_CATEGORIES.get(risk_type, {})
 alert_level = risk_cat.get("alert_level", "info")
 if alert_level == "immediate":
 risk_events["critical"] += 1
 elif alert_level == "high":
 risk_events["high"] += 1
 elif alert_level == "medium":
 risk_events["medium"] += 1
 else:
 risk_events["low"] += 1

 # Mobility score (0-100)
 mobility_activities = ["walking", "standing", "transferring", "exercising", "stretching"]
 mobility_count = sum(activity_counts.get(act, 0) for act in mobility_activities)
 immobility_count = sum(activity_counts.get(act, 0) for act in ["lying", "lying_in_bed", "still", "sleeping"])
 total_mobility = mobility_count + immobility_count
 mobility_score = (mobility_count / total_mobility * 100) if total_mobility > 0 else 0

 # Fall risk score (0-100)
 fall_risk_activities = ["tripping", "falling", "fallen", "bed_exit", "roaming", "unsteady"]
 fall_risk_count = sum(activity_counts.get(act, 0) for act in fall_risk_activities)
 fall_risk_score = min(100, fall_risk_count * 10)

 # Agitation score (0-100)
 agitation_activities = ["agitated", "restless", "thrashing", "rocking", "pulling_at_tubes"]
 agitation_count = sum(activity_counts.get(act, 0) for act in agitation_activities)
 agitation_score = min(100, agitation_count * 15)

 # Sleep quality score (0-100)
 sleep_count = activity_counts.get("sleeping", 0)
 restless_during_night = sum(1 for a in recent_activities
 if a["activity"] in ["restless", "turning_in_bed"]
 and self._is_night_time(a["timestamp"]))
 sleep_quality_score = max(0, 100 - restless_during_night * 5)

 # Long-term indicators
 long_term_indicators = self._compute_long_term_indicators(recent_activities, activity_counts)

 return {
 "patient_id": self.patient_id,
 "analysis_window_hours": self.analysis_window_hours,
 "generated_at": current_time,
 "total_activities_logged": len(recent_activities),
 "activity_distribution": activity_counts,
 "category_statistics": category_stats,
 "risk_events": risk_events,
 "clinical_scores": {
 "mobility_score": round(mobility_score, 1),
 "fall_risk_score": round(fall_risk_score, 1),
 "agitation_score": round(agitation_score, 1),
 "sleep_quality_score": round(sleep_quality_score, 1)
 },
 "long_term_indicators": long_term_indicators,
 "recommendations": self._generate_recommendations(
 mobility_score, fall_risk_score, agitation_score, risk_events
 )
 }

 def _is_night_time(self, timestamp: float) -> bool:
 """Check if timestamp is during night hours (22:00 - 06:00)."""
 import datetime
 dt = datetime.datetime.fromtimestamp(timestamp)
 return dt.hour >= 22 or dt.hour < 6

 def _compute_long_term_indicators(self, activities: List[Dict],
 activity_counts: Dict) -> Dict:
 """Compute long-term clinical indicators from activity data."""
 indicators = {}

 # Position change frequency (pressure ulcer risk)
 turn_count = activity_counts.get("turning_in_bed", 0)
 hours = self.analysis_window_hours
 indicators["position_changes_per_hour"] = turn_count / hours if hours > 0 else 0

 # Bed exit timing patterns
 bed_exits = [a for a in activities if a["activity"] == "bed_exit"]
 night_exits = sum(1 for a in bed_exits if self._is_night_time(a["timestamp"]))
 indicators["night_bed_exits"] = night_exits

 # Activity diversity (cognitive/engagement indicator)
 unique_activities = len(activity_counts)
 indicators["activity_diversity"] = unique_activities

 # Self-care independence
 self_care_count = sum(activity_counts.get(act, 0)
 for act in ["eating", "drinking", "grooming", "dressing"])
 indicators["self_care_activities"] = self_care_count

 # Communication attempts
 comm_count = sum(activity_counts.get(act, 0)
 for act in ["waving", "pointing", "gesturing", "talking"])
 indicators["communication_attempts"] = comm_count

 # Device interference events
 device_count = sum(activity_counts.get(act, 0)
 for act in ["pulling_at_tubes", "removing_oxygen_mask", "pulling_iv_devices"])
 indicators["device_interference_events"] = device_count

 # Neurological events
 neuro_count = sum(activity_counts.get(act, 0)
 for act in ["seizure", "convulsion", "tremor"])
 indicators["neurological_events"] = neuro_count

 return indicators

 def _generate_recommendations(self, mobility_score: float, fall_risk_score: float,
 agitation_score: float, risk_events: Dict) -> List[str]:
 """Generate clinical recommendations based on analysis."""
 recommendations = []

 if mobility_score < 30:
 recommendations.append("Consider mobilization protocol - low activity level detected")

 if fall_risk_score > 50:
 recommendations.append("Elevated fall risk - consider fall prevention interventions")

 if agitation_score > 40:
 recommendations.append("Agitation detected - assess for pain, anxiety, or delirium")

 if risk_events["critical"] > 0:
 recommendations.append(f"ATTENTION: {risk_events['critical']} critical events recorded - review incident log")

 return recommendations

 def get_hospital_scale_metrics(self) -> Dict:
 """
 Generate metrics suitable for hospital-wide dashboards.

 Returns:
 Dict with standardized metrics for hospital reporting
 """
 summary = self.generate_clinical_summary()
 if summary.get("status") == "no_data":
 return summary

 return {
 "patient_id": self.patient_id,
 "acuity_level": self._compute_acuity_level(summary),
 "fall_risk_category": self._categorize_fall_risk(summary["clinical_scores"]["fall_risk_score"]),
 "mobility_status": self._categorize_mobility(summary["clinical_scores"]["mobility_score"]),
 "needs_intervention": summary["risk_events"]["critical"] > 0 or summary["risk_events"]["high"] > 2,
 "last_critical_event": self._get_last_critical_event(),
 "trending": self._compute_trends(),
 "scores": summary["clinical_scores"],
 "recommendations": summary["recommendations"]
 }

 def _compute_acuity_level(self, summary: Dict) -> str:
 """Compute patient acuity level for nursing assignment."""
 scores = summary.get("clinical_scores", {})
 risk_events = summary.get("risk_events", {})

 if risk_events.get("critical", 0) > 0 or scores.get("fall_risk_score", 0) > 70:
 return "high"
 elif risk_events.get("high", 0) > 2 or scores.get("agitation_score", 0) > 50:
 return "medium"
 else:
 return "low"

 def _categorize_fall_risk(self, score: float) -> str:
 """Categorize fall risk for standardized reporting."""
 if score >= 70:
 return "high"
 elif score >= 40:
 return "moderate"
 else:
 return "low"

 def _categorize_mobility(self, score: float) -> str:
 """Categorize mobility status."""
 if score >= 70:
 return "independent"
 elif score >= 40:
 return "assisted"
 else:
 return "dependent"

 def _get_last_critical_event(self) -> Optional[Dict]:
 """Get the most recent critical event."""
 critical_activities = ACTIVITY_CATEGORIES.get("critical", [])
 for entry in reversed(self.activity_log):
 if entry["activity"] in critical_activities:
 return {
 "activity": entry["activity"],
 "timestamp": entry["timestamp"],
 "confidence": entry["confidence"]
 }
 return None

 def _compute_trends(self) -> Dict:
 """Compute trending indicators (improving/stable/declining)."""
 if len(self.activity_log) < 10:
 return {"status": "insufficient_data"}

 # Split data into halves
 mid = len(self.activity_log) // 2
 first_half = self.activity_log[:mid]
 second_half = self.activity_log[mid:]

 # Compute mobility trend
 mobility_acts = ["walking", "standing", "exercising"]
 first_mobility = sum(1 for a in first_half if a["activity"] in mobility_acts)
 second_mobility = sum(1 for a in second_half if a["activity"] in mobility_acts)

 if second_mobility > first_mobility * 1.2:
 mobility_trend = "improving"
 elif second_mobility < first_mobility * 0.8:
 mobility_trend = "declining"
 else:
 mobility_trend = "stable"

 return {
 "mobility": mobility_trend,
 "data_points": len(self.activity_log)
 }


def get_all_activity_names() -> List[str]:
 """Get list of all defined activity names."""
 return list(ACTIVITY_FEATURE_MAPPING.keys())


def get_activity_count() -> int:
 """Get total number of mapped activities."""
 return len(ACTIVITY_FEATURE_MAPPING)


def get_activities_by_detection_type(detection_type: str) -> List[str]:
 """Get activities by detection type (static, dynamic, transitional, behavioral)."""
 return [
 name for name, config in ACTIVITY_FEATURE_MAPPING.items()
 if config.get("detection_type") == detection_type
 ]


def get_activities_by_risk_type(risk_type: str) -> List[str]:
 """Get activities by risk type."""
 return [
 name for name, config in ACTIVITY_FEATURE_MAPPING.items()
 if config.get("risk_type") == risk_type
 ]


def get_critical_activities() -> List[str]:
 """Get list of critical activities requiring immediate response."""
 return ACTIVITY_CATEGORIES.get("critical", [])


def validate_activity_mapping_completeness() -> Dict:
 """
 Validate that all activities are properly mapped.
 Useful for hospital deployment verification.
 """
 from analytics.activity_definitions import ALL_ACTIVITIES

 defined_activities = set(ALL_ACTIVITIES.keys())
 mapped_activities = set(ACTIVITY_FEATURE_MAPPING.keys())

 missing = defined_activities - mapped_activities
 extra = mapped_activities - defined_activities

 return {
 "total_defined": len(defined_activities),
 "total_mapped": len(mapped_activities),
 "missing_mappings": list(missing),
 "extra_mappings": list(extra),
 "is_complete": len(missing) == 0,
 "coverage_percent": (len(mapped_activities) / len(defined_activities)) * 100 if defined_activities else 0
 }