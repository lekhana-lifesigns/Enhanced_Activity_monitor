# analytics/activity_definitions.py
# Comprehensive Activity Definitions for Human Activity Monitoring
# Defines all possible activities a human could perform in a clinical/patient monitoring context

import logging

log = logging.getLogger("activity_definitions")

# ============================================================================
# ACTIVITY CATEGORIES AND DEFINITIONS
# ============================================================================

# Basic Postural Activities
BASIC_POSTURAL = {
    "lying": {
        "name": "Lying Down",
        "subtypes": ["supine", "prone", "left_lateral", "right_lateral", "fetal"],
        "description": "Patient is in a horizontal position",
        "clinical_significance": "Normal rest state, but changes may indicate distress"
    },
    "sitting": {
        "name": "Sitting",
        "subtypes": ["upright", "slouched", "leaning_forward", "leaning_back", "edge_of_bed"],
        "description": "Patient is in a seated position",
        "clinical_significance": "May indicate readiness for mobility or respiratory issues"
    },
    "standing": {
        "name": "Standing",
        "subtypes": ["still", "swaying", "supported", "unsupported"],
        "description": "Patient is in an upright standing position",
        "clinical_significance": "Indicates mobility and balance capability"
    },
    "kneeling": {
        "name": "Kneeling",
        "subtypes": ["on_one_knee", "on_both_knees"],
        "description": "Patient is in a kneeling position",
        "clinical_significance": "Unusual position, may indicate fall or distress"
    },
    "crouching": {
        "name": "Crouching",
        "subtypes": ["squatting", "hunched"],
        "description": "Patient is in a low crouched position",
        "clinical_significance": "May indicate pain, distress, or fall recovery"
    }
}

# Locomotion Activities
LOCOMOTION = {
    "walking": {
        "name": "Walking",
        "subtypes": ["normal", "slow", "fast", "shuffling", "unsteady", "assisted"],
        "description": "Patient is moving by foot",
        "clinical_significance": "Indicates mobility level and gait quality"
    },
    "running": {
        "name": "Running",
        "subtypes": ["jogging", "sprinting"],
        "description": "Patient is moving quickly by foot",
        "clinical_significance": "Unusual in clinical setting, may indicate agitation or elopement risk"
    },
    "crawling": {
        "name": "Crawling",
        "subtypes": ["on_hands_knees", "army_crawl"],
        "description": "Patient is moving on hands and knees",
        "clinical_significance": "May indicate fall, weakness, or confusion"
    },
    "rolling": {
        "name": "Rolling",
        "subtypes": ["in_bed", "on_floor"],
        "description": "Patient is rolling their body",
        "clinical_significance": "May indicate restlessness, pain, or seizure activity"
    },
    "moving": {
        "name": "Moving",
        "subtypes": ["general_movement", "shifting", "repositioning"],
        "description": "Patient is in motion but not specifically walking or running",
        "clinical_significance": "Indicates general mobility and activity level"
    },
    "tripping": {
        "name": "Tripping",
        "subtypes": ["stumbling", "loss_of_balance", "near_fall"],
        "description": "Patient is tripping or losing balance",
        "clinical_significance": "HIGH: May indicate fall risk, requires immediate attention"
    },
    "roaming": {
        "name": "Roaming/Wandering",
        "subtypes": ["pacing", "aimless_walking", "elopement_risk"],
        "description": "Patient is moving around without clear purpose or destination",
        "clinical_significance": "HIGH: May indicate confusion, agitation, or elopement risk"
    }
}

# Bed-Related Activities
BED_ACTIVITIES = {
    "lying_in_bed": {
        "name": "Lying in Bed",
        "subtypes": ["supine", "prone", "side", "fetal"],
        "description": "Patient is lying down in bed",
        "clinical_significance": "Normal rest state"
    },
    "sitting_on_bed": {
        "name": "Sitting on Bed",
        "subtypes": ["edge", "center", "supported", "unsupported"],
        "description": "Patient is sitting on the bed",
        "clinical_significance": "May indicate preparation for transfer or mobility"
    },
    "bed_exit": {
        "name": "Bed Exit",
        "subtypes": ["attempting", "partial", "complete"],
        "description": "Patient is exiting or attempting to exit the bed",
        "clinical_significance": "CRITICAL: High fall risk, requires immediate attention"
    },
    "bed_entry": {
        "name": "Bed Entry",
        "subtypes": ["getting_into_bed", "lying_down"],
        "description": "Patient is entering or getting into bed",
        "clinical_significance": "Normal activity, but timing may indicate fatigue or distress"
    },
    "turning_in_bed": {
        "name": "Turning in Bed",
        "subtypes": ["left", "right", "frequent"],
        "description": "Patient is changing position while in bed",
        "clinical_significance": "May indicate restlessness, pain, or pressure relief"
    },
    "sitting_up_in_bed": {
        "name": "Sitting Up in Bed",
        "subtypes": ["with_assistance", "without_assistance"],
        "description": "Patient transitions from lying to sitting in bed",
        "clinical_significance": "Indicates mobility and strength"
    }
}

# Transfer Activities
TRANSFER_ACTIVITIES = {
    "transferring": {
        "name": "Transferring",
        "subtypes": ["bed_to_chair", "chair_to_bed", "bed_to_standing", "standing_to_bed"],
        "description": "Patient is moving between surfaces",
        "clinical_significance": "High fall risk period, requires monitoring"
    },
    "pivoting": {
        "name": "Pivoting",
        "subtypes": ["standing_pivot", "seated_pivot"],
        "description": "Patient is rotating their body",
        "clinical_significance": "Common during transfers, balance critical"
    }
}

# Upper Body Activities
UPPER_BODY = {
    "reaching": {
        "name": "Reaching",
        "subtypes": ["upward", "forward", "sideways", "behind"],
        "description": "Patient is extending arm(s) to reach for something",
        "clinical_significance": "May indicate need for assistance or risk of overreaching"
    },
    "waving": {
        "name": "Waving",
        "subtypes": ["hand_wave", "arm_wave"],
        "description": "Patient is moving hand/arm in waving motion",
        "clinical_significance": "May indicate communication attempt or agitation"
    },
    "pointing": {
        "name": "Pointing",
        "subtypes": ["with_finger", "with_arm"],
        "description": "Patient is pointing at something",
        "clinical_significance": "Communication gesture"
    },
    "arm_raising": {
        "name": "Arm Raising",
        "subtypes": ["one_arm", "both_arms", "overhead"],
        "description": "Patient is raising arm(s)",
        "clinical_significance": "May indicate stretching, exercise, or distress signal"
    },
    "hand_to_face": {
        "name": "Hand to Face",
        "subtypes": ["rubbing_eyes", "covering_face", "touching_mouth"],
        "description": "Patient brings hand to face",
        "clinical_significance": "May indicate fatigue, distress, or medical device interference"
    }
}

# Lower Body Activities
LOWER_BODY = {
    "leg_movement": {
        "name": "Leg Movement",
        "subtypes": ["kicking", "stretching", "crossing", "swinging"],
        "description": "Patient is moving legs",
        "clinical_significance": "May indicate restlessness, exercise, or discomfort"
    },
    "foot_movement": {
        "name": "Foot Movement",
        "subtypes": ["tapping", "fidgeting", "pointing"],
        "description": "Patient is moving feet",
        "clinical_significance": "May indicate anxiety, restlessness, or neurological activity"
    }
}

# Agitation and Distress Activities
AGITATION_DISTRESS = {
    "restless": {
        "name": "Restless",
        "subtypes": ["fidgeting", "constant_movement", "inability_to_sit_still"],
        "description": "Patient shows restless behavior",
        "clinical_significance": "May indicate pain, anxiety, or delirium"
    },
    "agitated": {
        "name": "Agitated",
        "subtypes": ["mild", "moderate", "severe", "combative"],
        "description": "Patient shows signs of agitation",
        "clinical_significance": "CRITICAL: Requires immediate assessment and intervention"
    },
    "pulling_at_tubes": {
        "name": "Pulling at Tubes",
        "subtypes": ["iv_line", "catheter", "oxygen", "feeding_tube"],
        "description": "Patient is attempting to remove medical devices",
        "clinical_significance": "CRITICAL: Safety risk, requires immediate intervention"
    },
    "thrashing": {
        "name": "Thrashing",
        "subtypes": ["arms", "legs", "full_body"],
        "description": "Patient is making violent or uncontrolled movements",
        "clinical_significance": "CRITICAL: May indicate seizure, severe agitation, or distress"
    },
    "rocking": {
        "name": "Rocking",
        "subtypes": ["forward_back", "side_to_side", "circular"],
        "description": "Patient is rocking their body",
        "clinical_significance": "May indicate self-soothing, agitation, or neurological condition"
    }
}

# Fall-Related Activities
FALL_ACTIVITIES = {
    "falling": {
        "name": "Falling",
        "subtypes": ["forward", "backward", "sideways", "from_bed", "from_chair", "from_standing"],
        "description": "Patient is in the process of falling",
        "clinical_significance": "CRITICAL: Immediate emergency response required"
    },
    "fallen": {
        "name": "Fallen",
        "subtypes": ["on_floor", "against_furniture", "partial_fall"],
        "description": "Patient has fallen and is on the ground",
        "clinical_significance": "CRITICAL: Immediate assessment and assistance required"
    },
    "getting_up_from_floor": {
        "name": "Getting Up from Floor",
        "subtypes": ["with_assistance", "without_assistance", "attempting"],
        "description": "Patient is attempting to rise from the floor",
        "clinical_significance": "May indicate fall recovery, requires monitoring"
    }
}

# Seizure and Neurological Activities
NEUROLOGICAL = {
    "seizure": {
        "name": "Seizure",
        "subtypes": ["tonic_clonic", "absence", "focal", "myoclonic"],
        "description": "Patient is experiencing seizure activity",
        "clinical_significance": "CRITICAL: Immediate medical intervention required"
    },
    "convulsion": {
        "name": "Convulsion",
        "subtypes": ["mild", "moderate", "severe"],
        "description": "Patient is experiencing convulsive movements",
        "clinical_significance": "CRITICAL: Emergency response required"
    },
    "tremor": {
        "name": "Tremor",
        "subtypes": ["hand", "arm", "leg", "full_body"],
        "description": "Patient has involuntary shaking",
        "clinical_significance": "May indicate medication side effects, neurological condition, or withdrawal"
    },
    "rigidity": {
        "name": "Rigidity",
        "subtypes": ["muscle_rigidity", "postural_rigidity"],
        "description": "Patient shows muscle stiffness or rigidity",
        "clinical_significance": "May indicate neurological condition or medication effect"
    }
}

# Respiratory Activities
RESPIRATORY = {
    "breathing": {
        "name": "Breathing",
        "subtypes": ["normal", "rapid", "shallow", "deep", "labored"],
        "description": "Patient respiratory activity",
        "clinical_significance": "Vital sign monitoring, changes may indicate distress"
    },
    "coughing": {
        "name": "Coughing",
        "subtypes": ["mild", "moderate", "severe", "persistent"],
        "description": "Patient is coughing",
        "clinical_significance": "May indicate respiratory infection, aspiration, or irritation"
    },
    "gasping": {
        "name": "Gasping",
        "subtypes": ["occasional", "frequent", "agonal"],
        "description": "Patient is gasping for air",
        "clinical_significance": "CRITICAL: Respiratory distress, immediate assessment required"
    }
}

# Eating and Drinking Activities
EATING_DRINKING = {
    "eating": {
        "name": "Eating",
        "subtypes": ["self_feeding", "assisted_feeding", "tube_feeding"],
        "description": "Patient is consuming food",
        "clinical_significance": "Nutritional intake monitoring, aspiration risk"
    },
    "drinking": {
        "name": "Drinking",
        "subtypes": ["from_cup", "from_straw", "from_bottle"],
        "description": "Patient is consuming liquids",
        "clinical_significance": "Hydration monitoring, aspiration risk"
    },
    "chewing": {
        "name": "Chewing",
        "subtypes": ["food", "gum", "other"],
        "description": "Patient is chewing",
        "clinical_significance": "May indicate eating or oral motor activity"
    }
}

# Personal Care Activities
PERSONAL_CARE = {
    "grooming": {
        "name": "Grooming",
        "subtypes": ["brushing_hair", "washing_face", "shaving"],
        "description": "Patient is performing personal grooming",
        "clinical_significance": "Indicates independence and self-care capability"
    },
    "dressing": {
        "name": "Dressing",
        "subtypes": ["putting_on", "taking_off", "adjusting"],
        "description": "Patient is changing or adjusting clothing",
        "clinical_significance": "Indicates independence and mobility"
    },
    "toileting": {
        "name": "Toileting",
        "subtypes": ["using_toilet", "using_bedpan", "incontinence"],
        "description": "Patient is using bathroom facilities",
        "clinical_significance": "Mobility and independence indicator, fall risk during transfers"
    }
}

# Social and Communication Activities
SOCIAL = {
    "talking": {
        "name": "Talking",
        "subtypes": ["to_staff", "to_visitor", "to_self", "on_phone"],
        "description": "Patient is engaged in conversation",
        "clinical_significance": "Communication and cognitive status indicator"
    },
    "gesturing": {
        "name": "Gesturing",
        "subtypes": ["pointing", "waving", "thumbs_up", "thumbs_down"],
        "description": "Patient is making communicative gestures",
        "clinical_significance": "Communication method, especially if verbal communication limited"
    },
    "waving": {
        "name": "Waving",
        "subtypes": ["hello", "goodbye", "attention_seeking"],
        "description": "Patient is waving",
        "clinical_significance": "Communication or attention-seeking behavior"
    }
}

# Exercise and Rehabilitation Activities
EXERCISE = {
    "exercising": {
        "name": "Exercising",
        "subtypes": ["arm_exercises", "leg_exercises", "walking_exercise", "stretching"],
        "description": "Patient is performing therapeutic exercises",
        "clinical_significance": "Rehabilitation progress, mobility improvement"
    },
    "stretching": {
        "name": "Stretching",
        "subtypes": ["arms", "legs", "torso", "full_body"],
        "description": "Patient is stretching",
        "clinical_significance": "Mobility and flexibility indicator"
    },
    "range_of_motion": {
        "name": "Range of Motion",
        "subtypes": ["active", "passive", "assisted"],
        "description": "Patient is performing range of motion exercises",
        "clinical_significance": "Rehabilitation and mobility assessment"
    }
}

# Inactive/Static States
INACTIVE = {
    "still": {
        "name": "Still",
        "subtypes": ["lying_still", "sitting_still", "standing_still"],
        "description": "Patient is not moving",
        "clinical_significance": "May indicate rest, sleep, or unresponsiveness"
    },
    "sleeping": {
        "name": "Sleeping",
        "subtypes": ["deep_sleep", "light_sleep", "restless_sleep"],
        "description": "Patient appears to be sleeping",
        "clinical_significance": "Normal rest state, but changes may indicate issues"
    },
    "unresponsive": {
        "name": "Unresponsive",
        "subtypes": ["to_voice", "to_touch", "to_pain"],
        "description": "Patient is not responding to stimuli",
        "clinical_significance": "CRITICAL: May indicate medical emergency, requires immediate assessment"
    }
}

# ============================================================================
# COMPREHENSIVE ACTIVITY LIST
# ============================================================================

ALL_ACTIVITIES = {
    **BASIC_POSTURAL,
    **LOCOMOTION,
    **BED_ACTIVITIES,
    **TRANSFER_ACTIVITIES,
    **UPPER_BODY,
    **LOWER_BODY,
    **AGITATION_DISTRESS,
    **FALL_ACTIVITIES,
    **NEUROLOGICAL,
    **RESPIRATORY,
    **EATING_DRINKING,
    **PERSONAL_CARE,
    **SOCIAL,
    **EXERCISE,
    **INACTIVE
}

# ============================================================================
# ACTIVITY PRIORITY LEVELS
# ============================================================================

CRITICAL_ACTIVITIES = [
    "falling", "fallen", "seizure", "convulsion", "agitated", "unresponsive",
    "gasping", "pulling_at_tubes", "thrashing", "bed_exit"
]

HIGH_PRIORITY_ACTIVITIES = [
    "getting_up_from_floor", "restless", "coughing", "tremor", "rigidity"
]

NORMAL_ACTIVITIES = [
    "lying", "sitting", "standing", "walking", "sleeping", "breathing",
    "eating", "drinking", "talking", "still"
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_activity_info(activity_key):
    """Get detailed information about an activity."""
    return ALL_ACTIVITIES.get(activity_key, None)

def get_activity_priority(activity_key):
    """Get priority level for an activity."""
    if activity_key in CRITICAL_ACTIVITIES:
        return "CRITICAL"
    elif activity_key in HIGH_PRIORITY_ACTIVITIES:
        return "HIGH"
    elif activity_key in NORMAL_ACTIVITIES:
        return "NORMAL"
    else:
        return "MEDIUM"

def get_activities_by_category(category):
    """Get all activities in a specific category."""
    category_map = {
        "postural": BASIC_POSTURAL,
        "locomotion": LOCOMOTION,
        "bed": BED_ACTIVITIES,
        "transfer": TRANSFER_ACTIVITIES,
        "upper_body": UPPER_BODY,
        "lower_body": LOWER_BODY,
        "agitation": AGITATION_DISTRESS,
        "fall": FALL_ACTIVITIES,
        "neurological": NEUROLOGICAL,
        "respiratory": RESPIRATORY,
        "eating": EATING_DRINKING,
        "personal_care": PERSONAL_CARE,
        "social": SOCIAL,
        "exercise": EXERCISE,
        "inactive": INACTIVE
    }
    return category_map.get(category, {})

def list_all_activities():
    """Get a list of all activity keys."""
    return list(ALL_ACTIVITIES.keys())

def get_activity_count():
    """Get total number of defined activities."""
    return len(ALL_ACTIVITIES)

