#!/usr/bin/env python3
"""
NTU RGB+D Action Class Mapping
Maps NTU RGB+D 60 and 120 action classes (C1-C120) to clinical activity definitions.
Note: Using 'C' prefix for Clinical action classes.
"""

# ============================================================================
# NTU RGB+D 60 Action Classes (C1-C60)
# ============================================================================

NTU_60_ACTIONS = {
 # Daily Actions (C1-C40)
 "C1": {"name": "drink water", "category": "daily", "clinical": "eating_drinking"},
 "C2": {"name": "eat meal", "category": "daily", "clinical": "eating_drinking"},
 "C3": {"name": "brush teeth", "category": "daily", "clinical": "personal_care"},
 "C4": {"name": "brush hair", "category": "daily", "clinical": "personal_care"},
 "C5": {"name": "drop", "category": "daily", "clinical": "fall_activities"},
 "C6": {"name": "pick up", "category": "daily", "clinical": "upper_body"},
 "C7": {"name": "throw", "category": "daily", "clinical": "upper_body"},
 "C8": {"name": "sit down", "category": "daily", "clinical": "basic_postural"},
 "C9": {"name": "stand up", "category": "daily", "clinical": "basic_postural"},
 "C10": {"name": "clapping", "category": "daily", "clinical": "upper_body"},
 "C11": {"name": "reading", "category": "daily", "clinical": "inactive"},
 "C12": {"name": "writing", "category": "daily", "clinical": "upper_body"},
 "C13": {"name": "tear up paper", "category": "daily", "clinical": "upper_body"},
 "C14": {"name": "put on jacket", "category": "daily", "clinical": "personal_care"},
 "C15": {"name": "take off jacket", "category": "daily", "clinical": "personal_care"},
 "C16": {"name": "put on a shoe", "category": "daily", "clinical": "personal_care"},
 "C17": {"name": "take off a shoe", "category": "daily", "clinical": "personal_care"},
 "C18": {"name": "put on glasses", "category": "daily", "clinical": "personal_care"},
 "C19": {"name": "take off glasses", "category": "daily", "clinical": "personal_care"},
 "C20": {"name": "put on a hat/cap", "category": "daily", "clinical": "personal_care"},
 "C21": {"name": "take off a hat/cap", "category": "daily", "clinical": "personal_care"},
 "C22": {"name": "cheer up", "category": "daily", "clinical": "social"},
 "C23": {"name": "hand waving", "category": "daily", "clinical": "social"},
 "C24": {"name": "kicking something", "category": "daily", "clinical": "lower_body"},
 "C25": {"name": "reach into pocket", "category": "daily", "clinical": "upper_body"},
 "C26": {"name": "hopping", "category": "daily", "clinical": "locomotion"},
 "C27": {"name": "jump up", "category": "daily", "clinical": "locomotion"},
 "C28": {"name": "phone call", "category": "daily", "clinical": "social"},
 "C29": {"name": "play with phone/tablet", "category": "daily", "clinical": "upper_body"},
 "C30": {"name": "type on a keyboard", "category": "daily", "clinical": "upper_body"},
 "C31": {"name": "point to something", "category": "daily", "clinical": "social"},
 "C32": {"name": "taking a selfie", "category": "daily", "clinical": "upper_body"},
 "C33": {"name": "check time (from watch)", "category": "daily", "clinical": "upper_body"},
 "C34": {"name": "rub two hands", "category": "daily", "clinical": "upper_body"},
 "C35": {"name": "nod head/bow", "category": "daily", "clinical": "social"},
 "C36": {"name": "shake head", "category": "daily", "clinical": "social"},
 "C37": {"name": "wipe face", "category": "daily", "clinical": "personal_care"},
 "C38": {"name": "salute", "category": "daily", "clinical": "social"},
 "C39": {"name": "put palms together", "category": "daily", "clinical": "upper_body"},
 "C40": {"name": "cross hands in front", "category": "daily", "clinical": "upper_body"},
 
 # Medical Conditions (C41-C49)
 "C41": {"name": "sneeze/cough", "category": "medical", "clinical": "respiratory"},
 "C42": {"name": "staggering", "category": "medical", "clinical": "neurological"},
 "C43": {"name": "falling down", "category": "medical", "clinical": "fall_activities"},
 "C44": {"name": "headache", "category": "medical", "clinical": "neurological"},
 "C45": {"name": "chest pain", "category": "medical", "clinical": "neurological"},
 "C46": {"name": "back pain", "category": "medical", "clinical": "neurological"},
 "C47": {"name": "neck pain", "category": "medical", "clinical": "neurological"},
 "C48": {"name": "nausea/vomiting", "category": "medical", "clinical": "neurological"},
 "C49": {"name": "fan self", "category": "medical", "clinical": "upper_body"},
 
 # Mutual Actions (C50-C60)
 "C50": {"name": "punch/slap", "category": "mutual", "clinical": "agitation_distress"},
 "C51": {"name": "kicking", "category": "mutual", "clinical": "agitation_distress"},
 "C52": {"name": "pushing", "category": "mutual", "clinical": "agitation_distress"},
 "C53": {"name": "pat on back", "category": "mutual", "clinical": "social"},
 "C54": {"name": "point finger", "category": "mutual", "clinical": "social"},
 "C55": {"name": "hugging", "category": "mutual", "clinical": "social"},
 "C56": {"name": "giving object", "category": "mutual", "clinical": "social"},
 "C57": {"name": "touch pocket", "category": "mutual", "clinical": "social"},
 "C58": {"name": "shaking hands", "category": "mutual", "clinical": "social"},
 "C59": {"name": "walking towards", "category": "mutual", "clinical": "locomotion"},
 "C60": {"name": "walking apart", "category": "mutual", "clinical": "locomotion"},
}

# ============================================================================
# NTU RGB+D 120 Additional Actions (C61-C120)
# ============================================================================

NTU_120_ADDITIONAL_ACTIONS = {
 # Daily Actions (C61-C102)
 "C61": {"name": "put on headphone", "category": "daily", "clinical": "personal_care"},
 "C62": {"name": "take off headphone", "category": "daily", "clinical": "personal_care"},
 "C63": {"name": "shoot at basket", "category": "daily", "clinical": "exercise"},
 "C64": {"name": "bounce ball", "category": "daily", "clinical": "exercise"},
 "C65": {"name": "tennis bat swing", "category": "daily", "clinical": "exercise"},
 "C66": {"name": "juggle table tennis ball", "category": "daily", "clinical": "exercise"},
 "C67": {"name": "hush", "category": "daily", "clinical": "social"},
 "C68": {"name": "flick hair", "category": "daily", "clinical": "personal_care"},
 "C69": {"name": "thumb up", "category": "daily", "clinical": "social"},
 "C70": {"name": "thumb down", "category": "daily", "clinical": "social"},
 "C71": {"name": "make OK sign", "category": "daily", "clinical": "social"},
 "C72": {"name": "make victory sign", "category": "daily", "clinical": "social"},
 "C73": {"name": "staple book", "category": "daily", "clinical": "upper_body"},
 "C74": {"name": "counting money", "category": "daily", "clinical": "upper_body"},
 "C75": {"name": "cutting nails", "category": "daily", "clinical": "personal_care"},
 "C76": {"name": "cutting paper", "category": "daily", "clinical": "upper_body"},
 "C77": {"name": "snap fingers", "category": "daily", "clinical": "upper_body"},
 "C78": {"name": "open bottle", "category": "daily", "clinical": "upper_body"},
 "C79": {"name": "sniff/smell", "category": "daily", "clinical": "upper_body"},
 "C80": {"name": "squat down", "category": "daily", "clinical": "basic_postural"},
 "C81": {"name": "toss a coin", "category": "daily", "clinical": "upper_body"},
 "C82": {"name": "fold paper", "category": "daily", "clinical": "upper_body"},
 "C83": {"name": "ball up paper", "category": "daily", "clinical": "upper_body"},
 "C84": {"name": "play magic cube", "category": "daily", "clinical": "upper_body"},
 "C85": {"name": "apply cream on face", "category": "daily", "clinical": "personal_care"},
 "C86": {"name": "apply cream on hand", "category": "daily", "clinical": "personal_care"},
 "C87": {"name": "put on bag", "category": "daily", "clinical": "personal_care"},
 "C88": {"name": "take off bag", "category": "daily", "clinical": "personal_care"},
 "C89": {"name": "put object into bag", "category": "daily", "clinical": "upper_body"},
 "C90": {"name": "take object out of bag", "category": "daily", "clinical": "upper_body"},
 "C91": {"name": "open a box", "category": "daily", "clinical": "upper_body"},
 "C92": {"name": "move heavy objects", "category": "daily", "clinical": "upper_body"},
 "C93": {"name": "shake fist", "category": "daily", "clinical": "agitation_distress"},
 "C94": {"name": "throw up cap/hat", "category": "daily", "clinical": "upper_body"},
 "C95": {"name": "capitulate", "category": "daily", "clinical": "social"},
 "C96": {"name": "cross arms", "category": "daily", "clinical": "upper_body"},
 "C97": {"name": "arm circles", "category": "daily", "clinical": "exercise"},
 "C98": {"name": "arm swings", "category": "daily", "clinical": "exercise"},
 "C99": {"name": "run on the spot", "category": "daily", "clinical": "exercise"},
 "C100": {"name": "butt kicks", "category": "daily", "clinical": "exercise"},
 "C101": {"name": "cross toe touch", "category": "daily", "clinical": "exercise"},
 "C102": {"name": "side kick", "category": "daily", "clinical": "exercise"},
 
 # Medical Conditions (C103-C105)
 "C103": {"name": "yawn", "category": "medical", "clinical": "respiratory"},
 "C104": {"name": "stretch oneself", "category": "medical", "clinical": "exercise"},
 "C105": {"name": "blow nose", "category": "medical", "clinical": "respiratory"},
 
 # Mutual Actions (C106-C120)
 "C106": {"name": "hit with object", "category": "mutual", "clinical": "agitation_distress"},
 "C107": {"name": "wield knife", "category": "mutual", "clinical": "agitation_distress"},
 "C108": {"name": "knock over", "category": "mutual", "clinical": "agitation_distress"},
 "C109": {"name": "grab stuff", "category": "mutual", "clinical": "upper_body"},
 "C110": {"name": "shoot with gun", "category": "mutual", "clinical": "upper_body"},
 "C111": {"name": "step on foot", "category": "mutual", "clinical": "lower_body"},
 "C112": {"name": "high-five", "category": "mutual", "clinical": "social"},
 "C113": {"name": "cheers and drink", "category": "mutual", "clinical": "social"},
 "C114": {"name": "carry object", "category": "mutual", "clinical": "upper_body"},
 "C115": {"name": "take a photo", "category": "mutual", "clinical": "upper_body"},
 "C116": {"name": "follow", "category": "mutual", "clinical": "locomotion"},
 "C117": {"name": "whisper", "category": "mutual", "clinical": "social"},
 "C118": {"name": "exchange things", "category": "mutual", "clinical": "social"},
 "C119": {"name": "support somebody", "category": "mutual", "clinical": "social"},
 "C120": {"name": "rock-paper-scissors", "category": "mutual", "clinical": "social"},
}

# ============================================================================
# Complete NTU RGB+D 120 Action Dictionary
# ============================================================================

NTU_120_ACTIONS = {**NTU_60_ACTIONS, **NTU_120_ADDITIONAL_ACTIONS}

# ============================================================================
# Mapping to System Activity Keys
# ============================================================================

def get_ntu_action_info(action_id: str) -> dict:
 """Get information about an NTU action class."""
 return NTU_120_ACTIONS.get(action_id, None)

def map_ntu_to_clinical_activity(action_id: str) -> str:
 """
 Map NTU action class to clinical activity key.
 
 Args:
 action_id: NTU action ID (e.g., "C1", "C43")
 
 Returns:
 Clinical activity key from activity_definitions.py
 """
 action_info = get_ntu_action_info(action_id)
 if not action_info:
 return "unknown"
 
 clinical_category = action_info.get("clinical", "unknown")
 
 # Map clinical categories to specific activity keys
 category_to_activity = {
 "eating_drinking": "drinking", # or "eating" based on specific action
 "personal_care": "grooming",
 "fall_activities": "fallen",
 "upper_body": "reaching",
 "basic_postural": "sitting", # or "standing", "lying" based on action
 "locomotion": "walking",
 "social": "talking",
 "lower_body": "leg_movement",
 "respiratory": "breathing",
 "neurological": "tremor",
 "agitation_distress": "agitated",
 "exercise": "exercising",
 "inactive": "still",
 }
 
 # Special mappings for specific actions (using C prefix)
 specific_mappings = {
 "C8": "sitting",
 "C9": "standing",
 "C43": "falling",
 "C41": "coughing",
 "C42": "tripping",
 "C50": "agitated",
 "C51": "agitated",
 "C52": "agitated",
 "C23": "waving",
 "C31": "pointing",
 "C26": "walking",
 "C27": "running",
 "C80": "crouching",
 }
 
 if action_id in specific_mappings:
 return specific_mappings[action_id]
 
 return category_to_activity.get(clinical_category, "unknown")

def get_all_ntu_actions_by_category(category: str = None) -> dict:
 """
 Get all NTU actions, optionally filtered by category.
 
 Args:
 category: "daily", "medical", or "mutual"
 
 Returns:
 Dictionary of action_id -> action_info
 """
 if category is None:
 return NTU_120_ACTIONS
 
 return {k: v for k, v in NTU_120_ACTIONS.items() if v.get("category") == category}

def get_clinical_relevance(action_id: str) -> tuple:
 """
 Get clinical relevance of an NTU action.
 
 Returns:
 (is_clinical, priority_level, clinical_category)
 """
 action_info = get_ntu_action_info(action_id)
 if not action_info:
 return (False, "NORMAL", "unknown")
 
 clinical_category = action_info.get("clinical", "unknown")
 
 # Critical actions (using C prefix)
 critical_actions = ["C43", "C50", "C51", "C52", "C106", "C107", "C108"]
 if action_id in critical_actions:
 return (True, "CRITICAL", clinical_category)
 
 # High priority actions (using C prefix)
 high_priority = ["C42", "C44", "C45", "C46", "C47", "C48", "C41", "C93"]
 if action_id in high_priority:
 return (True, "HIGH", clinical_category)
 
 # Medical conditions are always clinically relevant
 if action_info.get("category") == "medical":
 return (True, "HIGH", clinical_category)
 
 # Mutual actions involving aggression
 if action_info.get("category") == "mutual" and clinical_category == "agitation_distress":
 return (True, "HIGH", clinical_category)
 
 return (True, "NORMAL", clinical_category)
