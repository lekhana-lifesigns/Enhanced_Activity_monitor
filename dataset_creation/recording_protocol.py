"""
Clinical Dataset Recording Protocol
Defines structured protocols for recording custom clinical datasets.

2-Level Activity Hierarchy:
 Level 1 (TEMPORAL_CLASSES): 12 trainable temporal states - directly learned by GRU/GCN model
 Level 2 (CLINICAL_TAGS): Fine-grained clinical interpretations - derived from Level 1 + features

Level 1 classes are mutually exclusive per time window and annotated on video timelines.
Level 2 tags are optional metadata applied on top of Level 1 labels for clinical context.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


# ============================================================
# LEVEL 1: TEMPORAL CLASSES (12 trainable states)
# These are the PRIMARY annotation targets for temporal model training.
# Mutually exclusive per 3-second window.
# ============================================================

@dataclass
class TemporalClass:
 """Definition of a Level 1 temporal activity class."""
 class_id: int
 name: str
 short_code: str # 3-letter code for display
 color: str # Hex color for timeline
 description: str
 clinical_urgency: str # "low", "medium", "high", "critical"
 hotkey: str # Keyboard shortcut for annotation
 visual_criteria: List[str]
 level2_derivations: List[str] # What Level 2 tags can be derived from this


TEMPORAL_CLASSES = {
 "lying_still": TemporalClass(
 class_id=0, name="lying_still", short_code="LYS", color="#2196F3",
 description="Patient is horizontal with minimal movement",
 clinical_urgency="low", hotkey="0",
 visual_criteria=[
 "Body is horizontal (spine angle < 30 deg from horizontal)",
 "Motion energy near zero (limbs not moving)",
 "Stable keypoint positions across the window",
 ],
 level2_derivations=["pressure_ulcer_risk", "sleep_state", "breathing_shallow"],
 ),
 "lying_restless": TemporalClass(
 class_id=1, name="lying_restless", short_code="LYR", color="#FF9800",
 description="Patient is horizontal but with intermittent limb/body movement",
 clinical_urgency="low", hotkey="1",
 visual_criteria=[
 "Body horizontal but limbs moving intermittently",
 "Small positional shifts, fidgeting, leg movements",
 "NOT full body rotation (that is turning_in_bed)",
 ],
 level2_derivations=["cough_mild", "cough_moderate", "breathing_labored",
 "pain_guarding", "agitation_self_contact", "hand_wringing"],
 ),
 "turning_in_bed": TemporalClass(
 class_id=2, name="turning_in_bed", short_code="TIB", color="#9C27B0",
 description="Patient is rotating or repositioning significantly while in bed",
 clinical_urgency="low", hotkey="2",
 visual_criteria=[
 "Torso rotates laterally (supine to side, side to prone, etc.)",
 "Hips and shoulders shift > ~30cm laterally",
 "Duration: typically 2-10 seconds per turn",
 ],
 level2_derivations=["restlessness_pattern", "pain_response"],
 ),
 "sitting_stable": TemporalClass(
 class_id=3, name="sitting_stable", short_code="SIS", color="#4CAF50",
 description="Patient upright in seated position with stable posture",
 clinical_urgency="low", hotkey="3",
 visual_criteria=[
 "Torso upright (spine angle > 45 deg from horizontal)",
 "Hips and knees bent (seated geometry)",
 "Minimal sway (COM stays within ~5cm)",
 ],
 level2_derivations=["pain_bracing"],
 ),
 "sitting_unstable": TemporalClass(
 class_id=4, name="sitting_unstable", short_code="SIU", color="#FFC107",
 description="Patient seated but showing postural instability",
 clinical_urgency="medium", hotkey="4",
 visual_criteria=[
 "Seated posture with visible sway or rocking",
 "Repeated balance corrections (hands grabbing, torso shifting)",
 "Head/shoulders oscillating > ~10cm from center",
 ],
 level2_derivations=["fall_risk_elevated", "dizziness"],
 ),
 "bed_exit": TemporalClass(
 class_id=5, name="bed_exit", short_code="BEX", color="#F44336",
 description="Patient attempting to leave bed (sitting at edge, legs over side, or getting up)",
 clinical_urgency="high", hotkey="5",
 visual_criteria=[
 "Patient moves from lying/sitting center-bed to bed edge",
 "Legs swing over bed side",
 "Weight shift toward standing from bed",
 ],
 level2_derivations=["fall_risk_elevated", "elopement_risk"],
 ),
 "standing": TemporalClass(
 class_id=6, name="standing", short_code="STD", color="#00BCD4",
 description="Patient fully upright on feet, stationary or minimal weight shifting",
 clinical_urgency="low", hotkey="6",
 visual_criteria=[
 "Full vertical extent visible (head to feet)",
 "Legs straight (knees extended)",
 "Not translating across the room (that is walking)",
 ],
 level2_derivations=["orthostatic_risk"],
 ),
 "walking": TemporalClass(
 class_id=7, name="walking", short_code="WLK", color="#009688",
 description="Patient moving across space with rhythmic gait pattern",
 clinical_urgency="low", hotkey="7",
 visual_criteria=[
 "Alternating leg motion",
 "COM translating across frame",
 "Rhythmic pattern (not stumbling/falling)",
 ],
 level2_derivations=["gait_quality", "mobility_assessment"],
 ),
 "agitated": TemporalClass(
 class_id=8, name="agitated", short_code="AGT", color="#E91E63",
 description="High-energy, non-purposeful, sustained body movement",
 clinical_urgency="high", hotkey="8",
 visual_criteria=[
 "Sustained high motion energy across multiple joints",
 "Non-rhythmic, non-purposeful movement",
 "May include: thrashing, pulling, repeated reaching, rocking violently",
 ],
 level2_derivations=["cough_severe", "iv_manipulation", "catheter_touching",
 "wound_picking", "agitation_restlessness", "mask_removal",
 "tube_pulling", "delirium_pattern"],
 ),
 "convulsive": TemporalClass(
 class_id=9, name="convulsive", short_code="CNV", color="#D50000",
 description="Involuntary, rhythmic, high-frequency whole-body or limb movements",
 clinical_urgency="critical", hotkey="9",
 visual_criteria=[
 "Very high frequency limb/body oscillations",
 "Symmetric or semi-symmetric pattern",
 "Appears involuntary (not goal-directed)",
 "Very high jerk index",
 ],
 level2_derivations=["seizure_tonic_clonic", "seizure_myoclonic", "seizure_cluster"],
 ),
 "fallen": TemporalClass(
 class_id=10, name="fallen", short_code="FLN", color="#B71C1C",
 description="Patient has fallen or is in the process of an uncontrolled descent",
 clinical_urgency="critical", hotkey="f",
 visual_criteria=[
 "Rapid uncontrolled downward movement (not intentional lying down)",
 "Patient ends up at ground/floor level (not in bed)",
 "Preceded by loss of balance or sudden posture collapse",
 ],
 level2_derivations=["fall_from_bed", "fall_from_standing", "fall_injury_risk"],
 ),
 "occluded": TemporalClass(
 class_id=11, name="occluded", short_code="OCC", color="#9E9E9E",
 description="Patient not sufficiently visible for reliable classification",
 clinical_urgency="low", hotkey="o",
 visual_criteria=[
 "Fewer than 8 keypoints detected with confidence > 0.3",
 "Patient blocked by caregiver, equipment, or furniture",
 "Camera obstruction or extreme angle",
 ],
 level2_derivations=["caregiver_present", "equipment_occlusion"],
 ),
}

# Ordered list for model output mapping
TEMPORAL_CLASS_NAMES = [tc.name for tc in sorted(TEMPORAL_CLASSES.values(), key=lambda x: x.class_id)]
TEMPORAL_CLASS_IDS = {name: i for i, name in enumerate(TEMPORAL_CLASS_NAMES)}
NUM_TEMPORAL_CLASSES = len(TEMPORAL_CLASS_NAMES)


# ============================================================
# LEVEL 2: CLINICAL TAGS (derived interpretations)
# These are SECONDARY metadata tags applied on top of Level 1 labels.
# They are NOT direct model targets — derived from Level 1 + features + rules.
# ============================================================

@dataclass
class ClinicalTag:
 """Definition of a Level 2 clinical interpretation tag."""
 tag_id: str
 tag_name: str
 category: str # "respiratory", "agitation", "pain", "clinical_intervention", "safety"
 description: str
 parent_level1_classes: List[str] # Which Level 1 classes this can co-occur with
 detection_method: str # "feature_threshold", "temporal_pattern", "object_detection", "facial"
 clinical_significance: str
 annotation_optional: bool = True # Optional during annotation (can be derived automatically)


CLINICAL_TAGS = {
 # Respiratory (derived from feature[4] breath_rate_proxy)
 "cough_mild": ClinicalTag("cough_mild", "Mild Cough", "respiratory",
 "Light, occasional cough", ["lying_restless", "sitting_stable"],
 "feature_threshold", "May indicate mild respiratory irritation"),
 "cough_moderate": ClinicalTag("cough_moderate", "Moderate Cough", "respiratory",
 "Persistent cough with visible effort", ["lying_restless", "sitting_unstable"],
 "feature_threshold", "Indicates respiratory distress requiring monitoring"),
 "cough_severe": ClinicalTag("cough_severe", "Severe Cough", "respiratory",
 "Intense, prolonged coughing with distress", ["agitated"],
 "feature_threshold", "Critical - requires immediate clinical attention"),
 "breathing_shallow": ClinicalTag("breathing_shallow", "Shallow Breathing", "respiratory",
 "Rapid, shallow breathing with minimal chest expansion", ["lying_still", "lying_restless"],
 "feature_threshold", "May indicate respiratory distress or anxiety"),
 "breathing_labored": ClinicalTag("breathing_labored", "Labored Breathing", "respiratory",
 "Difficulty breathing with visible effort", ["lying_restless", "agitated"],
 "feature_threshold", "Critical respiratory distress requiring intervention"),

 # Agitation (derived from feature[0] motion_energy + feature[6] motor_entropy)
 "agitation_restlessness": ClinicalTag("agitation_restlessness", "Restlessness", "agitation",
 "Frequent position changes, fidgeting", ["lying_restless", "agitated"],
 "temporal_pattern", "May indicate discomfort, anxiety, or delirium"),
 "hand_wringing": ClinicalTag("hand_wringing", "Hand Wringing", "agitation",
 "Repetitive hand rubbing or wringing", ["lying_restless", "sitting_stable"],
 "feature_threshold", "Common sign of anxiety or pain"),
 "self_contact": ClinicalTag("self_contact", "Self-Contact", "agitation",
 "Touching face, head, arms; self-hugging", ["lying_restless", "agitated"],
 "feature_threshold", "Indicates distress, pain, or need for comfort"),

 # Pain (derived from posture asymmetry + facial cues)
 "pain_grimacing": ClinicalTag("pain_grimacing", "Facial Grimacing", "pain",
 "Facial expressions indicating pain", ["lying_restless", "sitting_stable", "agitated"],
 "facial", "Direct indicator of pain experience"),
 "pain_guarding": ClinicalTag("pain_guarding", "Guarding", "pain",
 "Protective body positioning or tensing", ["lying_restless", "sitting_stable"],
 "feature_threshold", "Indicates pain location and severity"),
 "pain_bracing": ClinicalTag("pain_bracing", "Bracing", "pain",
 "Holding onto objects for support during pain", ["sitting_stable", "standing"],
 "feature_threshold", "Indicates acute pain episode"),

 # Clinical Intervention (derived from hand_proximity + object detection)
 "iv_manipulation": ClinicalTag("iv_manipulation", "IV Line Manipulation", "clinical_intervention",
 "Touching, pulling, or removing IV line", ["agitated", "lying_restless"],
 "object_detection", "Critical safety event - risk of injury or infection"),
 "catheter_touching": ClinicalTag("catheter_touching", "Catheter Touching", "clinical_intervention",
 "Touching catheter or attempting to remove", ["agitated", "lying_restless"],
 "object_detection", "Critical safety event - infection risk"),
 "wound_picking": ClinicalTag("wound_picking", "Wound Picking", "clinical_intervention",
 "Touching or picking at wound or surgical site", ["agitated", "lying_restless"],
 "object_detection", "Risk of infection and delayed healing"),
 "mask_removal": ClinicalTag("mask_removal", "Mask Removal", "clinical_intervention",
 "Attempting to remove oxygen mask or face covering", ["agitated"],
 "object_detection", "Risk of oxygen desaturation"),
 "tube_pulling": ClinicalTag("tube_pulling", "Tube Pulling", "clinical_intervention",
 "Pulling at chest tubes, feeding tubes, drains", ["agitated"],
 "object_detection", "Critical device interference"),

 # Safety (derived from temporal patterns over longer windows)
 "fall_risk_elevated": ClinicalTag("fall_risk_elevated", "Elevated Fall Risk", "safety",
 "Multiple bed exits or instability episodes", ["bed_exit", "sitting_unstable"],
 "temporal_pattern", "Requires increased monitoring"),
 "elopement_risk": ClinicalTag("elopement_risk", "Elopement Risk", "safety",
 "Repeated standing + walking toward exit", ["standing", "walking", "bed_exit"],
 "temporal_pattern", "Patient may attempt to leave"),
 "delirium_pattern": ClinicalTag("delirium_pattern", "Delirium Pattern", "safety",
 "Prolonged agitation + sleep disruption pattern", ["agitated", "lying_restless"],
 "temporal_pattern", "Requires clinical assessment for delirium"),
 "pressure_ulcer_risk": ClinicalTag("pressure_ulcer_risk", "Pressure Ulcer Risk", "safety",
 "Prolonged immobility (> 2 hours same position)", ["lying_still"],
 "temporal_pattern", "Requires repositioning intervention"),
}

# Legacy mapping: old CLINICAL_ACTIVITIES → new Level 1 + Level 2
LEGACY_TO_LEVEL1_MAPPING = {
 "cough_mild": "lying_restless",
 "cough_moderate": "lying_restless",
 "cough_severe": "agitated",
 "agitation_restlessness": "lying_restless", # or "agitated" if high energy
 "agitation_hand_wringing": "lying_restless",
 "agitation_self_contact": "lying_restless",
 "pain_facial_grimacing": "lying_restless",
 "pain_guarding": "lying_restless",
 "pain_bracing": "sitting_stable",
 "breathing_shallow": "lying_still",
 "breathing_labored": "lying_restless",
 "iv_manipulation": "agitated",
 "catheter_touching": "agitated",
 "wound_picking": "lying_restless",
}


# ============================================================
# LEGACY: Original CLINICAL_ACTIVITIES (kept for backward compatibility)
# These are now Level 2 clinical interpretations, not model targets.
# ============================================================

@dataclass
class ActivityDefinition:
 """Definition of a recordable activity (Level 2 clinical interpretation)."""
 activity_id: str
 activity_name: str
 category: str # "respiratory", "agitation", "pain", "clinical_intervention"
 description: str
 duration_min: float # Minimum duration in seconds
 duration_max: float # Maximum duration in seconds
 intensity_levels: List[str] # e.g., ["mild", "moderate", "severe"]
 clinical_significance: str
 recording_guidelines: List[str]
 annotation_tags: List[str] # Tags to capture during annotation
 level1_mapping: str = "" # Primary Level 1 temporal class this maps to


@dataclass
class RecordingSession:
 """Metadata for a recording session."""
 session_id: str
 participant_id: str
 date: str
 time: str
 location: str
 camera_setup: Dict[str, any]
 activities_recorded: List[str]
 consent_obtained: bool
 irb_protocol_id: str
 researcher_id: str
 notes: str


# Define clinical activities to record
CLINICAL_ACTIVITIES = {
 # Respiratory Activities
 "cough_mild": ActivityDefinition(
 activity_id="cough_mild",
 activity_name="Mild Cough",
 category="respiratory",
 description="Light, occasional cough without significant respiratory distress",
 duration_min=1.0,
 duration_max=5.0,
 intensity_levels=["mild"],
 clinical_significance="May indicate mild respiratory irritation or early infection",
 recording_guidelines=[
 "Record natural cough episodes",
 "Capture full torso and head in frame",
 "Record at least 3-5 repetitions",
 "Ensure audio is captured if available",
 "Note environmental factors (dust, temperature)"
 ],
 annotation_tags=["frequency", "productive", "dry", "body_position"],
 level1_mapping="lying_restless",
 ),

 "cough_moderate": ActivityDefinition(
 activity_id="cough_moderate",
 activity_name="Moderate Cough",
 category="respiratory",
 description="Persistent cough with visible effort and possible discomfort",
 duration_min=2.0,
 duration_max=10.0,
 intensity_levels=["moderate"],
 clinical_significance="Indicates respiratory distress requiring monitoring",
 recording_guidelines=[
 "Record extended coughing episodes",
 "Capture body movements and posture changes",
 "Note any protective behaviors (hand to chest)",
 "Record at least 3-5 repetitions"
 ],
 annotation_tags=["frequency", "productive", "body_movement", "discomfort_level"]
 ),

 "cough_severe": ActivityDefinition(
 activity_id="cough_severe",
 activity_name="Severe Cough",
 category="respiratory",
 description="Intense, prolonged coughing with significant distress",
 duration_min=3.0,
 duration_max=15.0,
 intensity_levels=["severe"],
 clinical_significance="Critical - requires immediate clinical attention",
 recording_guidelines=[
 "Record full episode from start to recovery",
 "Capture breathing patterns before and after",
 "Note any interventions or assistance provided",
 "Record facial expressions and body tension"
 ],
 annotation_tags=["frequency", "productive", "gasping", "assistance_needed", "recovery_time"]
 ),

 # Agitation Behaviors
 "agitation_restlessness": ActivityDefinition(
 activity_id="agitation_restlessness",
 activity_name="Restlessness",
 category="agitation",
 description="Frequent position changes, fidgeting, inability to remain still",
 duration_min=5.0,
 duration_max=30.0,
 intensity_levels=["mild", "moderate", "severe"],
 clinical_significance="May indicate discomfort, anxiety, or delirium",
 recording_guidelines=[
 "Record continuous behavior over extended period",
 "Capture whole body movements",
 "Note frequency of position changes",
 "Record at least 2-3 episodes per intensity level"
 ],
 annotation_tags=["position_changes_per_minute", "intensity", "triggers", "body_parts_involved"]
 ),

 "agitation_hand_wringing": ActivityDefinition(
 activity_id="agitation_hand_wringing",
 activity_name="Hand Wringing",
 category="agitation",
 description="Repetitive hand rubbing or wringing movements",
 duration_min=3.0,
 duration_max=20.0,
 intensity_levels=["mild", "moderate", "severe"],
 clinical_significance="Common sign of anxiety or pain",
 recording_guidelines=[
 "Focus camera on upper body and hands",
 "Capture hand movement details",
 "Record multiple repetitions",
 "Note facial expressions simultaneously"
 ],
 annotation_tags=["repetitions_per_minute", "intensity", "hand_pressure", "facial_expression"]
 ),

 "agitation_self_contact": ActivityDefinition(
 activity_id="agitation_self_contact",
 activity_name="Self-Contact / Self-Soothing",
 category="agitation",
 description="Touching face, head, arms; self-hugging; protective postures",
 duration_min=2.0,
 duration_max=15.0,
 intensity_levels=["mild", "moderate", "severe"],
 clinical_significance="Indicates distress, pain, or need for comfort",
 recording_guidelines=[
 "Record full body to capture all contact points",
 "Note which body parts are touched",
 "Record duration and frequency",
 "Capture context (before/after events)"
 ],
 annotation_tags=["contact_locations", "duration", "frequency", "intensity", "context"]
 ),

 # Pain Behaviors
 "pain_facial_grimacing": ActivityDefinition(
 activity_id="pain_facial_grimacing",
 activity_name="Facial Grimacing",
 category="pain",
 description="Facial expressions indicating pain: frowning, eye closing, nose wrinkling",
 duration_min=1.0,
 duration_max=10.0,
 intensity_levels=["mild", "moderate", "severe"],
 clinical_significance="Direct indicator of pain experience",
 recording_guidelines=[
 "High-quality face capture required",
 "Ensure good lighting on face",
 "Record at least 5 repetitions per intensity",
 "Capture baseline neutral expression first"
 ],
 annotation_tags=["pain_scale_1_10", "facial_features", "duration", "triggers"]
 ),

 "pain_guarding": ActivityDefinition(
 activity_id="pain_guarding",
 activity_name="Guarding / Protective Posture",
 category="pain",
 description="Protective body positioning, tensing, or shielding of painful area",
 duration_min=2.0,
 duration_max=20.0,
 intensity_levels=["mild", "moderate", "severe"],
 clinical_significance="Indicates pain location and severity",
 recording_guidelines=[
 "Capture full body posture",
 "Record movements leading to guarding",
 "Note which body area is protected",
 "Record at least 3 episodes"
 ],
 annotation_tags=["body_area", "intensity", "movement_trigger", "duration"]
 ),

 "pain_bracing": ActivityDefinition(
 activity_id="pain_bracing",
 activity_name="Bracing",
 category="pain",
 description="Holding onto bed rails, furniture, or body parts for support during pain",
 duration_min=2.0,
 duration_max=15.0,
 intensity_levels=["mild", "moderate", "severe"],
 clinical_significance="Indicates acute pain episode",
 recording_guidelines=[
 "Record grip strength indicators (hand whitening)",
 "Capture full body tension",
 "Note what is being grasped",
 "Record facial expressions simultaneously"
 ],
 annotation_tags=["grip_location", "intensity", "duration", "facial_expression"]
 ),

 # Respiratory Patterns
 "breathing_shallow": ActivityDefinition(
 activity_id="breathing_shallow",
 activity_name="Shallow Breathing",
 category="respiratory",
 description="Rapid, shallow breathing with minimal chest expansion",
 duration_min=10.0,
 duration_max=60.0,
 intensity_levels=["mild", "moderate", "severe"],
 clinical_significance="May indicate respiratory distress or anxiety",
 recording_guidelines=[
 "Record chest and abdomen movements",
 "Count breaths per minute",
 "Note depth of breathing",
 "Record for extended period (30+ seconds)"
 ],
 annotation_tags=["respiratory_rate", "depth", "rhythm", "associated_symptoms"]
 ),

 "breathing_labored": ActivityDefinition(
 activity_id="breathing_labored",
 activity_name="Labored Breathing",
 category="respiratory",
 description="Difficulty breathing with visible effort, use of accessory muscles",
 duration_min=10.0,
 duration_max=60.0,
 intensity_levels=["moderate", "severe"],
 clinical_significance="Critical respiratory distress requiring intervention",
 recording_guidelines=[
 "Record chest, neck, and shoulder movements",
 "Capture accessory muscle use",
 "Note nasal flaring if visible",
 "Record facial expressions of distress"
 ],
 annotation_tags=["respiratory_rate", "accessory_muscles", "nasal_flaring", "oxygen_saturation"]
 ),

 # Clinical Interventions (Safety-Critical)
 "iv_manipulation": ActivityDefinition(
 activity_id="iv_manipulation",
 activity_name="IV Line Manipulation",
 category="clinical_intervention",
 description="Patient touching, pulling, or attempting to remove IV line",
 duration_min=2.0,
 duration_max=10.0,
 intensity_levels=["mild", "moderate", "severe"],
 clinical_significance="Critical safety event - risk of injury or infection",
 recording_guidelines=[
 "Record hand movements toward IV site",
 "Capture full arm and IV location",
 "Note intentionality (deliberate vs. accidental)",
 "Record at least 3 scenarios"
 ],
 annotation_tags=["hand_contact", "intentionality", "success", "intervention_needed"]
 ),

 "catheter_touching": ActivityDefinition(
 activity_id="catheter_touching",
 activity_name="Catheter Touching",
 category="clinical_intervention",
 description="Patient touching catheter or attempting to remove",
 duration_min=2.0,
 duration_max=10.0,
 intensity_levels=["mild", "moderate", "severe"],
 clinical_significance="Critical safety event - infection risk",
 recording_guidelines=[
 "Record hand movements toward catheter area",
 "Note privacy considerations",
 "Record intentionality",
 "Document intervention timing"
 ],
 annotation_tags=["hand_contact", "intentionality", "intervention_time"]
 ),

 "wound_picking": ActivityDefinition(
 activity_id="wound_picking",
 activity_name="Wound Picking / Interference",
 category="clinical_intervention",
 description="Patient touching, scratching, or picking at wound or surgical site",
 duration_min=1.0,
 duration_max=10.0,
 intensity_levels=["mild", "moderate", "severe"],
 clinical_significance="Risk of infection and delayed healing",
 recording_guidelines=[
 "Record hand movements toward wound area",
 "Note frequency and intensity",
 "Capture facial expressions",
 "Document intervention effectiveness"
 ],
 annotation_tags=["wound_location", "frequency", "intensity", "intervention_time"]
 ),
}


class RecordingProtocolManager:
 """Manages recording protocols and sessions."""

 def __init__(self, protocols_dir: str = "dataset_creation/protocols"):
 self.protocols_dir = Path(protocols_dir)
 self.protocols_dir.mkdir(parents=True, exist_ok=True)

 # Save activity definitions
 self.activities = CLINICAL_ACTIVITIES
 self._save_activity_definitions()

 def _save_activity_definitions(self):
 """Save activity definitions to JSON."""
 activities_dict = {
 act_id: asdict(activity)
 for act_id, activity in self.activities.items()
 }

 output_file = self.protocols_dir / "activity_definitions.json"
 with open(output_file, 'w') as f:
 json.dump(activities_dict, f, indent=2)

 def get_activity(self, activity_id: str) -> Optional[ActivityDefinition]:
 """Get activity definition by ID."""
 return self.activities.get(activity_id)

 def get_activities_by_category(self, category: str) -> List[ActivityDefinition]:
 """Get all activities in a category."""
 return [
 activity for activity in self.activities.values()
 if activity.category == category
 ]

 def create_recording_session(self,
 participant_id: str,
 activities: List[str],
 irb_protocol_id: str,
 researcher_id: str,
 location: str = "Clinical Lab",
 notes: str = "") -> RecordingSession:
 """
 Create a new recording session.

 Args:
 participant_id: Participant identifier (anonymized)
 activities: List of activity IDs to record
 irb_protocol_id: IRB protocol identifier
 researcher_id: Researcher identifier
 location: Recording location
 notes: Additional notes

 Returns:
 RecordingSession object
 """
 now = datetime.now()
 session_id = f"session_{participant_id}_{now.strftime('%Y%m%d_%H%M%S')}"

 session = RecordingSession(
 session_id=session_id,
 participant_id=participant_id,
 date=now.strftime('%Y-%m-%d'),
 time=now.strftime('%H:%M:%S'),
 location=location,
 camera_setup={
 "resolution": [1280, 720],
 "fps": 30,
 "camera_model": "Logitech C920",
 "lighting": "LED 5500K",
 "distance_to_subject": "2.0m"
 },
 activities_recorded=activities,
 consent_obtained=True, # Must be verified before recording
 irb_protocol_id=irb_protocol_id,
 researcher_id=researcher_id,
 notes=notes
 )

 # Save session metadata
 self._save_session(session)

 return session

 def _save_session(self, session: RecordingSession):
 """Save session metadata to JSON."""
 session_file = self.protocols_dir / f"{session.session_id}_metadata.json"
 with open(session_file, 'w') as f:
 json.dump(asdict(session), f, indent=2)

 def generate_consent_form(self, output_path: str):
 """Generate informed consent form template."""
 consent_template = """
INFORMED CONSENT FORM
Clinical Activity Monitoring Research Study

IRB Protocol ID: [IRB_PROTOCOL_ID]
Principal Investigator: [PI_NAME]
Institution: [INSTITUTION]

STUDY PURPOSE:
This research study aims to develop and validate an automated system for monitoring
patient activities and detecting clinical events such as falls, respiratory distress,
and agitation behaviors. Your participation will help improve patient safety monitoring.

PROCEDURES:
If you agree to participate, you will be video recorded performing various activities
including:
- Normal daily activities (sitting, standing, walking)
- Simulated respiratory events (coughing, breathing patterns)
- Movement behaviors (restlessness, position changes)

The recording will last approximately 30-60 minutes. All recordings are performed in
a controlled clinical environment with appropriate supervision.

RISKS:
The risks are minimal and similar to those of everyday life. There is a small risk of:
- Fatigue from performing activities
- Minor discomfort during simulated activities
- Privacy concerns (mitigated through data de-identification)

BENEFITS:
You may not directly benefit from this study, but your participation will contribute
to improving patient safety monitoring systems.

CONFIDENTIALITY:
All data will be de-identified and stored securely. Video recordings will be used only
for research purposes and destroyed after analysis. Your name will not appear in any
publications or presentations.

VOLUNTARY PARTICIPATION:
Your participation is completely voluntary. You may withdraw at any time without penalty.

CONTACTS:
For questions about the study: [RESEARCHER_CONTACT]
For questions about your rights as a research participant: [IRB_CONTACT]

CONSENT:
I have read and understood this consent form. I agree to participate in this study.

Participant Name: ____________________
Signature: ____________________
Date: ____________________

Researcher Name: ____________________
Signature: ____________________
Date: ____________________
"""

 output_path = Path(output_path)
 output_path.parent.mkdir(parents=True, exist_ok=True)

 with open(output_path, 'w') as f:
 f.write(consent_template)

 def generate_recording_checklist(self, session: RecordingSession, output_path: str):
 """Generate pre-recording checklist."""
 checklist = f"""
RECORDING SESSION CHECKLIST

Session ID: {session.session_id}
Participant ID: {session.participant_id}
Date: {session.date}
Researcher: {session.researcher_id}

PRE-RECORDING CHECKLIST:
[ ] Informed consent obtained and signed
[ ] Participant ID verified (de-identified)
[ ] Camera positioned correctly (2m distance, eye level)
[ ] Lighting verified (5500K, no shadows)
[ ] Audio recording enabled (if applicable)
[ ] Background clear and uncluttered
[ ] Privacy ensured (door closed, signage posted)
[ ] Emergency protocols reviewed
[ ] Recording equipment tested

ACTIVITIES TO RECORD:
"""
 for activity_id in session.activities_recorded:
 activity = self.get_activity(activity_id)
 if activity:
 checklist += f"\n[ ] {activity.activity_name} ({activity.category})"
 checklist += f"\n - Duration: {activity.duration_min}-{activity.duration_max}s"
 checklist += f"\n - Intensity levels: {', '.join(activity.intensity_levels)}"
 checklist += f"\n - Repetitions: ___/3 minimum"
 checklist += "\n"

 checklist += """
POST-RECORDING CHECKLIST:
[ ] All activities recorded successfully
[ ] Video files saved with correct naming
[ ] Metadata file created
[ ] Participant debriefed
[ ] Data backed up to secure storage
[ ] Equipment cleaned and stored

NOTES:
_____________________________________________________________________________
_____________________________________________________________________________
_____________________________________________________________________________

Researcher Signature: ____________________ Date: ____________________
"""

 output_path = Path(output_path)
 with open(output_path, 'w') as f:
 f.write(checklist)


# Convenience function
def create_recording_protocol(protocols_dir: str = "dataset_creation/protocols") -> RecordingProtocolManager:
 """Create and initialize recording protocol manager."""
 return RecordingProtocolManager(protocols_dir)
