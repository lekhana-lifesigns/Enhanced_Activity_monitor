# dataset_creation/data_collector.py
"""
Data Collection for Biomedical HPE Dataset Creation (Phase 1)
Collects and annotates data for motor development, rehabilitation, and gait analysis.
"""

import cv2
import numpy as np
import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yaml

log = logging.getLogger("data_collector")

class BiomedicalDataCollector:
 """
 Collects annotated data for biomedical HPE applications.
 Supports motor development assessment, rehabilitation monitoring, and gait analysis.
 """

 def __init__(self, output_dir: str, config: Dict):
 self.output_dir = output_dir
 self.config = config
 self.collection_type = config.get("collection_type", "motor_development") # motor_development, rehabilitation, gait_analysis

 # Create output directories
 self.frames_dir = os.path.join(output_dir, "frames")
 self.annotations_dir = os.path.join(output_dir, "annotations")
 os.makedirs(self.frames_dir, exist_ok=True)
 os.makedirs(self.annotations_dir, exist_ok=True)

 # Data collection parameters
 self.fps = config.get("fps", 30)
 self.max_frames_per_session = config.get("max_frames_per_session", 3000) # 100 seconds at 30fps
 self.frame_interval = 1.0 / self.fps

 # Session tracking
 self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
 self.frame_count = 0
 self.session_data = []

 # Annotation metadata
 self.metadata = {
 "session_id": self.session_id,
 "collection_type": self.collection_type,
 "fps": self.fps,
 "start_time": datetime.now().isoformat(),
 "frames_collected": 0,
 "annotations": []
 }

 log.info(f"Data collector initialized for {self.collection_type} (session: {self.session_id})")

 def collect_frame(self, frame: np.ndarray, keypoints_2d: np.ndarray,
 keypoints_3d: Optional[np.ndarray] = None,
 metadata: Optional[Dict] = None) -> bool:
 """
 Collect a frame with pose data and metadata.

 Args:
 frame: RGB frame
 keypoints_2d: 2D keypoints (17, 3)
 keypoints_3d: Optional 3D keypoints (17, 4)
 metadata: Additional metadata (subject_id, activity, etc.)

 Returns:
 True if frame collected, False if session limit reached
 """
 if self.frame_count >= self.max_frames_per_session:
 log.info("Session frame limit reached")
 return False

 # Generate frame filename
 frame_filename = f"{self.frame_count:03d}.jpg"
 frame_path = os.path.join(self.frames_dir, frame_filename)

 # Save frame
 cv2.imwrite(frame_path, frame)

 # Prepare annotation data
 annotation = {
 "frame_id": self.frame_count,
 "timestamp": time.time(),
 "frame_path": frame_filename,
 "keypoints_2d": keypoints_2d.tolist(),
 "keypoints_3d": keypoints_3d.tolist() if keypoints_3d is not None else None,
 "metadata": metadata or {}
 }

 # Add collection-type specific annotations
 if self.collection_type == "motor_development":
 annotation.update(self._get_motor_dev_annotations(keypoints_2d, keypoints_3d, metadata))
 elif self.collection_type == "rehabilitation":
 annotation.update(self._get_rehab_annotations(keypoints_2d, keypoints_3d, metadata))
 elif self.collection_type == "gait_analysis":
 annotation.update(self._get_gait_annotations(keypoints_2d, keypoints_3d, metadata))

 # Store annotation
 self.session_data.append(annotation)
 self.frame_count += 1

 return True

 def _get_motor_dev_annotations(self, kps_2d: np.ndarray, kps_3d: Optional[np.ndarray],
 metadata: Optional[Dict]) -> Dict:
 """Get motor development specific annotations."""
 annotations = {
 "subject_age_months": metadata.get("subject_age_months", 0) if metadata else 0,
 "movement_type": metadata.get("movement_type", "unknown") if metadata else "unknown",
 "quality_score": 0.0, # Would be computed by assessor
 "abnormality_flags": []
 }

 # Add joint angles for motor development analysis
 if kps_2d.shape[0] >= 17:
 annotations["joint_angles"] = {
 "left_elbow": self._compute_angle(kps_2d, 9, 7, 5), # wrist-elbow-shoulder
 "right_elbow": self._compute_angle(kps_2d, 10, 8, 6),
 "left_knee": self._compute_angle(kps_2d, 15, 13, 11), # ankle-knee-hip
 "right_knee": self._compute_angle(kps_2d, 16, 14, 12),
 "left_hip": self._compute_angle(kps_2d, 13, 11, 5), # knee-hip-shoulder
 "right_hip": self._compute_angle(kps_2d, 14, 12, 6)
 }

 return annotations

 def _get_rehab_annotations(self, kps_2d: np.ndarray, kps_3d: Optional[np.ndarray],
 metadata: Optional[Dict]) -> Dict:
 """Get rehabilitation specific annotations."""
 annotations = {
 "exercise_type": metadata.get("exercise_type", "unknown") if metadata else "unknown",
 "rep_count": metadata.get("rep_count", 0) if metadata else 0,
 "form_score": 0.0, # Would be computed by monitor
 "range_of_motion": 0.0
 }

 # Add exercise-specific metrics
 exercise = annotations["exercise_type"]
 if exercise == "upper_limb" and kps_2d.shape[0] >= 17:
 annotations["joint_angles"] = {
 "left_shoulder": self._compute_angle(kps_2d, 7, 5, 11), # elbow-shoulder-hip
 "right_shoulder": self._compute_angle(kps_2d, 8, 6, 12),
 "left_elbow": self._compute_angle(kps_2d, 9, 7, 5),
 "right_elbow": self._compute_angle(kps_2d, 10, 8, 6)
 }
 elif exercise == "lower_limb" and kps_2d.shape[0] >= 17:
 annotations["joint_angles"] = {
 "left_hip": self._compute_angle(kps_2d, 13, 11, 5),
 "right_hip": self._compute_angle(kps_2d, 14, 12, 6),
 "left_knee": self._compute_angle(kps_2d, 15, 13, 11),
 "right_knee": self._compute_angle(kps_2d, 16, 14, 12)
 }

 return annotations

 def _get_gait_annotations(self, kps_2d: np.ndarray, kps_3d: Optional[np.ndarray],
 metadata: Optional[Dict]) -> Dict:
 """Get gait analysis specific annotations."""
 annotations = {
 "gait_phase": metadata.get("gait_phase", "unknown") if metadata else "unknown",
 "step_count": metadata.get("step_count", 0) if metadata else 0,
 "cadence": 0.0,
 "step_length": 0.0
 }

 # Add spatiotemporal parameters
 if kps_2d.shape[0] >= 17:
 # Calculate basic gait metrics
 left_ankle = kps_2d[15][:2] if kps_2d[15][2] > 0.5 else None
 right_ankle = kps_2d[16][:2] if kps_2d[16][2] > 0.5 else None

 if left_ankle is not None and right_ankle is not None:
 # Simple step length estimation (would need calibration)
 annotations["ankle_positions"] = {
 "left": left_ankle.tolist(),
 "right": right_ankle.tolist()
 }

 return annotations

 def save_session(self):
 """Save collected session data."""
 # Save metadata
 self.metadata["frames_collected"] = self.frame_count
 self.metadata["end_time"] = datetime.now().isoformat()
 self.metadata["annotations"] = self.session_data

 metadata_path = os.path.join(self.annotations_dir, f"session_{self.session_id}.json")
 with open(metadata_path, 'w') as f:
 json.dump(self.metadata, f, indent=2)

 # Save individual frame annotations
 for annotation in self.session_data:
 frame_id = annotation["frame_id"]
 ann_path = os.path.join(self.annotations_dir, f"{frame_id:06d}.json")
 with open(ann_path, 'w') as f:
 json.dump(annotation, f, indent=2)

 log.info(f"Session saved: {self.frame_count} frames, {len(self.session_data)} annotations")
 return metadata_path

 def _compute_angle(self, kps: np.ndarray, joint1: int, joint2: int, joint3: int) -> float:
 """Compute angle at joint2 formed by joint1-joint2-joint3."""
 if (kps[joint1][2] < 0.5 or kps[joint2][2] < 0.5 or kps[joint3][2] < 0.5):
 return 0.0

 v1 = kps[joint1][:2] - kps[joint2][:2]
 v2 = kps[joint3][:2] - kps[joint2][:2]

 n1 = np.linalg.norm(v1)
 n2 = np.linalg.norm(v2)
 if n1 == 0.0 or n2 == 0.0:
 return 0.0 # Return safe default for zero-length vectors
 
 cos_angle = np.dot(v1, v2) / (n1 * n2)
 cos_angle = np.clip(cos_angle, -1, 1)
 return np.degrees(np.arccos(cos_angle))


class DatasetManager:
 """
 Manages multiple data collection sessions and creates training datasets.
 """

 def __init__(self, base_dir: str):
 self.base_dir = base_dir
 self.sessions = []

 # Load existing sessions
 self._load_sessions()

 def _load_sessions(self):
 """Load existing session metadata."""
 annotations_dir = os.path.join(self.base_dir, "annotations")
 if not os.path.exists(annotations_dir):
 return

 for filename in os.listdir(annotations_dir):
 if filename.startswith("session_") and filename.endswith(".json"):
 session_path = os.path.join(annotations_dir, filename)
 try:
 with open(session_path, 'r') as f:
 session_data = json.load(f)
 self.sessions.append(session_data)
 except Exception as e:
 log.warning(f"Failed to load session {filename}: {e}")

 def create_dataset(self, output_path: str, collection_types: List[str] = None,
 min_frames_per_session: int = 100) -> Dict:
 """
 Create a training dataset from collected sessions.

 Args:
 output_path: Path to save the dataset
 collection_types: Types of data to include (None for all)
 min_frames_per_session: Minimum frames required per session

 Returns:
 Dataset statistics
 """
 # Filter sessions
 filtered_sessions = []
 for session in self.sessions:
 if collection_types and session["collection_type"] not in collection_types:
 continue
 if session["frames_collected"] < min_frames_per_session:
 continue
 filtered_sessions.append(session)

 # Create dataset structure
 dataset = {
 "info": {
 "description": "Biomedical HPE Dataset",
 "version": "1.0",
 "created": datetime.now().isoformat(),
 "collection_types": collection_types or ["all"]
 },
 "sessions": filtered_sessions,
 "statistics": self._compute_dataset_stats(filtered_sessions)
 }

 # Save dataset
 os.makedirs(os.path.dirname(output_path), exist_ok=True)
 with open(output_path, 'w') as f:
 json.dump(dataset, f, indent=2)

 log.info(f"Dataset created: {len(filtered_sessions)} sessions, saved to {output_path}")
 return dataset["statistics"]

 def _compute_dataset_stats(self, sessions: List[Dict]) -> Dict:
 """Compute dataset statistics."""
 stats = {
 "total_sessions": len(sessions),
 "total_frames": sum(s["frames_collected"] for s in sessions),
 "collection_types": {},
 "subject_age_distribution": {},
 "exercise_distribution": {}
 }

 for session in sessions:
 # Collection types
 coll_type = session["collection_type"]
 stats["collection_types"][coll_type] = stats["collection_types"].get(coll_type, 0) + 1

 # Subject ages (for motor development)
 if "annotations" in session and session["annotations"]:
 for ann in session["annotations"]:
 if "subject_age_months" in ann:
 age = ann["subject_age_months"]
 age_group = f"{age//6 * 6}-{(age//6 + 1)*6 - 1} months"
 stats["subject_age_distribution"][age_group] = \
 stats["subject_age_distribution"].get(age_group, 0) + 1

 # Exercises (for rehabilitation)
 if "exercise_type" in ann:
 exercise = ann["exercise_type"]
 stats["exercise_distribution"][exercise] = \
 stats["exercise_distribution"].get(exercise, 0) + 1

 return stats