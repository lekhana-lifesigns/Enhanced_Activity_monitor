"""
Clinical Dataset Recorder
Records video + metadata for custom clinical activity datasets.

Integrates with existing camera system to record:
- Raw video frames
- Pose keypoints
- Activity annotations
- Session metadata
"""

import cv2
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from dataset_creation.recording_protocol import RecordingSession, RecordingProtocolManager, CLINICAL_ACTIVITIES

log = logging.getLogger("dataset_recorder")


class DatasetRecorder:
 """
 Records clinical activity datasets with synchronized video and metadata.
 """

 def __init__(self,
 output_dir: str = "custom_datasets",
 camera_idx: int = 0,
 resolution: Tuple[int, int] = (1280, 720),
 fps: int = 30):
 """
 Initialize dataset recorder.

 Args:
 output_dir: Directory to save recorded datasets
 camera_idx: Camera device index
 resolution: Video resolution (width, height)
 fps: Target frames per second
 """
 self.output_dir = Path(output_dir)
 self.output_dir.mkdir(parents=True, exist_ok=True)

 self.camera_idx = camera_idx
 self.resolution = resolution
 self.fps = fps

 # Recording state
 self.is_recording = False
 self.current_session = None
 self.current_activity = None
 self.recording_start_time = None

 # Data buffers
 self.frames = []
 self.keypoints = []
 self.timestamps = []
 self.annotations = []

 # Camera
 self.cap = None

 # Protocol manager
 self.protocol_manager = RecordingProtocolManager()

 log.info(f"Dataset recorder initialized. Output: {self.output_dir}")

 def initialize_camera(self) -> bool:
 """
 Initialize camera capture.

 Returns:
 True if successful
 """
 try:
 self.cap = cv2.VideoCapture(self.camera_idx)

 if not self.cap.isOpened():
 log.error(f"Failed to open camera {self.camera_idx}")
 return False

 # Set camera properties
 self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
 self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
 self.cap.set(cv2.CAP_PROP_FPS, self.fps)

 # Verify settings
 actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

 log.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")

 return True

 except Exception as e:
 log.error(f"Error initializing camera: {e}")
 return False

 def start_session(self,
 participant_id: str,
 activities: List[str],
 irb_protocol_id: str,
 researcher_id: str,
 notes: str = "") -> Optional[RecordingSession]:
 """
 Start a new recording session.

 Args:
 participant_id: Participant identifier (anonymized)
 activities: List of activity IDs to record
 irb_protocol_id: IRB protocol identifier
 researcher_id: Researcher identifier
 notes: Session notes

 Returns:
 RecordingSession object or None if failed
 """
 if self.current_session:
 log.warning("Session already active. Stop current session first.")
 return None

 # Create session
 self.current_session = self.protocol_manager.create_recording_session(
 participant_id=participant_id,
 activities=activities,
 irb_protocol_id=irb_protocol_id,
 researcher_id=researcher_id,
 notes=notes
 )

 # Create session directory
 session_dir = self.output_dir / self.current_session.session_id
 session_dir.mkdir(parents=True, exist_ok=True)

 log.info(f"Started session: {self.current_session.session_id}")

 return self.current_session

 def start_activity_recording(self,
 activity_id: str,
 intensity: str = "moderate",
 notes: str = "") -> bool:
 """
 Start recording an activity.

 Args:
 activity_id: Activity ID from CLINICAL_ACTIVITIES
 intensity: Intensity level (mild, moderate, severe)
 notes: Recording notes

 Returns:
 True if started successfully
 """
 if not self.current_session:
 log.error("No active session. Start a session first.")
 return False

 if self.is_recording:
 log.warning("Already recording. Stop current recording first.")
 return False

 # Validate activity
 activity_def = self.protocol_manager.get_activity(activity_id)
 if not activity_def:
 log.error(f"Unknown activity: {activity_id}")
 return False

 if intensity not in activity_def.intensity_levels:
 log.warning(f"Invalid intensity '{intensity}' for {activity_id}. Using default.")
 intensity = activity_def.intensity_levels[0]

 # Initialize buffers
 self.frames = []
 self.keypoints = []
 self.timestamps = []
 self.annotations = []

 # Set state
 self.current_activity = {
 'activity_id': activity_id,
 'activity_name': activity_def.activity_name,
 'category': activity_def.category,
 'intensity': intensity,
 'notes': notes,
 'start_time': time.time()
 }

 self.is_recording = True
 self.recording_start_time = time.time()

 log.info(f"Started recording: {activity_def.activity_name} ({intensity})")

 return True

 def record_frame(self, frame: np.ndarray, keypoints: Optional[List] = None) -> bool:
 """
 Record a single frame with optional keypoints.

 Args:
 frame: Video frame (BGR format)
 keypoints: Optional pose keypoints [[x, y, conf], ...]

 Returns:
 True if recorded successfully
 """
 if not self.is_recording:
 return False

 # Store frame
 self.frames.append(frame.copy())

 # Store keypoints (if provided)
 if keypoints is not None:
 self.keypoints.append(keypoints)
 else:
 self.keypoints.append([])

 # Store timestamp
 elapsed = time.time() - self.recording_start_time
 self.timestamps.append(elapsed)

 return True

 def add_annotation(self, annotation_type: str, value: any, timestamp: Optional[float] = None):
 """
 Add an annotation at current or specified timestamp.

 Args:
 annotation_type: Type of annotation (e.g., "pain_scale", "trigger", "note")
 value: Annotation value
 timestamp: Optional timestamp (uses current if not provided)
 """
 if not self.is_recording:
 return

 if timestamp is None:
 timestamp = time.time() - self.recording_start_time

 self.annotations.append({
 'timestamp': timestamp,
 'type': annotation_type,
 'value': value
 })

 log.info(f"Added annotation: {annotation_type}={value} @ {timestamp:.2f}s")

 def stop_activity_recording(self, save: bool = True) -> Optional[Dict]:
 """
 Stop recording current activity.

 Args:
 save: Whether to save the recording

 Returns:
 Recording metadata or None if failed
 """
 if not self.is_recording:
 log.warning("No active recording")
 return None

 self.is_recording = False

 # Calculate duration
 duration = time.time() - self.recording_start_time

 # Create metadata
 metadata = {
 'session_id': self.current_session.session_id,
 'participant_id': self.current_session.participant_id,
 'activity_id': self.current_activity['activity_id'],
 'activity_name': self.current_activity['activity_name'],
 'category': self.current_activity['category'],
 'intensity': self.current_activity['intensity'],
 'duration': duration,
 'num_frames': len(self.frames),
 'fps': len(self.frames) / duration if duration > 0 else 0,
 'resolution': self.resolution,
 'annotations': self.annotations,
 'notes': self.current_activity['notes'],
 'timestamp': datetime.now().isoformat()
 }

 log.info(f"Stopped recording: {metadata['activity_name']} ({duration:.2f}s, {len(self.frames)} frames)")

 # Save if requested
 if save:
 self._save_recording(metadata)

 return metadata

 def _save_recording(self, metadata: Dict):
 """Save recording to disk."""
 session_dir = self.output_dir / self.current_session.session_id
 activity_dir = session_dir / metadata['category']
 activity_dir.mkdir(parents=True, exist_ok=True)

 # Generate filename
 timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
 filename_base = f"{metadata['activity_id']}_{metadata['intensity']}_{timestamp}"

 # Save video
 video_path = activity_dir / f"{filename_base}.avi"
 self._save_video(video_path)

 # Save keypoints
 keypoints_path = activity_dir / f"{filename_base}_keypoints.json"
 self._save_keypoints(keypoints_path)

 # Save metadata
 metadata_path = activity_dir / f"{filename_base}_metadata.json"
 with open(metadata_path, 'w') as f:
 json.dump(metadata, f, indent=2)

 log.info(f"Saved recording to {activity_dir}")

 def _save_video(self, output_path: Path):
 """Save frames as video file."""
 if not self.frames:
 return

 fourcc = cv2.VideoWriter_fourcc(*'XVID')
 out = cv2.VideoWriter(
 str(output_path),
 fourcc,
 self.fps,
 self.resolution
 )

 for frame in self.frames:
 out.write(frame)

 out.release()
 log.info(f"Saved video: {output_path} ({len(self.frames)} frames)")

 def _save_keypoints(self, output_path: Path):
 """Save keypoints as JSON."""
 keypoints_data = {
 'num_frames': len(self.keypoints),
 'timestamps': self.timestamps,
 'keypoints': self.keypoints
 }

 with open(output_path, 'w') as f:
 json.dump(keypoints_data, f, indent=2)

 def stop_session(self):
 """Stop current recording session."""
 if self.is_recording:
 self.stop_activity_recording(save=True)

 self.current_session = None
 log.info("Session stopped")

 def release(self):
 """Release camera resources."""
 if self.cap:
 self.cap.release()
 log.info("Camera released")

 def preview_mode(self):
 """
 Run camera preview mode with recording controls.

 Controls:
 - SPACE: Start/stop recording
 - A: Add annotation
 - Q: Quit
 - S: Save current recording
 """
 if not self.initialize_camera():
 log.error("Failed to initialize camera")
 return

 log.info("Preview mode started")
 log.info("Controls: SPACE=start/stop, A=annotate, S=save, Q=quit")

 while True:
 ret, frame = self.cap.read()
 if not ret:
 break

 # Record frame if active
 if self.is_recording:
 self.record_frame(frame)

 # Draw recording indicator
 cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
 cv2.putText(frame, "REC", (50, 40),
 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

 # Show duration
 duration = time.time() - self.recording_start_time
 cv2.putText(frame, f"{duration:.1f}s", (50, 70),
 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

 # Show activity info
 if self.current_activity:
 info_text = f"{self.current_activity['activity_name']} ({self.current_activity['intensity']})"
 cv2.putText(frame, info_text, (10, frame.shape[0] - 20),
 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

 # Display frame
 cv2.imshow('Dataset Recorder', frame)

 # Handle keyboard input
 key = cv2.waitKey(1) & 0xFF

 if key == ord('q'):
 break
 elif key == ord(' '): # Space - toggle recording
 if self.is_recording:
 self.stop_activity_recording(save=True)
 log.info("Recording stopped")
 else:
 log.warning("Use start_activity_recording() to start recording")
 elif key == ord('a'): # Add annotation
 if self.is_recording:
 self.add_annotation("manual_marker", "user_marked")
 elif key == ord('s'): # Save
 if self.is_recording:
 self.stop_activity_recording(save=True)

 self.release()
 cv2.destroyAllWindows()


def quick_record_activity(activity_id: str,
 participant_id: str,
 intensity: str = "moderate",
 duration: float = 10.0,
 camera_idx: int = 0) -> Optional[Dict]:
 """
 Quick function to record a single activity.

 Args:
 activity_id: Activity ID from CLINICAL_ACTIVITIES
 participant_id: Participant identifier
 intensity: Intensity level
 duration: Recording duration in seconds
 camera_idx: Camera device index

 Returns:
 Recording metadata or None if failed
 """
 recorder = DatasetRecorder(camera_idx=camera_idx)

 if not recorder.initialize_camera():
 return None

 # Start session
 session = recorder.start_session(
 participant_id=participant_id,
 activities=[activity_id],
 irb_protocol_id="IRB-2024-001",
 researcher_id="researcher_01"
 )

 if not session:
 return None

 # Start recording
 if not recorder.start_activity_recording(activity_id, intensity):
 return None

 log.info(f"Recording {activity_id} for {duration}s...")

 # Record frames
 start_time = time.time()
 while (time.time() - start_time) < duration:
 ret, frame = recorder.cap.read()
 if ret:
 recorder.record_frame(frame)

 # Stop and save
 metadata = recorder.stop_activity_recording(save=True)

 recorder.release()

 return metadata
