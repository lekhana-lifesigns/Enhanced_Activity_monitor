#!/usr/bin/env python3
"""
Video Processor for NTU RGB+D Action Classification
Processes 1-minute video files to extract keypoints and classify actions.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import json
from collections import deque

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.pose.pose_estimator import PoseEstimator
from pipeline.detectors.yolo_segmentation_detector import YOLOSegmentationDetector
from analytics.enhanced_activity_classifier import EnhancedActivityClassifier
from dataset_creation.ntu_action_mapping import (
 NTU_120_ACTIONS, get_ntu_action_info, map_ntu_to_clinical_activity
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("video_processor")


class VideoProcessor:
 """
 Process 1-minute videos to extract keypoints and classify NTU RGB+D actions.
 """
 
 def __init__(self, 
 pose_model: str = "movenet_lightning",
 detector_model: str = "yolo11n.pt",
 fps: int = 30):
 """
 Initialize video processor.
 
 Args:
 pose_model: Pose estimation model ("movenet_lightning" or "mediapipe")
 detector_model: YOLO model path
 fps: Target FPS for processing (will sample frames accordingly)
 """
 self.fps = fps
 self.pose_estimator = None
 self.detector = None
 self.activity_classifier = None
 
 # Initialize pose estimator
 try:
 from pipeline.pose.pose_estimator import PoseEstimator
 self.pose_estimator = PoseEstimator(model_type=pose_model)
 log.info(f" Pose estimator initialized: {pose_model}")
 except Exception as e:
 log.error(f" Failed to initialize pose estimator: {e}")
 raise
 
 # Initialize detector
 try:
 from pipeline.detectors.yolo_segmentation_detector import YOLOSegmentationDetector
 self.detector = YOLOSegmentationDetector(model_path=detector_model)
 log.info(f" Detector initialized: {detector_model}")
 except Exception as e:
 log.warning(f" Detector initialization failed: {e}, continuing without detection")
 
 # Initialize activity classifier
 try:
 self.activity_classifier = EnhancedActivityClassifier()
 log.info(" Activity classifier initialized")
 except Exception as e:
 log.warning(f" Activity classifier initialization failed: {e}")
 
 def extract_keypoints_from_video(self, 
 video_path: str,
 output_path: Optional[str] = None,
 sample_rate: int = 1) -> Dict:
 """
 Extract keypoints from a video file.
 
 Args:
 video_path: Path to video file
 output_path: Optional path to save keypoints (JSON/NPZ)
 sample_rate: Process every Nth frame (1 = all frames)
 
 Returns:
 Dictionary with:
 - keypoints: List of keypoint arrays [T, 17, 3]
 - frames: Frame numbers processed
 - fps: Video FPS
 - duration: Video duration in seconds
 - metadata: Video metadata
 """
 log.info(f" Processing video: {video_path}")
 
 cap = cv2.VideoCapture(video_path)
 if not cap.isOpened():
 raise ValueError(f"Cannot open video: {video_path}")
 
 # Get video properties
 video_fps = cap.get(cv2.CAP_PROP_FPS)
 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 duration = total_frames / video_fps if video_fps > 0 else 0
 
 log.info(f"Video properties: {width}x{height}, {video_fps:.2f} FPS, {duration:.2f}s, {total_frames} frames")
 
 # Calculate frame sampling
 frames_to_process = list(range(0, total_frames, sample_rate))
 log.info(f"Processing {len(frames_to_process)} frames (sample_rate={sample_rate})")
 
 keypoints_list = []
 frame_numbers = []
 bboxes_list = []
 
 frame_idx = 0
 while cap.isOpened():
 ret, frame = cap.read()
 if not ret:
 break
 
 # Only process sampled frames
 if frame_idx not in frames_to_process:
 frame_idx += 1
 continue
 
 # Detect person
 person_detected = False
 bbox = None
 
 if self.detector:
 try:
 detections = self.detector.detect(frame)
 if detections and len(detections) > 0:
 # Use first person detection
 person_detection = detections[0]
 bbox = person_detection.get('bbox', None)
 person_detected = True
 except Exception as e:
 log.debug(f"Detection failed on frame {frame_idx}: {e}")
 
 # Extract keypoints
 if self.pose_estimator:
 try:
 if person_detected and bbox:
 # Crop to person bbox for better pose estimation
 x1, y1, x2, y2 = map(int, bbox[:4])
 x1, y1 = max(0, x1), max(0, y1)
 x2, y2 = min(width, x2), min(height, y2)
 person_crop = frame[y1:y2, x1:x2]
 kps = self.pose_estimator.estimate(person_crop)
 
 # Adjust keypoints back to full frame coordinates
 if kps is not None and len(kps) > 0:
 for kp in kps:
 kp[0] += x1 # x coordinate
 kp[1] += y1 # y coordinate
 else:
 # Estimate on full frame
 kps = self.pose_estimator.estimate(frame)
 
 if kps is not None and len(kps) > 0:
 # Convert to numpy array format [17, 3] where 3 = [x, y, confidence]
 kps_array = np.array(kps)
 if kps_array.shape[0] == 17: # COCO format
 keypoints_list.append(kps_array)
 frame_numbers.append(frame_idx)
 bboxes_list.append(bbox)
 except Exception as e:
 log.debug(f"Pose estimation failed on frame {frame_idx}: {e}")
 
 frame_idx += 1
 
 # Progress logging
 if frame_idx % 100 == 0:
 log.info(f"Processed {frame_idx}/{total_frames} frames...")
 
 cap.release()
 
 if len(keypoints_list) == 0:
 log.warning(" No keypoints extracted from video")
 return None
 
 # Convert to numpy array [T, 17, 3]
 keypoints_array = np.array(keypoints_list)
 
 log.info(f" Extracted {len(keypoints_list)} keypoint frames")
 
 result = {
 "keypoints": keypoints_array, # [T, 17, 3]
 "frames": frame_numbers,
 "bboxes": bboxes_list,
 "fps": video_fps,
 "duration": duration,
 "total_frames": total_frames,
 "processed_frames": len(keypoints_list),
 "metadata": {
 "video_path": video_path,
 "width": width,
 "height": height,
 "sample_rate": sample_rate,
 }
 }
 
 # Save if output path provided
 if output_path:
 self._save_keypoints(result, output_path)
 
 return result
 
 def classify_video_actions(self, 
 keypoints_data: Dict,
 window_size: int = 30) -> Dict:
 """
 Classify actions in video using extracted keypoints.
 
 Args:
 keypoints_data: Output from extract_keypoints_from_video
 window_size: Number of frames for temporal window
 
 Returns:
 Dictionary with:
 - predictions: List of (frame_idx, activity, confidence, ntu_action_id)
 - dominant_action: Most frequent action
 - action_timeline: Action changes over time
 """
 if self.activity_classifier is None:
 log.warning("Activity classifier not available")
 return None
 
 keypoints = keypoints_data["keypoints"]
 frame_numbers = keypoints_data["frames"]
 
 log.info(f" Classifying actions from {len(keypoints)} keypoint frames")
 
 predictions = []
 kps_history = deque(maxlen=window_size)
 
 for i, kps_frame in enumerate(keypoints):
 # Add to history
 kps_history.append(kps_frame.tolist())
 
 # Classify activity
 if len(kps_history) >= 5: # Need some history
 try:
 activity_result = self.activity_classifier.classify_activity(
 kps=kps_frame.tolist(),
 kps_history=list(kps_history)
 )
 
 activity = activity_result.get("activity", "unknown")
 confidence = activity_result.get("confidence", 0.0)
 
 # Map to NTU action (simplified - would need more sophisticated mapping)
 ntu_action_id = self._map_activity_to_ntu(activity)
 
 predictions.append({
 "frame_idx": frame_numbers[i],
 "activity": activity,
 "confidence": float(confidence),
 "ntu_action_id": ntu_action_id,
 "priority": activity_result.get("priority", "NORMAL")
 })
 except Exception as e:
 log.debug(f"Classification failed on frame {i}: {e}")
 
 # Find dominant action
 if predictions:
 activity_counts = {}
 for pred in predictions:
 activity = pred["activity"]
 activity_counts[activity] = activity_counts.get(activity, 0) + 1
 
 dominant_action = max(activity_counts.items(), key=lambda x: x[1])
 
 result = {
 "predictions": predictions,
 "dominant_action": {
 "activity": dominant_action[0],
 "count": dominant_action[1],
 "percentage": dominant_action[1] / len(predictions) * 100
 },
 "total_frames_classified": len(predictions),
 "action_distribution": activity_counts
 }
 
 log.info(f" Classified {len(predictions)} frames")
 log.info(f"Dominant action: {dominant_action[0]} ({dominant_action[1]}/{len(predictions)} frames)")
 
 return result
 
 return None
 
 def _map_activity_to_ntu(self, activity: str) -> Optional[str]:
 """Map system activity to NTU action ID."""
 # Reverse mapping - find NTU action that matches activity
 # Note: Using "C" prefix for Clinical action classes (C1-C120)
 activity_to_ntu = {
 "sitting": "C8",
 "standing": "C9",
 "walking": "C26",
 "running": "C27",
 "falling": "C43",
 "fallen": "C43",
 "coughing": "C41",
 "waving": "C23",
 "pointing": "C31",
 "reaching": "C6",
 "drinking": "C1",
 "eating": "C2",
 "agitated": "C50",
 "tripping": "C42",
 }
 
 return activity_to_ntu.get(activity, None)
 
 def _save_keypoints(self, data: Dict, output_path: str):
 """Save extracted keypoints to file."""
 output_path = Path(output_path)
 output_path.parent.mkdir(parents=True, exist_ok=True)
 
 if output_path.suffix == ".npz":
 np.savez(
 output_path,
 keypoints=data["keypoints"],
 frames=np.array(data["frames"]),
 fps=data["fps"],
 duration=data["duration"],
 metadata=json.dumps(data["metadata"])
 )
 elif output_path.suffix == ".json":
 # Convert numpy arrays to lists for JSON
 json_data = {
 "keypoints": data["keypoints"].tolist(),
 "frames": data["frames"],
 "fps": data["fps"],
 "duration": data["duration"],
 "metadata": data["metadata"]
 }
 with open(output_path, 'w') as f:
 json.dump(json_data, f, indent=2)
 else:
 # Default to NPZ
 np.savez(output_path.with_suffix(".npz"), **data)
 
 log.info(f" Saved keypoints to: {output_path}")
 
 def process_video(self, 
 video_path: str,
 output_dir: Optional[str] = None,
 classify: bool = True,
 sample_rate: int = 1) -> Dict:
 """
 Complete video processing pipeline.
 
 Args:
 video_path: Path to video file
 output_dir: Directory to save outputs
 classify: Whether to classify actions
 sample_rate: Frame sampling rate
 
 Returns:
 Complete processing results
 """
 video_path = Path(video_path)
 if not video_path.exists():
 raise FileNotFoundError(f"Video not found: {video_path}")
 
 # Extract keypoints
 keypoints_output = None
 if output_dir:
 output_dir = Path(output_dir)
 output_dir.mkdir(parents=True, exist_ok=True)
 keypoints_output = output_dir / f"{video_path.stem}_keypoints.npz"
 
 keypoints_data = self.extract_keypoints_from_video(
 str(video_path),
 output_path=str(keypoints_output) if keypoints_output else None,
 sample_rate=sample_rate
 )
 
 if keypoints_data is None:
 return {"error": "Failed to extract keypoints"}
 
 result = {
 "video_path": str(video_path),
 "keypoints": keypoints_data,
 "classification": None
 }
 
 # Classify actions
 if classify and self.activity_classifier:
 classification_result = self.classify_video_actions(keypoints_data)
 result["classification"] = classification_result
 
 # Save classification results
 if output_dir and classification_result:
 classification_output = output_dir / f"{video_path.stem}_classification.json"
 with open(classification_output, 'w') as f:
 json.dump(classification_result, f, indent=2)
 log.info(f" Saved classification to: {classification_output}")
 
 return result


def process_video_batch(video_dir: str, 
 output_dir: str,
 video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv"],
 sample_rate: int = 1):
 """
 Process multiple videos in a directory.
 
 Args:
 video_dir: Directory containing videos
 output_dir: Output directory for results
 video_extensions: List of video file extensions to process
 sample_rate: Frame sampling rate
 """
 video_dir = Path(video_dir)
 output_dir = Path(output_dir)
 output_dir.mkdir(parents=True, exist_ok=True)
 
 # Find all video files
 video_files = []
 for ext in video_extensions:
 video_files.extend(video_dir.glob(f"*{ext}"))
 video_files.extend(video_dir.glob(f"*{ext.upper()}"))
 
 log.info(f" Found {len(video_files)} video files")
 
 # Initialize processor
 processor = VideoProcessor()
 
 # Process each video
 results = []
 for i, video_file in enumerate(video_files, 1):
 log.info(f"\n{'='*60}")
 log.info(f"Processing video {i}/{len(video_files)}: {video_file.name}")
 log.info(f"{'='*60}")
 
 try:
 result = processor.process_video(
 str(video_file),
 output_dir=str(output_dir / video_file.stem),
 classify=True,
 sample_rate=sample_rate
 )
 results.append(result)
 except Exception as e:
 log.error(f" Failed to process {video_file.name}: {e}")
 results.append({"video": str(video_file), "error": str(e)})
 
 # Save batch summary
 summary = {
 "total_videos": len(video_files),
 "successful": sum(1 for r in results if "error" not in r),
 "failed": sum(1 for r in results if "error" in r),
 "results": results
 }
 
 summary_path = output_dir / "batch_processing_summary.json"
 with open(summary_path, 'w') as f:
 json.dump(summary, f, indent=2)
 
 log.info(f"\n{'='*60}")
 log.info(f" Batch processing complete!")
 log.info(f"Successfully processed: {summary['successful']}/{summary['total_videos']}")
 log.info(f"Summary saved to: {summary_path}")
 
 return summary


if __name__ == "__main__":
 import argparse
 
 parser = argparse.ArgumentParser(description="Process videos for NTU RGB+D action classification")
 parser.add_argument("--video", type=str, help="Path to video file")
 parser.add_argument("--video_dir", type=str, help="Directory containing videos (batch mode)")
 parser.add_argument("--output_dir", type=str, default="./video_output", help="Output directory")
 parser.add_argument("--sample_rate", type=int, default=1, help="Process every Nth frame")
 parser.add_argument("--no_classify", action="store_true", help="Skip action classification")
 
 args = parser.parse_args()
 
 if args.video_dir:
 # Batch processing
 process_video_batch(
 video_dir=args.video_dir,
 output_dir=args.output_dir,
 sample_rate=args.sample_rate
 )
 elif args.video:
 # Single video processing
 processor = VideoProcessor()
 result = processor.process_video(
 video_path=args.video,
 output_dir=args.output_dir,
 classify=not args.no_classify,
 sample_rate=args.sample_rate
 )
 print("\n" + "="*60)
 print("Processing Results:")
 print("="*60)
 print(json.dumps(result, indent=2, default=str))
 else:
 parser.print_help()
