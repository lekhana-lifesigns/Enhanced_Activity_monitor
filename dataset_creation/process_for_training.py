#!/usr/bin/env python3
"""
Process Video for Annotation and Training
Complete workflow: Extract keypoints → Validate → Annotate → Prepare for training
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("process_training")

try:
 import cv2
except ImportError:
 log.error("OpenCV not installed. Install with: pip install opencv-python")
 sys.exit(1)


class TrainingDataProcessor:
 """
 Process videos for annotation and training.
 """
 
 def __init__(self, output_dir: str = "./training_data"):
 """
 Initialize processor.
 
 Args:
 output_dir: Base output directory for training data
 """
 self.output_dir = Path(output_dir)
 self.output_dir.mkdir(parents=True, exist_ok=True)
 
 # Create subdirectories
 self.keypoints_dir = self.output_dir / "keypoints"
 self.annotations_dir = self.output_dir / "annotations"
 self.frames_dir = self.output_dir / "frames"
 self.metadata_dir = self.output_dir / "metadata"
 
 for dir_path in [self.keypoints_dir, self.annotations_dir, self.frames_dir, self.metadata_dir]:
 dir_path.mkdir(parents=True, exist_ok=True)
 
 def extract_keypoints_simple(self, video_path: str, sample_rate: int = 1) -> Dict:
 """
 Extract keypoints from video (simplified version without pose estimator).
 This extracts frame data for manual annotation.
 
 Args:
 video_path: Path to video file
 sample_rate: Process every Nth frame
 
 Returns:
 Dictionary with video metadata and frame paths
 """
 video_path = Path(video_path)
 log.info(f" Processing video: {video_path.name}")
 
 cap = cv2.VideoCapture(str(video_path))
 if not cap.isOpened():
 raise ValueError(f"Cannot open video: {video_path}")
 
 # Get video properties
 width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 fps = cap.get(cv2.CAP_PROP_FPS)
 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 duration = total_frames / fps if fps > 0 else 0
 
 log.info(f"Video: {width}×{height}, {fps:.2f} FPS, {duration:.2f}s, {total_frames} frames")
 
 # Extract frames
 frame_data = []
 frame_idx = 0
 saved_count = 0
 
 video_stem = video_path.stem
 
 while cap.isOpened():
 ret, frame = cap.read()
 if not ret:
 break
 
 # Sample frames
 if frame_idx % sample_rate == 0:
 # Save frame
 frame_filename = f"{video_stem}_frame_{frame_idx:06d}.jpg"
 frame_path = self.frames_dir / frame_filename
 
 cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
 
 frame_data.append({
 "frame_number": frame_idx,
 "frame_filename": frame_filename,
 "frame_path": str(frame_path.relative_to(self.output_dir)),
 "timestamp": frame_idx / fps if fps > 0 else 0,
 "needs_annotation": True
 })
 
 saved_count += 1
 if saved_count % 100 == 0:
 log.info(f" Extracted {saved_count} frames...")
 
 frame_idx += 1
 
 cap.release()
 
 log.info(f" Extracted {len(frame_data)} frames")
 
 return {
 "video_path": str(video_path),
 "video_name": video_path.name,
 "properties": {
 "width": width,
 "height": height,
 "fps": fps,
 "total_frames": total_frames,
 "duration": duration,
 "extracted_frames": len(frame_data)
 },
 "frames": frame_data,
 "sample_rate": sample_rate
 }
 
 def create_annotation_template(self, video_data: Dict, activity: str, action_id: str = None) -> Dict:
 """
 Create annotation template for video.
 
 Args:
 video_data: Output from extract_keypoints_simple
 activity: Activity name (e.g., "coughing")
 action_id: NTU action ID (e.g., "C41")
 
 Returns:
 Annotation template dictionary
 """
 from dataset_creation.ntu_action_mapping import get_ntu_action_info
 
 # Get action info if action_id provided
 action_info = None
 if action_id:
 action_info = get_ntu_action_info(action_id)
 
 annotation = {
 "video_id": Path(video_data["video_path"]).stem,
 "video_path": video_data["video_path"],
 "activity": activity,
 "action_id": action_id,
 "action_info": action_info,
 "created_at": datetime.now().isoformat(),
 "video_properties": video_data["properties"],
 "frames": [],
 "annotations": {
 "keypoints_2d": [], # Will be filled during annotation
 "keypoints_3d": [], # Optional
 "bbox": [], # Bounding boxes
 "activity_labels": [], # Per-frame activity labels
 "quality_scores": [] # Annotation quality scores
 },
 "metadata": {
 "annotator": None,
 "annotation_date": None,
 "annotation_status": "pending", # pending, in_progress, completed
 "notes": ""
 }
 }
 
 # Add frame entries
 for frame_info in video_data["frames"]:
 annotation["frames"].append({
 "frame_number": frame_info["frame_number"],
 "frame_filename": frame_info["frame_filename"],
 "frame_path": frame_info["frame_path"],
 "timestamp": frame_info["timestamp"],
 "keypoints_2d": None, # To be annotated
 "keypoints_3d": None, # Optional
 "bbox": None,
 "activity_label": activity,
 "confidence": None
 })
 
 return annotation
 
 def save_annotation_template(self, annotation: Dict) -> Path:
 """Save annotation template to file."""
 annotation_file = self.annotations_dir / f"{annotation['video_id']}_annotation.json"
 
 with open(annotation_file, 'w') as f:
 json.dump(annotation, f, indent=2)
 
 log.info(f" Saved annotation template: {annotation_file}")
 return annotation_file
 
 def process_video_for_training(self,
 video_path: str,
 activity: str,
 action_id: Optional[str] = None,
 sample_rate: int = 1) -> Dict:
 """
 Complete workflow: Extract → Create annotation template → Save.
 
 Args:
 video_path: Path to video file
 activity: Activity name
 action_id: NTU action ID (optional)
 sample_rate: Frame sampling rate
 
 Returns:
 Processing results
 """
 log.info(f"\n{'='*60}")
 log.info(f"Processing video for training: {Path(video_path).name}")
 log.info(f"{'='*60}")
 
 # Step 1: Extract frames
 log.info("\n Step 1: Extracting frames...")
 video_data = self.extract_keypoints_simple(video_path, sample_rate=sample_rate)
 
 # Step 2: Create annotation template
 log.info("\n Step 2: Creating annotation template...")
 annotation = self.create_annotation_template(video_data, activity, action_id)
 
 # Step 3: Save annotation template
 log.info("\n Step 3: Saving annotation template...")
 annotation_file = self.save_annotation_template(annotation)
 
 # Step 4: Create metadata summary
 log.info("\n Step 4: Creating metadata summary...")
 metadata = {
 "video_path": video_path,
 "activity": activity,
 "action_id": action_id,
 "total_frames": video_data["properties"]["total_frames"],
 "extracted_frames": len(video_data["frames"]),
 "sample_rate": sample_rate,
 "annotation_file": str(annotation_file.relative_to(self.output_dir)),
 "created_at": datetime.now().isoformat()
 }
 
 metadata_file = self.metadata_dir / f"{Path(video_path).stem}_metadata.json"
 with open(metadata_file, 'w') as f:
 json.dump(metadata, f, indent=2)
 
 log.info(f" Saved metadata: {metadata_file}")
 
 result = {
 "video_data": video_data,
 "annotation": annotation,
 "annotation_file": str(annotation_file),
 "metadata_file": str(metadata_file),
 "output_dir": str(self.output_dir)
 }
 
 log.info(f"\n Processing complete!")
 log.info(f" Frames extracted: {len(video_data['frames'])}")
 log.info(f" Annotation template: {annotation_file.name}")
 log.info(f" Output directory: {self.output_dir}")
 
 return result


def main():
 """Main function for command-line usage."""
 import argparse
 
 parser = argparse.ArgumentParser(description="Process video for annotation and training")
 parser.add_argument("video", type=str, help="Path to video file")
 parser.add_argument("--activity", type=str, required=True, help="Activity name (e.g., 'coughing')")
 parser.add_argument("--action_id", type=str, help="NTU action ID (e.g., 'C41')")
 parser.add_argument("--output_dir", type=str, default="./training_data", help="Output directory")
 parser.add_argument("--sample_rate", type=int, default=1, help="Process every Nth frame")
 
 args = parser.parse_args()
 
 # Initialize processor
 processor = TrainingDataProcessor(output_dir=args.output_dir)
 
 # Process video
 result = processor.process_video_for_training(
 video_path=args.video,
 activity=args.activity,
 action_id=args.action_id,
 sample_rate=args.sample_rate
 )
 
 print(f"\n{'='*60}")
 print("Processing Summary")
 print(f"{'='*60}")
 print(f"Video: {Path(args.video).name}")
 print(f"Activity: {args.activity}")
 print(f"Action ID: {args.action_id or 'Not specified'}")
 print(f"Frames extracted: {len(result['video_data']['frames'])}")
 print(f"Annotation file: {result['annotation_file']}")
 print(f"Output directory: {result['output_dir']}")
 print(f"\nNext steps:")
 print(f"1. Review frames in: {processor.frames_dir}")
 print(f"2. Annotate using: {result['annotation_file']}")
 print(f"3. Use annotated data for training")


if __name__ == "__main__":
 main()
