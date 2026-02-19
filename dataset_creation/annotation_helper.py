#!/usr/bin/env python3
"""
Annotation Helper Tool
Helps annotate videos by providing utilities for keypoint annotation.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("annotation_helper")


class AnnotationHelper:
 """
 Helper class for video annotation.
 """
 
 def __init__(self, annotation_file: str):
 """
 Initialize annotation helper.
 
 Args:
 annotation_file: Path to annotation JSON file
 """
 self.annotation_file = Path(annotation_file)
 self.annotation = self.load_annotation()
 
 def load_annotation(self) -> Dict:
 """Load annotation from file."""
 with open(self.annotation_file, 'r') as f:
 return json.load(f)
 
 def save_annotation(self):
 """Save annotation to file."""
 with open(self.annotation_file, 'w') as f:
 json.dump(self.annotation, f, indent=2)
 log.info(f" Saved annotation: {self.annotation_file}")
 
 def get_pending_frames(self) -> List[Dict]:
 """Get frames that need annotation."""
 return [
 frame for frame in self.annotation["frames"]
 if frame.get("keypoints_2d") is None
 ]
 
 def get_annotated_frames(self) -> List[Dict]:
 """Get frames that are already annotated."""
 return [
 frame for frame in self.annotation["frames"]
 if frame.get("keypoints_2d") is not None
 ]
 
 def update_frame_annotation(self,
 frame_number: int,
 keypoints_2d: List,
 keypoints_3d: Optional[List] = None,
 bbox: Optional[List] = None,
 confidence: Optional[float] = None):
 """
 Update annotation for a specific frame.
 
 Args:
 frame_number: Frame number
 keypoints_2d: 2D keypoints [17, 3] format
 keypoints_3d: Optional 3D keypoints
 bbox: Optional bounding box [x1, y1, x2, y2]
 confidence: Optional confidence score
 """
 for frame in self.annotation["frames"]:
 if frame["frame_number"] == frame_number:
 frame["keypoints_2d"] = keypoints_2d
 if keypoints_3d:
 frame["keypoints_3d"] = keypoints_3d
 if bbox:
 frame["bbox"] = bbox
 if confidence:
 frame["confidence"] = confidence
 log.info(f" Updated annotation for frame {frame_number}")
 return
 
 log.warning(f" Frame {frame_number} not found")
 
 def mark_annotation_complete(self, annotator: str, notes: str = ""):
 """Mark annotation as complete."""
 self.annotation["metadata"]["annotator"] = annotator
 self.annotation["metadata"]["annotation_date"] = __import__("datetime").datetime.now().isoformat()
 self.annotation["metadata"]["annotation_status"] = "completed"
 self.annotation["metadata"]["notes"] = notes
 log.info(" Annotation marked as complete")
 
 def get_statistics(self) -> Dict:
 """Get annotation statistics."""
 total_frames = len(self.annotation["frames"])
 annotated_frames = len(self.get_annotated_frames())
 pending_frames = len(self.get_pending_frames())
 
 return {
 "total_frames": total_frames,
 "annotated_frames": annotated_frames,
 "pending_frames": pending_frames,
 "completion_percentage": (annotated_frames / total_frames * 100) if total_frames > 0 else 0,
 "status": self.annotation["metadata"]["annotation_status"]
 }
 
 def print_statistics(self):
 """Print annotation statistics."""
 stats = self.get_statistics()
 print("\n" + "="*60)
 print("Annotation Statistics")
 print("="*60)
 print(f"Video: {self.annotation['video_id']}")
 print(f"Activity: {self.annotation['activity']}")
 print(f"Total frames: {stats['total_frames']}")
 print(f"Annotated: {stats['annotated_frames']}")
 print(f"Pending: {stats['pending_frames']}")
 print(f"Completion: {stats['completion_percentage']:.1f}%")
 print(f"Status: {stats['status']}")
 print("="*60)


def create_training_dataset(annotation_file: str, output_file: str):
 """
 Convert annotation to training dataset format.
 
 Args:
 annotation_file: Path to annotation JSON
 output_file: Path to save training dataset (NPZ format)
 """
 import numpy as np
 
 helper = AnnotationHelper(annotation_file)
 annotation = helper.annotation
 
 # Extract annotated frames
 annotated_frames = helper.get_annotated_frames()
 
 if len(annotated_frames) == 0:
 log.error("No annotated frames found!")
 return
 
 # Prepare data arrays
 keypoints_2d_list = []
 keypoints_3d_list = []
 activities = []
 frame_numbers = []
 
 for frame in annotated_frames:
 if frame.get("keypoints_2d") is not None:
 keypoints_2d_list.append(frame["keypoints_2d"])
 if frame.get("keypoints_3d"):
 keypoints_3d_list.append(frame["keypoints_3d"])
 activities.append(frame.get("activity_label", annotation["activity"]))
 frame_numbers.append(frame["frame_number"])
 
 # Convert to numpy arrays
 keypoints_2d_array = np.array(keypoints_2d_list) # [N, 17, 3]
 
 # Create dataset
 dataset = {
 "keypoints_2d": keypoints_2d_array,
 "activities": np.array(activities),
 "frame_numbers": np.array(frame_numbers),
 "video_id": annotation["video_id"],
 "activity": annotation["activity"],
 "action_id": annotation.get("action_id"),
 "metadata": annotation["metadata"]
 }
 
 if len(keypoints_3d_list) > 0:
 dataset["keypoints_3d"] = np.array(keypoints_3d_list)
 
 # Save dataset
 np.savez(output_file, **dataset)
 log.info(f" Saved training dataset: {output_file}")
 log.info(f" Samples: {len(keypoints_2d_list)}")
 log.info(f" Shape: {keypoints_2d_array.shape}")
 
 return dataset


if __name__ == "__main__":
 import argparse
 
 parser = argparse.ArgumentParser(description="Annotation Helper Tool")
 parser.add_argument("annotation_file", type=str, help="Path to annotation JSON file")
 parser.add_argument("--stats", action="store_true", help="Show statistics")
 parser.add_argument("--create-dataset", type=str, help="Create training dataset (output file)")
 parser.add_argument("--mark-complete", type=str, help="Mark annotation as complete (annotator name)")
 
 args = parser.parse_args()
 
 helper = AnnotationHelper(args.annotation_file)
 
 if args.stats:
 helper.print_statistics()
 
 if args.mark_complete:
 helper.mark_annotation_complete(args.mark_complete)
 helper.save_annotation()
 
 if args.create_dataset:
 create_training_dataset(args.annotation_file, args.create_dataset)
