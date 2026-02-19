"""
Clinical Dataset Exporter
Exports annotated clinical datasets in standard formats for training.

Supported formats:
1. System format (JSON) - for direct use with Enhanced Activity Monitor
2. COCO format - for pose estimation training
3. CSV format - for activity classification training
4. Video clips - segmented by annotation
"""

import json
import cv2
import csv
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from dataset_creation.recording_protocol import CLINICAL_ACTIVITIES

log = logging.getLogger("dataset_exporter")


class DatasetExporter:
 """
 Exports clinical datasets in various standard formats.
 """

 def __init__(self):
 pass

 def export_to_system_format(self,
 dataset_path: str,
 output_path: str,
 split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
 """
 Export dataset in system format (compatible with Enhanced Activity Monitor).

 Args:
 dataset_path: Path to recorded dataset
 output_path: Output directory
 split_ratio: (train, val, test) split ratios
 """
 dataset_path = Path(dataset_path)
 output_path = Path(output_path)
 output_path.mkdir(parents=True, exist_ok=True)

 log.info(f"Exporting dataset to system format: {output_path}")

 # Find all annotation files
 annotation_files = list(dataset_path.rglob("*_annotations.json"))

 if not annotation_files:
 log.warning("No annotation files found")
 return

 # Collect all samples
 all_samples = []

 for annot_file in annotation_files:
 samples = self._load_annotated_samples(annot_file)
 all_samples.extend(samples)

 log.info(f"Collected {len(all_samples)} samples")

 # Split into train/val/test
 np.random.shuffle(all_samples)

 train_size = int(len(all_samples) * split_ratio[0])
 val_size = int(len(all_samples) * split_ratio[1])

 train_samples = all_samples[:train_size]
 val_samples = all_samples[train_size:train_size + val_size]
 test_samples = all_samples[train_size + val_size:]

 # Save splits
 self._save_split(train_samples, output_path / "train.json", "train")
 self._save_split(val_samples, output_path / "val.json", "val")
 self._save_split(test_samples, output_path / "test.json", "test")

 # Save metadata
 self._save_export_metadata(output_path, len(train_samples), len(val_samples), len(test_samples))

 log.info(f"Exported: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")

 def _load_annotated_samples(self, annotation_file: Path) -> List[Dict]:
 """Load samples from annotation file."""
 with open(annotation_file, 'r') as f:
 data = json.load(f)

 video_path = Path(data.get('video_path', annotation_file.parent / annotation_file.stem.replace('_annotations', '.avi')))
 annotations = data.get('annotations', [])
 fps = data.get('fps', 30)

 samples = []

 # Load keypoints if available
 keypoints_file = annotation_file.with_name(annotation_file.stem.replace('_annotations', '_keypoints') + '.json')
 keypoints_data = None

 if keypoints_file.exists():
 with open(keypoints_file, 'r') as f:
 keypoints_data = json.load(f)

 # Extract samples for each annotation
 for annot in annotations:
 start_frame = annot['start_frame']
 end_frame = annot['end_frame']

 # Extract activity ID from activity string
 activity_full = annot['activity']
 activity_id = self._extract_activity_id(activity_full)

 for frame_idx in range(start_frame, end_frame + 1):
 # Get keypoints if available
 if keypoints_data and frame_idx < len(keypoints_data.get('keypoints', [])):
 keypoints = keypoints_data['keypoints'][frame_idx]
 else:
 keypoints = [] # Placeholder

 sample = {
 'frame': frame_idx,
 'timestamp': frame_idx / fps,
 'keypoints': keypoints,
 'activity': activity_id,
 'intensity': annot.get('intensity', 'moderate'),
 'tags': annot.get('tags', []),
 'pain_scale': annot.get('pain_scale', 0),
 'metadata': {
 'video_path': str(video_path),
 'annotation_file': str(annotation_file),
 'source': 'custom_dataset'
 }
 }

 samples.append(sample)

 return samples

 def _extract_activity_id(self, activity_full: str) -> str:
 """Extract activity ID from full activity string."""
 # activity_full format: "Activity Name (category)"
 # Need to find matching activity ID

 for activity_id, activity_def in CLINICAL_ACTIVITIES.items():
 if activity_def.activity_name in activity_full:
 return activity_id

 # Fallback: use activity name directly
 return activity_full.split('(')[0].strip().lower().replace(' ', '_')

 def _save_split(self, samples: List[Dict], output_file: Path, split_name: str):
 """Save dataset split."""
 with open(output_file, 'w') as f:
 json.dump({
 'split': split_name,
 'num_samples': len(samples),
 'samples': samples
 }, f, indent=2)

 def _save_export_metadata(self, output_path: Path, train_count: int, val_count: int, test_count: int):
 """Save export metadata."""
 metadata = {
 'export_date': datetime.now().isoformat(),
 'format': 'system_format',
 'splits': {
 'train': train_count,
 'val': val_count,
 'test': test_count
 },
 'total_samples': train_count + val_count + test_count
 }

 with open(output_path / "export_metadata.json", 'w') as f:
 json.dump(metadata, f, indent=2)

 def export_to_csv(self,
 dataset_path: str,
 output_path: str,
 include_keypoints: bool = False):
 """
 Export dataset to CSV format for activity classification.

 Args:
 dataset_path: Path to recorded dataset
 output_path: Output CSV file path
 include_keypoints: Whether to include keypoint columns
 """
 dataset_path = Path(dataset_path)
 output_path = Path(output_path)
 output_path.parent.mkdir(parents=True, exist_ok=True)

 log.info(f"Exporting dataset to CSV: {output_path}")

 # Find all annotation files
 annotation_files = list(dataset_path.rglob("*_annotations.json"))

 # Collect all samples
 all_samples = []
 for annot_file in annotation_files:
 samples = self._load_annotated_samples(annot_file)
 all_samples.extend(samples)

 # Write CSV
 with open(output_path, 'w', newline='') as f:
 if include_keypoints and all_samples and all_samples[0]['keypoints']:
 # Include keypoint columns
 num_keypoints = len(all_samples[0]['keypoints'])
 keypoint_cols = []
 for i in range(num_keypoints):
 keypoint_cols.extend([f'kp{i}_x', f'kp{i}_y', f'kp{i}_conf'])

 fieldnames = ['frame', 'timestamp', 'activity', 'intensity', 'pain_scale'] + keypoint_cols
 else:
 fieldnames = ['frame', 'timestamp', 'activity', 'intensity', 'pain_scale', 'tags']

 writer = csv.DictWriter(f, fieldnames=fieldnames)
 writer.writeheader()

 for sample in all_samples:
 row = {
 'frame': sample['frame'],
 'timestamp': sample['timestamp'],
 'activity': sample['activity'],
 'intensity': sample['intensity'],
 'pain_scale': sample.get('pain_scale', 0)
 }

 if include_keypoints and sample['keypoints']:
 # Flatten keypoints
 for i, kp in enumerate(sample['keypoints']):
 if len(kp) >= 2:
 row[f'kp{i}_x'] = kp[0]
 row[f'kp{i}_y'] = kp[1]
 row[f'kp{i}_conf'] = kp[2] if len(kp) > 2 else 0.0
 else:
 row[f'kp{i}_x'] = 0.0
 row[f'kp{i}_y'] = 0.0
 row[f'kp{i}_conf'] = 0.0
 else:
 row['tags'] = ','.join(sample.get('tags', []))

 writer.writerow(row)

 log.info(f"Exported {len(all_samples)} samples to CSV")

 def export_video_clips(self,
 dataset_path: str,
 output_path: str,
 padding_frames: int = 10):
 """
 Export annotated segments as individual video clips.

 Args:
 dataset_path: Path to recorded dataset
 output_path: Output directory for clips
 padding_frames: Number of frames to add before/after annotation
 """
 dataset_path = Path(dataset_path)
 output_path = Path(output_path)
 output_path.mkdir(parents=True, exist_ok=True)

 log.info(f"Exporting video clips: {output_path}")

 # Find all annotation files
 annotation_files = list(dataset_path.rglob("*_annotations.json"))

 clip_count = 0

 for annot_file in annotation_files:
 with open(annot_file, 'r') as f:
 data = json.load(f)

 video_path = Path(data.get('video_path', annot_file.parent / annot_file.stem.replace('_annotations', '.avi')))

 if not video_path.exists():
 log.warning(f"Video not found: {video_path}")
 continue

 cap = cv2.VideoCapture(str(video_path))
 if not cap.isOpened():
 log.warning(f"Failed to open video: {video_path}")
 continue

 fps = cap.get(cv2.CAP_PROP_FPS)
 width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

 annotations = data.get('annotations', [])

 for i, annot in enumerate(annotations):
 start_frame = max(0, annot['start_frame'] - padding_frames)
 end_frame = annot['end_frame'] + padding_frames

 activity_id = self._extract_activity_id(annot['activity'])
 intensity = annot.get('intensity', 'moderate')

 # Create clip directory
 clip_dir = output_path / activity_id / intensity
 clip_dir.mkdir(parents=True, exist_ok=True)

 # Generate clip filename
 clip_name = f"{video_path.stem}_clip{i:03d}_{start_frame}-{end_frame}.avi"
 clip_path = clip_dir / clip_name

 # Extract and save clip
 self._extract_video_clip(cap, clip_path, start_frame, end_frame, fps, width, height)

 # Save clip metadata
 clip_metadata = {
 'clip_path': str(clip_path),
 'source_video': str(video_path),
 'start_frame': start_frame,
 'end_frame': end_frame,
 'activity': activity_id,
 'intensity': intensity,
 'tags': annot.get('tags', []),
 'pain_scale': annot.get('pain_scale', 0),
 'notes': annot.get('notes', '')
 }

 metadata_path = clip_path.with_suffix('.json')
 with open(metadata_path, 'w') as f:
 json.dump(clip_metadata, f, indent=2)

 clip_count += 1

 cap.release()

 log.info(f"Exported {clip_count} video clips")

 def _extract_video_clip(self, cap: cv2.VideoCapture, output_path: Path,
 start_frame: int, end_frame: int, fps: float,
 width: int, height: int):
 """Extract video clip from larger video."""
 fourcc = cv2.VideoWriter_fourcc(*'XVID')
 out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

 for frame_idx in range(start_frame, end_frame + 1):
 cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
 ret, frame = cap.read()

 if ret:
 out.write(frame)

 out.release()

 def create_training_manifest(self,
 dataset_path: str,
 output_path: str):
 """
 Create training manifest with dataset statistics and file paths.

 Args:
 dataset_path: Path to exported dataset
 output_path: Output manifest file
 """
 dataset_path = Path(dataset_path)
 output_path = Path(output_path)

 # Load splits
 train_data = self._load_json_safe(dataset_path / "train.json")
 val_data = self._load_json_safe(dataset_path / "val.json")
 test_data = self._load_json_safe(dataset_path / "test.json")

 # Calculate statistics
 all_samples = []
 if train_data:
 all_samples.extend(train_data.get('samples', []))
 if val_data:
 all_samples.extend(val_data.get('samples', []))
 if test_data:
 all_samples.extend(test_data.get('samples', []))

 # Activity distribution
 activity_counts = {}
 for sample in all_samples:
 activity = sample.get('activity', 'unknown')
 activity_counts[activity] = activity_counts.get(activity, 0) + 1

 # Intensity distribution
 intensity_counts = {}
 for sample in all_samples:
 intensity = sample.get('intensity', 'unknown')
 intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1

 # Create manifest
 manifest = {
 'dataset_path': str(dataset_path),
 'created': datetime.now().isoformat(),
 'total_samples': len(all_samples),
 'splits': {
 'train': train_data.get('num_samples', 0) if train_data else 0,
 'val': val_data.get('num_samples', 0) if val_data else 0,
 'test': test_data.get('num_samples', 0) if test_data else 0
 },
 'activity_distribution': activity_counts,
 'intensity_distribution': intensity_counts,
 'files': {
 'train': str(dataset_path / "train.json"),
 'val': str(dataset_path / "val.json"),
 'test': str(dataset_path / "test.json")
 }
 }

 with open(output_path, 'w') as f:
 json.dump(manifest, f, indent=2)

 log.info(f"Training manifest created: {output_path}")

 return manifest

 def _load_json_safe(self, file_path: Path) -> Optional[Dict]:
 """Safely load JSON file."""
 try:
 if file_path.exists():
 with open(file_path, 'r') as f:
 return json.load(f)
 except Exception as e:
 log.error(f"Error loading {file_path}: {e}")

 return None


def export_dataset(dataset_path: str,
 output_dir: str,
 formats: List[str] = ["system", "csv", "clips"],
 split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
 """
 Convenience function to export dataset in multiple formats.

 Args:
 dataset_path: Path to recorded dataset
 output_dir: Output directory
 formats: List of export formats ("system", "csv", "clips", "coco")
 split_ratio: Train/val/test split ratios

 Returns:
 Dictionary with export paths
 """
 output_dir = Path(output_dir)
 output_dir.mkdir(parents=True, exist_ok=True)

 exporter = DatasetExporter()
 exports = {}

 log.info(f"Exporting dataset from {dataset_path}")
 log.info(f"Formats: {', '.join(formats)}")

 if "system" in formats:
 system_dir = output_dir / "system_format"
 exporter.export_to_system_format(dataset_path, str(system_dir), split_ratio)
 exports['system'] = str(system_dir)

 # Create training manifest
 manifest_path = system_dir / "training_manifest.json"
 exporter.create_training_manifest(str(system_dir), str(manifest_path))
 exports['manifest'] = str(manifest_path)

 if "csv" in formats:
 csv_path = output_dir / "dataset.csv"
 exporter.export_to_csv(dataset_path, str(csv_path), include_keypoints=True)
 exports['csv'] = str(csv_path)

 if "clips" in formats:
 clips_dir = output_dir / "video_clips"
 exporter.export_video_clips(dataset_path, str(clips_dir))
 exports['clips'] = str(clips_dir)

 log.info("Export complete!")
 log.info(f"Outputs: {json.dumps(exports, indent=2)}")

 return exports
