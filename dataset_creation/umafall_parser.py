"""
UMAFall Dataset Parser
Parses the UMAFall fall detection dataset.

Dataset structure:
- Subject folders (Subject_01, Subject_02, etc.)
- Activity trials with IMU data
- Labels: ADL (Activities of Daily Living) vs Falls

Reference: https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset_zip/4214283
"""

import os
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from dataset_creation.clinical_dataset_manager import ClinicalDatasetManager, DataFormatConverter

log = logging.getLogger("umafall_parser")


class UMAFallParser(ClinicalDatasetManager):
 """Parser for UMAFall dataset."""

 # UMAFall activity mapping
 ACTIVITY_MAP = {
 # ADL Activities
 'walking': 'walking',
 'standing': 'standing',
 'sitting': 'sitting',
 'lying': 'lying_in_bed',
 'picking_up': 'bending_forward',
 'sitting_down': 'sitting_down',
 'standing_up': 'standing_up',
 'going_upstairs': 'climbing_stairs',
 'going_downstairs': 'descending_stairs',
 'jumping': 'jumping',

 # Fall Activities
 'fall_forward': 'fall',
 'fall_backward': 'fall',
 'fall_lateral': 'fall',
 'fall_sitting': 'fall',
 'fall_lying': 'fall'
 }

 def __init__(self, cache_dir: str = "clinical_datasets"):
 super().__init__("UMAFall", cache_dir)
 self.dataset_path = None
 self.subjects = []

 # UMAFall-specific metadata
 self.metadata.update({
 'num_sensors': 2, # Waist + chest accelerometers
 'sample_rate': 20, # 20 Hz
 'sensor_locations': ['waist', 'chest'],
 'activities': list(self.ACTIVITY_MAP.keys())
 })

 def load_dataset(self, dataset_path: str) -> bool:
 """
 Load UMAFall dataset from local path.

 Expected structure:
 dataset_path/
 Subject_01/
 Activity_01/
 data.csv
 info.txt
 Activity_02/
 ...
 Subject_02/
 ...

 Args:
 dataset_path: Path to UMAFall dataset directory

 Returns:
 True if loaded successfully
 """
 self.dataset_path = Path(dataset_path)

 if not self.dataset_path.exists():
 log.error(f"Dataset path does not exist: {dataset_path}")
 return False

 # Find all subject directories
 self.subjects = sorted([d for d in self.dataset_path.iterdir() if d.is_dir() and d.name.startswith('Subject')])

 if not self.subjects:
 log.error(f"No subject directories found in {dataset_path}")
 return False

 self.metadata['num_subjects'] = len(self.subjects)
 log.info(f"Loaded UMAFall dataset: {len(self.subjects)} subjects")

 return True

 def get_subject_list(self) -> List[str]:
 """Get list of available subject IDs."""
 return [s.name for s in self.subjects]

 def parse_subject(self, subject_id: str) -> Dict[str, Any]:
 """
 Parse data for a single subject.

 Args:
 subject_id: Subject identifier (e.g., "Subject_01")

 Returns:
 Dictionary with:
 - subject_id: Subject ID
 - trials: List of activity trials
 - metadata: Subject metadata
 """
 subject_dir = self.dataset_path / subject_id

 if not subject_dir.exists():
 log.error(f"Subject directory not found: {subject_dir}")
 return {}

 trials = []

 # Iterate through activity directories
 activity_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir()])

 for activity_dir in activity_dirs:
 trial = self._parse_trial(activity_dir, subject_id)
 if trial:
 trials.append(trial)

 log.info(f"Parsed {subject_id}: {len(trials)} trials")

 return {
 'subject_id': subject_id,
 'trials': trials,
 'metadata': {
 'num_trials': len(trials)
 }
 }

 def _parse_trial(self, trial_dir: Path, subject_id: str) -> Optional[Dict]:
 """
 Parse a single activity trial.

 Args:
 trial_dir: Path to trial directory
 subject_id: Subject ID

 Returns:
 Dictionary with trial data or None if parsing failed
 """
 # Look for data files (CSV or TXT)
 data_files = list(trial_dir.glob('*.csv')) + list(trial_dir.glob('*.txt'))

 if not data_files:
 log.warning(f"No data files found in {trial_dir}")
 return None

 # Parse the first data file found
 data_file = data_files[0]

 try:
 # Read IMU data
 imu_data = self._read_imu_data(data_file)

 if imu_data is None or len(imu_data) == 0:
 return None

 # Extract activity label from directory name
 activity_name = trial_dir.name.lower()
 activity_label = self._infer_activity_label(activity_name)
 is_fall = self._is_fall_activity(activity_name)

 return {
 'trial_id': trial_dir.name,
 'subject_id': subject_id,
 'activity': activity_label,
 'is_fall': is_fall,
 'imu_data': imu_data,
 'duration': len(imu_data) / self.metadata['sample_rate'],
 'num_samples': len(imu_data)
 }

 except Exception as e:
 log.error(f"Error parsing trial {trial_dir}: {e}")
 return None

 def _read_imu_data(self, data_file: Path) -> Optional[np.ndarray]:
 """
 Read IMU data from CSV/TXT file.

 Expected format: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
 (Format may vary - this is a generic parser)

 Args:
 data_file: Path to data file

 Returns:
 IMU data as numpy array (N x 6) or None if failed
 """
 try:
 # Try to read as CSV
 data = []
 with open(data_file, 'r') as f:
 reader = csv.reader(f)
 for row in reader:
 # Skip header or empty rows
 if not row or any(not val.strip() for val in row):
 continue

 # Try to parse as floats
 try:
 values = [float(val) for val in row if val.strip()]
 if len(values) >= 3: # At minimum need 3 acceleration values
 data.append(values[:6]) # Take first 6 values (acc + gyro)
 except ValueError:
 continue # Skip non-numeric rows

 if not data:
 return None

 return np.array(data)

 except Exception as e:
 log.error(f"Error reading IMU data from {data_file}: {e}")
 return None

 def _infer_activity_label(self, activity_name: str) -> str:
 """
 Infer activity label from trial directory name.

 Args:
 activity_name: Trial directory name

 Returns:
 Activity label in system format
 """
 activity_name = activity_name.lower()

 # Check against known activities
 for key, value in self.ACTIVITY_MAP.items():
 if key in activity_name:
 return value

 # Default mappings
 if 'walk' in activity_name:
 return 'walking'
 elif 'sit' in activity_name:
 return 'sitting'
 elif 'stand' in activity_name:
 return 'standing'
 elif 'lie' in activity_name or 'lying' in activity_name:
 return 'lying_in_bed'
 elif 'fall' in activity_name:
 return 'fall'
 else:
 return 'unknown'

 def _is_fall_activity(self, activity_name: str) -> bool:
 """Check if activity is a fall."""
 return 'fall' in activity_name.lower()

 def convert_to_system_format(self, subject_data: Dict) -> List[Dict]:
 """
 Convert UMAFall data to system format.

 Since UMAFall has IMU data but not keypoints, we need to:
 1. Use IMU data to estimate activity/fall
 2. Generate placeholder keypoints (or use IMU-to-pose model if available)

 Args:
 subject_data: Raw subject data from parse_subject()

 Returns:
 List of samples in system format
 """
 samples = []
 converter = DataFormatConverter()

 for trial in subject_data.get('trials', []):
 imu_data = trial['imu_data']
 activity = trial['activity']
 is_fall = trial['is_fall']

 # Convert each IMU sample to a system sample
 for frame_idx, imu_sample in enumerate(imu_data):
 # Estimate keypoints from IMU (placeholder for now)
 # In production, use a trained IMU-to-pose model
 keypoints = converter.imu_to_keypoints(imu_sample)

 sample = {
 'frame': frame_idx,
 'timestamp': frame_idx / self.metadata['sample_rate'],
 'keypoints': keypoints,
 'activity': activity,
 'fall_detected': is_fall,
 'metadata': {
 'subject_id': trial['subject_id'],
 'trial_id': trial['trial_id'],
 'imu_data': imu_sample.tolist(),
 'source': 'umafall'
 }
 }

 samples.append(sample)

 return samples


def load_umafall_dataset(dataset_path: str, output_dir: str = None) -> UMAFallParser:
 """
 Convenience function to load and process UMAFall dataset.

 Args:
 dataset_path: Path to UMAFall dataset
 output_dir: Output directory for processed data

 Returns:
 UMAFallParser instance
 """
 parser = UMAFallParser()

 if not parser.load_dataset(dataset_path):
 log.error("Failed to load UMAFall dataset")
 return parser

 log.info(f"Processing {len(parser.get_subject_list())} subjects...")

 # Process all subjects
 for subject_id in parser.get_subject_list():
 subject_data = parser.parse_subject(subject_id)
 processed_data = parser.convert_to_system_format(subject_data)

 if output_dir:
 parser.save_processed_data(subject_id, processed_data, output_dir)

 # Log statistics
 stats = parser.get_activity_statistics(processed_data)
 log.info(f"{subject_id}: {stats['total_samples']} samples, {stats['fall_samples']} falls ({stats['fall_percentage']:.1f}%)")

 return parser
