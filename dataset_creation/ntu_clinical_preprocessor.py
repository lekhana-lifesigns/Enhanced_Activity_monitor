#!/usr/bin/env python3
"""
NTU RGB+D 120 Dataset Preprocessor for Clinical Activity Monitoring
Integrates the large-scale NTU dataset with your edge AI clinical system.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ntu_preprocessor")

class NTUClinicalPreprocessor:
 """
 Preprocesses NTU RGB+D 120 dataset for clinical activity monitoring.
 Converts 25-joint skeletons to 17-keypoint format and filters clinical actions.
 """

 def __init__(self, ntu_root_path: str, output_path: str):
 self.ntu_root = Path(ntu_root_path)
 self.output_path = Path(output_path)
 self.output_path.mkdir(parents=True, exist_ok=True)

 # NTU skeleton joint mapping (25 joints) to COCO format (17 keypoints)
 self.ntu_to_coco_mapping = self._create_joint_mapping()

 # Clinical action classes from NTU RGB+D 120
 self.clinical_actions = self._define_clinical_actions()

 # Statistics tracking
 self.stats = {
 'total_samples': 0,
 'clinical_samples': 0,
 'processed_sequences': 0,
 'conversion_errors': 0
 }

 def _create_joint_mapping(self) -> Dict[int, int]:
 """
 Map NTU 25-joint skeleton to COCO 17-keypoint format.
 NTU joints: https://github.com/shahroudy/NTURGB-D
 COCO keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
 """
 return {
 # COCO keypoints (0-16) <- NTU joints (1-25)
 0: 3, # nose <- head
 1: 5, # left_eye <- left_shoulder (approximation)
 2: 9, # right_eye <- right_shoulder (approximation)
 3: 4, # left_ear <- neck (approximation)
 4: 4, # right_ear <- neck (approximation)
 5: 5, # left_shoulder <- left_shoulder
 6: 9, # right_shoulder <- right_shoulder
 7: 6, # left_elbow <- left_elbow
 8: 10, # right_elbow <- right_elbow
 9: 7, # left_wrist <- left_wrist
 10: 11, # right_wrist <- right_wrist
 11: 13, # left_hip <- left_hip
 12: 17, # right_hip <- right_hip
 13: 14, # left_knee <- left_knee
 14: 18, # right_knee <- right_knee
 15: 15, # left_ankle <- left_ankle
 16: 19, # right_ankle <- right_ankle
 }

 def _define_clinical_actions(self) -> Dict[str, List[str]]:
 """
 Define clinical action categories from NTU RGB+D 120.
 Maps action names to clinical relevance.
 """
 return {
 'falls': ['falling down'],
 'parkinsons': ['staggering', 'trembling'],
 'cardiac': ['chest pain', 'chest pain'],
 'respiratory': ['sneezing', 'coughing', 'blowing nose'],
 'neurological': ['headache', 'dizzy'],
 'gastrointestinal': ['vomiting condition'],
 'mobility': ['standing up', 'sitting down', 'walking', 'running'],
 'agitation': ['kicking', 'punching', 'throwing'],
 'rehabilitation': ['arm circles', 'arm swings', 'jumping', 'hopping']
 }

 def convert_skeleton_format(self, ntu_skeleton: np.ndarray) -> np.ndarray:
 """
 Convert NTU 25-joint skeleton to COCO 17-keypoint format.

 Args:
 ntu_skeleton: [T, 25, 3] - T frames, 25 joints, (x,y,z)

 Returns:
 coco_keypoints: [T, 17, 3] - T frames, 17 keypoints, (x,y,confidence)
 """
 try:
 T, num_joints, coords = ntu_skeleton.shape
 assert num_joints == 25, f"Expected 25 joints, got {num_joints}"

 # Initialize COCO format keypoints
 coco_keypoints = np.zeros((T, 17, 3))

 for coco_idx, ntu_idx in self.ntu_to_coco_mapping.items():
 # Copy x,y coordinates
 coco_keypoints[:, coco_idx, :2] = ntu_skeleton[:, ntu_idx, :2]
 # Set confidence to 1.0 (assuming NTU skeletons are reliable)
 coco_keypoints[:, coco_idx, 2] = 1.0

 return coco_keypoints

 except Exception as e:
 log.error(f"Skeleton conversion error: {e}")
 self.stats['conversion_errors'] += 1
 return None

 def is_clinical_action(self, action_name: str) -> Tuple[bool, str]:
 """
 Check if action is clinically relevant.

 Args:
 action_name: NTU action class name

 Returns:
 (is_clinical, clinical_category)
 """
 for category, actions in self.clinical_actions.items():
 if any(action.lower() in action_name.lower() for action in actions):
 return True, category
 return False, "non_clinical"

 def process_sample(self, sample_path: Path) -> Optional[Dict]:
 """
 Process a single NTU sample.

 Args:
 sample_path: Path to NTU sample file

 Returns:
 Processed sample dict or None if invalid
 """
 try:
 # Load NTU sample (assuming .npz format with skeleton data)
 sample_data = np.load(sample_path)
 ntu_skeleton = sample_data['skeleton'] # [T, 25, 3]
 action_name = sample_data['action'].item()

 self.stats['total_samples'] += 1

 # Check if clinically relevant
 is_clinical, category = self.is_clinical_action(action_name)
 if not is_clinical:
 return None

 self.stats['clinical_samples'] += 1

 # Convert skeleton format
 coco_keypoints = self.convert_skeleton_format(ntu_skeleton)
 if coco_keypoints is None:
 return None

 # Create processed sample
 processed_sample = {
 'original_action': action_name,
 'clinical_category': category,
 'keypoints': coco_keypoints, # [T, 17, 3]
 'sequence_length': len(coco_keypoints),
 'sample_id': sample_path.stem,
 'metadata': {
 'source': 'ntu_rgb_d_120',
 'conversion': 'ntu_25_to_coco_17',
 'clinical_relevance': True
 }
 }

 self.stats['processed_sequences'] += 1
 return processed_sample

 except Exception as e:
 log.error(f"Error processing {sample_path}: {e}")
 return None

 def process_dataset(self, limit_samples: Optional[int] = None) -> Dict:
 """
 Process entire NTU dataset and save clinical samples.

 Args:
 limit_samples: Limit number of samples to process (for testing)

 Returns:
 Processing statistics
 """
 log.info(" Starting NTU RGB+D 120 Clinical Preprocessing")
 log.info(f"Input: {self.ntu_root}")
 log.info(f"Output: {self.output_path}")

 # Find all NTU sample files
 sample_files = list(self.ntu_root.rglob("*.npz")) # Assuming .npz format
 log.info(f"Found {len(sample_files)} NTU samples")

 if limit_samples:
 sample_files = sample_files[:limit_samples]
 log.info(f"Limited to {limit_samples} samples for testing")

 # Process samples
 clinical_samples = []
 for i, sample_file in enumerate(sample_files):
 if i % 1000 == 0:
 log.info(f"Processed {i}/{len(sample_files)} samples")

 processed = self.process_sample(sample_file)
 if processed:
 clinical_samples.append(processed)

 # Save processed data
 self._save_processed_data(clinical_samples)

 # Print statistics
 self._print_statistics()

 return self.stats

 def _save_processed_data(self, clinical_samples: List[Dict]):
 """Save processed clinical samples to disk."""
 log.info(f" Saving {len(clinical_samples)} clinical samples")

 # Save as individual files for memory efficiency
 samples_dir = self.output_path / "clinical_samples"
 samples_dir.mkdir(exist_ok=True)

 for sample in clinical_samples:
 sample_id = sample['sample_id']
 np.savez(
 samples_dir / f"{sample_id}.npz",
 keypoints=sample['keypoints'],
 clinical_category=sample['clinical_category'],
 metadata=str(sample['metadata'])
 )

 # Save summary
 summary = {
 'total_clinical_samples': len(clinical_samples),
 'categories': {},
 'statistics': self.stats
 }

 for sample in clinical_samples:
 cat = sample['clinical_category']
 summary['categories'][cat] = summary['categories'].get(cat, 0) + 1

 with open(self.output_path / "clinical_dataset_summary.yaml", 'w') as f:
 yaml.dump(summary, f, default_flow_style=False)

 def _print_statistics(self):
 """Print processing statistics."""
 log.info(" NTU RGB+D 120 Clinical Preprocessing Complete")
 log.info("=" * 60)
 log.info(f"Total NTU samples processed: {self.stats['total_samples']}")
 log.info(f"Clinical samples extracted: {self.stats['clinical_samples']}")
 log.info(f"Successfully processed sequences: {self.stats['processed_sequences']}")
 log.info(f"Conversion errors: {self.stats['conversion_errors']}")

 clinical_percentage = (self.stats['clinical_samples'] / self.stats['total_samples'] * 100) if self.stats['total_samples'] > 0 else 0
 log.info(".1f")

def create_clinical_training_dataset(ntu_path: str, output_path: str, limit_samples: Optional[int] = None):
 """
 Main function to create clinical training dataset from NTU RGB+D 120.

 Usage:
 create_clinical_training_dataset(
 ntu_path="/path/to/ntu_rgb_d_120",
 output_path="./clinical_dataset",
 limit_samples=1000 # For testing
 )
 """
 preprocessor = NTUClinicalPreprocessor(ntu_path, output_path)
 stats = preprocessor.process_dataset(limit_samples=limit_samples)

 log.info(" Clinical dataset creation complete!")
 log.info(f" Output saved to: {output_path}")

 return stats

if __name__ == "__main__":
 # Example usage
 import argparse

 parser = argparse.ArgumentParser(description="NTU RGB+D 120 Clinical Preprocessor")
 parser.add_argument("--ntu_path", required=True, help="Path to NTU RGB+D 120 dataset")
 parser.add_argument("--output_path", required=True, help="Output path for clinical dataset")
 parser.add_argument("--limit_samples", type=int, help="Limit samples for testing")

 args = parser.parse_args()

 create_clinical_training_dataset(
 ntu_path=args.ntu_path,
 output_path=args.output_path,
 limit_samples=args.limit_samples
 )