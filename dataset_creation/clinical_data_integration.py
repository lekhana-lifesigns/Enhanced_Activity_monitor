# dataset_creation/clinical_data_integration.py
"""
Clinical Data Integration for Enhanced Activity Monitor
Integrates real clinical datasets with the existing activity monitoring pipeline.
Provides practical examples for clinical data collection and validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import duckdb

from dataset_creation.clinical_data_loader import ClinicalDataLoader

log = logging.getLogger("clinical_integration")

class ClinicalDataIntegration:
 """
 Integrates clinical datasets with the activity monitoring system.
 Focuses on real clinical data sources and validation.
 """

 def __init__(self):
 self.data_loader = ClinicalDataLoader()
 self.clinical_sources = self._get_clinical_data_sources()

 def _get_clinical_data_sources(self) -> Dict[str, Dict]:
 """Get information about available clinical data sources."""
 return {
 'mimic_iii': {
 'description': 'MIMIC-III Clinical Database',
 'access': 'Requires PhysioNet credentialed access',
 'url': 'https://physionet.org/content/mimiciii/1.4/',
 'data_types': ['vital_signs', 'medications', 'lab_results'],
 'clinical_relevance': 'High - Real ICU patient data'
 },
 'mimic_iv': {
 'description': 'MIMIC-IV Clinical Database',
 'access': 'Requires PhysioNet credentialed access',
 'url': 'https://physionet.org/content/mimiciv/2.2/',
 'data_types': ['vital_signs', 'medications', 'procedures'],
 'clinical_relevance': 'Very High - Latest ICU data'
 },
 'eicu_crd': {
 'description': 'eICU Collaborative Research Database',
 'access': 'Requires PhysioNet credentialed access',
 'url': 'https://physionet.org/content/eicu-crd/2.0/',
 'data_types': ['vital_signs', 'apache_scores', 'treatments'],
 'clinical_relevance': 'High - Multi-center ICU data'
 },
 'hirid': {
 'description': 'HiRID - High Resolution ICU Database',
 'access': 'Research collaboration required',
 'url': 'https://hirid.intensivecare.ai/',
 'data_types': ['high_freq_vitals', 'interventions'],
 'clinical_relevance': 'Very High - High-frequency monitoring'
 },
 'public_datasets': {
 'description': 'Public Activity Recognition Datasets',
 'access': 'Freely available',
 'sources': [
 'Kinetics-700 (action recognition)',
 'UCF101 (human actions)',
 'HMDB51 (human motions)',
 'NTU RGB+D (3D actions)'
 ],
 'clinical_relevance': 'Medium - Good for transfer learning'
 }
 }

 def demonstrate_clinical_data_loading(self):
 """Demonstrate how to load clinical data using your example pattern."""
 log.info(" Clinical Data Loading Demonstration")
 log.info("=" * 50)

 # Example 1: Using DuckDB with local clinical data
 log.info("\n1. DuckDB Local Clinical Data Example:")
 log.info("""
# If you have local clinical data in Parquet format:
import duckdb
conn = duckdb.connect()

# Load local clinical dataset
clinical_data = conn.execute('''
 SELECT *
 FROM 'local_clinical_data/*.parquet'
 WHERE activity_type IN ('walking', 'falling', 'sitting')
''').df()

print(f"Loaded {len(clinical_data)} clinical samples")
 """)

 # Example 2: Using the ClinicalDataLoader
 log.info("\n2. ClinicalDataLoader Usage:")
 log.info("""
from dataset_creation.clinical_data_loader import ClinicalDataLoader

loader = ClinicalDataLoader()

# Load activity data for transfer learning
activity_data = loader.load_activity_data(
 activities=['walking', 'running', 'falling'],
 limit=1000
)

# Convert to training format
training_data = loader.convert_to_training_format(activity_data)

# Train your clinical model
# (Use the training script: python training/train_clinical_model.py)
 """)

 # Example 3: Integrating with existing pipeline
 log.info("\n3. Integration with Activity Monitor Pipeline:")
 log.info("""
# In your inference_pipeline.py, you can add clinical validation:

def _validate_with_clinical_data(self, prediction, keypoints):
 '''Validate predictions against clinical data patterns.'''

 # Load clinical reference data
 loader = ClinicalDataLoader()
 clinical_refs = loader.load_activity_data(['walking'], limit=100)

 # Compare against clinical patterns
 clinical_score = self._compute_clinical_similarity(
 keypoints, clinical_refs
 )

 return clinical_score > 0.8 # Clinical confidence threshold
 """)

 def show_clinical_data_sources(self):
 """Display available clinical data sources."""
 log.info("\n Available Clinical Data Sources")
 log.info("=" * 50)

 for source, info in self.clinical_sources.items():
 log.info(f"\n {source.upper()}")
 log.info(f" Description: {info['description']}")
 log.info(f" Access: {info['access']}")
 log.info(f" URL: {info.get('url', 'N/A')}")
 log.info(f" Clinical Relevance: {info['clinical_relevance']}")

 if 'data_types' in info:
 log.info(f" Data Types: {', '.join(info['data_types'])}")
 if 'sources' in info:
 log.info(f" Available Datasets: {', '.join(info['sources'])}")

 def create_clinical_validation_dataset(self, output_path: str = "clinical_validation_data"):
 """Create a validation dataset from available sources."""
 log.info("Creating clinical validation dataset...")

 output_dir = Path(output_path)
 output_dir.mkdir(exist_ok=True)

 # Try to load available public datasets for validation
 validation_data = {
 'activities': [],
 'keypoints': [],
 'metadata': []
 }

 try:
 # Load some activity data for validation
 activity_data = self.data_loader.load_activity_data(
 activities=['walking', 'running'],
 limit=100
 )

 if not activity_data.empty:
 training_format = self.data_loader.convert_to_training_format(activity_data)

 validation_data['activities'].extend(training_format['activities'])
 validation_data['keypoints'].extend(training_format['keypoints'])
 validation_data['metadata'].extend(training_format['metadata'])

 log.info(f"Created validation dataset with {len(validation_data['activities'])} samples")

 # Save validation data
 np.savez(
 output_dir / "clinical_validation.npz",
 activities=np.array(validation_data['activities']),
 keypoints=np.array(validation_data['keypoints']),
 metadata=validation_data['metadata']
 )

 log.info(f"Saved validation data to {output_dir / 'clinical_validation.npz'}")

 except Exception as e:
 log.warning(f"Could not create validation dataset: {e}")
 log.info("Using synthetic validation data for demonstration...")

 # Create synthetic validation data
 self._create_synthetic_validation_data(output_dir)

 def _create_synthetic_validation_data(self, output_dir: Path):
 """Create synthetic validation data for demonstration."""
 log.info("Creating synthetic clinical validation data...")

 # Generate synthetic clinical activity data
 activities = ['walking', 'sitting', 'standing', 'falling', 'lying_down']
 n_samples = 50

 synthetic_data = {
 'activities': np.random.choice(activities, n_samples),
 'keypoints': np.random.rand(n_samples, 17, 3), # 17 keypoints, x,y,conf
 'metadata': [{'synthetic': True, 'id': i} for i in range(n_samples)]
 }

 # Save synthetic data
 np.savez(
 output_dir / "synthetic_validation.npz",
 **synthetic_data
 )

 log.info(f"Created synthetic validation dataset with {n_samples} samples")
 log.info(" Note: This is synthetic data. Use real clinical datasets for production.")

 def integrate_with_pipeline(self):
 """Show how to integrate clinical data with the existing pipeline."""
 log.info("\n Pipeline Integration Guide")
 log.info("=" * 50)

 integration_code = '''
# Add to inference_pipeline.py

def _load_clinical_references(self):
 """Load clinical reference data for validation."""
 try:
 from dataset_creation.clinical_data_loader import ClinicalDataLoader
 loader = ClinicalDataLoader()

 # Load clinical reference patterns
 self.clinical_references = loader.load_activity_data(
 activities=['walking', 'sitting', 'standing'],
 limit=500
 )

 log.info(f"Loaded {len(self.clinical_references)} clinical reference samples")
 except Exception as e:
 log.warning(f"Could not load clinical references: {e}")
 self.clinical_references = None

def _validate_clinical_consistency(self, keypoints, activity_prediction):
 """Validate prediction against clinical patterns."""
 if self.clinical_references is None:
 return 0.5 # Neutral confidence

 # Compare keypoints against clinical reference distributions
 clinical_similarity = self._compute_pose_similarity(
 keypoints, self.clinical_references
 )

 return clinical_similarity
 '''

 log.info("Add this code to your inference_pipeline.py:")
 log.info(integration_code)


def main():
 """Main demonstration function."""
 logging.basicConfig(level=logging.INFO)

 integrator = ClinicalDataIntegration()

 # Show available data sources
 integrator.show_clinical_data_sources()

 # Demonstrate data loading
 integrator.demonstrate_clinical_data_loading()

 # Create validation dataset
 integrator.create_clinical_validation_dataset()

 # Show pipeline integration
 integrator.integrate_with_pipeline()

 log.info("\n Clinical Data Integration Setup Complete")
 log.info("\nNext Steps:")
 log.info("1. Obtain access to clinical datasets (MIMIC-III/IV, eICU)")
 log.info("2. Convert clinical data to activity monitoring format")
 log.info("3. Train models with: python training/train_clinical_model.py")
 log.info("4. Integrate validation with your pipeline")


if __name__ == "__main__":
 main()