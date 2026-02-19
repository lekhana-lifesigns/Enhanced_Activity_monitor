"""
UMAFall Fall Detection Validation Script

Usage:
 python scripts/validate_umafall.py --dataset-path /path/to/umafall --output-dir validation_results

This script:
1. Loads UMAFall dataset
2. Converts to system format
3. Validates fall detection algorithm
4. Generates clinical validation report
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset_creation.umafall_parser import UMAFallParser
from pipeline.validation.fall_detection_validator import validate_fall_detection_on_dataset

# Setup logging
logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("validate_umafall")


def main():
 parser = argparse.ArgumentParser(description='Validate fall detection on UMAFall dataset')
 parser.add_argument('--dataset-path', type=str, required=True,
 help='Path to UMAFall dataset directory')
 parser.add_argument('--output-dir', type=str, default='validation_results',
 help='Output directory for validation results')
 parser.add_argument('--subjects', type=str, default=None,
 help='Comma-separated list of subject IDs to process (e.g., "Subject_01,Subject_02"). If not provided, processes all subjects.')
 parser.add_argument('--cache-dir', type=str, default='clinical_datasets',
 help='Directory to cache processed data')

 args = parser.parse_args()

 log.info("=" * 70)
 log.info("UMAFall Fall Detection Validation")
 log.info("=" * 70)

 # Initialize parser
 log.info("Loading UMAFall dataset...")
 umafall_parser = UMAFallParser(cache_dir=args.cache_dir)

 if not umafall_parser.load_dataset(args.dataset_path):
 log.error("Failed to load UMAFall dataset. Please check the dataset path.")
 return 1

 # Get subject list
 all_subjects = umafall_parser.get_subject_list()
 log.info(f"Found {len(all_subjects)} subjects in dataset")

 # Filter subjects if specified
 if args.subjects:
 subject_filter = args.subjects.split(',')
 subjects_to_process = [s for s in all_subjects if s in subject_filter]
 if not subjects_to_process:
 log.error(f"No matching subjects found. Available: {', '.join(all_subjects)}")
 return 1
 log.info(f"Processing {len(subjects_to_process)} specified subjects: {', '.join(subjects_to_process)}")
 else:
 subjects_to_process = all_subjects
 log.info(f"Processing all {len(subjects_to_process)} subjects")

 # Process subjects and collect all samples
 log.info("Processing subjects and converting to system format...")
 all_samples = []

 for subject_id in subjects_to_process:
 log.info(f"Processing {subject_id}...")

 # Check if already processed
 cached_data = umafall_parser.load_processed_data(subject_id)

 if cached_data:
 log.info(f" Loaded cached data: {len(cached_data)} samples")
 samples = cached_data
 else:
 # Parse and convert
 subject_data = umafall_parser.parse_subject(subject_id)

 if not subject_data or not subject_data.get('trials'):
 log.warning(f" No data found for {subject_id}, skipping")
 continue

 samples = umafall_parser.convert_to_system_format(subject_data)

 # Cache processed data
 umafall_parser.save_processed_data(subject_id, samples)
 log.info(f" Processed and cached {len(samples)} samples")

 # Get statistics
 stats = umafall_parser.get_activity_statistics(samples)
 log.info(f" Activities: {len(stats['activities'])}, Falls: {stats['fall_samples']} ({stats['fall_percentage']:.1f}%)")

 all_samples.extend(samples)

 log.info(f"\nTotal samples across all subjects: {len(all_samples)}")

 if not all_samples:
 log.error("No samples to validate!")
 return 1

 # Run validation
 log.info("\n" + "=" * 70)
 log.info("Running Fall Detection Validation...")
 log.info("=" * 70 + "\n")

 output_path = Path(args.output_dir) / "umafall_validation_report.json"
 output_path.parent.mkdir(parents=True, exist_ok=True)

 report = validate_fall_detection_on_dataset(
 dataset_samples=all_samples,
 output_path=str(output_path),
 dataset_name="UMAFall"
 )

 # Print clinical interpretation
 log.info("\n" + "=" * 70)
 log.info("CLINICAL INTERPRETATION")
 log.info("=" * 70)

 interpretation = report['clinical_interpretation']
 log.info(f"\nSensitivity: {interpretation['sensitivity_interpretation']}")
 log.info(f"Specificity: {interpretation['specificity_interpretation']}")
 log.info(f"\nOverall: {interpretation['overall_assessment']}")

 log.info("\n" + "=" * 70)
 log.info("RECOMMENDATIONS")
 log.info("=" * 70)
 for i, rec in enumerate(report['recommendations'], 1):
 log.info(f"{i}. {rec}")

 log.info("\n" + "=" * 70)
 log.info(f"Validation complete! Report saved to: {output_path}")
 log.info("=" * 70)

 return 0


if __name__ == "__main__":
 sys.exit(main())
