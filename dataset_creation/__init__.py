"""
Clinical Dataset Integration Module

This module provides tools for integrating external clinical datasets
for validation and benchmarking of the Enhanced Activity Monitor system.

Available Parsers:
- UMAFallParser: Parser for UMAFall fall detection dataset

Usage:
 from dataset_creation.umafall_parser import load_umafall_dataset

 parser = load_umafall_dataset('/path/to/umafall', output_dir='validation_results')
"""

from dataset_creation.clinical_dataset_manager import ClinicalDatasetManager, DataFormatConverter

__all__ = ['ClinicalDatasetManager', 'DataFormatConverter']
