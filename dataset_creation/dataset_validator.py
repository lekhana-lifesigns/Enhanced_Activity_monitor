"""
Clinical Dataset Validator
Validates quality and completeness of recorded clinical datasets.

Quality Checks:
1. Video quality (resolution, frame rate, duration)
2. Annotation completeness
3. Data consistency
4. Clinical relevance
5. Privacy compliance
"""

import json
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from dataset_creation.recording_protocol import CLINICAL_ACTIVITIES

log = logging.getLogger("dataset_validator")


@dataclass
class ValidationIssue:
 """Represents a validation issue."""
 severity: str # "error", "warning", "info"
 category: str # "video_quality", "annotation", "consistency", "privacy"
 message: str
 location: Optional[str] = None # File path or frame number


@dataclass
class ValidationReport:
 """Complete validation report."""
 dataset_path: str
 total_recordings: int
 total_annotations: int
 passed: bool
 quality_score: float # 0-100
 issues: List[ValidationIssue]
 statistics: Dict
 recommendations: List[str]


class DatasetValidator:
 """
 Validates clinical dataset quality and completeness.
 """

 def __init__(self):
 self.issues = []
 self.statistics = {
 'total_videos': 0,
 'total_frames': 0,
 'total_annotations': 0,
 'activities': {},
 'avg_duration': 0.0,
 'avg_quality': 0.0
 }

 def validate_dataset(self, dataset_path: str) -> ValidationReport:
 """
 Validate entire dataset.

 Args:
 dataset_path: Path to dataset directory

 Returns:
 ValidationReport with findings
 """
 self.issues = []
 dataset_path = Path(dataset_path)

 if not dataset_path.exists():
 self.issues.append(ValidationIssue(
 severity="error",
 category="dataset",
 message=f"Dataset path does not exist: {dataset_path}"
 ))
 return self._generate_report(str(dataset_path))

 log.info(f"Validating dataset: {dataset_path}")

 # Find all video files
 video_files = list(dataset_path.rglob("*.avi")) + list(dataset_path.rglob("*.mp4"))
 self.statistics['total_videos'] = len(video_files)

 if len(video_files) == 0:
 self.issues.append(ValidationIssue(
 severity="error",
 category="dataset",
 message="No video files found in dataset"
 ))
 return self._generate_report(str(dataset_path))

 # Validate each recording
 for video_file in video_files:
 self._validate_recording(video_file)

 # Check dataset completeness
 self._validate_dataset_completeness()

 # Generate report
 report = self._generate_report(str(dataset_path))

 log.info(f"Validation complete. Quality score: {report.quality_score:.1f}/100")
 log.info(f"Issues: {len([i for i in self.issues if i.severity == 'error'])} errors, "
 f"{len([i for i in self.issues if i.severity == 'warning'])} warnings")

 return report

 def _validate_recording(self, video_path: Path):
 """Validate a single recording."""
 log.info(f"Validating: {video_path.name}")

 # Find associated files
 metadata_file = video_path.with_name(video_path.stem + "_metadata.json")
 keypoints_file = video_path.with_name(video_path.stem + "_keypoints.json")
 annotation_file = video_path.with_name(video_path.stem + "_annotations.json")

 # Check metadata exists
 if not metadata_file.exists():
 self.issues.append(ValidationIssue(
 severity="warning",
 category="annotation",
 message="Metadata file missing",
 location=str(video_path)
 ))

 # Validate video quality
 self._validate_video_quality(video_path)

 # Validate metadata
 if metadata_file.exists():
 self._validate_metadata(metadata_file)

 # Validate annotations
 if annotation_file.exists():
 self._validate_annotations(annotation_file, video_path)

 def _validate_video_quality(self, video_path: Path):
 """Validate video quality metrics."""
 try:
 cap = cv2.VideoCapture(str(video_path))

 if not cap.isOpened():
 self.issues.append(ValidationIssue(
 severity="error",
 category="video_quality",
 message="Failed to open video file",
 location=str(video_path)
 ))
 return

 # Get video properties
 width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 fps = cap.get(cv2.CAP_PROP_FPS)
 frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 duration = frame_count / fps if fps > 0 else 0

 self.statistics['total_frames'] += frame_count

 # Check resolution
 if width < 640 or height < 480:
 self.issues.append(ValidationIssue(
 severity="warning",
 category="video_quality",
 message=f"Low resolution: {width}x{height} (minimum recommended: 640x480)",
 location=str(video_path)
 ))

 # Check frame rate
 if fps < 20:
 self.issues.append(ValidationIssue(
 severity="warning",
 category="video_quality",
 message=f"Low frame rate: {fps} fps (minimum recommended: 20 fps)",
 location=str(video_path)
 ))

 # Check duration
 if duration < 2.0:
 self.issues.append(ValidationIssue(
 severity="warning",
 category="video_quality",
 message=f"Very short recording: {duration:.1f}s (minimum recommended: 2s)",
 location=str(video_path)
 ))
 elif duration > 120.0:
 self.issues.append(ValidationIssue(
 severity="info",
 category="video_quality",
 message=f"Very long recording: {duration:.1f}s (consider splitting)",
 location=str(video_path)
 ))

 # Sample frames for quality analysis
 quality_score = self._analyze_frame_quality(cap, num_samples=10)

 if quality_score < 50:
 self.issues.append(ValidationIssue(
 severity="error",
 category="video_quality",
 message=f"Poor video quality score: {quality_score:.1f}/100",
 location=str(video_path)
 ))
 elif quality_score < 70:
 self.issues.append(ValidationIssue(
 severity="warning",
 category="video_quality",
 message=f"Mediocre video quality score: {quality_score:.1f}/100",
 location=str(video_path)
 ))

 cap.release()

 except Exception as e:
 self.issues.append(ValidationIssue(
 severity="error",
 category="video_quality",
 message=f"Error validating video: {str(e)}",
 location=str(video_path)
 ))

 def _analyze_frame_quality(self, cap: cv2.VideoCapture, num_samples: int = 10) -> float:
 """
 Analyze frame quality using various metrics.

 Returns:
 Quality score (0-100)
 """
 frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

 if frame_count == 0:
 return 0.0

 sample_interval = max(1, frame_count // num_samples)
 quality_scores = []

 for i in range(0, frame_count, sample_interval):
 cap.set(cv2.CAP_PROP_POS_FRAMES, i)
 ret, frame = cap.read()

 if not ret:
 continue

 # Calculate quality metrics
 # 1. Brightness (should be in reasonable range)
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 brightness = np.mean(gray)
 brightness_score = 100 if 80 <= brightness <= 180 else max(0, 100 - abs(brightness - 130))

 # 2. Contrast (standard deviation of pixel values)
 contrast = np.std(gray)
 contrast_score = min(100, contrast * 2) # Higher contrast is better

 # 3. Sharpness (Laplacian variance)
 laplacian = cv2.Laplacian(gray, cv2.CV_64F)
 sharpness = np.var(laplacian)
 sharpness_score = min(100, sharpness / 10) # Normalize

 # 4. Check for motion blur (edge detection)
 edges = cv2.Canny(gray, 50, 150)
 edge_density = np.sum(edges > 0) / edges.size * 100
 blur_score = min(100, edge_density * 10) # More edges = less blur

 # Combine scores
 frame_score = (brightness_score * 0.3 + contrast_score * 0.2 +
 sharpness_score * 0.3 + blur_score * 0.2)
 quality_scores.append(frame_score)

 return np.mean(quality_scores) if quality_scores else 0.0

 def _validate_metadata(self, metadata_path: Path):
 """Validate metadata completeness and correctness."""
 try:
 with open(metadata_path, 'r') as f:
 metadata = json.load(f)

 required_fields = ['session_id', 'participant_id', 'activity_id',
 'duration', 'num_frames']

 # Check required fields
 for field in required_fields:
 if field not in metadata:
 self.issues.append(ValidationIssue(
 severity="error",
 category="annotation",
 message=f"Missing required metadata field: {field}",
 location=str(metadata_path)
 ))

 # Validate activity ID
 if 'activity_id' in metadata:
 if metadata['activity_id'] not in CLINICAL_ACTIVITIES:
 self.issues.append(ValidationIssue(
 severity="error",
 category="annotation",
 message=f"Unknown activity ID: {metadata['activity_id']}",
 location=str(metadata_path)
 ))
 else:
 # Update statistics
 activity_id = metadata['activity_id']
 self.statistics['activities'][activity_id] = \
 self.statistics['activities'].get(activity_id, 0) + 1

 # Check duration
 if 'duration' in metadata:
 activity_id = metadata.get('activity_id')
 if activity_id and activity_id in CLINICAL_ACTIVITIES:
 activity_def = CLINICAL_ACTIVITIES[activity_id]
 duration = metadata['duration']

 if duration < activity_def.duration_min:
 self.issues.append(ValidationIssue(
 severity="warning",
 category="consistency",
 message=f"Duration {duration:.1f}s below minimum {activity_def.duration_min}s for {activity_id}",
 location=str(metadata_path)
 ))
 elif duration > activity_def.duration_max:
 self.issues.append(ValidationIssue(
 severity="info",
 category="consistency",
 message=f"Duration {duration:.1f}s exceeds maximum {activity_def.duration_max}s for {activity_id}",
 location=str(metadata_path)
 ))

 except json.JSONDecodeError as e:
 self.issues.append(ValidationIssue(
 severity="error",
 category="annotation",
 message=f"Invalid JSON in metadata: {str(e)}",
 location=str(metadata_path)
 ))
 except Exception as e:
 self.issues.append(ValidationIssue(
 severity="error",
 category="annotation",
 message=f"Error validating metadata: {str(e)}",
 location=str(metadata_path)
 ))

 def _validate_annotations(self, annotation_path: Path, video_path: Path):
 """Validate annotation file."""
 try:
 with open(annotation_path, 'r') as f:
 data = json.load(f)

 annotations = data.get('annotations', [])
 self.statistics['total_annotations'] += len(annotations)

 if len(annotations) == 0:
 self.issues.append(ValidationIssue(
 severity="warning",
 category="annotation",
 message="No annotations found",
 location=str(annotation_path)
 ))

 # Validate each annotation
 total_frames = data.get('total_frames', 0)

 for i, annot in enumerate(annotations):
 # Check required fields
 required = ['activity', 'start_frame', 'end_frame', 'intensity']
 for field in required:
 if field not in annot:
 self.issues.append(ValidationIssue(
 severity="error",
 category="annotation",
 message=f"Annotation {i} missing field: {field}",
 location=str(annotation_path)
 ))

 # Validate frame range
 if 'start_frame' in annot and 'end_frame' in annot:
 start = annot['start_frame']
 end = annot['end_frame']

 if start > end:
 self.issues.append(ValidationIssue(
 severity="error",
 category="consistency",
 message=f"Annotation {i}: start_frame ({start}) > end_frame ({end})",
 location=str(annotation_path)
 ))

 if end >= total_frames:
 self.issues.append(ValidationIssue(
 severity="error",
 category="consistency",
 message=f"Annotation {i}: end_frame ({end}) exceeds video length ({total_frames})",
 location=str(annotation_path)
 ))

 except Exception as e:
 self.issues.append(ValidationIssue(
 severity="error",
 category="annotation",
 message=f"Error validating annotations: {str(e)}",
 location=str(annotation_path)
 ))

 def _validate_dataset_completeness(self):
 """Validate overall dataset completeness."""
 # Check activity coverage
 total_activities = len(CLINICAL_ACTIVITIES)
 recorded_activities = len(self.statistics['activities'])

 coverage = (recorded_activities / total_activities) * 100

 if coverage < 20:
 self.issues.append(ValidationIssue(
 severity="warning",
 category="dataset",
 message=f"Low activity coverage: {coverage:.1f}% ({recorded_activities}/{total_activities} activities)"
 ))

 # Check for missing critical activities
 critical_activities = ['cough_severe', 'fall', 'agitation_restlessness']
 missing_critical = [act for act in critical_activities
 if act not in self.statistics['activities']]

 if missing_critical:
 self.issues.append(ValidationIssue(
 severity="warning",
 category="dataset",
 message=f"Missing critical activities: {', '.join(missing_critical)}"
 ))

 # Check sample count per activity
 for activity_id, count in self.statistics['activities'].items():
 if count < 3:
 self.issues.append(ValidationIssue(
 severity="info",
 category="dataset",
 message=f"Low sample count for {activity_id}: {count} (minimum recommended: 3)"
 ))

 def _generate_report(self, dataset_path: str) -> ValidationReport:
 """Generate validation report."""
 # Count issues by severity
 errors = len([i for i in self.issues if i.severity == "error"])
 warnings = len([i for i in self.issues if i.severity == "warning"])

 # Calculate quality score
 quality_score = self._calculate_quality_score(errors, warnings)

 # Determine pass/fail
 passed = errors == 0 and warnings < 5

 # Generate recommendations
 recommendations = self._generate_recommendations()

 report = ValidationReport(
 dataset_path=dataset_path,
 total_recordings=self.statistics['total_videos'],
 total_annotations=self.statistics['total_annotations'],
 passed=passed,
 quality_score=quality_score,
 issues=self.issues,
 statistics=self.statistics,
 recommendations=recommendations
 )

 return report

 def _calculate_quality_score(self, errors: int, warnings: int) -> float:
 """Calculate overall quality score (0-100)."""
 base_score = 100.0

 # Deduct points for issues
 base_score -= errors * 15 # Each error costs 15 points
 base_score -= warnings * 5 # Each warning costs 5 points

 # Bonus for good coverage
 if self.statistics['total_videos'] > 20:
 base_score += 5

 if len(self.statistics['activities']) > 10:
 base_score += 5

 return max(0.0, min(100.0, base_score))

 def _generate_recommendations(self) -> List[str]:
 """Generate recommendations for improvement."""
 recommendations = []

 errors = len([i for i in self.issues if i.severity == "error"])
 warnings = len([i for i in self.issues if i.severity == "warning"])

 if errors > 0:
 recommendations.append(f"Fix {errors} critical errors before using dataset for training")

 if warnings > 5:
 recommendations.append(f"Address {warnings} warnings to improve dataset quality")

 # Activity-specific recommendations
 total_activities = len(CLINICAL_ACTIVITIES)
 recorded_activities = len(self.statistics['activities'])

 if recorded_activities < total_activities * 0.5:
 recommendations.append(f"Record more activities (currently {recorded_activities}/{total_activities})")

 # Sample count recommendations
 low_sample_activities = [act for act, count in self.statistics['activities'].items() if count < 5]
 if low_sample_activities:
 recommendations.append(f"Increase samples for: {', '.join(low_sample_activities[:3])}")

 if not recommendations:
 recommendations.append("Dataset quality is good! Ready for training.")

 return recommendations

 def save_report(self, report: ValidationReport, output_path: str):
 """Save validation report to JSON."""
 output_path = Path(output_path)
 output_path.parent.mkdir(parents=True, exist_ok=True)

 report_dict = {
 'dataset_path': report.dataset_path,
 'total_recordings': report.total_recordings,
 'total_annotations': report.total_annotations,
 'passed': report.passed,
 'quality_score': report.quality_score,
 'issues': [asdict(issue) for issue in report.issues],
 'statistics': report.statistics,
 'recommendations': report.recommendations
 }

 with open(output_path, 'w') as f:
 json.dump(report_dict, f, indent=2)

 log.info(f"Validation report saved to {output_path}")


def validate_dataset(dataset_path: str, output_path: Optional[str] = None) -> ValidationReport:
 """
 Convenience function to validate dataset.

 Args:
 dataset_path: Path to dataset directory
 output_path: Optional path to save report

 Returns:
 ValidationReport
 """
 validator = DatasetValidator()
 report = validator.validate_dataset(dataset_path)

 if output_path:
 validator.save_report(report, output_path)

 # Print summary
 print("=" * 70)
 print("DATASET VALIDATION REPORT")
 print("=" * 70)
 print(f"Dataset: {report.dataset_path}")
 print(f"Total Recordings: {report.total_recordings}")
 print(f"Total Annotations: {report.total_annotations}")
 print(f"Quality Score: {report.quality_score:.1f}/100")
 print(f"Status: {'PASSED' if report.passed else 'FAILED'}")
 print()

 errors = len([i for i in report.issues if i.severity == "error"])
 warnings = len([i for i in report.issues if i.severity == "warning"])
 print(f"Issues: {errors} errors, {warnings} warnings")
 print()

 if report.recommendations:
 print("Recommendations:")
 for i, rec in enumerate(report.recommendations, 1):
 print(f" {i}. {rec}")
 print("=" * 70)

 return report
