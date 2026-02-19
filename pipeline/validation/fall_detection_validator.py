"""
Fall Detection Validation Framework
Validates fall detection algorithm against clinical datasets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

# Import fall detection
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from analytics.fall_detection import FallDetector

log = logging.getLogger("fall_detection_validator")


class FallDetectionValidator:
 """
 Validates fall detection algorithm against ground truth data.
 Calculates clinical metrics: sensitivity, specificity, precision, recall, F1, ROC.
 """

 def __init__(self, fall_detector: Optional[FallDetector] = None):
 """
 Initialize validator.

 Args:
 fall_detector: Fall detector instance (creates default if None)
 """
 self.fall_detector = fall_detector or FallDetector()

 # Results storage
 self.predictions = []
 self.ground_truth = []
 self.detailed_results = []

 def validate_samples(self, samples: List[Dict], window_size: int = 10) -> Dict:
 """
 Validate fall detection on a list of samples.

 Args:
 samples: List of samples in system format from dataset parser
 window_size: Number of frames for history window

 Returns:
 Validation results dictionary
 """
 self.predictions = []
 self.ground_truth = []
 self.detailed_results = []

 # Process samples with sliding window
 for i in range(len(samples)):
 sample = samples[i]

 # Get history window
 start_idx = max(0, i - window_size)
 history_samples = samples[start_idx:i]

 # Extract keypoints and history
 keypoints = sample['keypoints']
 keypoints_history = [s['keypoints'] for s in history_samples]

 # Get posture state (infer from activity)
 posture_state = self._infer_posture_from_activity(sample['activity'])

 # Frame shape (assume normalized or use default)
 frame_shape = (720, 1280) # Default HD resolution

 # Run fall detection
 result = self.fall_detector.detect_fall(
 kps=keypoints,
 kps_history=keypoints_history,
 posture_state=posture_state,
 frame_shape=frame_shape
 )

 # Store prediction and ground truth
 prediction = result['fall_detected']
 truth = sample['fall_detected']

 self.predictions.append(prediction)
 self.ground_truth.append(truth)

 # Store detailed result
 self.detailed_results.append({
 'frame': sample['frame'],
 'timestamp': sample['timestamp'],
 'prediction': prediction,
 'ground_truth': truth,
 'confidence': result['confidence'],
 'indicators': result['indicators'],
 'activity': sample['activity'],
 'metadata': sample.get('metadata', {})
 })

 # Calculate metrics
 metrics = self.calculate_metrics()

 log.info(f"Validation complete: {len(samples)} samples")
 log.info(f"Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")

 return {
 'metrics': metrics,
 'detailed_results': self.detailed_results,
 'num_samples': len(samples)
 }

 def calculate_metrics(self) -> Dict:
 """
 Calculate clinical validation metrics.

 Returns:
 Dictionary with:
 - accuracy: Overall accuracy
 - sensitivity (recall): True positive rate
 - specificity: True negative rate
 - precision (PPV): Positive predictive value
 - npv: Negative predictive value
 - f1_score: F1 score
 - confusion_matrix: TP, TN, FP, FN
 """
 if not self.predictions or not self.ground_truth:
 return {}

 predictions = np.array(self.predictions)
 ground_truth = np.array(self.ground_truth)

 # Calculate confusion matrix
 tp = np.sum((predictions == True) & (ground_truth == True))
 tn = np.sum((predictions == False) & (ground_truth == False))
 fp = np.sum((predictions == True) & (ground_truth == False))
 fn = np.sum((predictions == False) & (ground_truth == True))

 total = len(predictions)

 # Calculate metrics
 accuracy = (tp + tn) / total if total > 0 else 0.0

 # Sensitivity (Recall, True Positive Rate)
 sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

 # Specificity (True Negative Rate)
 specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

 # Precision (Positive Predictive Value)
 precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

 # Negative Predictive Value
 npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

 # F1 Score
 f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

 return {
 'accuracy': float(accuracy),
 'sensitivity': float(sensitivity),
 'specificity': float(specificity),
 'precision': float(precision),
 'npv': float(npv),
 'f1_score': float(f1_score),
 'confusion_matrix': {
 'true_positive': int(tp),
 'true_negative': int(tn),
 'false_positive': int(fp),
 'false_negative': int(fn)
 },
 'total_samples': int(total)
 }

 def calculate_per_subject_metrics(self) -> Dict[str, Dict]:
 """
 Calculate metrics per subject.

 Returns:
 Dictionary mapping subject_id to metrics
 """
 # Group results by subject
 subject_results = defaultdict(lambda: {'predictions': [], 'ground_truth': []})

 for result in self.detailed_results:
 subject_id = result['metadata'].get('subject_id', 'unknown')
 subject_results[subject_id]['predictions'].append(result['prediction'])
 subject_results[subject_id]['ground_truth'].append(result['ground_truth'])

 # Calculate metrics per subject
 per_subject_metrics = {}

 for subject_id, data in subject_results.items():
 # Temporarily set predictions and ground truth
 original_pred = self.predictions
 original_truth = self.ground_truth

 self.predictions = data['predictions']
 self.ground_truth = data['ground_truth']

 metrics = self.calculate_metrics()
 per_subject_metrics[subject_id] = metrics

 # Restore original data
 self.predictions = original_pred
 self.ground_truth = original_truth

 return per_subject_metrics

 def calculate_roc_curve(self, num_thresholds: int = 100) -> Dict:
 """
 Calculate ROC curve data.

 Args:
 num_thresholds: Number of threshold points

 Returns:
 Dictionary with:
 - thresholds: List of thresholds
 - tpr: True positive rates
 - fpr: False positive rates
 - auc: Area under curve
 """
 confidences = [r['confidence'] for r in self.detailed_results]
 ground_truth = [r['ground_truth'] for r in self.detailed_results]

 thresholds = np.linspace(0, 1, num_thresholds)
 tpr_list = []
 fpr_list = []

 for threshold in thresholds:
 predictions = [conf >= threshold for conf in confidences]

 tp = sum((p == True) & (gt == True) for p, gt in zip(predictions, ground_truth))
 fn = sum((p == False) & (gt == True) for p, gt in zip(predictions, ground_truth))
 fp = sum((p == True) & (gt == False) for p, gt in zip(predictions, ground_truth))
 tn = sum((p == False) & (gt == False) for p, gt in zip(predictions, ground_truth))

 tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
 fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

 tpr_list.append(tpr)
 fpr_list.append(fpr)

 # Calculate AUC using trapezoidal rule
 auc = np.trapz(tpr_list, fpr_list)

 return {
 'thresholds': thresholds.tolist(),
 'tpr': tpr_list,
 'fpr': fpr_list,
 'auc': float(abs(auc)) # Ensure positive AUC
 }

 def _infer_posture_from_activity(self, activity: str) -> str:
 """
 Infer posture state from activity label.

 Args:
 activity: Activity label

 Returns:
 Posture state string
 """
 activity_lower = activity.lower()

 if 'lying' in activity_lower or 'supine' in activity_lower:
 return 'supine'
 elif 'sitting' in activity_lower:
 return 'sitting'
 elif 'standing' in activity_lower or 'walking' in activity_lower:
 return 'standing'
 elif 'fall' in activity_lower:
 return 'supine' # After fall, likely lying down
 else:
 return 'unknown'

 def generate_report(self, output_path: str, dataset_name: str = "Dataset"):
 """
 Generate comprehensive validation report.

 Args:
 output_path: Path to save report (JSON)
 dataset_name: Name of dataset for report
 """
 output_path = Path(output_path)
 output_path.parent.mkdir(parents=True, exist_ok=True)

 # Overall metrics
 overall_metrics = self.calculate_metrics()

 # Per-subject metrics
 per_subject_metrics = self.calculate_per_subject_metrics()

 # ROC curve data
 roc_data = self.calculate_roc_curve()

 # Generate report
 report = {
 'dataset': dataset_name,
 'validation_timestamp': self._get_timestamp(),
 'overall_metrics': overall_metrics,
 'per_subject_metrics': per_subject_metrics,
 'roc_curve': roc_data,
 'clinical_interpretation': self._generate_clinical_interpretation(overall_metrics),
 'recommendations': self._generate_recommendations(overall_metrics)
 }

 # Save report
 with open(output_path, 'w') as f:
 json.dump(report, f, indent=2)

 log.info(f"Validation report saved to {output_path}")

 return report

 def _generate_clinical_interpretation(self, metrics: Dict) -> Dict:
 """Generate clinical interpretation of metrics."""
 sensitivity = metrics.get('sensitivity', 0.0)
 specificity = metrics.get('specificity', 0.0)

 interpretation = {
 'sensitivity_interpretation': '',
 'specificity_interpretation': '',
 'overall_assessment': ''
 }

 # Sensitivity interpretation
 if sensitivity >= 0.95:
 interpretation['sensitivity_interpretation'] = 'Excellent - Detects 95%+ of actual falls (high clinical safety)'
 elif sensitivity >= 0.90:
 interpretation['sensitivity_interpretation'] = 'Good - Detects 90%+ of actual falls'
 elif sensitivity >= 0.80:
 interpretation['sensitivity_interpretation'] = 'Acceptable - Detects 80%+ of actual falls (may miss some falls)'
 else:
 interpretation['sensitivity_interpretation'] = 'Poor - Misses >20% of falls (clinical risk)'

 # Specificity interpretation
 if specificity >= 0.95:
 interpretation['specificity_interpretation'] = 'Excellent - Very few false alarms (<5%)'
 elif specificity >= 0.90:
 interpretation['specificity_interpretation'] = 'Good - Few false alarms (<10%)'
 elif specificity >= 0.80:
 interpretation['specificity_interpretation'] = 'Acceptable - Some false alarms (10-20%)'
 else:
 interpretation['specificity_interpretation'] = 'Poor - Many false alarms (>20%, alert fatigue risk)'

 # Overall assessment
 if sensitivity >= 0.90 and specificity >= 0.90:
 interpretation['overall_assessment'] = 'Clinical grade performance - Ready for deployment'
 elif sensitivity >= 0.85 and specificity >= 0.85:
 interpretation['overall_assessment'] = 'Good performance - Minor tuning recommended'
 elif sensitivity >= 0.80:
 interpretation['overall_assessment'] = 'Acceptable performance - Further validation recommended'
 else:
 interpretation['overall_assessment'] = 'Insufficient performance - Requires improvement before clinical use'

 return interpretation

 def _generate_recommendations(self, metrics: Dict) -> List[str]:
 """Generate recommendations based on metrics."""
 recommendations = []

 sensitivity = metrics.get('sensitivity', 0.0)
 specificity = metrics.get('specificity', 0.0)
 precision = metrics.get('precision', 0.0)

 if sensitivity < 0.90:
 recommendations.append(f"Increase sensitivity (currently {sensitivity:.2%}). Consider lowering detection thresholds.")

 if specificity < 0.90:
 recommendations.append(f"Reduce false positives (specificity: {specificity:.2%}). Consider increasing confidence thresholds or adding temporal smoothing.")

 if precision < 0.80:
 recommendations.append(f"Improve precision (currently {precision:.2%}). Too many false alarms may cause alert fatigue.")

 if sensitivity >= 0.95 and specificity >= 0.95:
 recommendations.append("Excellent performance! Consider real-world clinical testing.")

 if not recommendations:
 recommendations.append("Performance meets clinical standards. Proceed with extended validation.")

 return recommendations

 def _get_timestamp(self) -> str:
 """Get current timestamp."""
 from datetime import datetime
 return datetime.now().isoformat()


def validate_fall_detection_on_dataset(dataset_samples: List[Dict],
 output_path: str,
 dataset_name: str = "Clinical Dataset") -> Dict:
 """
 Convenience function to validate fall detection on a dataset.

 Args:
 dataset_samples: List of processed samples from dataset parser
 output_path: Path to save validation report
 dataset_name: Name of dataset

 Returns:
 Validation report
 """
 validator = FallDetectionValidator()

 log.info(f"Validating fall detection on {dataset_name} ({len(dataset_samples)} samples)...")

 # Run validation
 results = validator.validate_samples(dataset_samples)

 # Generate report
 report = validator.generate_report(output_path, dataset_name)

 # Print summary
 metrics = results['metrics']
 log.info("=" * 60)
 log.info(f"VALIDATION RESULTS - {dataset_name}")
 log.info("=" * 60)
 log.info(f"Accuracy: {metrics['accuracy']:.1%}")
 log.info(f"Sensitivity: {metrics['sensitivity']:.1%} (True Positive Rate)")
 log.info(f"Specificity: {metrics['specificity']:.1%} (True Negative Rate)")
 log.info(f"Precision: {metrics['precision']:.1%} (Positive Predictive Value)")
 log.info(f"F1 Score: {metrics['f1_score']:.3f}")
 log.info(f"AUC: {report['roc_curve']['auc']:.3f}")
 log.info("=" * 60)

 return report
