# pipeline/validation/clinical_validator.py
# Clinical Validation and Metrics for ICU Activity Monitoring
# Implements medical-grade evaluation metrics and validation protocols

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import time

log = logging.getLogger("clinical_validator")

class ClinicalMetrics:
 """
 Clinical-grade metrics for ICU activity monitoring validation.
 Implements sensitivity, specificity, precision, and clinical relevance scores.
 """

 def __init__(self):
 self.reset()

 def reset(self):
 """Reset all metrics."""
 self.true_positives = defaultdict(int)
 self.false_positives = defaultdict(int)
 self.true_negatives = defaultdict(int)
 self.false_negatives = defaultdict(int)
 self.prediction_counts = defaultdict(int)
 self.ground_truth_counts = defaultdict(int)
 self.latency_measurements = []
 self.accuracy_measurements = []

 def update(self, prediction: str, ground_truth: str, latency_ms: float = 0.0):
 """
 Update metrics with a new prediction.

 Args:
 prediction: Predicted activity label
 ground_truth: Ground truth activity label
 latency_ms: Prediction latency in milliseconds
 """
 self.prediction_counts[prediction] += 1
 self.ground_truth_counts[ground_truth] += 1

 if prediction == ground_truth:
 self.true_positives[prediction] += 1
 # Count true negatives for other classes
 for label in self.ground_truth_counts.keys():
 if label != prediction:
 self.true_negatives[label] += 1
 else:
 self.false_positives[prediction] += 1
 self.false_negatives[ground_truth] += 1

 if latency_ms > 0:
 self.latency_measurements.append(latency_ms)

 def compute_metrics(self) -> Dict[str, Any]:
 """
 Compute comprehensive clinical metrics.

 Returns:
 Dictionary with precision, recall, F1-score, specificity, etc.
 """
 metrics = {}

 # Per-class metrics
 labels = set(list(self.prediction_counts.keys()) + list(self.ground_truth_counts.keys()))

 for label in labels:
 tp = self.true_positives[label]
 fp = self.false_positives[label]
 fn = self.false_negatives[label]
 tn = self.true_negatives.get(label, 0)

 # Precision = TP / (TP + FP)
 precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

 # Recall = TP / (TP + FN)
 recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

 # Specificity = TN / (TN + FP)
 specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

 # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
 f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

 metrics[label] = {
 'precision': precision,
 'recall': recall,
 'specificity': specificity,
 'f1_score': f1,
 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
 }

 # Overall metrics
 total_tp = sum(self.true_positives.values())
 total_fp = sum(self.false_positives.values())
 total_fn = sum(self.false_negatives.values())

 overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
 overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
 overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

 metrics['overall'] = {
 'precision': overall_precision,
 'recall': overall_recall,
 'f1_score': overall_f1,
 'total_predictions': sum(self.prediction_counts.values()),
 'total_ground_truth': sum(self.ground_truth_counts.values())
 }

 # Performance metrics
 if self.latency_measurements:
 metrics['performance'] = {
 'mean_latency_ms': np.mean(self.latency_measurements),
 'median_latency_ms': np.median(self.latency_measurements),
 'p95_latency_ms': np.percentile(self.latency_measurements, 95),
 'p99_latency_ms': np.percentile(self.latency_measurements, 99),
 'max_latency_ms': np.max(self.latency_measurements),
 'min_latency_ms': np.min(self.latency_measurements)
 }

 # Clinical relevance scores (ICU-specific)
 metrics['clinical'] = self._compute_clinical_scores(metrics)

 return metrics

 def _compute_clinical_scores(self, metrics: Dict) -> Dict[str, float]:
 """
 Compute clinically relevant scores for ICU monitoring.

 Critical activities: fall_detected, convulsion, pain_response
 Important activities: agitation, restlessness, delirium
 """
 clinical_weights = {
 'fall_detected': 1.0, # Critical - immediate intervention needed
 'convulsion': 1.0, # Critical - medical emergency
 'pain_response': 0.9, # High priority - patient comfort
 'agitation': 0.7, # Important - may indicate distress
 'restlessness': 0.6, # Moderate - monitor closely
 'delirium': 0.8, # High - cognitive impairment
 'calm': 0.3 # Baseline - good but monitor
 }

 weighted_f1 = 0.0
 total_weight = 0.0

 for label, weight in clinical_weights.items():
 if label in metrics and isinstance(metrics[label], dict):
 f1 = metrics[label].get('f1_score', 0.0)
 weighted_f1 += f1 * weight
 total_weight += weight

 clinical_f1 = weighted_f1 / total_weight if total_weight > 0 else 0.0

 # False negative rate for critical conditions (very bad if missed)
 critical_labels = ['fall_detected', 'convulsion', 'pain_response']
 critical_fn_rate = 0.0
 critical_count = 0

 for label in critical_labels:
 if label in metrics and isinstance(metrics[label], dict):
 fn = metrics[label].get('fn', 0)
 tp = metrics[label].get('tp', 0)
 total_critical = fn + tp
 if total_critical > 0:
 critical_fn_rate += fn / total_critical
 critical_count += 1

 avg_critical_fn_rate = critical_fn_rate / critical_count if critical_count > 0 else 0.0

 return {
 'weighted_clinical_f1': clinical_f1,
 'critical_false_negative_rate': avg_critical_fn_rate,
 'clinical_monitoring_score': 1.0 - avg_critical_fn_rate # Higher is better
 }


class ClinicalValidator:
 """
 Clinical validation system for ICU activity monitoring.
 Provides real-time validation and offline evaluation capabilities.
 """

 def __init__(self, enable_real_time_validation: bool = True):
 self.metrics = ClinicalMetrics()
 self.enable_real_time_validation = enable_real_time_validation
 self.validation_history = []
 self.alerts = []

 # Validation thresholds (clinically acceptable ranges)
 self.thresholds = {
 'min_f1_critical': 0.85, # Critical activities must be >85% F1
 'max_critical_fn_rate': 0.10, # <10% false negatives for critical conditions
 'max_latency_ms': 100.0, # <100ms latency for real-time use
 'min_clinical_score': 0.90 # >90% clinical monitoring score
 }

 log.info("Clinical validator initialized")

 def validate_prediction(self, prediction_result: Dict, ground_truth: Optional[str] = None) -> Dict:
 """
 Validate a prediction result against clinical standards.

 Args:
 prediction_result: Prediction result dict from pipeline
 ground_truth: Ground truth label (if available)

 Returns:
 Validation result with pass/fail status and recommendations
 """
 validation = {
 'timestamp': time.time(),
 'passed': True,
 'issues': [],
 'recommendations': [],
 'confidence_score': 0.0
 }

 # Extract prediction data
 predicted_label = prediction_result.get('label', 'unknown')
 confidence = prediction_result.get('confidence', 0.0)
 latency = prediction_result.get('inference_ms', 0.0)

 # Update metrics if ground truth available
 if ground_truth:
 self.metrics.update(predicted_label, ground_truth, latency)

 # Clinical validation checks
 validation['confidence_score'] = self._assess_prediction_confidence(prediction_result)

 # Check latency
 if latency > self.thresholds['max_latency_ms']:
 validation['passed'] = False
 validation['issues'].append(f"High latency: {latency:.1f}ms > {self.thresholds['max_latency_ms']}ms")
 validation['recommendations'].append("Consider model optimization or hardware upgrade")

 # Check confidence for critical predictions
 if predicted_label in ['fall_detected', 'convulsion', 'pain_response'] and confidence < 0.8:
 validation['passed'] = False
 validation['issues'].append(f"Low confidence for critical prediction: {confidence:.2f} < 0.8")
 validation['recommendations'].append("Review model training data for critical conditions")

 # Check for unrealistic predictions
 if self._is_unrealistic_prediction(prediction_result):
 validation['passed'] = False
 validation['issues'].append("Prediction appears unrealistic")
 validation['recommendations'].append("Verify sensor data quality and environmental conditions")

 # Store validation result
 self.validation_history.append(validation)

 # Generate alerts for critical issues
 if not validation['passed']:
 self._generate_alert(validation)

 return validation

 def _assess_prediction_confidence(self, prediction_result: Dict) -> float:
 """
 Assess overall confidence in the prediction result.
 Considers multiple factors beyond just the confidence score.
 """
 confidence = prediction_result.get('confidence', 0.0)
 label = prediction_result.get('label', 'unknown')

 # Base confidence from model
 base_score = confidence

 # Adjust for clinical context
 clinical_multipliers = {
 'fall_detected': 1.2, # Boost critical detections
 'convulsion': 1.2,
 'pain_response': 1.1,
 'agitation': 1.0,
 'restlessness': 1.0,
 'delirium': 1.0,
 'calm': 0.9 # Slightly penalize baseline predictions
 }

 clinical_multiplier = clinical_multipliers.get(label, 1.0)
 adjusted_score = min(base_score * clinical_multiplier, 1.0)

 # Factor in additional evidence
 person_present = prediction_result.get('person_present', False)
 if not person_present:
 adjusted_score *= 0.5 # Reduce confidence if no person detected

 # Check keypoint quality
 kps = prediction_result.get('kps')
 if kps is not None and len(kps) > 0:
 valid_kps = sum(1 for kp in kps if len(kp) >= 3 and kp[2] > 0.5)
 kp_quality = valid_kps / len(kps)
 adjusted_score *= (0.5 + 0.5 * kp_quality) # Blend with keypoint quality

 return adjusted_score

 def _is_unrealistic_prediction(self, prediction_result: Dict) -> bool:
 """
 Check if prediction appears unrealistic based on context.
 """
 label = prediction_result.get('label', 'unknown')

 # Check for impossible transitions
 if hasattr(self, '_previous_label'):
 impossible_transitions = [
 ('calm', 'convulsion'), # Unrealistic instant transitions
 ('fall_detected', 'calm')
 ]

 for from_label, to_label in impossible_transitions:
 if self._previous_label == from_label and label == to_label:
 return True

 self._previous_label = label

 # Check environmental context
 bed_info = prediction_result.get('bed_info')
 if bed_info and label in ['fall_detected'] and not prediction_result.get('person_on_bed', True):
 # Fall detected but person not on bed - might be false positive
 return True

 return False

 def _generate_alert(self, validation: Dict):
 """Generate clinical alerts for validation failures."""
 alert = {
 'timestamp': validation['timestamp'],
 'severity': 'high' if any('critical' in issue.lower() for issue in validation['issues']) else 'medium',
 'issues': validation['issues'],
 'recommendations': validation['recommendations']
 }

 self.alerts.append(alert)

 # Log critical alerts
 if alert['severity'] == 'high':
 log.error(f"CRITICAL VALIDATION FAILURE: {alert['issues']}")

 def get_validation_report(self) -> Dict:
 """
 Generate comprehensive validation report.

 Returns:
 Full validation report with metrics and recommendations
 """
 current_metrics = self.metrics.compute_metrics()

 report = {
 'metrics': current_metrics,
 'validation_history': self.validation_history[-100:], # Last 100 validations
 'alerts': self.alerts[-50:], # Last 50 alerts
 'thresholds': self.thresholds,
 'compliance_status': self._check_compliance(current_metrics),
 'recommendations': self._generate_recommendations(current_metrics)
 }

 return report

 def _check_compliance(self, metrics: Dict) -> Dict[str, bool]:
 """Check compliance with clinical standards."""
 clinical = metrics.get('clinical', {})
 performance = metrics.get('performance', {})

 return {
 'critical_f1_compliant': clinical.get('weighted_clinical_f1', 0) >= self.thresholds['min_f1_critical'],
 'false_negative_compliant': clinical.get('critical_false_negative_rate', 1) <= self.thresholds['max_critical_fn_rate'],
 'latency_compliant': performance.get('p95_latency_ms', 1000) <= self.thresholds['max_latency_ms'],
 'clinical_score_compliant': clinical.get('clinical_monitoring_score', 0) >= self.thresholds['min_clinical_score']
 }

 def _generate_recommendations(self, metrics: Dict) -> List[str]:
 """Generate improvement recommendations based on metrics."""
 recommendations = []

 clinical = metrics.get('clinical', {})
 performance = metrics.get('performance', {})

 if clinical.get('critical_false_negative_rate', 0) > self.thresholds['max_critical_fn_rate']:
 recommendations.append("Improve detection of critical conditions - consider additional training data")

 if clinical.get('weighted_clinical_f1', 0) < self.thresholds['min_f1_critical']:
 recommendations.append("Enhance model accuracy for clinically important activities")

 if performance.get('p95_latency_ms', 0) > self.thresholds['max_latency_ms']:
 recommendations.append("Optimize model for lower latency - consider quantization or model pruning")

 if not recommendations:
 recommendations.append("System performing within clinical standards")

 return recommendations