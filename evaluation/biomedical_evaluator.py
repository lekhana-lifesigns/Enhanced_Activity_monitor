# evaluation/biomedical_evaluator.py
"""
Biomedical HPE Evaluation and Benchmarking (Phase 2)
Evaluates performance on motor development, rehabilitation, and gait analysis tasks.
"""

import numpy as np
import logging
import os
import json
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd

log = logging.getLogger("biomedical_evaluator")

class BiomedicalHPEEvaluator:
 """
 Evaluates biomedical HPE system performance on clinical tasks.
 Computes accuracy, clinical correlations, and performance metrics.
 """

 def __init__(self, output_dir: str):
 self.output_dir = output_dir
 os.makedirs(output_dir, exist_ok=True)

 # Evaluation results
 self.results = {
 "motor_development": {},
 "rehabilitation": {},
 "gait_analysis": {},
 "overall": {}
 }

 # Performance metrics
 self.metrics = {
 "latency": [],
 "throughput": [],
 "accuracy": [],
 "clinical_correlation": []
 }

 def evaluate_motor_development(self, predictions: List[Dict],
 ground_truth: List[Dict]) -> Dict:
 """
 Evaluate motor development assessment performance.

 Args:
 predictions: List of prediction results from biomedical HPE
 ground_truth: List of ground truth annotations

 Returns:
 Evaluation metrics
 """
 log.info("Evaluating motor development assessment...")

 results = {
 "movement_quality_accuracy": 0.0,
 "abnormality_detection": {},
 "age_group_performance": {},
 "clinical_correlation": 0.0
 }

 if not predictions or not ground_truth:
 log.warning("No data for motor development evaluation")
 return results

 # Extract predictions and ground truth
 pred_qualities = []
 true_qualities = []
 pred_risks = []
 true_risks = []

 for pred, gt in zip(predictions, ground_truth):
 # Movement quality assessment
 if "motor_dev" in pred.get("results", {}):
 motor_result = pred["results"]["motor_dev"]
 if isinstance(motor_result, dict) and "classification" in motor_result:
 pred_quality = motor_result["classification"].get("overall_risk", "unknown")
 pred_qualities.append(pred_quality)

 # Ground truth quality (would come from clinical assessment)
 true_quality = gt.get("clinical_assessment", {}).get("risk_level", "unknown")
 true_qualities.append(true_quality)

 # Compute accuracy metrics
 if pred_qualities and true_qualities:
 # Convert to binary classification (normal vs abnormal)
 pred_binary = [1 if p in ["high", "medium"] else 0 for p in pred_qualities]
 true_binary = [1 if t in ["high", "medium"] else 0 for t in true_qualities]

 results["abnormality_detection"] = {
 "accuracy": np.mean(np.array(pred_binary) == np.array(true_binary)),
 "precision": np.sum(np.array(pred_binary) & np.array(true_binary)) / max(1, np.sum(pred_binary)),
 "recall": np.sum(np.array(pred_binary) & np.array(true_binary)) / max(1, np.sum(true_binary))
 }

 # Age-group specific performance
 age_groups = defaultdict(list)
 for pred, gt in zip(predictions, ground_truth):
 age = gt.get("subject_age_months", 0)
 age_group = f"{age//6 * 6}-{(age//6 + 1)*6 - 1} months"

 if "motor_dev" in pred.get("results", {}):
 motor_result = pred["results"]["motor_dev"]
 if isinstance(motor_result, dict) and "classification" in motor_result:
 pred_risk = motor_result["classification"].get("overall_risk", "low")
 true_risk = gt.get("clinical_assessment", {}).get("risk_level", "low")

 age_groups[age_group].append((pred_risk, true_risk))

 for age_group, pairs in age_groups.items():
 if pairs:
 accuracies = [1 if p == t else 0 for p, t in pairs]
 results["age_group_performance"][age_group] = np.mean(accuracies)

 self.results["motor_development"] = results
 return results

 def evaluate_rehabilitation(self, predictions: List[Dict],
 ground_truth: List[Dict]) -> Dict:
 """
 Evaluate rehabilitation monitoring performance.
 """
 log.info("Evaluating rehabilitation monitoring...")

 results = {
 "exercise_recognition_accuracy": 0.0,
 "form_assessment_accuracy": 0.0,
 "rep_counting_accuracy": 0.0,
 "exercise_types": {}
 }

 if not predictions or not ground_truth:
 log.warning("No data for rehabilitation evaluation")
 return results

 # Extract predictions and ground truth
 exercise_accuracies = defaultdict(list)
 form_scores = []
 rep_counts = []

 for pred, gt in zip(predictions, ground_truth):
 if "rehab" in pred.get("results", {}):
 rehab_result = pred["results"]["rehab"]
 if isinstance(rehab_result, dict):
 # Exercise type recognition
 pred_exercise = rehab_result.get("exercise_type", "unknown")
 true_exercise = gt.get("exercise_type", "unknown")

 if pred_exercise == true_exercise:
 exercise_accuracies[pred_exercise].append(1)
 else:
 exercise_accuracies[pred_exercise].append(0)

 # Form assessment
 pred_form = rehab_result.get("metrics", {}).get("accuracy_score", 0.5)
 true_form = gt.get("form_score", 0.5)
 form_scores.append((pred_form, true_form))

 # Rep counting
 pred_reps = rehab_result.get("rep_count", 0)
 true_reps = gt.get("rep_count", 0)
 rep_counts.append((pred_reps, true_reps))

 # Compute metrics
 if exercise_accuracies:
 for exercise, accuracies in exercise_accuracies.items():
 results["exercise_types"][exercise] = np.mean(accuracies)
 results["exercise_recognition_accuracy"] = np.mean([acc for accs in exercise_accuracies.values() for acc in accs])

 if form_scores:
 form_diffs = [abs(p - t) for p, t in form_scores]
 results["form_assessment_accuracy"] = 1.0 - np.mean(form_diffs) # Lower difference = higher accuracy

 if rep_counts:
 rep_diffs = [abs(p - t) for p, t in rep_counts]
 results["rep_counting_accuracy"] = 1.0 - min(1.0, np.mean(rep_diffs) / 5.0) # Normalize by max expected error

 self.results["rehabilitation"] = results
 return results

 def evaluate_gait_analysis(self, predictions: List[Dict],
 ground_truth: List[Dict]) -> Dict:
 """
 Evaluate gait analysis performance.
 """
 log.info("Evaluating gait analysis...")

 results = {
 "spatiotemporal_accuracy": {},
 "abnormality_detection": {},
 "cadence_error": 0.0,
 "step_length_error": 0.0,
 "walking_speed_error": 0.0
 }

 if not predictions or not ground_truth:
 log.warning("No data for gait analysis evaluation")
 return results

 # Extract gait parameters
 cadence_errors = []
 step_length_errors = []
 speed_errors = []

 for pred, gt in zip(predictions, ground_truth):
 if "gait" in pred.get("results", {}):
 gait_result = pred["results"]["gait"]
 if isinstance(gait_result, dict) and "assessment" in gait_result:
 # Spatiotemporal parameters
 gait_metrics = gait_result.get("gait_parameters", {})

 # Compare with ground truth
 true_cadence = gt.get("gait_metrics", {}).get("cadence", 0)
 pred_cadence = gait_metrics.get("cadence_mean", 0)
 if true_cadence > 0 and pred_cadence > 0:
 cadence_errors.append(abs(pred_cadence - true_cadence) / true_cadence)

 true_step_length = gt.get("gait_metrics", {}).get("step_length", 0)
 pred_step_length = gait_metrics.get("step_length_mean", 0)
 if true_step_length > 0 and pred_step_length > 0:
 step_length_errors.append(abs(pred_step_length - true_step_length) / true_step_length)

 true_speed = gt.get("gait_metrics", {}).get("walking_speed", 0)
 pred_speed = gait_metrics.get("walking_speed_mean", 0)
 if true_speed > 0 and pred_speed > 0:
 speed_errors.append(abs(pred_speed - true_speed) / true_speed)

 # Compute average errors
 if cadence_errors:
 results["cadence_error"] = np.mean(cadence_errors)
 if step_length_errors:
 results["step_length_error"] = np.mean(step_length_errors)
 if speed_errors:
 results["walking_speed_error"] = np.mean(speed_errors)

 # Overall spatiotemporal accuracy (lower error = higher accuracy)
 all_errors = cadence_errors + step_length_errors + speed_errors
 if all_errors:
 results["spatiotemporal_accuracy"]["overall"] = 1.0 - min(1.0, np.mean(all_errors))

 self.results["gait_analysis"] = results
 return results

 def evaluate_performance(self, latency_measurements: List[float],
 throughput_measurements: List[float]) -> Dict:
 """
 Evaluate system performance metrics.
 """
 log.info("Evaluating system performance...")

 performance = {
 "latency": {
 "mean": np.mean(latency_measurements) if latency_measurements else 0,
 "std": np.std(latency_measurements) if latency_measurements else 0,
 "p95": np.percentile(latency_measurements, 95) if latency_measurements else 0,
 "p99": np.percentile(latency_measurements, 99) if latency_measurements else 0
 },
 "throughput": {
 "mean": np.mean(throughput_measurements) if throughput_measurements else 0,
 "std": np.std(throughput_measurements) if throughput_measurements else 0
 }
 }

 self.metrics["latency"] = latency_measurements
 self.metrics["throughput"] = throughput_measurements

 return performance

 def generate_report(self) -> Dict:
 """Generate comprehensive evaluation report."""
 report = {
 "timestamp": time.time(),
 "evaluation_results": self.results,
 "performance_metrics": self.metrics,
 "summary": self._generate_summary(),
 "recommendations": self._generate_recommendations()
 }

 # Save report
 report_path = os.path.join(self.output_dir, "evaluation_report.json")
 with open(report_path, 'w') as f:
 json.dump(report, f, indent=2)

 # Generate plots
 self._generate_plots()

 log.info(f"Evaluation report saved to {report_path}")
 return report

 def _generate_summary(self) -> Dict:
 """Generate evaluation summary."""
 summary = {
 "overall_performance": "unknown",
 "strengths": [],
 "weaknesses": [],
 "clinical_readiness": "not_ready"
 }

 # Assess overall performance
 scores = []

 # Motor development
 if self.results["motor_development"].get("abnormality_detection", {}).get("accuracy"):
 scores.append(self.results["motor_development"]["abnormality_detection"]["accuracy"])

 # Rehabilitation
 if self.results["rehabilitation"].get("exercise_recognition_accuracy"):
 scores.append(self.results["rehabilitation"]["exercise_recognition_accuracy"])
 if self.results["rehabilitation"].get("form_assessment_accuracy"):
 scores.append(self.results["rehabilitation"]["form_assessment_accuracy"])

 # Gait analysis
 if self.results["gait_analysis"].get("spatiotemporal_accuracy", {}).get("overall"):
 scores.append(self.results["gait_analysis"]["spatiotemporal_accuracy"]["overall"])

 if scores:
 avg_score = np.mean(scores)
 if avg_score >= 0.9:
 summary["overall_performance"] = "excellent"
 summary["clinical_readiness"] = "ready"
 elif avg_score >= 0.8:
 summary["overall_performance"] = "good"
 summary["clinical_readiness"] = "conditionally_ready"
 elif avg_score >= 0.7:
 summary["overall_performance"] = "acceptable"
 summary["clinical_readiness"] = "needs_improvement"
 else:
 summary["overall_performance"] = "poor"
 summary["clinical_readiness"] = "not_ready"

 # Identify strengths and weaknesses
 if self.results["motor_development"].get("abnormality_detection", {}).get("accuracy", 0) > 0.8:
 summary["strengths"].append("Strong motor development abnormality detection")
 else:
 summary["weaknesses"].append("Motor development assessment needs improvement")

 if self.results["rehabilitation"].get("form_assessment_accuracy", 0) > 0.8:
 summary["strengths"].append("Good rehabilitation form assessment")
 else:
 summary["weaknesses"].append("Rehabilitation form assessment needs improvement")

 if self.results["gait_analysis"].get("spatiotemporal_accuracy", {}).get("overall", 0) > 0.8:
 summary["strengths"].append("Accurate gait spatiotemporal analysis")
 else:
 summary["weaknesses"].append("Gait analysis accuracy needs improvement")

 return summary

 def _generate_recommendations(self) -> List[str]:
 """Generate improvement recommendations."""
 recommendations = []

 # Based on performance gaps
 if self.results["motor_development"].get("abnormality_detection", {}).get("accuracy", 0) < 0.8:
 recommendations.append("Improve motor development feature extraction and classification models")
 recommendations.append("Collect more diverse infant pose data for training")

 if self.results["rehabilitation"].get("form_assessment_accuracy", 0) < 0.8:
 recommendations.append("Enhance exercise form detection algorithms")
 recommendations.append("Add more exercise types and variations to training data")

 if self.results["gait_analysis"].get("spatiotemporal_accuracy", {}).get("overall", 0) < 0.8:
 recommendations.append("Improve gait parameter estimation algorithms")
 recommendations.append("Add ground truth calibration for spatiotemporal measurements")

 # Performance recommendations
 if self.metrics["latency"] and np.mean(self.metrics["latency"]) > 100:
 recommendations.append("Optimize model inference for lower latency")
 recommendations.append("Consider TensorRT optimization for deployment")

 if not recommendations:
 recommendations.append("System performance is good, consider expanding to additional clinical applications")

 return recommendations

 def _generate_plots(self):
 """Generate evaluation plots."""
 try:
 # Performance plots
 if self.metrics["latency"]:
 plt.figure(figsize=(10, 6))
 plt.hist(self.metrics["latency"], bins=50, alpha=0.7)
 plt.xlabel("Latency (ms)")
 plt.ylabel("Frequency")
 plt.title("Inference Latency Distribution")
 plt.savefig(os.path.join(self.output_dir, "latency_distribution.png"))
 plt.close()

 # Clinical accuracy plots
 fig, axes = plt.subplots(1, 3, figsize=(15, 5))

 # Motor development
 if self.results["motor_development"].get("age_group_performance"):
 age_groups = list(self.results["motor_development"]["age_group_performance"].keys())
 accuracies = list(self.results["motor_development"]["age_group_performance"].values())
 axes[0].bar(age_groups, accuracies)
 axes[0].set_title("Motor Development by Age Group")
 axes[0].set_ylabel("Accuracy")
 axes[0].tick_params(axis='x', rotation=45)

 # Rehabilitation
 if self.results["rehabilitation"].get("exercise_types"):
 exercises = list(self.results["rehabilitation"]["exercise_types"].keys())
 accuracies = list(self.results["rehabilitation"]["exercise_types"].values())
 axes[1].bar(exercises, accuracies)
 axes[1].set_title("Rehabilitation by Exercise Type")
 axes[1].set_ylabel("Accuracy")
 axes[1].tick_params(axis='x', rotation=45)

 # Gait analysis
 gait_errors = [
 self.results["gait_analysis"].get("cadence_error", 0),
 self.results["gait_analysis"].get("step_length_error", 0),
 self.results["gait_analysis"].get("walking_speed_error", 0)
 ]
 if any(gait_errors):
 axes[2].bar(["Cadence", "Step Length", "Walking Speed"], gait_errors)
 axes[2].set_title("Gait Analysis Errors")
 axes[2].set_ylabel("Relative Error")

 plt.tight_layout()
 plt.savefig(os.path.join(self.output_dir, "clinical_accuracy.png"))
 plt.close()

 except Exception as e:
 log.warning(f"Failed to generate plots: {e}")


def run_evaluation_suite(predictions_file: str, ground_truth_file: str,
 output_dir: str) -> Dict:
 """
 Run complete evaluation suite.

 Args:
 predictions_file: Path to predictions JSON file
 ground_truth_file: Path to ground truth JSON file
 output_dir: Output directory for results

 Returns:
 Evaluation results
 """
 evaluator = BiomedicalHPEEvaluator(output_dir)

 # Load data
 with open(predictions_file, 'r') as f:
 predictions = json.load(f)

 with open(ground_truth_file, 'r') as f:
 ground_truth = json.load(f)

 # Run evaluations
 evaluator.evaluate_motor_development(predictions, ground_truth)
 evaluator.evaluate_rehabilitation(predictions, ground_truth)
 evaluator.evaluate_gait_analysis(predictions, ground_truth)

 # Generate report
 report = evaluator.generate_report()

 return report