# analytics/regulatory_validation_protocol.py
"""
Regulatory Validation Protocol for EAC System
Comprehensive validation framework for medical device certification.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

log = logging.getLogger("validation_protocol")

class ValidationProtocol:
 """
 Regulatory validation protocol for clinical activity monitoring system.
 Implements IEC 62304, ISO 14971, and FDA guidance for medical devices.
 """

 def __init__(self):
 self.protocol_version = "1.0"
 self.start_date = datetime.now()
 self.phases = self._define_phases()
 self.metrics = self._define_metrics()
 self.test_cases = self._define_test_cases()

 def _define_phases(self) -> Dict[str, Dict]:
 """Define validation phases per regulatory requirements."""
 return {
 "technical_validation": {
 "name": "Technical Validation",
 "duration_weeks": 2,
 "objectives": [
 "Verify feature extraction accuracy (>95% keypoint detection)",
 "Validate temporal windowing logic",
 "Test edge case handling (occlusion, lighting)",
 "Benchmark computational performance (<50ms/frame)",
 "Verify data security and privacy controls"
 ],
 "deliverables": [
 "Technical validation report",
 "Performance benchmark results",
 "Edge case test matrix",
 "Security assessment report"
 ],
 "success_criteria": {
 "keypoint_accuracy": 0.95,
 "temporal_accuracy": 0.90,
 "performance_target": 50, # ms/frame
 "uptime_target": 0.995
 }
 },

 "clinical_pilot": {
 "name": "Clinical Pilot Study",
 "duration_weeks": 4,
 "objectives": [
 "Deploy in low-acuity ward with nurse supervision",
 "Collect real patient data with ground truth annotations",
 "Validate alert accuracy against clinical observations",
 "Assess nurse workflow integration",
 "Evaluate false positive/negative rates",
 "Measure response times and clinical outcomes"
 ],
 "deliverables": [
 "Clinical pilot report",
 "Alert accuracy analysis",
 "Nurse feedback survey",
 "Workflow integration assessment",
 "Safety incident log"
 ],
 "success_criteria": {
 "alert_sensitivity": 0.90,
 "alert_specificity": 0.95,
 "nurse_satisfaction": 0.80,
 "response_time_avg": 30, # seconds
 "adverse_events": 0
 }
 },

 "regulatory_compliance": {
 "name": "Regulatory Compliance & Submission",
 "duration_weeks": 6,
 "objectives": [
 "Conduct formal validation per IEC 62304",
 "Generate traceability matrix",
 "Perform risk management per ISO 14971",
 "Prepare FDA 510(k) submission documentation",
 "Complete cybersecurity assessment",
 "Finalize user documentation and training materials"
 ],
 "deliverables": [
 "IEC 62304 validation report",
 "ISO 14971 risk management report",
 "FDA 510(k) submission package",
 "Traceability matrix",
 "User manual and training materials",
 "Post-market surveillance plan"
 ],
 "success_criteria": {
 "iec_compliance": True,
 "risk_residual": "acceptable",
 "documentation_complete": True,
 "fda_review_ready": True
 }
 }
 }

 def _define_metrics(self) -> Dict[str, Dict]:
 """Define quantitative metrics for validation."""
 return {
 "alert_performance": {
 "sensitivity": "TP / (TP + FN)",
 "specificity": "TN / (TN + FP)",
 "precision": "TP / (TP + FP)",
 "f1_score": "2 * precision * recall / (precision + recall)",
 "auc_roc": "Area under ROC curve"
 },

 "temporal_performance": {
 "detection_latency": "Time from event to alert (seconds)",
 "response_time": "Time from alert to nurse action (seconds)",
 "temporal_accuracy": "Correct temporal classification rate"
 },

 "system_performance": {
 "throughput": "Frames processed per second",
 "latency": "End-to-end processing time (ms)",
 "uptime": "System availability percentage",
 "resource_usage": "CPU/GPU/memory utilization"
 },

 "clinical_outcomes": {
 "fall_prevention": "Reduction in fall incidents",
 "response_efficiency": "Time to clinical intervention",
 "patient_safety": "Adverse event rate",
 "staff_satisfaction": "Nurse workflow impact score"
 }
 }

 def _define_test_cases(self) -> Dict[str, List[Dict]]:
 """Define comprehensive test cases for validation."""
 return {
 "functional_tests": [
 {
 "id": "FT-001",
 "name": "Activity Detection Accuracy",
 "description": "Verify correct detection of target activities",
 "test_data": "Annotated video dataset",
 "acceptance_criteria": "≥90% accuracy for each activity",
 "risk_level": "Critical"
 },
 {
 "id": "FT-002",
 "name": "Temporal Window Logic",
 "description": "Validate persistence and observation window logic",
 "test_data": "Synthetic temporal sequences",
 "acceptance_criteria": "Correct alert triggering based on duration/persistence",
 "risk_level": "Critical"
 },
 {
 "id": "FT-003",
 "name": "Alert Thresholds",
 "description": "Verify clinical threshold implementation",
 "test_data": "Feature vector test cases",
 "acceptance_criteria": "Correct risk level assignment",
 "risk_level": "Critical"
 }
 ],

 "performance_tests": [
 {
 "id": "PT-001",
 "name": "Processing Latency",
 "description": "Measure end-to-end processing time",
 "test_data": "High-resolution video streams",
 "acceptance_criteria": "<50ms average latency",
 "risk_level": "Major"
 },
 {
 "id": "PT-002",
 "name": "Concurrent Users",
 "description": "Test multi-patient monitoring capability",
 "test_data": "Multiple simultaneous video streams",
 "acceptance_criteria": "Maintain performance with 10+ patients",
 "risk_level": "Major"
 }
 ],

 "safety_tests": [
 {
 "id": "ST-001",
 "name": "False Positive Rate",
 "description": "Measure inappropriate alerts",
 "test_data": "Normal activity recordings",
 "acceptance_criteria": "<5% false positive rate",
 "risk_level": "Critical"
 },
 {
 "id": "ST-002",
 "name": "Critical Alert Detection",
 "description": "Verify detection of oxygen mask removal, IV pulls",
 "test_data": "Critical event simulations",
 "acceptance_criteria": "100% detection rate",
 "risk_level": "Critical"
 }
 ],

 "usability_tests": [
 {
 "id": "UT-001",
 "name": "Nurse Interface",
 "description": "Evaluate alert display and explanation clarity",
 "test_data": "Nurse feedback surveys",
 "acceptance_criteria": ">80% satisfaction rating",
 "risk_level": "Major"
 },
 {
 "id": "UT-002",
 "name": "Workflow Integration",
 "description": "Assess impact on clinical workflow",
 "test_data": "Time-motion studies",
 "acceptance_criteria": "No increase in nurse workload",
 "risk_level": "Major"
 }
 ]
 }

 def generate_validation_plan(self) -> Dict[str, Any]:
 """Generate comprehensive validation plan."""
 return {
 "protocol_info": {
 "version": self.protocol_version,
 "generated_date": self.start_date.isoformat(),
 "total_duration_weeks": sum(p["duration_weeks"] for p in self.phases.values()),
 "regulatory_standards": ["IEC 62304", "ISO 14971", "FDA 510(k)"]
 },
 "phases": self.phases,
 "metrics": self.metrics,
 "test_cases": self.test_cases,
 "data_requirements": self._get_data_requirements(),
 "risk_mitigation": self._get_risk_mitigation()
 }

 def _get_data_requirements(self) -> Dict[str, Any]:
 """Define data requirements for validation."""
 return {
 "patient_diversity": {
 "count": 100,
 "demographics": {
 "age_range": "18-95 years",
 "gender_distribution": "Balanced",
 "clinical_conditions": ["Post-surgical", "Medical", "ICU", "Rehabilitation"]
 }
 },
 "environmental_conditions": {
 "lighting": ["Daylight", "Artificial", "Low light", "Variable"],
 "camera_angles": ["Overhead", "Side", "Multiple views"],
 "backgrounds": ["Hospital room", "ICU bed", "Various clutter levels"],
 "occlusion_scenarios": ["Partial", "Heavy", "Equipment interference"]
 },
 "temporal_coverage": {
 "duration": "24/7 monitoring",
 "activity_patterns": ["Peak hours", "Night shifts", "Weekend patterns"],
 "event_types": ["Normal activities", "Critical events", "False triggers"]
 },
 "annotation_requirements": {
 "expert_annotators": "Registered nurses with 2+ years experience",
 "annotation_tools": "Custom clinical annotation interface",
 "inter_annotator_agreement": ">0.85 kappa score",
 "ground_truth_validation": "Double annotation with adjudication"
 }
 }

 def _get_risk_mitigation(self) -> Dict[str, List[str]]:
 """Define risk mitigation strategies."""
 return {
 "false_negatives": [
 "Multi-modal sensor fusion",
 "Conservative threshold setting",
 "Redundant detection algorithms",
 "Regular threshold calibration"
 ],
 "false_positives": [
 "Clinical threshold validation",
 "Nurse feedback integration",
 "Patient-specific baseline learning",
 "Alert fatigue monitoring"
 ],
 "system_failures": [
 "Hardware redundancy",
 "Automatic failover mechanisms",
 "Real-time system monitoring",
 "Regular maintenance schedules"
 ],
 "privacy_breaches": [
 "On-device processing only",
 "No cloud storage of video",
 "HIPAA-compliant data handling",
 "Regular security audits"
 ]
 }

 def validate_phase_completion(self, phase_name: str, results: Dict[str, Any]) -> Dict[str, Any]:
 """
 Validate completion of a validation phase.

 Args:
 phase_name: Name of the phase
 results: Validation results dictionary

 Returns:
 Validation report
 """
 phase = self.phases.get(phase_name)
 if not phase:
 return {"error": f"Unknown phase: {phase_name}"}

 success_criteria = phase["success_criteria"]
 validation_report = {
 "phase": phase_name,
 "timestamp": datetime.now().isoformat(),
 "criteria_met": {},
 "overall_success": True,
 "recommendations": []
 }

 for criterion, target in success_criteria.items():
 actual = results.get(criterion)
 if actual is None:
 validation_report["criteria_met"][criterion] = False
 validation_report["overall_success"] = False
 validation_report["recommendations"].append(f"Missing result for {criterion}")
 else:
 met = self._check_criterion(criterion, actual, target)
 validation_report["criteria_met"][criterion] = met
 if not met:
 validation_report["overall_success"] = False
 validation_report["recommendations"].append(
 f"{criterion}: achieved {actual}, target {target}"
 )

 return validation_report

 def _check_criterion(self, criterion: str, actual: Any, target: Any) -> bool:
 """Check if a criterion is met."""
 if isinstance(target, (int, float)):
 if isinstance(actual, (int, float)):
 return actual >= target
 return False
 elif isinstance(target, bool):
 return actual == target
 else:
 # String or other comparison
 return actual == target

 def export_protocol(self, filepath: str):
 """Export validation protocol to JSON file."""
 protocol = self.generate_validation_plan()
 with open(filepath, 'w') as f:
 json.dump(protocol, f, indent=2, default=str)
 log.info(f"Validation protocol exported to {filepath}")

# Global protocol instance
_validation_protocol = None

def get_validation_protocol() -> ValidationProtocol:
 """Get or create global validation protocol instance."""
 global _validation_protocol
 if _validation_protocol is None:
 _validation_protocol = ValidationProtocol()
 return _validation_protocol