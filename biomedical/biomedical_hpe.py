# biomedical/biomedical_hpe.py
"""
Biomedical Human Pose Estimation System
Integrates motor development, rehabilitation, and gait analysis.

Based on research from:
- Markerless HPE for biomedical applications (Frontiers review)
- Clinical-grade pose estimation methods
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time

from .motor_development.motor_assessment import MotorDevelopmentAssessor
from .rehabilitation.rehab_monitor import RehabilitationMonitor
from .gait_analysis.gait_analyzer import GaitAnalyzer

log = logging.getLogger("biomedical_hpe")

class BiomedicalHPESystem:
 """
 Integrated biomedical HPE system for clinical applications.
 Supports motor development assessment, rehabilitation monitoring, and gait analysis.
 """

 def __init__(self, config: Dict):
 self.config = config
 self.active_modules = {}

 # Initialize modules based on configuration
 if config.get("enable_motor_development", False):
 self.active_modules["motor_dev"] = MotorDevelopmentAssessor(
 window_size=config.get("motor_dev_window", 300),
 fps=config.get("fps", 30)
 )

 if config.get("enable_rehabilitation", False):
 self.active_modules["rehab"] = RehabilitationMonitor(
 exercise_type=config.get("exercise_type", "upper_limb"),
 target_reps=config.get("target_reps", 10),
 session_duration=config.get("session_duration", 300)
 )

 if config.get("enable_gait_analysis", False):
 self.active_modules["gait"] = GaitAnalyzer(
 fps=config.get("fps", 30),
 analysis_window=config.get("gait_window", 300)
 )

 log.info("Biomedical HPE system initialized with modules: %s", list(self.active_modules.keys()))

 def process_frame(self, keypoints_2d: np.ndarray,
 keypoints_3d: Optional[np.ndarray] = None,
 timestamp: Optional[float] = None) -> Dict:
 """
 Process a frame through all active biomedical modules.

 Args:
 keypoints_2d: 2D keypoints (17, 3) - x, y, confidence
 keypoints_3d: Optional 3D keypoints (17, 4) - x, y, z, confidence
 timestamp: Frame timestamp

 Returns:
 Dict with results from all active modules
 """
 results = {
 "timestamp": timestamp or time.time(),
 "modules_active": list(self.active_modules.keys()),
 "results": {}
 }

 # Process through each active module
 for module_name, module in self.active_modules.items():
 try:
 if module_name == "motor_dev":
 module_result = module.assess_movement_quality(keypoints_2d, keypoints_3d)
 elif module_name == "rehab":
 module_result = module.monitor_exercise(keypoints_2d, keypoints_3d)
 elif module_name == "gait":
 module_result = module.analyze_gait(keypoints_2d, keypoints_3d, timestamp)

 results["results"][module_name] = module_result

 except Exception as e:
 log.error(f"Error in {module_name} module: {e}")
 results["results"][module_name] = {"error": str(e)}

 # Generate integrated clinical insights
 results["clinical_insights"] = self._generate_clinical_insights(results["results"])

 return results

 def _generate_clinical_insights(self, module_results: Dict) -> Dict:
 """Generate integrated clinical insights from all modules."""
 insights = {
 "overall_assessment": "normal",
 "risk_level": "low",
 "key_findings": [],
 "recommendations": [],
 "alerts": []
 }

 # Analyze motor development
 if "motor_dev" in module_results:
 motor_result = module_results["motor_dev"]
 if isinstance(motor_result, dict) and "classification" in motor_result:
 classification = motor_result["classification"]
 risk = classification.get("overall_risk", "unknown")

 if risk == "high":
 insights["overall_assessment"] = "concerning"
 insights["risk_level"] = "high"
 insights["key_findings"].append("Abnormal motor development patterns detected")
 insights["recommendations"].append("Consult pediatric neurologist for evaluation")
 insights["alerts"].append("High risk of neurological disorder")

 elif risk == "medium":
 insights["risk_level"] = "medium"
 insights["key_findings"].append("Some motor development concerns noted")

 # Analyze rehabilitation progress
 if "rehab" in module_results:
 rehab_result = module_results["rehab"]
 if isinstance(rehab_result, dict):
 accuracy = rehab_result.get("metrics", {}).get("accuracy_score", 0.5)
 reps_completed = rehab_result.get("rep_count", 0)
 target_reps = rehab_result.get("target_reps", 10)

 if accuracy < 0.7:
 insights["key_findings"].append("Poor exercise form detected")
 insights["recommendations"].append("Focus on proper technique")

 if reps_completed >= target_reps:
 insights["key_findings"].append("Exercise goals achieved")
 insights["recommendations"].append("Consider increasing difficulty level")

 # Analyze gait
 if "gait" in module_results:
 gait_result = module_results["gait"]
 if isinstance(gait_result, dict) and "assessment" in gait_result:
 assessment = gait_result["assessment"]
 quality = assessment.get("overall_quality", "unknown")

 if quality == "abnormal":
 insights["overall_assessment"] = "concerning"
 insights["risk_level"] = "high"
 insights["key_findings"].extend(assessment.get("abnormalities", []))
 insights["recommendations"].extend(assessment.get("recommendations", []))
 insights["alerts"].append("Gait abnormalities detected")

 elif quality == "mildly_abnormal":
 if insights["risk_level"] == "low":
 insights["risk_level"] = "medium"
 insights["key_findings"].extend(assessment.get("abnormalities", []))

 # Overall assessment logic
 if insights["risk_level"] == "high":
 insights["overall_assessment"] = "requires_attention"
 elif insights["risk_level"] == "medium":
 insights["overall_assessment"] = "monitor_closely"
 else:
 insights["overall_assessment"] = "normal"

 return insights

 def get_module_status(self) -> Dict:
 """Get status of all modules."""
 status = {}
 for module_name, module in self.active_modules.items():
 if hasattr(module, 'rep_count'):
 status[module_name] = {
 "reps_completed": module.rep_count,
 "active": True
 }
 elif hasattr(module, 'pose_history'):
 status[module_name] = {
 "frames_processed": len(module.pose_history),
 "active": True
 }
 else:
 status[module_name] = {"active": True}

 return status

 def reset_session(self):
 """Reset all modules for new session."""
 for module in self.active_modules.values():
 if hasattr(module, 'reset_analysis'):
 module.reset_analysis()
 elif hasattr(module, 'pose_history'):
 module.pose_history.clear()
 if hasattr(module, 'rep_count'):
 module.rep_count = 0
 if hasattr(module, 'step_events'):
 module.step_events.clear()

 log.info("Biomedical HPE session reset")

 def configure_modules(self, config_updates: Dict):
 """Update configuration for modules."""
 self.config.update(config_updates)

 # Reinitialize modules if needed
 if "enable_motor_development" in config_updates:
 if config_updates["enable_motor_development"]:
 if "motor_dev" not in self.active_modules:
 self.active_modules["motor_dev"] = MotorDevelopmentAssessor(
 window_size=self.config.get("motor_dev_window", 300),
 fps=self.config.get("fps", 30)
 )
 else:
 # Remove module when flag is False
 if "motor_dev" in self.active_modules:
 module = self.active_modules["motor_dev"]
 if hasattr(module, 'close'):
 module.close()
 elif hasattr(module, 'teardown'):
 module.teardown()
 del self.active_modules["motor_dev"]

 if "enable_rehabilitation" in config_updates:
 if config_updates["enable_rehabilitation"]:
 if "rehab" not in self.active_modules:
 self.active_modules["rehab"] = RehabilitationMonitor(
 exercise_type=self.config.get("exercise_type", "upper_limb"),
 target_reps=self.config.get("target_reps", 10),
 session_duration=self.config.get("session_duration", 300)
 )
 else:
 # Remove module when flag is False
 if "rehab" in self.active_modules:
 module = self.active_modules["rehab"]
 if hasattr(module, 'close'):
 module.close()
 elif hasattr(module, 'teardown'):
 module.teardown()
 del self.active_modules["rehab"]

 if "enable_gait_analysis" in config_updates:
 if config_updates["enable_gait_analysis"]:
 if "gait" not in self.active_modules:
 self.active_modules["gait"] = GaitAnalyzer(
 fps=self.config.get("fps", 30),
 analysis_window=self.config.get("gait_window", 300)
 )
 else:
 # Remove module when flag is False
 if "gait" in self.active_modules:
 module = self.active_modules["gait"]
 if hasattr(module, 'close'):
 module.close()
 elif hasattr(module, 'teardown'):
 module.teardown()
 del self.active_modules["gait"]

 log.info("Modules reconfigured: %s", list(self.active_modules.keys()))