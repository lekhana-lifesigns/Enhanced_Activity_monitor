# pipeline/validation/capture_validator.py
"""
Capture Validation Module for EAM Data Collection & Runtime Inference.

Enforces the EAM Data Collection Checklist constraints:
- Distance band validation (CLOSE: 1.8m, MID: 2.8m, FAR: 3.5m)
- ROI/crop sanity checks
- Body part visibility requirements
- Quality gates for feature extraction

Used by both:
1. Data collection scripts (pre-capture validation)
2. Inference pipeline (runtime frame validation)
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from enum import Enum

log = logging.getLogger("validator")


class DistanceBand(Enum):
 """Camera distance bands per EAM protocol."""
 CLOSE = "close" # 1.7-1.9m (target 1.8m) - Upper body & device interaction
 MID = "mid" # 2.6-3.0m (target 2.8m) - Posture & bed movement
 FAR = "far" # 3.2-4.5m (target 3.5m) - Exit risk & mobility


@dataclass
class DistanceConfig:
 """Distance band configuration."""
 min_meters: float
 max_meters: float
 target_meters: float

 # Expected body coverage ratios at this distance
 min_body_height_ratio: float # Minimum body height as fraction of frame
 max_body_height_ratio: float # Maximum body height as fraction of frame


# Distance band specifications from checklist
DISTANCE_BANDS = {
 DistanceBand.CLOSE: DistanceConfig(
 min_meters=1.7, max_meters=1.9, target_meters=1.8,
 min_body_height_ratio=0.6, max_body_height_ratio=0.95
 ),
 DistanceBand.MID: DistanceConfig(
 min_meters=2.6, max_meters=3.0, target_meters=2.8,
 min_body_height_ratio=0.4, max_body_height_ratio=0.75
 ),
 DistanceBand.FAR: DistanceConfig(
 min_meters=3.2, max_meters=4.5, target_meters=3.5,
 min_body_height_ratio=0.25, max_body_height_ratio=0.55
 ),
}


# Required keypoints per activity type
KEYPOINT_REQUIREMENTS = {
 "device_interaction": ["nose", "left_shoulder", "right_shoulder",
 "left_wrist", "right_wrist"],
 "posture": ["nose", "left_shoulder", "right_shoulder",
 "left_hip", "right_hip"],
 "exit_risk": ["left_hip", "right_hip", "left_knee", "right_knee",
 "left_ankle", "right_ankle"],
 "respiratory": ["nose", "left_shoulder", "right_shoulder"],
 "agitation": ["nose", "left_shoulder", "right_shoulder",
 "left_wrist", "right_wrist", "left_hip", "right_hip"],
}

# COCO keypoint indices
KEYPOINT_INDICES = {
 "nose": 0,
 "left_eye": 1, "right_eye": 2,
 "left_ear": 3, "right_ear": 4,
 "left_shoulder": 5, "right_shoulder": 6,
 "left_elbow": 7, "right_elbow": 8,
 "left_wrist": 9, "right_wrist": 10,
 "left_hip": 11, "right_hip": 12,
 "left_knee": 13, "right_knee": 14,
 "left_ankle": 15, "right_ankle": 16,
}


@dataclass
class ValidationResult:
 """Result of frame/ROI validation."""
 valid: bool
 reason: str = ""
 warnings: List[str] = None
 metrics: Dict[str, float] = None

 def __post_init__(self):
 if self.warnings is None:
 self.warnings = []
 if self.metrics is None:
 self.metrics = {}


class CaptureValidator:
 """
 Validates frames and ROIs against EAM data collection protocol.

 Enforces:
 - ROI sanity constraints (minimum size, aspect ratio)
 - Body part visibility (based on activity type)
 - Distance band consistency
 - Quality gates for feature extraction
 """

 def __init__(self,
 min_roi_width_ratio: float = 0.4,
 min_roi_height_ratio: float = 0.5,
 min_keypoint_confidence: float = 0.3,
 min_valid_keypoint_ratio: float = 0.4):
 """
 Initialize validator with constraints.

 Args:
 min_roi_width_ratio: Minimum ROI width as fraction of frame width
 min_roi_height_ratio: Minimum ROI height as fraction of frame height
 min_keypoint_confidence: Minimum confidence for valid keypoint
 min_valid_keypoint_ratio: Minimum fraction of keypoints that must be valid
 """
 self.min_roi_width_ratio = min_roi_width_ratio
 self.min_roi_height_ratio = min_roi_height_ratio
 self.min_keypoint_confidence = min_keypoint_confidence
 self.min_valid_keypoint_ratio = min_valid_keypoint_ratio

 def validate_roi(self,
 roi: Tuple[int, int, int, int],
 frame_shape: Tuple[int, int],
 keypoints: Optional[List] = None) -> ValidationResult:
 """
 Validate ROI against capture constraints.

 Args:
 roi: (x1, y1, x2, y2) region of interest
 frame_shape: (height, width) of frame
 keypoints: Optional keypoints to check visibility

 Returns:
 ValidationResult with validity and diagnostics
 """
 x1, y1, x2, y2 = roi
 frame_h, frame_w = frame_shape[:2]

 warnings = []
 metrics = {}

 # Calculate ROI dimensions
 roi_w = x2 - x1
 roi_h = y2 - y1

 metrics["roi_width"] = roi_w
 metrics["roi_height"] = roi_h
 metrics["roi_width_ratio"] = roi_w / frame_w
 metrics["roi_height_ratio"] = roi_h / frame_h

 # Check minimum size constraints
 if roi_w / frame_w < self.min_roi_width_ratio:
 return ValidationResult(
 valid=False,
 reason=f"ROI too narrow: {roi_w/frame_w:.2%} < {self.min_roi_width_ratio:.0%}",
 metrics=metrics
 )

 if roi_h / frame_h < self.min_roi_height_ratio:
 return ValidationResult(
 valid=False,
 reason=f"ROI too short: {roi_h/frame_h:.2%} < {self.min_roi_height_ratio:.0%}",
 metrics=metrics
 )

 # Check if ROI is bottom-heavy (head likely cut off)
 roi_center_y = (y1 + y2) / 2
 frame_center_y = frame_h / 2

 if y1 > frame_h * 0.4: # ROI starts in lower 60% of frame
 warnings.append(f"ROI bottom-heavy (y1={y1}, may cut off head/torso)")

 # Check aspect ratio (should be roughly portrait for person)
 aspect_ratio = roi_w / roi_h if roi_h > 0 else 0
 metrics["aspect_ratio"] = aspect_ratio

 if aspect_ratio > 2.0: # Very wide ROI
 warnings.append(f"ROI aspect ratio unusual: {aspect_ratio:.2f} (very wide)")

 # If keypoints provided, check critical visibility
 if keypoints is not None:
 kp_result = self._check_keypoint_visibility(keypoints, roi, frame_shape)
 if not kp_result.valid:
 return kp_result
 warnings.extend(kp_result.warnings)
 metrics.update(kp_result.metrics)

 return ValidationResult(
 valid=True,
 reason="ROI valid",
 warnings=warnings,
 metrics=metrics
 )

 def validate_keypoints(self,
 keypoints: List,
 activity_type: str = "posture") -> ValidationResult:
 """
 Validate keypoints for activity type.

 Args:
 keypoints: List of (x, y, confidence) tuples
 activity_type: Type of activity being monitored

 Returns:
 ValidationResult with validity and diagnostics
 """
 if keypoints is None or len(keypoints) < 5:
 return ValidationResult(
 valid=False,
 reason=f"Insufficient keypoints: {len(keypoints) if keypoints else 0} < 5"
 )

 metrics = {}
 warnings = []

 # Count valid keypoints
 valid_count = sum(1 for kp in keypoints
 if len(kp) >= 3 and kp[2] > self.min_keypoint_confidence)
 total_count = len(keypoints)
 valid_ratio = valid_count / total_count

 metrics["valid_keypoints"] = valid_count
 metrics["total_keypoints"] = total_count
 metrics["valid_ratio"] = valid_ratio

 # Check minimum valid ratio
 if valid_ratio < self.min_valid_keypoint_ratio:
 return ValidationResult(
 valid=False,
 reason=f"Low keypoint quality: {valid_ratio:.0%} valid < {self.min_valid_keypoint_ratio:.0%}",
 metrics=metrics
 )

 # Check activity-specific requirements
 required = KEYPOINT_REQUIREMENTS.get(activity_type, KEYPOINT_REQUIREMENTS["posture"])
 missing = []

 for kp_name in required:
 idx = KEYPOINT_INDICES.get(kp_name)
 if idx is not None and idx < len(keypoints):
 kp = keypoints[idx]
 if len(kp) < 3 or kp[2] < self.min_keypoint_confidence:
 missing.append(kp_name)

 if missing:
 missing_str = ", ".join(missing)
 if len(missing) > len(required) // 2:
 return ValidationResult(
 valid=False,
 reason=f"Missing critical keypoints for {activity_type}: {missing_str}",
 metrics=metrics
 )
 warnings.append(f"Missing some keypoints for {activity_type}: {missing_str}")

 # Calculate average confidence
 confidences = [kp[2] for kp in keypoints if len(kp) >= 3]
 if confidences:
 metrics["avg_confidence"] = np.mean(confidences)
 metrics["min_confidence"] = np.min(confidences)

 return ValidationResult(
 valid=True,
 reason="Keypoints valid",
 warnings=warnings,
 metrics=metrics
 )

 def validate_zoom_request(self,
 current_roi: Tuple[int, int, int, int],
 proposed_roi: Tuple[int, int, int, int],
 frame_shape: Tuple[int, int],
 keypoints: Optional[List] = None) -> ValidationResult:
 """
 Validate proposed zoom change.

 Args:
 current_roi: Current (x1, y1, x2, y2) or None for full frame
 proposed_roi: Proposed (x1, y1, x2, y2)
 frame_shape: (height, width) of frame
 keypoints: Current keypoints for visibility check

 Returns:
 ValidationResult indicating if zoom should be applied
 """
 # First validate the proposed ROI itself
 roi_result = self.validate_roi(proposed_roi, frame_shape, keypoints)
 if not roi_result.valid:
 return roi_result

 # Check if keypoints have critical points visible
 if keypoints is not None:
 # Must have head/shoulders for any zoom
 has_head = self._has_keypoint(keypoints, "nose")
 has_shoulders = (self._has_keypoint(keypoints, "left_shoulder") or
 self._has_keypoint(keypoints, "right_shoulder"))

 if not (has_head or has_shoulders):
 return ValidationResult(
 valid=False,
 reason="Cannot zoom: no head or shoulders detected",
 metrics=roi_result.metrics
 )

 return roi_result

 def _check_keypoint_visibility(self,
 keypoints: List,
 roi: Tuple[int, int, int, int],
 frame_shape: Tuple[int, int]) -> ValidationResult:
 """Check if critical keypoints are within ROI."""
 x1, y1, x2, y2 = roi
 frame_h, frame_w = frame_shape[:2]

 inside_count = 0
 critical_missing = []

 for i, kp in enumerate(keypoints):
 if len(kp) < 3 or kp[2] < self.min_keypoint_confidence:
 continue

 # Convert normalized coords to pixel coords
 kp_x = kp[0] * frame_w if kp[0] <= 1.0 else kp[0]
 kp_y = kp[1] * frame_h if kp[1] <= 1.0 else kp[1]

 # Check if inside ROI
 if x1 <= kp_x <= x2 and y1 <= kp_y <= y2:
 inside_count += 1
 elif i in [0, 5, 6, 11, 12]: # nose, shoulders, hips
 kp_name = [k for k, v in KEYPOINT_INDICES.items() if v == i]
 if kp_name:
 critical_missing.append(kp_name[0])

 metrics = {"keypoints_in_roi": inside_count}
 warnings = []

 if critical_missing:
 warnings.append(f"Critical keypoints outside ROI: {', '.join(critical_missing)}")

 return ValidationResult(valid=True, warnings=warnings, metrics=metrics)

 def _has_keypoint(self, keypoints: List, name: str) -> bool:
 """Check if keypoint exists with sufficient confidence."""
 idx = KEYPOINT_INDICES.get(name)
 if idx is None or idx >= len(keypoints):
 return False
 kp = keypoints[idx]
 return len(kp) >= 3 and kp[2] > self.min_keypoint_confidence

 def estimate_distance_band(self,
 keypoints: List,
 frame_shape: Tuple[int, int]) -> Optional[DistanceBand]:
 """
 Estimate camera distance band from body size in frame.

 Args:
 keypoints: Detected keypoints
 frame_shape: (height, width) of frame

 Returns:
 Estimated DistanceBand or None if cannot determine
 """
 if keypoints is None or len(keypoints) < 12:
 return None

 frame_h, frame_w = frame_shape[:2]

 # Calculate body height from shoulder to ankle (or hip if no ankle)
 shoulder_y = None
 foot_y = None

 # Get shoulder position
 for idx in [5, 6]: # left/right shoulder
 if idx < len(keypoints) and keypoints[idx][2] > self.min_keypoint_confidence:
 y = keypoints[idx][1]
 y = y * frame_h if y <= 1.0 else y
 shoulder_y = y
 break

 # Get foot/ankle position
 for idx in [15, 16, 13, 14, 11, 12]: # ankles, knees, hips
 if idx < len(keypoints) and keypoints[idx][2] > self.min_keypoint_confidence:
 y = keypoints[idx][1]
 y = y * frame_h if y <= 1.0 else y
 foot_y = y
 break

 if shoulder_y is None or foot_y is None:
 return None

 body_height = abs(foot_y - shoulder_y)
 body_height_ratio = body_height / frame_h

 # Match to distance band
 for band, config in DISTANCE_BANDS.items():
 if config.min_body_height_ratio <= body_height_ratio <= config.max_body_height_ratio:
 return band

 # Fallback based on thresholds
 if body_height_ratio > 0.55:
 return DistanceBand.CLOSE
 elif body_height_ratio > 0.35:
 return DistanceBand.MID
 else:
 return DistanceBand.FAR


# Singleton instance for pipeline use
_validator_instance = None

def get_validator() -> CaptureValidator:
 """Get singleton validator instance."""
 global _validator_instance
 if _validator_instance is None:
 _validator_instance = CaptureValidator()
 return _validator_instance
