# pipeline/validation/__init__.py
"""
Validation module for EAM capture and inference.

Provides:
- CaptureValidator: Frame/ROI validation against clinical protocol
- DistanceBand: Camera distance specifications
- ValidationResult: Structured validation output
"""

from .capture_validator import (
 CaptureValidator,
 ValidationResult,
 DistanceBand,
 DistanceConfig,
 DISTANCE_BANDS,
 KEYPOINT_REQUIREMENTS,
 KEYPOINT_INDICES,
 get_validator,
)

__all__ = [
 "CaptureValidator",
 "ValidationResult",
 "DistanceBand",
 "DistanceConfig",
 "DISTANCE_BANDS",
 "KEYPOINT_REQUIREMENTS",
 "KEYPOINT_INDICES",
 "get_validator",
]
