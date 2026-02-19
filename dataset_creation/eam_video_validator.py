#!/usr/bin/env python3
"""
EAM Video Validator
Validates video files against EAM dataset requirements.
Based on EAM video metadata analysis requirements.
"""

import cv2
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

log = logging.getLogger("eam_validator")


@dataclass
class EAMVideoRequirements:
 """EAM video requirements configuration."""
 resolution_width: int = 1920
 resolution_height: int = 1080
 fps_min: float = 28.0
 fps_max: float = 32.0
 fps_target: float = 30.0
 duration_min: float = 5.0 # seconds
 duration_max: float = 15.0 # seconds (for mask removal activities)
 aspect_ratio: float = 16.0 / 9.0
 format_extensions: List[str] = None
 audio_required: bool = False
 jpeg_quality: int = 90
 
 def __post_init__(self):
 if self.format_extensions is None:
 self.format_extensions = [".mp4", ".avi", ".mov", ".mkv"]


@dataclass
class VideoValidationResult:
 """Result of video validation."""
 video_path: str
 passed: bool
 checks: Dict[str, bool]
 properties: Dict[str, float]
 errors: List[str]
 warnings: List[str]
 grade: str # A+, A, B, C, D, F
 
 def to_dict(self) -> Dict:
 """Convert to dictionary."""
 return {
 "video_path": self.video_path,
 "passed": self.passed,
 "checks": self.checks,
 "properties": self.properties,
 "errors": self.errors,
 "warnings": self.warnings,
 "grade": self.grade
 }


class EAMVideoValidator:
 """
 Validates videos against EAM dataset requirements.
 """
 
 def __init__(self, requirements: Optional[EAMVideoRequirements] = None):
 """
 Initialize validator.
 
 Args:
 requirements: Custom requirements (uses defaults if None)
 """
 self.requirements = requirements or EAMVideoRequirements()
 
 def validate_video(self, video_path: str) -> VideoValidationResult:
 """
 Validate a video file against EAM requirements.
 
 Args:
 video_path: Path to video file
 
 Returns:
 VideoValidationResult with validation details
 """
 video_path = Path(video_path)
 
 if not video_path.exists():
 return VideoValidationResult(
 video_path=str(video_path),
 passed=False,
 checks={},
 properties={},
 errors=[f"File not found: {video_path}"],
 warnings=[],
 grade="F"
 )
 
 # Open video
 cap = cv2.VideoCapture(str(video_path))
 if not cap.isOpened():
 return VideoValidationResult(
 video_path=str(video_path),
 passed=False,
 checks={},
 properties={},
 errors=[f"Cannot open video: {video_path}"],
 warnings=[],
 grade="F"
 )
 
 # Get video properties
 width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 fps = cap.get(cv2.CAP_PROP_FPS)
 frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 duration = frame_count / fps if fps > 0 else 0
 file_size_mb = video_path.stat().st_size / (1024 * 1024)
 
 # Check audio channels (approximate - OpenCV doesn't directly provide this)
 # We'll check file size as proxy (videos with audio are larger)
 has_audio = self._check_audio_presence(video_path, duration)
 
 cap.release()
 
 # Calculate aspect ratio
 aspect_ratio = width / height if height > 0 else 0
 
 # Perform validation checks
 # Audio check: 
 # - If audio_required=False: pass if no audio (not has_audio)
 # - If audio_required=True: pass if audio exists (has_audio)
 audio_check_passed = (not has_audio) if not self.requirements.audio_required else has_audio
 
 checks = {
 "resolution_1920x1080": width == self.requirements.resolution_width and 
 height == self.requirements.resolution_height,
 "frame_rate_acceptable": self.requirements.fps_min <= fps <= self.requirements.fps_max,
 "duration_valid": self.requirements.duration_min <= duration <= self.requirements.duration_max,
 "aspect_ratio_16_9": abs(aspect_ratio - self.requirements.aspect_ratio) < 0.01,
 "format_supported": video_path.suffix.lower() in self.requirements.format_extensions,
 "no_audio": audio_check_passed,
 }
 
 # Collect errors and warnings
 errors = []
 warnings = []
 
 if not checks["resolution_1920x1080"]:
 errors.append(f"Resolution {width}×{height} != required {self.requirements.resolution_width}×{self.requirements.resolution_height}")
 
 if not checks["frame_rate_acceptable"]:
 warnings.append(f"Frame rate {fps:.2f} fps outside ideal range ({self.requirements.fps_min}-{self.requirements.fps_max})")
 
 if not checks["duration_valid"]:
 errors.append(f"Duration {duration:.2f}s outside required range ({self.requirements.duration_min}-{self.requirements.duration_max}s)")
 
 if not checks["aspect_ratio_16_9"]:
 errors.append(f"Aspect ratio {aspect_ratio:.2f} != required {self.requirements.aspect_ratio:.2f}")
 
 if not checks["format_supported"]:
 warnings.append(f"Format {video_path.suffix} may not be optimal")
 
 if has_audio and not self.requirements.audio_required:
 warnings.append("Audio present but not required (wastes storage space)")
 
 if not has_audio and self.requirements.audio_required:
 errors.append("Audio required but not detected in video")
 
 # Calculate grade
 passed = all(checks.values())
 grade = self._calculate_grade(checks, errors, warnings)
 
 properties = {
 "width": width,
 "height": height,
 "fps": fps,
 "frame_count": frame_count,
 "duration": duration,
 "aspect_ratio": aspect_ratio,
 "file_size_mb": file_size_mb,
 "has_audio": has_audio
 }
 
 return VideoValidationResult(
 video_path=str(video_path),
 passed=passed,
 checks=checks,
 properties=properties,
 errors=errors,
 warnings=warnings,
 grade=grade
 )
 
 def _check_audio_presence(self, video_path: Path, duration: float) -> bool:
 """
 Check if video has audio track.
 Uses file size heuristic: videos with audio are typically larger.
 """
 file_size_mb = video_path.stat().st_size / (1024 * 1024)
 
 # Heuristic: For 1920x1080 video at 30fps, ~1MB per second without audio
 # With audio (stereo), expect ~1.2-1.5MB per second
 expected_size_no_audio = duration * 1.0 # MB
 expected_size_with_audio = duration * 1.3 # MB
 
 # If file size is significantly larger than expected, likely has audio
 return file_size_mb > expected_size_with_audio
 
 def _calculate_grade(self, checks: Dict[str, bool], errors: List[str], warnings: List[str]) -> str:
 """Calculate overall grade."""
 critical_checks = ["resolution_1920x1080", "duration_valid", "aspect_ratio_16_9"]
 important_checks = ["frame_rate_acceptable", "format_supported"]
 
 # Check critical requirements
 critical_passed = all(checks.get(c, False) for c in critical_checks)
 important_passed = all(checks.get(c, False) for c in important_checks)
 
 if critical_passed and important_passed and len(errors) == 0:
 if len(warnings) == 0:
 return "A+"
 elif len(warnings) <= 1:
 return "A"
 else:
 return "B"
 elif critical_passed and len(errors) == 0:
 return "C"
 elif critical_passed:
 return "D"
 else:
 return "F"
 
 def print_validation_report(self, result: VideoValidationResult):
 """Print formatted validation report."""
 print("=" * 60)
 print("EAM VIDEO VALIDATION REPORT")
 print("=" * 60)
 print(f"\n Video: {result.video_path}")
 print(f" Grade: {result.grade}")
 print(f" Status: {'PASSED' if result.passed else 'FAILED'}")
 
 print(f"\n Video Properties:")
 props = result.properties
 print(f" Resolution: {int(props['width'])}×{int(props['height'])}")
 print(f" Frame rate: {props['fps']:.2f} fps")
 print(f" Duration: {props['duration']:.2f} seconds")
 print(f" Frame count: {int(props['frame_count'])}")
 print(f" Aspect ratio: {props['aspect_ratio']:.2f}")
 print(f" File size: {props['file_size_mb']:.2f} MB")
 print(f" Audio: {'Present' if props.get('has_audio') else 'Not detected'}")
 
 print(f"\n Validation Checks:")
 for check_name, passed in result.checks.items():
 status = "" if passed else ""
 check_display = check_name.replace("_", " ").title()
 print(f" {status} {check_display}: {passed}")
 
 if result.errors:
 print(f"\n Errors:")
 for error in result.errors:
 print(f" • {error}")
 
 if result.warnings:
 print(f"\n Warnings:")
 for warning in result.warnings:
 print(f" • {warning}")
 
 print("\n" + "=" * 60)
 
 def validate_batch(self, video_dir: str) -> List[VideoValidationResult]:
 """
 Validate all videos in a directory.
 
 Args:
 video_dir: Directory containing videos
 
 Returns:
 List of validation results
 """
 video_dir = Path(video_dir)
 results = []
 
 # Find all video files
 video_files = []
 for ext in self.requirements.format_extensions:
 video_files.extend(video_dir.glob(f"*{ext}"))
 video_files.extend(video_dir.glob(f"*{ext.upper()}"))
 
 log.info(f"Found {len(video_files)} video files to validate")
 
 for video_file in video_files:
 result = self.validate_video(str(video_file))
 results.append(result)
 
 return results


def validate_eam_video(video_path: str, print_report: bool = True) -> VideoValidationResult:
 """
 Convenience function to validate a video.
 
 Args:
 video_path: Path to video file
 print_report: Whether to print validation report
 
 Returns:
 VideoValidationResult
 """
 validator = EAMVideoValidator()
 result = validator.validate_video(video_path)
 
 if print_report:
 validator.print_validation_report(result)
 
 return result


if __name__ == "__main__":
 import argparse
 
 parser = argparse.ArgumentParser(description="Validate video against EAM requirements")
 parser.add_argument("video", type=str, help="Path to video file")
 parser.add_argument("--no-report", action="store_true", help="Don't print report")
 
 args = parser.parse_args()
 
 result = validate_eam_video(args.video, print_report=not args.no_report)
 
 # Exit with error code if validation failed
 exit(0 if result.passed else 1)
