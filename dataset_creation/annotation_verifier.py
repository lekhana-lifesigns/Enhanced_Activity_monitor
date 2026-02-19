#!/usr/bin/env python3
"""
Annotation Verifier
Processes clinical videos and images through YOLO detection + pose estimation,
generates annotated frames with overlays, and enforces per-frame verification
before saving to the training set.

Workflow:
 1. Input: video or image directory
 2. Run YOLO detection (person bbox) + YOLO-pose (17 keypoints)
 3. Draw overlays: bbox, skeleton, label, quality score
 4. Auto-QA: accept/reject/flag frames based on quality gates
 5. Save annotated frames + verification manifest (JSON)
 6. Export only verified frames to training-ready NPZ

Usage:
 # Process a single video
 python annotation_verifier.py --video clinical_validation_data/video\ for\ detection/COUGHING\ -\ 1.8\ M.mp4

 # Process all videos in directory
 python annotation_verifier.py --video-dir clinical_validation_data/video\ for\ detection/

 # Process standalone images
 python annotation_verifier.py --image-dir clinical_validation_data/video\ for\ detection/ --pattern "*.jpg"

 # Export verified data only
 python annotation_verifier.py --export-verified training_data/verified/manifest.json --output training_data/verified_npz/
"""

import sys
import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("annotation_verifier")

# 12-class schema (must match temporal_annotation_schema.yaml)
CLASS_NAMES = [
 "lying_still", "lying_restless", "turning_in_bed",
 "sitting_stable", "sitting_unstable", "bed_exit",
 "standing", "walking", "agitated",
 "convulsive", "fallen", "occluded",
]

CLASS_COLORS = {
 "lying_still": (33, 150, 243), # Blue
 "lying_restless": (255, 152, 0), # Orange
 "turning_in_bed": (156, 39, 176), # Purple
 "sitting_stable": (76, 175, 80), # Green
 "sitting_unstable": (255, 193, 7), # Amber
 "bed_exit": (244, 67, 54), # Red
 "standing": (0, 188, 212), # Cyan
 "walking": (0, 150, 136), # Teal
 "agitated": (233, 30, 99), # Pink
 "convulsive": (213, 0, 0), # Dark Red
 "fallen": (183, 28, 28), # Dark Red variant
 "occluded": (158, 158, 158), # Grey
}

# Video filename -> label mapping (from temporal_annotation_schema.yaml)
VIDEO_LABEL_MAP = {
 "AGITATED": "agitated",
 "RESTLESS": "lying_restless",
 "THRASHING": "agitated",
 "COUGHING": "lying_restless",
 "HAND TO FACE": "lying_restless",
 "HAND TO CHEST TUBE": "agitated",
 "HAND TO IV": "lying_restless",
 "HAND TO IV LINE": "lying_restless",
 "HAND TO MASK": "agitated",
 "PULLING AT TUBES": "agitated",
 "REMOVAL OF MASK": "agitated",
 "LABOURED BREATHING": "lying_restless",
 "RESPIRATORY DISTRESS": "agitated",
}

# Image filename prefix -> label mapping
IMAGE_LABEL_MAP = {
 "cough": "lying_restless",
 "agitat": "agitated",
 "agiat": "agitated", # Handle typo in agiated06.webp
 "restless": "lying_restless",
 "fall": "fallen",
 "lying": "lying_still",
 "sitting": "sitting_stable",
 "standing": "standing",
 "walking": "walking",
}

# COCO skeleton connections for visualization
SKELETON_EDGES = [
 (0, 1), (0, 2), (1, 3), (2, 4), # Head
 (5, 6), # Shoulders
 (5, 7), (7, 9), # Left arm
 (6, 8), (8, 10), # Right arm
 (5, 11), (6, 12), # Torso
 (11, 12), # Hips
 (11, 13), (13, 15), # Left leg
 (12, 14), (14, 16), # Right leg
]


@dataclass
class FrameVerification:
 """Verification result for a single frame."""
 frame_idx: int
 frame_file: str
 label: str
 label_id: int
 status: str # "accepted", "rejected", "review", "relabeled"
 person_detected: bool
 person_confidence: float
 num_keypoints_valid: int # conf > 0.3
 num_keypoints_high: int # conf > 0.5
 keypoint_avg_conf: float
 bbox: List[float] = field(default_factory=list) # [x1, y1, x2, y2]
 quality_score: float = 0.0
 rejection_reason: str = ""
 notes: str = ""

 def to_dict(self) -> Dict:
 return asdict(self)


@dataclass
class VerificationManifest:
 """Complete verification manifest for a video/image set."""
 source_path: str
 source_type: str # "video" or "images"
 created_at: str
 label: str
 total_frames: int
 accepted: int
 rejected: int
 review: int
 avg_quality: float
 frames: List[Dict] = field(default_factory=list)

 def to_dict(self) -> Dict:
 return asdict(self)


def infer_label_from_filename(filename: str) -> Tuple[str, int]:
 """
 Infer activity label from video/image filename.

 Returns:
 (label_name, label_id) tuple. label_id = -1 if unknown.
 """
 name_upper = Path(filename).stem.upper()

 # Check video mapping first (longer names first for specificity)
 for prefix in sorted(VIDEO_LABEL_MAP.keys(), key=len, reverse=True):
 if prefix in name_upper:
 label = VIDEO_LABEL_MAP[prefix]
 return label, CLASS_NAMES.index(label)

 # Check image prefix mapping
 name_lower = Path(filename).stem.lower()
 for prefix, label in IMAGE_LABEL_MAP.items():
 if name_lower.startswith(prefix):
 return label, CLASS_NAMES.index(label)

 return "unknown", -1


def compute_quality_score(
 person_conf: float,
 num_kps_valid: int,
 num_kps_high: int,
 avg_kp_conf: float,
 frame_brightness: float = 0.5
) -> float:
 """
 Compute a composite quality score (0-1) for a frame.

 Weights:
 - Person detection confidence: 25%
 - Keypoint count (valid): 25%
 - Keypoint count (high-conf): 20%
 - Average keypoint confidence: 20%
 - Frame brightness: 10%
 """
 kp_valid_score = min(num_kps_valid / 12.0, 1.0)
 kp_high_score = min(num_kps_high / 10.0, 1.0)
 brightness_score = 1.0 - abs(frame_brightness - 0.5) * 2 # Penalize very dark/bright

 score = (
 0.25 * person_conf +
 0.25 * kp_valid_score +
 0.20 * kp_high_score +
 0.20 * avg_kp_conf +
 0.10 * brightness_score
 )
 return round(min(max(score, 0.0), 1.0), 3)


def classify_frame_status(quality_score: float, num_kps_valid: int, person_conf: float) -> str:
 """
 Auto-classify frame verification status based on quality gates.

 Returns: "accepted", "rejected", or "review"
 """
 # Auto-reject: terrible quality
 if not person_conf or person_conf < 0.25:
 return "rejected"
 if num_kps_valid < 5:
 return "rejected"

 # Auto-accept: high quality
 if quality_score >= 0.65 and num_kps_valid >= 10 and person_conf >= 0.6:
 return "accepted"

 # Everything else needs manual review
 return "review"


def draw_skeleton_overlay(frame, keypoints, bbox, label, quality_score, status, person_conf):
 """
 Draw detection overlay on frame: bbox, skeleton, label, quality info.

 Args:
 frame: BGR image (numpy array, modified in-place)
 keypoints: (17, 3) array [x, y, conf] in pixel coordinates
 bbox: [x1, y1, x2, y2] in pixel coordinates
 label: Activity label string
 quality_score: Float 0-1
 status: "accepted", "rejected", "review"
 person_conf: Person detection confidence
 """
 import cv2

 h, w = frame.shape[:2]
 color = CLASS_COLORS.get(label, (200, 200, 200))
 # BGR conversion
 color_bgr = (color[2], color[1], color[0])

 status_colors = {
 "accepted": (0, 200, 0), # Green
 "rejected": (0, 0, 200), # Red
 "review": (0, 200, 255), # Yellow
 }
 status_color = status_colors.get(status, (200, 200, 200))

 # Draw bounding box
 if bbox and len(bbox) == 4:
 x1, y1, x2, y2 = [int(v) for v in bbox]
 cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

 # Draw skeleton
 if keypoints is not None:
 # Draw keypoints
 for i, (x, y, conf) in enumerate(keypoints):
 if conf > 0.3:
 px, py = int(x), int(y)
 radius = 4 if conf > 0.5 else 2
 kp_color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
 cv2.circle(frame, (px, py), radius, kp_color, -1)

 # Draw skeleton edges
 for i, j in SKELETON_EDGES:
 if keypoints[i][2] > 0.3 and keypoints[j][2] > 0.3:
 pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
 pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
 cv2.line(frame, pt1, pt2, color_bgr, 2)

 # Draw info panel (top-left)
 panel_h = 90
 cv2.rectangle(frame, (0, 0), (350, panel_h), (0, 0, 0), -1)
 cv2.rectangle(frame, (0, 0), (350, panel_h), status_color, 2)

 cv2.putText(frame, f"Label: {label}", (10, 20),
 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
 cv2.putText(frame, f"Quality: {quality_score:.2f} Person: {person_conf:.2f}", (10, 42),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 cv2.putText(frame, f"Keypoints: {int(sum(1 for k in keypoints if k[2] > 0.3))}/17 valid", (10, 62),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 cv2.putText(frame, f"Status: {status.upper()}", (10, 82),
 cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

 return frame


def process_video(
 video_path: str,
 output_dir: str,
 label_override: str = None,
 fps: int = 15,
 save_all_frames: bool = True
) -> VerificationManifest:
 """
 Process a video: detect persons + pose per frame, verify, save annotated frames.

 Args:
 video_path: Path to input video
 output_dir: Directory to save annotated frames + manifest
 label_override: Override auto-detected label
 fps: Target extraction FPS
 save_all_frames: Save all frames (True) or only accepted (False)

 Returns:
 VerificationManifest with per-frame results
 """
 import cv2
 from ultralytics import YOLO

 video_path = Path(video_path)
 if not video_path.exists():
 raise FileNotFoundError(f"Video not found: {video_path}")

 # Infer label
 if label_override:
 label = label_override
 label_id = CLASS_NAMES.index(label) if label in CLASS_NAMES else -1
 else:
 label, label_id = infer_label_from_filename(video_path.name)

 log.info("Processing video: %s -> label: %s (id: %d)", video_path.name, label, label_id)

 # Setup output
 video_id = video_path.stem.replace(" ", "_").replace("-", "_").lower()
 frame_dir = Path(output_dir) / video_id / "frames"
 annotated_dir = Path(output_dir) / video_id / "annotated"
 frame_dir.mkdir(parents=True, exist_ok=True)
 annotated_dir.mkdir(parents=True, exist_ok=True)

 # Load models
 pose_model = YOLO("yolo11n-pose.pt")
 log.info("YOLO11-pose model loaded")

 # Open video
 cap = cv2.VideoCapture(str(video_path))
 if not cap.isOpened():
 raise ValueError(f"Cannot open video: {video_path}")

 video_fps = cap.get(cv2.CAP_PROP_FPS)
 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 frame_skip = max(1, round(video_fps / fps))

 log.info("Video: %.1f FPS, %d frames, extracting at %d FPS (skip=%d)",
 video_fps, total_frames, fps, frame_skip)

 frame_verifications = []
 frame_idx = 0
 extract_idx = 0

 while True:
 ret, frame = cap.read()
 if not ret:
 break

 if frame_idx % frame_skip != 0:
 frame_idx += 1
 continue

 h, w = frame.shape[:2]

 # Run pose estimation
 results = pose_model(frame, verbose=False)
 res = results[0]

 person_detected = False
 person_conf = 0.0
 keypoints = np.zeros((17, 3), dtype=np.float32)
 bbox = []

 if hasattr(res, 'keypoints') and res.keypoints is not None and len(res.keypoints) > 0:
 kps_data = res.keypoints.data # (N_persons, 17, 3)
 boxes_data = res.boxes

 if len(kps_data) > 0:
 # Select person with highest mean keypoint confidence
 mean_confs = kps_data[:, :, 2].mean(dim=1)
 best_idx = mean_confs.argmax().item()
 keypoints = kps_data[best_idx].cpu().numpy() # (17, 3) in pixel coords
 person_detected = True

 if boxes_data is not None and len(boxes_data) > 0:
 person_conf = float(boxes_data.conf[best_idx].cpu())
 box = boxes_data.xyxy[best_idx].cpu().numpy()
 bbox = box.tolist()
 else:
 person_conf = float(mean_confs[best_idx].cpu())

 # Compute quality metrics
 num_kps_valid = int(np.sum(keypoints[:, 2] > 0.3))
 num_kps_high = int(np.sum(keypoints[:, 2] > 0.5))
 avg_kp_conf = float(np.mean(keypoints[:, 2])) if person_detected else 0.0

 # Frame brightness
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
 brightness = float(np.mean(gray)) / 255.0

 quality_score = compute_quality_score(
 person_conf, num_kps_valid, num_kps_high, avg_kp_conf, brightness
 )

 # Classify status
 status = classify_frame_status(quality_score, num_kps_valid, person_conf)

 rejection_reason = ""
 if status == "rejected":
 if person_conf < 0.25:
 rejection_reason = f"person_conf too low ({person_conf:.2f})"
 elif num_kps_valid < 5:
 rejection_reason = f"too few keypoints ({num_kps_valid}/17)"

 # Save raw frame
 frame_filename = f"{video_id}_frame_{extract_idx:06d}.jpg"
 cv2.imwrite(str(frame_dir / frame_filename), frame)

 # Draw overlay and save annotated frame
 annotated = frame.copy()
 draw_skeleton_overlay(
 annotated, keypoints, bbox, label, quality_score, status, person_conf
 )
 annotated_filename = f"{video_id}_annotated_{extract_idx:06d}.jpg"

 if save_all_frames or status == "accepted":
 cv2.imwrite(str(annotated_dir / annotated_filename), annotated)

 # Record verification
 verification = FrameVerification(
 frame_idx=extract_idx,
 frame_file=frame_filename,
 label=label,
 label_id=label_id,
 status=status,
 person_detected=person_detected,
 person_confidence=round(person_conf, 4),
 num_keypoints_valid=num_kps_valid,
 num_keypoints_high=num_kps_high,
 keypoint_avg_conf=round(avg_kp_conf, 4),
 bbox=[round(v, 1) for v in bbox],
 quality_score=quality_score,
 rejection_reason=rejection_reason,
 )
 frame_verifications.append(verification)

 extract_idx += 1
 frame_idx += 1

 cap.release()

 # Build manifest
 accepted_count = sum(1 for f in frame_verifications if f.status == "accepted")
 rejected_count = sum(1 for f in frame_verifications if f.status == "rejected")
 review_count = sum(1 for f in frame_verifications if f.status == "review")
 avg_quality = np.mean([f.quality_score for f in frame_verifications]) if frame_verifications else 0.0

 manifest = VerificationManifest(
 source_path=str(video_path),
 source_type="video",
 created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
 label=label,
 total_frames=len(frame_verifications),
 accepted=accepted_count,
 rejected=rejected_count,
 review=review_count,
 avg_quality=round(float(avg_quality), 3),
 frames=[f.to_dict() for f in frame_verifications],
 )

 # Save manifest
 manifest_path = Path(output_dir) / video_id / "manifest.json"
 with open(manifest_path, 'w') as f:
 json.dump(manifest.to_dict(), f, indent=2)

 log.info("=" * 60)
 log.info("Verification complete: %s", video_path.name)
 log.info(" Total frames: %d", len(frame_verifications))
 log.info(" Accepted: %d (%.1f%%)", accepted_count,
 100 * accepted_count / max(len(frame_verifications), 1))
 log.info(" Rejected: %d (%.1f%%)", rejected_count,
 100 * rejected_count / max(len(frame_verifications), 1))
 log.info(" Review needed: %d (%.1f%%)", review_count,
 100 * review_count / max(len(frame_verifications), 1))
 log.info(" Avg quality: %.3f", avg_quality)
 log.info(" Manifest: %s", manifest_path)
 log.info("=" * 60)

 return manifest


def process_images(
 image_dir: str,
 output_dir: str,
 pattern: str = "*.jpg",
 label_override: str = None
) -> VerificationManifest:
 """
 Process a directory of images: detect persons + pose, verify, save annotated.

 Args:
 image_dir: Directory containing images
 output_dir: Output directory
 pattern: Glob pattern for images
 label_override: Override auto-detected label

 Returns:
 VerificationManifest with per-image results
 """
 import cv2
 from ultralytics import YOLO

 image_dir = Path(image_dir)
 extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
 image_files = []
 for ext in extensions:
 image_files.extend(sorted(image_dir.glob(ext)))

 if not image_files:
 log.warning("No images found in %s", image_dir)
 return None

 log.info("Found %d images in %s", len(image_files), image_dir)

 # Setup output
 annotated_dir = Path(output_dir) / "annotated"
 annotated_dir.mkdir(parents=True, exist_ok=True)

 # Load model
 pose_model = YOLO("yolo11n-pose.pt")
 log.info("YOLO11-pose model loaded")

 frame_verifications = []

 for idx, img_path in enumerate(image_files):
 frame = cv2.imread(str(img_path))
 if frame is None:
 log.warning("Cannot read image: %s", img_path)
 continue

 # Infer label per image
 if label_override:
 label = label_override
 label_id = CLASS_NAMES.index(label) if label in CLASS_NAMES else -1
 else:
 label, label_id = infer_label_from_filename(img_path.name)

 h, w = frame.shape[:2]

 # Run pose
 results = pose_model(frame, verbose=False)
 res = results[0]

 person_detected = False
 person_conf = 0.0
 keypoints = np.zeros((17, 3), dtype=np.float32)
 bbox = []

 if hasattr(res, 'keypoints') and res.keypoints is not None and len(res.keypoints) > 0:
 kps_data = res.keypoints.data
 boxes_data = res.boxes

 if len(kps_data) > 0:
 mean_confs = kps_data[:, :, 2].mean(dim=1)
 best_idx = mean_confs.argmax().item()
 keypoints = kps_data[best_idx].cpu().numpy()
 person_detected = True

 if boxes_data is not None and len(boxes_data) > 0:
 person_conf = float(boxes_data.conf[best_idx].cpu())
 box = boxes_data.xyxy[best_idx].cpu().numpy()
 bbox = box.tolist()
 else:
 person_conf = float(mean_confs[best_idx].cpu())

 num_kps_valid = int(np.sum(keypoints[:, 2] > 0.3))
 num_kps_high = int(np.sum(keypoints[:, 2] > 0.5))
 avg_kp_conf = float(np.mean(keypoints[:, 2])) if person_detected else 0.0

 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
 brightness = float(np.mean(gray)) / 255.0

 quality_score = compute_quality_score(
 person_conf, num_kps_valid, num_kps_high, avg_kp_conf, brightness
 )
 status = classify_frame_status(quality_score, num_kps_valid, person_conf)

 rejection_reason = ""
 if status == "rejected":
 if person_conf < 0.25:
 rejection_reason = f"person_conf too low ({person_conf:.2f})"
 elif num_kps_valid < 5:
 rejection_reason = f"too few keypoints ({num_kps_valid}/17)"

 # Draw overlay
 annotated = frame.copy()
 draw_skeleton_overlay(
 annotated, keypoints, bbox, label, quality_score, status, person_conf
 )
 out_name = f"{img_path.stem}_annotated{img_path.suffix}"
 cv2.imwrite(str(annotated_dir / out_name), annotated)

 verification = FrameVerification(
 frame_idx=idx,
 frame_file=img_path.name,
 label=label,
 label_id=label_id,
 status=status,
 person_detected=person_detected,
 person_confidence=round(person_conf, 4),
 num_keypoints_valid=num_kps_valid,
 num_keypoints_high=num_kps_high,
 keypoint_avg_conf=round(avg_kp_conf, 4),
 bbox=[round(v, 1) for v in bbox],
 quality_score=quality_score,
 rejection_reason=rejection_reason,
 )
 frame_verifications.append(verification)

 if (idx + 1) % 10 == 0:
 log.info(" Processed %d/%d images...", idx + 1, len(image_files))

 # Build manifest
 accepted_count = sum(1 for f in frame_verifications if f.status == "accepted")
 rejected_count = sum(1 for f in frame_verifications if f.status == "rejected")
 review_count = sum(1 for f in frame_verifications if f.status == "review")
 avg_quality = np.mean([f.quality_score for f in frame_verifications]) if frame_verifications else 0.0

 manifest = VerificationManifest(
 source_path=str(image_dir),
 source_type="images",
 created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
 label="mixed",
 total_frames=len(frame_verifications),
 accepted=accepted_count,
 rejected=rejected_count,
 review=review_count,
 avg_quality=round(float(avg_quality), 3),
 frames=[f.to_dict() for f in frame_verifications],
 )

 manifest_path = Path(output_dir) / "manifest.json"
 with open(manifest_path, 'w') as f:
 json.dump(manifest.to_dict(), f, indent=2)

 log.info("=" * 60)
 log.info("Image verification complete: %s", image_dir)
 log.info(" Total images: %d", len(frame_verifications))
 log.info(" Accepted: %d | Rejected: %d | Review: %d", accepted_count, rejected_count, review_count)
 log.info(" Avg quality: %.3f", avg_quality)
 log.info("=" * 60)

 return manifest


def process_all_clinical_videos(video_dir: str, output_dir: str) -> List[VerificationManifest]:
 """
 Batch-process all clinical validation videos.

 Args:
 video_dir: Directory containing .mp4 files
 output_dir: Output directory for verified data

 Returns:
 List of VerificationManifest objects
 """
 video_dir = Path(video_dir)
 video_files = sorted(
 list(video_dir.glob("*.mp4")) +
 list(video_dir.glob("*.avi")) +
 list(video_dir.glob("*.mov"))
 )

 if not video_files:
 log.error("No video files found in %s", video_dir)
 return []

 log.info("Found %d video files to process", len(video_files))

 manifests = []
 for i, video_path in enumerate(video_files, 1):
 log.info("\n[%d/%d] Processing: %s", i, len(video_files), video_path.name)
 try:
 manifest = process_video(str(video_path), output_dir)
 manifests.append(manifest)
 except Exception as e:
 log.exception("Failed to process %s: %s", video_path.name, e)

 # Print batch summary
 total_frames = sum(m.total_frames for m in manifests)
 total_accepted = sum(m.accepted for m in manifests)
 total_rejected = sum(m.rejected for m in manifests)
 total_review = sum(m.review for m in manifests)

 log.info("\n" + "=" * 60)
 log.info("BATCH SUMMARY")
 log.info("=" * 60)
 log.info("Videos processed: %d", len(manifests))
 log.info("Total frames: %d", total_frames)
 log.info("Accepted: %d (%.1f%%)", total_accepted, 100 * total_accepted / max(total_frames, 1))
 log.info("Rejected: %d (%.1f%%)", total_rejected, 100 * total_rejected / max(total_frames, 1))
 log.info("Review needed: %d (%.1f%%)", total_review, 100 * total_review / max(total_frames, 1))

 # Per-label distribution
 label_counts = Counter()
 for m in manifests:
 for f in m.frames:
 if f["status"] == "accepted":
 label_counts[f["label"]] += 1

 log.info("\nAccepted frames by label:")
 for label_name in CLASS_NAMES:
 count = label_counts.get(label_name, 0)
 if count > 0:
 log.info(" %-18s: %d", label_name, count)

 # Save batch manifest
 batch_manifest_path = Path(output_dir) / "batch_manifest.json"
 with open(batch_manifest_path, 'w') as f:
 json.dump({
 "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
 "total_videos": len(manifests),
 "total_frames": total_frames,
 "total_accepted": total_accepted,
 "total_rejected": total_rejected,
 "total_review": total_review,
 "label_distribution": dict(label_counts),
 "videos": [m.to_dict() for m in manifests],
 }, f, indent=2)

 log.info("Batch manifest: %s", batch_manifest_path)

 return manifests


def export_verified_to_npz(
 manifest_path: str,
 output_dir: str,
 status_filter: str = "accepted"
):
 """
 Export verified frames from manifest to training-ready NPZ.
 Only frames matching status_filter are exported.

 Args:
 manifest_path: Path to verification manifest JSON
 output_dir: Output directory for NPZ files
 status_filter: Only export frames with this status ("accepted" or "accepted,review")
 """
 from dataset_creation.temporal_annotation_converter import compute_icu_features

 manifest_path = Path(manifest_path)
 with open(manifest_path) as f:
 manifest = json.load(f)

 allowed_statuses = set(status_filter.split(","))
 log.info("Exporting frames with status in: %s", allowed_statuses)

 # Load YOLO for keypoint re-extraction from saved frames
 import cv2
 from ultralytics import YOLO
 pose_model = YOLO("yolo11n-pose.pt")

 source_dir = manifest_path.parent
 frame_dir = source_dir / "frames"

 all_keypoints = []
 all_labels = []
 exported_count = 0

 for frame_info in manifest.get("frames", []):
 if frame_info["status"] not in allowed_statuses:
 continue

 label_id = frame_info["label_id"]
 if label_id < 0:
 continue

 frame_file = frame_dir / frame_info["frame_file"]
 if not frame_file.exists():
 continue

 # Re-extract keypoints from raw frame
 frame = cv2.imread(str(frame_file))
 if frame is None:
 continue

 results = pose_model(frame, verbose=False)
 res = results[0]

 if hasattr(res, 'keypoints') and res.keypoints is not None and len(res.keypoints) > 0:
 kps_data = res.keypoints.data
 if len(kps_data) > 0:
 mean_confs = kps_data[:, :, 2].mean(dim=1)
 best_idx = mean_confs.argmax().item()
 kps = kps_data[best_idx].cpu().numpy()

 # Normalize to [0, 1]
 h, w = frame.shape[:2]
 kps[:, 0] /= w
 kps[:, 1] /= h

 all_keypoints.append(kps)
 all_labels.append(label_id)
 exported_count += 1

 if not all_keypoints:
 log.warning("No valid frames to export")
 return

 keypoints = np.array(all_keypoints, dtype=np.float32) # (N, 17, 3)
 labels = np.array(all_labels, dtype=np.int64)

 # Compute features for each frame (single-frame window)
 features = np.zeros((len(keypoints), 9), dtype=np.float32)
 for i in range(len(keypoints)):
 # Use a small window around this frame for feature computation
 start = max(0, i - 5)
 end = min(len(keypoints), i + 6)
 window = keypoints[start:end]
 window_features = compute_icu_features(window)
 # Take the features for the center frame
 center_idx = min(i - start, len(window_features) - 1)
 features[i] = window_features[center_idx]

 # Save NPZ
 output_path = Path(output_dir)
 output_path.mkdir(parents=True, exist_ok=True)

 source_name = manifest.get("label", "unknown").replace(" ", "_")
 npz_path = output_path / f"verified_{source_name}.npz"
 np.savez_compressed(
 npz_path,
 features=features,
 labels=labels,
 keypoints=keypoints,
 class_names=np.array(CLASS_NAMES),
 schema_version="1.0.0",
 )

 log.info("Exported %d verified frames to %s", exported_count, npz_path)
 log.info("Label distribution:")
 for cls_id in range(len(CLASS_NAMES)):
 count = int((labels == cls_id).sum())
 if count > 0:
 log.info(" [%2d] %-18s: %d", cls_id, CLASS_NAMES[cls_id], count)


def main():
 parser = argparse.ArgumentParser(
 description="Annotation Verifier - per-frame QA for clinical training data"
 )
 parser.add_argument("--video", type=str, help="Process a single video file")
 parser.add_argument("--video-dir", type=str, help="Process all videos in directory")
 parser.add_argument("--image-dir", type=str, help="Process images in directory")
 parser.add_argument("--output", type=str, default="training_data/verified",
 help="Output directory (default: training_data/verified)")
 parser.add_argument("--label", type=str, help="Override label for all frames")
 parser.add_argument("--fps", type=int, default=15, help="Extraction FPS for videos")
 parser.add_argument("--export-verified", type=str,
 help="Export verified frames from manifest to NPZ")
 parser.add_argument("--export-output", type=str, default="training_data/verified_npz",
 help="Output directory for exported NPZ")
 parser.add_argument("--status-filter", type=str, default="accepted",
 help="Status filter for export (default: accepted)")

 args = parser.parse_args()

 if args.export_verified:
 export_verified_to_npz(args.export_verified, args.export_output, args.status_filter)
 elif args.video:
 process_video(args.video, args.output, label_override=args.label, fps=args.fps)
 elif args.video_dir:
 process_all_clinical_videos(args.video_dir, args.output)
 elif args.image_dir:
 process_images(args.image_dir, args.output, label_override=args.label)
 else:
 parser.print_help()


if __name__ == "__main__":
 main()
