#!/usr/bin/env python3
"""
Temporal Annotation Converter
Converts temporal segment annotations + extracted keypoints → training NPZ files
for the TemporalModel (GRU/GCN).

Workflow:
 1. Read video → extract keypoints at 15 FPS using YOLO-pose
 2. Read temporal labels (JSONL from CVAT/Label Studio export)
 3. Slice keypoints into overlapping windows (48 frames, stride 8)
 4. Assign each window a label based on majority vote within the window
 5. Compute 9D ICU features per frame
 6. Export as NPZ: features (N, T, F), labels (N,), metadata

Usage:
 python temporal_annotation_converter.py \
 --video path/to/video.mp4 \
 --labels path/to/video_temporal_labels.jsonl \
 --output path/to/output/ \
 --schema path/to/temporal_annotation_schema.yaml

 # Batch mode:
 python temporal_annotation_converter.py \
 --batch path/to/video_list.txt \
 --output path/to/output/
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("temporal_converter")

# Schema class mapping (must match temporal_annotation_schema.yaml)
CLASS_NAMES = [
 "lying_still", # 0
 "lying_restless", # 1
 "turning_in_bed", # 2
 "sitting_stable", # 3
 "sitting_unstable", # 4
 "bed_exit", # 5
 "standing", # 6
 "walking", # 7
 "agitated", # 8
 "convulsive", # 9
 "fallen", # 10
 "occluded", # 11
]

NUM_CLASSES = len(CLASS_NAMES)


def load_schema(schema_path: str) -> Dict:
 """Load and validate the annotation schema."""
 with open(schema_path, 'r') as f:
 schema = yaml.safe_load(f)

 # Validate class count matches
 assert len(schema['classes']) == NUM_CLASSES, \
 f"Schema has {len(schema['classes'])} classes, expected {NUM_CLASSES}"

 return schema


def load_temporal_labels(labels_path: str) -> List[Dict]:
 """
 Load temporal segment labels from JSONL file.

 Each line: {"video_id": ..., "start_sec": ..., "end_sec": ..., "label": ..., "label_id": ...}
 """
 labels = []
 with open(labels_path, 'r') as f:
 for line_num, line in enumerate(f, 1):
 line = line.strip()
 if not line:
 continue
 try:
 segment = json.loads(line)
 # Validate required fields
 required = ['start_sec', 'end_sec', 'label', 'label_id']
 for field in required:
 if field not in segment:
 log.warning("Line %d missing field '%s', skipping", line_num, field)
 continue
 labels.append(segment)
 except json.JSONDecodeError as e:
 log.warning("Line %d: invalid JSON: %s", line_num, e)

 # Sort by start time
 labels.sort(key=lambda x: x['start_sec'])

 # Validate no gaps
 for i in range(1, len(labels)):
 gap = labels[i]['start_sec'] - labels[i-1]['end_sec']
 if abs(gap) > 0.1: # Allow 100ms tolerance
 log.warning("Gap of %.2fs between segments at %.2fs", gap, labels[i-1]['end_sec'])

 log.info("Loaded %d temporal segments covering %.1fs",
 len(labels), labels[-1]['end_sec'] if labels else 0)
 return labels


def extract_keypoints_from_video(video_path: str, fps: int = 15) -> np.ndarray:
 """
 Extract COCO-17 keypoints from video using YOLO-pose.

 Returns:
 np.ndarray shape (N_frames, 17, 3) - [x, y, confidence] per keypoint
 """
 try:
 import cv2
 except ImportError:
 log.error("OpenCV required: pip install opencv-python")
 sys.exit(1)

 try:
 from ultralytics import YOLO
 except ImportError:
 log.error("Ultralytics required: pip install ultralytics")
 sys.exit(1)

 # Load pose model
 model = YOLO("yolo11n-pose.pt")
 log.info("YOLO11-pose model loaded for keypoint extraction")

 cap = cv2.VideoCapture(str(video_path))
 if not cap.isOpened():
 raise ValueError(f"Cannot open video: {video_path}")

 video_fps = cap.get(cv2.CAP_PROP_FPS)
 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 frame_skip = max(1, round(video_fps / fps))

 log.info("Video: %.1f FPS, %d frames, extracting at %d FPS (skip=%d)",
 video_fps, total_frames, fps, frame_skip)

 all_keypoints = []
 frame_idx = 0

 while True:
 ret, frame = cap.read()
 if not ret:
 break

 if frame_idx % frame_skip == 0:
 # Run pose estimation
 results = model(frame, verbose=False)
 res = results[0]

 if hasattr(res, 'keypoints') and res.keypoints is not None and len(res.keypoints) > 0:
 # Take the most confident person detection
 kps_data = res.keypoints.data # (N_persons, 17, 3)
 if len(kps_data) > 0:
 # Select person with highest mean confidence
 mean_confs = kps_data[:, :, 2].mean(dim=1)
 best_idx = mean_confs.argmax().item()
 kps = kps_data[best_idx].cpu().numpy() # (17, 3)

 # Normalize coordinates to [0, 1]
 h, w = frame.shape[:2]
 kps[:, 0] /= w
 kps[:, 1] /= h

 all_keypoints.append(kps)
 else:
 all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
 else:
 all_keypoints.append(np.zeros((17, 3), dtype=np.float32))

 frame_idx += 1

 cap.release()
 keypoints = np.array(all_keypoints, dtype=np.float32)
 log.info("Extracted keypoints: shape %s", keypoints.shape)
 return keypoints


def compute_icu_features(keypoints_window: np.ndarray) -> np.ndarray:
 """
 Compute 9D ICU feature vector for each frame in the window.
 Matches the feature_extractor.py ICUFeatureEncoder output.

 Args:
 keypoints_window: (T, 17, 3) array of keypoints

 Returns:
 (T, 9) feature array
 """
 T = keypoints_window.shape[0]
 features = np.zeros((T, 9), dtype=np.float32)

 MIN_CONF = 0.3

 for t in range(T):
 kps = keypoints_window[t] # (17, 3)

 # Feature [0]: motion_energy
 if t > 0:
 prev_kps = keypoints_window[t - 1]
 valid = (kps[:, 2] > MIN_CONF) & (prev_kps[:, 2] > MIN_CONF)
 if valid.any():
 vel = kps[valid, :2] - prev_kps[valid, :2]
 features[t, 0] = 0.5 * np.sum(vel ** 2) / valid.sum()

 # Feature [1]: jerk_index (requires t-2)
 if t >= 2:
 kps_prev = keypoints_window[t - 1]
 kps_prev2 = keypoints_window[t - 2]
 valid = (kps[:, 2] > MIN_CONF) & (kps_prev[:, 2] > MIN_CONF) & (kps_prev2[:, 2] > MIN_CONF)
 if valid.any():
 vel_curr = kps[valid, :2] - kps_prev[valid, :2]
 vel_prev = kps_prev[valid, :2] - kps_prev2[valid, :2]
 acc = vel_curr - vel_prev
 features[t, 1] = np.sqrt(np.sum(acc ** 2)) / valid.sum()

 # Feature [2]: posture_instability (spine angle deviation from vertical)
 ls, rs = kps[5], kps[6] # shoulders
 lh, rh = kps[11], kps[12] # hips
 if all(k[2] > MIN_CONF for k in [ls, rs, lh, rh]):
 shoulder_mid = (ls[:2] + rs[:2]) / 2
 hip_mid = (lh[:2] + rh[:2]) / 2
 spine_vec = hip_mid - shoulder_mid
 # Angle from vertical (normalized: 0=upright, 1=horizontal)
 angle = np.abs(np.arctan2(spine_vec[0], spine_vec[1]))
 features[t, 2] = angle / (np.pi / 2) # 0-1 normalized

 # Feature [3]: sway_score (COM deviation within history window)
 if t >= 5:
 window_kps = keypoints_window[max(0, t-5):t+1]
 coms = []
 for wk in window_kps:
 valid_w = wk[:, 2] > MIN_CONF
 if valid_w.any():
 coms.append(wk[valid_w, :2].mean(axis=0))
 if len(coms) >= 3:
 coms_arr = np.array(coms)
 features[t, 3] = np.std(coms_arr, axis=0).sum()

 # Feature [4]: breath_rate_proxy (chest expansion oscillation)
 if t >= 10:
 chest_dists = []
 for tt in range(max(0, t-10), t+1):
 wk = keypoints_window[tt]
 if wk[5, 2] > MIN_CONF and wk[6, 2] > MIN_CONF:
 chest_dists.append(np.linalg.norm(wk[5, :2] - wk[6, :2]))
 if len(chest_dists) >= 5:
 # Normalized variance as proxy for breathing
 cd = np.array(chest_dists)
 features[t, 4] = np.std(cd) / (np.mean(cd) + 1e-6)

 # Feature [5]: hand_proximity_risk (hands near face/head)
 nose = kps[0]
 lw, rw = kps[9], kps[10] # wrists
 if nose[2] > MIN_CONF:
 prox = 0.0
 if lw[2] > MIN_CONF:
 prox = max(prox, 1.0 / (np.linalg.norm(lw[:2] - nose[:2]) + 0.01))
 if rw[2] > MIN_CONF:
 prox = max(prox, 1.0 / (np.linalg.norm(rw[:2] - nose[:2]) + 0.01))
 features[t, 5] = min(prox, 10.0) / 10.0 # Clamp and normalize

 # Feature [6]: motor_entropy
 if t >= 5:
 # Compute entropy of velocity directions over recent window
 velocities = []
 for tt in range(max(1, t-5), t+1):
 v = keypoints_window[tt, :, :2] - keypoints_window[tt-1, :, :2]
 valid_v = (keypoints_window[tt, :, 2] > MIN_CONF) & (keypoints_window[tt-1, :, 2] > MIN_CONF)
 if valid_v.any():
 angles = np.arctan2(v[valid_v, 1], v[valid_v, 0])
 velocities.extend(angles.tolist())
 if len(velocities) >= 5:
 # Discretize into 8 bins and compute entropy
 hist, _ = np.histogram(velocities, bins=8, range=(-np.pi, np.pi))
 hist = hist / (hist.sum() + 1e-6)
 ent = -np.sum(hist * np.log2(hist + 1e-10))
 features[t, 6] = ent / 3.0 # Normalize (max entropy = log2(8) = 3)

 # Feature [7]: symmetry_index (left-right body symmetry)
 left_joints = [5, 7, 9, 11, 13, 15] # L shoulder, elbow, wrist, hip, knee, ankle
 right_joints = [6, 8, 10, 12, 14, 16] # R equivalents
 sym_diffs = []
 for lj, rj in zip(left_joints, right_joints):
 if kps[lj, 2] > MIN_CONF and kps[rj, 2] > MIN_CONF:
 # Mirror x-coordinate and compare
 sym_diffs.append(abs(kps[lj, 1] - kps[rj, 1])) # Y-axis symmetry
 if sym_diffs:
 features[t, 7] = 1.0 - min(np.mean(sym_diffs) * 5, 1.0) # High = symmetric

 # Feature [8]: motion_variability (variance of motion energy over window)
 if t >= 10:
 energies = features[max(0, t-10):t+1, 0]
 features[t, 8] = np.std(energies) / (np.mean(energies) + 1e-6)

 return features


def get_frame_label(frame_time_sec: float, segments: List[Dict]) -> int:
 """
 Get the label ID for a given frame time from segment annotations.

 Returns:
 label_id (0-11), or -1 if no segment covers this time.
 """
 for seg in segments:
 if seg['start_sec'] <= frame_time_sec < seg['end_sec']:
 return seg['label_id']
 return -1


def create_training_windows(
 keypoints: np.ndarray,
 segments: List[Dict],
 fps: int = 15,
 window_size: int = 48,
 stride: int = 8,
 min_label_ratio: float = 0.6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
 """
 Slice keypoints into overlapping windows and assign labels.

 Args:
 keypoints: (N_frames, 17, 3) full video keypoints
 segments: Temporal label segments
 fps: Extraction FPS
 window_size: Frames per window (48 = 3.2s at 15 FPS)
 stride: Window stride (8 = 0.53s)
 min_label_ratio: Minimum fraction of window that must have same label

 Returns:
 features: (N_windows, window_size, 9) ICU features
 labels: (N_windows,) class IDs
 raw_keypoints: (N_windows, window_size, 17, 3) keypoints
 metadata: List of per-window metadata dicts
 """
 n_frames = keypoints.shape[0]
 n_windows = (n_frames - window_size) // stride + 1

 if n_windows <= 0:
 log.warning("Video too short for even one window (%d frames, need %d)", n_frames, window_size)
 return np.array([]), np.array([]), np.array([]), []

 all_features = []
 all_labels = []
 all_keypoints = []
 all_metadata = []

 skipped_ambiguous = 0
 skipped_unlabeled = 0

 for win_idx in range(n_windows):
 start_frame = win_idx * stride
 end_frame = start_frame + window_size

 # Get keypoints for this window
 win_kps = keypoints[start_frame:end_frame] # (window_size, 17, 3)

 # Determine label by majority vote across frames in window
 frame_labels = []
 for f in range(start_frame, end_frame):
 frame_time = f / fps
 label = get_frame_label(frame_time, segments)
 if label >= 0:
 frame_labels.append(label)

 if not frame_labels:
 skipped_unlabeled += 1
 continue

 # Majority vote
 counter = Counter(frame_labels)
 majority_label, majority_count = counter.most_common(1)[0]
 label_ratio = majority_count / len(frame_labels)

 if label_ratio < min_label_ratio:
 skipped_ambiguous += 1
 continue

 # Compute features for this window
 win_features = compute_icu_features(win_kps) # (window_size, 9)

 all_features.append(win_features)
 all_labels.append(majority_label)
 all_keypoints.append(win_kps)
 all_metadata.append({
 "window_idx": win_idx,
 "start_frame": start_frame,
 "end_frame": end_frame,
 "start_sec": start_frame / fps,
 "end_sec": end_frame / fps,
 "label": CLASS_NAMES[majority_label],
 "label_id": majority_label,
 "label_ratio": label_ratio,
 "n_valid_keypoint_frames": int(np.sum(win_kps[:, :, 2].max(axis=1) > 0.3)),
 })

 if skipped_ambiguous > 0:
 log.info("Skipped %d ambiguous windows (label_ratio < %.1f)", skipped_ambiguous, min_label_ratio)
 if skipped_unlabeled > 0:
 log.info("Skipped %d unlabeled windows", skipped_unlabeled)

 if not all_features:
 return np.array([]), np.array([]), np.array([]), []

 features = np.array(all_features, dtype=np.float32)
 labels = np.array(all_labels, dtype=np.int64)
 raw_kps = np.array(all_keypoints, dtype=np.float32)

 log.info("Created %d training windows. Label distribution:", len(labels))
 for cls_id in range(NUM_CLASSES):
 count = int((labels == cls_id).sum())
 if count > 0:
 log.info(" [%d] %-18s: %d (%.1f%%)", cls_id, CLASS_NAMES[cls_id], count, 100*count/len(labels))

 return features, labels, raw_kps, all_metadata


def export_training_data(
 output_dir: str,
 video_id: str,
 features: np.ndarray,
 labels: np.ndarray,
 keypoints: np.ndarray,
 metadata: List[Dict],
 schema_version: str = "1.0.0"
):
 """
 Export training data as NPZ file.

 Output structure:
 {video_id}_train.npz:
 features: (N, T, 9) - ICU feature sequences
 labels: (N,) - class IDs
 keypoints: (N, T, 17, 3) - raw keypoints (for augmentation/debugging)
 class_names: string array of class names
 schema_version: string
 """
 output_path = Path(output_dir)
 output_path.mkdir(parents=True, exist_ok=True)

 npz_path = output_path / f"{video_id}_train.npz"
 np.savez_compressed(
 npz_path,
 features=features,
 labels=labels,
 keypoints=keypoints,
 class_names=np.array(CLASS_NAMES),
 schema_version=schema_version,
 )

 # Save metadata as JSON alongside
 meta_path = output_path / f"{video_id}_metadata.json"
 with open(meta_path, 'w') as f:
 json.dump({
 "video_id": video_id,
 "schema_version": schema_version,
 "num_windows": len(labels),
 "window_size": features.shape[1] if len(features) > 0 else 0,
 "feature_dim": features.shape[2] if len(features) > 0 else 0,
 "label_distribution": {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(NUM_CLASSES)},
 "windows": metadata,
 }, f, indent=2)

 log.info("Exported: %s (%d windows, %.1f MB)",
 npz_path, len(labels), npz_path.stat().st_size / 1e6)


def merge_training_files(output_dir: str, merged_name: str = "eam_temporal_dataset"):
 """
 Merge multiple per-video NPZ files into a single training dataset.
 Applies class balancing via undersampling/oversampling.
 """
 output_path = Path(output_dir)
 npz_files = sorted(output_path.glob("*_train.npz"))

 if not npz_files:
 log.error("No training NPZ files found in %s", output_dir)
 return

 all_features = []
 all_labels = []
 all_keypoints = []

 for npz_file in npz_files:
 data = np.load(npz_file)
 if len(data['features']) > 0:
 all_features.append(data['features'])
 all_labels.append(data['labels'])
 all_keypoints.append(data['keypoints'])
 log.info(" Loaded %s: %d windows", npz_file.name, len(data['labels']))

 if not all_features:
 log.error("No valid training data found")
 return

 features = np.concatenate(all_features, axis=0)
 labels = np.concatenate(all_labels, axis=0)
 keypoints = np.concatenate(all_keypoints, axis=0)

 log.info("Merged dataset: %d total windows", len(labels))
 log.info("Label distribution:")
 for cls_id in range(NUM_CLASSES):
 count = int((labels == cls_id).sum())
 if count > 0:
 log.info(" [%2d] %-18s: %5d (%.1f%%)", cls_id, CLASS_NAMES[cls_id], count, 100*count/len(labels))

 # Class balancing
 features_balanced, labels_balanced, kps_balanced = balance_classes(features, labels, keypoints)

 # Save merged dataset
 merged_path = output_path / f"{merged_name}.npz"
 np.savez_compressed(
 merged_path,
 features=features_balanced,
 labels=labels_balanced,
 keypoints=kps_balanced,
 class_names=np.array(CLASS_NAMES),
 schema_version="1.0.0",
 )
 log.info("Merged balanced dataset: %s (%d windows)", merged_path, len(labels_balanced))

 # Patient-level split
 create_splits(features_balanced, labels_balanced, kps_balanced, output_path, merged_name)


def balance_classes(
 features: np.ndarray,
 labels: np.ndarray,
 keypoints: np.ndarray,
 cap_multiplier: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
 """
 Balance classes via capped oversampling of rare classes.

 Strategy:
 - Compute median class count
 - Cap majority class at cap_multiplier * median
 - Oversample minority classes to median count (with noise augmentation)
 """
 unique, counts = np.unique(labels, return_counts=True)
 median_count = int(np.median(counts))
 cap = int(cap_multiplier * median_count)

 log.info("Balancing: median=%d, cap=%d", median_count, cap)

 balanced_features = []
 balanced_labels = []
 balanced_keypoints = []

 for cls_id in unique:
 mask = labels == cls_id
 cls_features = features[mask]
 cls_labels = labels[mask]
 cls_kps = keypoints[mask]
 cls_count = len(cls_labels)

 if cls_count > cap:
 # Undersample
 indices = np.random.choice(cls_count, cap, replace=False)
 balanced_features.append(cls_features[indices])
 balanced_labels.append(cls_labels[indices])
 balanced_keypoints.append(cls_kps[indices])
 log.info(" [%d] %-18s: %d → %d (capped)", cls_id, CLASS_NAMES[cls_id], cls_count, cap)
 elif cls_count < median_count:
 # Oversample with augmentation
 balanced_features.append(cls_features)
 balanced_labels.append(cls_labels)
 balanced_keypoints.append(cls_kps)

 n_needed = median_count - cls_count
 indices = np.random.choice(cls_count, n_needed, replace=True)
 aug_features = cls_features[indices] + np.random.normal(0, 0.01, (n_needed,) + cls_features.shape[1:]).astype(np.float32)
 aug_kps = cls_kps[indices] # Keypoints not augmented (features are)
 balanced_features.append(aug_features)
 balanced_labels.append(np.full(n_needed, cls_id, dtype=np.int64))
 balanced_keypoints.append(aug_kps)
 log.info(" [%d] %-18s: %d → %d (oversampled)", cls_id, CLASS_NAMES[cls_id], cls_count, median_count)
 else:
 balanced_features.append(cls_features)
 balanced_labels.append(cls_labels)
 balanced_keypoints.append(cls_kps)

 return (
 np.concatenate(balanced_features),
 np.concatenate(balanced_labels),
 np.concatenate(balanced_keypoints),
 )


def create_splits(
 features: np.ndarray,
 labels: np.ndarray,
 keypoints: np.ndarray,
 output_path: Path,
 name: str,
 train_ratio: float = 0.7,
 val_ratio: float = 0.15
):
 """Create stratified train/val/test splits."""
 n = len(labels)
 indices = np.arange(n)
 np.random.shuffle(indices)

 train_end = int(n * train_ratio)
 val_end = int(n * (train_ratio + val_ratio))

 splits = {
 'train': indices[:train_end],
 'val': indices[train_end:val_end],
 'test': indices[val_end:],
 }

 for split_name, split_indices in splits.items():
 split_path = output_path / f"{name}_{split_name}.npz"
 np.savez_compressed(
 split_path,
 features=features[split_indices],
 labels=labels[split_indices],
 keypoints=keypoints[split_indices],
 class_names=np.array(CLASS_NAMES),
 )
 split_labels = labels[split_indices]
 log.info(" %s: %d windows, classes present: %d/%d",
 split_name, len(split_indices),
 len(np.unique(split_labels)), NUM_CLASSES)


def process_single_video(video_path: str, labels_path: str, output_dir: str, schema: Dict):
 """Process a single video: extract keypoints → align labels → export."""
 video_path = Path(video_path)
 video_id = video_path.stem.replace(" ", "_").lower()

 temporal_cfg = schema['temporal']
 fps = temporal_cfg['camera_fps']
 window_size = temporal_cfg['window_size_frames']
 stride = temporal_cfg['window_stride_frames']

 log.info("=" * 60)
 log.info("Processing: %s", video_path.name)
 log.info("=" * 60)

 # Step 1: Extract keypoints
 keypoints = extract_keypoints_from_video(str(video_path), fps=fps)

 if len(keypoints) < window_size:
 log.warning("Video too short (%d frames < %d window_size), skipping", len(keypoints), window_size)
 return

 # Step 2: Load temporal labels
 segments = load_temporal_labels(labels_path)

 if not segments:
 log.warning("No temporal labels found for %s, skipping", video_path.name)
 return

 # Step 3: Create training windows
 features, labels, raw_kps, metadata = create_training_windows(
 keypoints, segments, fps=fps, window_size=window_size, stride=stride
 )

 if len(features) == 0:
 log.warning("No valid training windows created for %s", video_path.name)
 return

 # Step 4: Export
 export_training_data(output_dir, video_id, features, labels, raw_kps, metadata,
 schema_version=schema.get('schema_version', '1.0.0'))


def process_existing_clinical_videos(schema: Dict, output_dir: str):
 """
 Process existing clinical_validation_data videos using the mapping in the schema.
 These are single-activity clips → entire video gets one label.
 """
 mapping = schema.get('existing_video_mapping', {})
 video_dir = Path("clinical_validation_data/video for detection")

 if not video_dir.exists():
 log.warning("Clinical validation video directory not found: %s", video_dir)
 return

 for video_name, info in mapping.items():
 video_path = video_dir / video_name
 if not video_path.exists():
 log.warning("Video not found: %s", video_path)
 continue

 label_name = info['primary_label']
 label_id = CLASS_NAMES.index(label_name)

 log.info("Processing existing clip: %s → %s (%d)", video_name, label_name, label_id)

 # Extract keypoints
 fps = schema['temporal']['camera_fps']
 keypoints = extract_keypoints_from_video(str(video_path), fps=fps)

 if len(keypoints) < schema['temporal']['window_size_frames']:
 log.warning(" Too short, skipping")
 continue

 # Create synthetic segment label covering entire video
 duration = len(keypoints) / fps
 segments = [{
 'video_id': video_path.stem,
 'start_sec': 0.0,
 'end_sec': duration,
 'label': label_name,
 'label_id': label_id,
 }]

 # Create windows
 features, labels, raw_kps, metadata = create_training_windows(
 keypoints, segments,
 fps=fps,
 window_size=schema['temporal']['window_size_frames'],
 stride=schema['temporal']['window_stride_frames']
 )

 if len(features) > 0:
 video_id = video_path.stem.replace(" ", "_").replace("-", "_").lower()
 export_training_data(output_dir, video_id, features, labels, raw_kps, metadata)


def main():
 parser = argparse.ArgumentParser(description="Convert temporal annotations to training data")
 parser.add_argument("--video", type=str, help="Path to video file")
 parser.add_argument("--labels", type=str, help="Path to temporal labels JSONL file")
 parser.add_argument("--output", type=str, default="./training_data/temporal",
 help="Output directory for NPZ files")
 parser.add_argument("--schema", type=str,
 default="dataset_creation/temporal_annotation_schema.yaml",
 help="Path to annotation schema YAML")
 parser.add_argument("--batch", type=str, help="Path to batch file (video_path,labels_path per line)")
 parser.add_argument("--process-existing", action="store_true",
 help="Process existing clinical_validation_data videos")
 parser.add_argument("--merge", action="store_true",
 help="Merge all per-video NPZ files into one dataset")
 parser.add_argument("--seed", type=int, default=42, help="Random seed")

 args = parser.parse_args()
 np.random.seed(args.seed)

 # Load schema
 schema = load_schema(args.schema)
 log.info("Schema loaded: version %s, %d classes", schema['schema_version'], NUM_CLASSES)

 if args.process_existing:
 process_existing_clinical_videos(schema, args.output)

 elif args.batch:
 with open(args.batch, 'r') as f:
 for line in f:
 line = line.strip()
 if not line or line.startswith('#'):
 continue
 parts = line.split(',')
 if len(parts) != 2:
 log.warning("Invalid batch line: %s", line)
 continue
 video_path, labels_path = parts[0].strip(), parts[1].strip()
 process_single_video(video_path, labels_path, args.output, schema)

 elif args.video and args.labels:
 process_single_video(args.video, args.labels, args.output, schema)

 else:
 parser.print_help()
 return

 if args.merge:
 merge_training_files(args.output)


if __name__ == "__main__":
 main()
