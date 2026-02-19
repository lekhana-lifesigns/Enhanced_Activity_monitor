#!/usr/bin/env python3
"""
Quick diagnostic script to test camera and YOLO pose separately.
Run this to identify where the issue is occurring.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time

def test_camera(camera_idx=0, resolution=(1280, 720)):
 """Test if camera can be opened and read frames."""
 print(f"\n{'='*60}")
 print(f"CAMERA TEST (index={camera_idx})")
 print(f"{'='*60}")

 cap = cv2.VideoCapture(camera_idx)

 if not cap.isOpened():
 print(f"[FAIL] Cannot open camera {camera_idx}")
 print("Possible causes:")
 print(" - Camera not connected")
 print(" - Camera in use by another application")
 print(" - Wrong camera index (try 1, 2, etc.)")
 return None

 print(f"[OK] Camera {camera_idx} opened")

 # Set resolution
 cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

 actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 print(f"[INFO] Resolution: requested {resolution}, actual {actual_w}x{actual_h}")

 # Try to read frames
 success_count = 0
 fail_count = 0

 print("\nReading 10 test frames...")
 for i in range(10):
 ret, frame = cap.read()
 if ret and frame is not None:
 success_count += 1
 print(f" Frame {i+1}: OK - shape {frame.shape}")
 else:
 fail_count += 1
 print(f" Frame {i+1}: FAILED")
 time.sleep(0.1)

 cap.release()

 print(f"\nCamera test result: {success_count}/10 frames OK")
 if success_count < 5:
 print("[FAIL] Camera is unstable")
 return None
 else:
 print("[OK] Camera working")
 return True


def test_yolo_detection(camera_idx=0):
 """Test YOLO detection on camera frames."""
 print(f"\n{'='*60}")
 print("YOLO DETECTION TEST")
 print(f"{'='*60}")

 try:
 from ultralytics import YOLO
 print("[OK] Ultralytics imported")
 except ImportError as e:
 print(f"[FAIL] Cannot import ultralytics: {e}")
 return None

 # Load detection model
 det_model_path = "yolo11n.pt"
 if not os.path.exists(det_model_path):
 print(f"[FAIL] Detection model not found: {det_model_path}")
 return None

 try:
 det_model = YOLO(det_model_path)
 print(f"[OK] Detection model loaded: {det_model_path}")
 except Exception as e:
 print(f"[FAIL] Cannot load detection model: {e}")
 return None

 # Test detection on camera
 cap = cv2.VideoCapture(camera_idx)
 if not cap.isOpened():
 print(f"[FAIL] Cannot open camera for detection test")
 return None

 print("\nRunning detection on 5 frames...")
 detections_found = 0

 for i in range(5):
 ret, frame = cap.read()
 if not ret or frame is None:
 print(f" Frame {i+1}: Camera read failed")
 continue

 results = det_model.predict(source=frame, conf=0.5, verbose=False)[0]
 boxes = results.boxes

 person_count = sum(1 for box in boxes if box.cls == 0) # class 0 = person
 print(f" Frame {i+1}: {len(boxes)} detections, {person_count} persons")

 if person_count > 0:
 detections_found += 1

 time.sleep(0.2)

 cap.release()

 if detections_found > 0:
 print(f"[OK] Detection working ({detections_found}/5 frames had persons)")
 return True
 else:
 print("[WARN] No persons detected - ensure you are in frame")
 return False


def test_yolo_pose(camera_idx=0):
 """Test YOLO pose estimation on camera frames."""
 print(f"\n{'='*60}")
 print("YOLO POSE TEST")
 print(f"{'='*60}")

 try:
 from ultralytics import YOLO
 except ImportError as e:
 print(f"[FAIL] Cannot import ultralytics: {e}")
 return None

 # Load pose model
 pose_model_path = "yolo11n-pose.pt"
 if not os.path.exists(pose_model_path):
 print(f"[FAIL] Pose model not found: {pose_model_path}")
 return None

 try:
 pose_model = YOLO(pose_model_path)
 print(f"[OK] Pose model loaded: {pose_model_path}")
 except Exception as e:
 print(f"[FAIL] Cannot load pose model: {e}")
 return None

 # Test pose on camera
 cap = cv2.VideoCapture(camera_idx)
 if not cap.isOpened():
 print(f"[FAIL] Cannot open camera for pose test")
 return None

 print("\nRunning pose estimation on 5 frames...")
 poses_found = 0

 for i in range(5):
 ret, frame = cap.read()
 if not ret or frame is None:
 print(f" Frame {i+1}: Camera read failed")
 continue

 results = pose_model.predict(source=frame, conf=0.5, verbose=False)[0]

 if results.keypoints is not None and len(results.keypoints.data) > 0:
 kps = results.keypoints.data[0] # First person
 valid_kps = sum(1 for kp in kps if kp[2] > 0.3) # Confidence > 0.3
 print(f" Frame {i+1}: {len(kps)} keypoints, {valid_kps} valid (conf>0.3)")

 if valid_kps >= 7: # At least 40% of 17 keypoints
 poses_found += 1
 else:
 print(f" Frame {i+1}: No keypoints detected")

 time.sleep(0.2)

 cap.release()

 if poses_found > 0:
 print(f"[OK] Pose estimation working ({poses_found}/5 frames had valid poses)")
 return True
 else:
 print("[WARN] No valid poses detected - ensure you are fully visible in frame")
 return False


def test_feature_extraction():
 """Test feature extraction with dummy keypoints."""
 print(f"\n{'='*60}")
 print("FEATURE EXTRACTION TEST")
 print(f"{'='*60}")

 try:
 from pipeline.pose.feature_extractor import ICUFeatureEncoder
 print("[OK] Feature extractor imported")
 except ImportError as e:
 print(f"[FAIL] Cannot import feature extractor: {e}")
 return None

 encoder = ICUFeatureEncoder()

 # Create dummy keypoints (17 COCO keypoints with good confidence)
 dummy_kps = [(0.5 + i*0.01, 0.3 + i*0.02, 0.9) for i in range(17)]

 features = encoder.extract(dummy_kps, frame_shape=(720, 1280))

 if features is not None:
 print(f"[OK] Feature extraction working - {len(features)}D features")
 print(f" Features: {features[:5]}... (first 5)")
 return True
 else:
 print("[FAIL] Feature extraction returned None")
 return False


if __name__ == "__main__":
 import argparse
 parser = argparse.ArgumentParser()
 parser.add_argument("--camera", type=int, default=0, help="Camera index")
 args = parser.parse_args()

 print("=" * 60)
 print("EAM DIAGNOSTIC TEST")
 print("=" * 60)
 print(f"Camera index: {args.camera}")
 print("Stand in front of camera with full body visible")
 print("=" * 60)

 results = {}

 # Test each component
 results["camera"] = test_camera(args.camera)

 if results["camera"]:
 results["detection"] = test_yolo_detection(args.camera)
 results["pose"] = test_yolo_pose(args.camera)

 results["features"] = test_feature_extraction()

 # Summary
 print(f"\n{'='*60}")
 print("SUMMARY")
 print(f"{'='*60}")

 for component, status in results.items():
 icon = "[OK]" if status else "[FAIL]" if status is False else "[SKIP]"
 print(f" {component}: {icon}")

 if all(v for v in results.values() if v is not None):
 print("\nAll tests passed! Pipeline should work.")
 else:
 print("\nSome tests failed. Fix the issues above before running EAM.")
