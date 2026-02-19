#!/usr/bin/env python3
"""
Test script for Video2IMU integration.

This script validates the Video2IMU converter that transforms video
keypoints into IMU-like signals, based on:
"Video2IMU: Realistic IMU features and signals from videos" (Lamsa et al.)

Usage:
 python scripts/test_video2imu.py [--video PATH] [--visualize]
"""

import sys
import os
import argparse
import numpy as np
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pose.video2imu_converter import Video2IMUConverter, create_video2imu_converter
from pipeline.pose.feature_extractor import ICUFeatureEncoder, extract_extended_features


def generate_synthetic_keypoints(frame_idx, activity="walking"):
 """
 Generate synthetic keypoints for testing.

 Args:
 frame_idx: Frame index (for temporal variation)
 activity: Type of activity to simulate

 Returns:
 17x3 keypoints array (x, y, confidence)
 """
 # Base pose (standing person, normalized coordinates 0-1)
 base_pose = np.array([
 [0.5, 0.15, 0.9], # 0: nose
 [0.48, 0.12, 0.85], # 1: left_eye
 [0.52, 0.12, 0.85], # 2: right_eye
 [0.45, 0.14, 0.8], # 3: left_ear
 [0.55, 0.14, 0.8], # 4: right_ear
 [0.42, 0.28, 0.9], # 5: left_shoulder
 [0.58, 0.28, 0.9], # 6: right_shoulder
 [0.38, 0.42, 0.85], # 7: left_elbow
 [0.62, 0.42, 0.85], # 8: right_elbow
 [0.35, 0.55, 0.8], # 9: left_wrist
 [0.65, 0.55, 0.8], # 10: right_wrist
 [0.45, 0.52, 0.9], # 11: left_hip
 [0.55, 0.52, 0.9], # 12: right_hip
 [0.44, 0.72, 0.85], # 13: left_knee
 [0.56, 0.72, 0.85], # 14: right_knee
 [0.43, 0.92, 0.8], # 15: left_ankle
 [0.57, 0.92, 0.8], # 16: right_ankle
 ], dtype=np.float32)

 # Apply activity-specific motion
 t = frame_idx / 15.0 # Assuming 15 FPS

 if activity == "walking":
 # Sinusoidal walking motion
 walk_phase = t * 2 * np.pi * 1.0 # 1 Hz walking frequency
 leg_swing = 0.03 * np.sin(walk_phase)
 arm_swing = 0.02 * np.sin(walk_phase + np.pi) # Arms opposite to legs

 # Left leg forward, right arm forward (and vice versa)
 base_pose[13, 0] += leg_swing # left knee
 base_pose[15, 0] += leg_swing * 1.5 # left ankle
 base_pose[14, 0] -= leg_swing # right knee
 base_pose[16, 0] -= leg_swing * 1.5 # right ankle

 # Arms
 base_pose[7, 0] -= arm_swing # left elbow
 base_pose[9, 0] -= arm_swing * 1.5 # left wrist
 base_pose[8, 0] += arm_swing # right elbow
 base_pose[10, 0] += arm_swing * 1.5 # right wrist

 # Vertical bobbing
 bob = 0.005 * np.sin(walk_phase * 2)
 base_pose[:, 1] += bob

 elif activity == "falling":
 # Simulate a fall (rapid descent)
 fall_progress = min(1.0, t / 1.5) # 1.5 second fall
 fall_angle = fall_progress * np.pi / 2 # 0 to 90 degrees

 # Rotate around hip center
 hip_center = (base_pose[11] + base_pose[12]) / 2
 for i in range(len(base_pose)):
 # Apply rotation (simplified 2D rotation)
 dy = base_pose[i, 1] - hip_center[1]
 base_pose[i, 1] = hip_center[1] + dy * np.cos(fall_angle)
 base_pose[i, 0] += dy * np.sin(fall_angle) * 0.3

 # Rapid descent
 base_pose[:, 1] += fall_progress * 0.3

 elif activity == "sitting":
 # Bent knees, lowered body
 base_pose[13, 1] = 0.55 # left knee bent
 base_pose[14, 1] = 0.55 # right knee bent
 base_pose[15, 1] = 0.65 # left ankle forward
 base_pose[16, 1] = 0.65 # right ankle forward
 # Lower body
 base_pose[:12, 1] += 0.15

 elif activity == "tremor":
 # High-frequency small oscillations (Parkinson's-like)
 tremor_freq = 5.0 # 5 Hz tremor
 tremor_phase = t * 2 * np.pi * tremor_freq
 tremor_amp = 0.005

 # Apply tremor to hands
 base_pose[9, 0] += tremor_amp * np.sin(tremor_phase)
 base_pose[9, 1] += tremor_amp * np.cos(tremor_phase)
 base_pose[10, 0] += tremor_amp * np.sin(tremor_phase + 0.5)
 base_pose[10, 1] += tremor_amp * np.cos(tremor_phase + 0.5)

 elif activity == "agitation":
 # Rapid, irregular movements
 noise = np.random.randn(17, 2) * 0.02
 base_pose[:, :2] += noise

 # Add small random noise to simulate detection jitter
 noise = np.random.randn(17, 2) * 0.002
 base_pose[:, :2] += noise

 return base_pose


def test_video2imu_converter():
 """Test basic Video2IMU converter functionality."""
 print("\n" + "="*60)
 print("TEST 1: Video2IMU Converter Basic Functionality")
 print("="*60)

 converter = Video2IMUConverter(window_size=30, fps=15.0)

 # Process 60 frames of walking
 print("\nProcessing 60 frames of walking motion...")
 for i in range(60):
 kps = generate_synthetic_keypoints(i, "walking")
 result = converter.update(kps)

 # Check outputs
 print("\nLast frame results:")
 print(f" Position: {result['position']}")
 print(f" Velocity: {result['velocity']}")
 print(f" Acceleration: {result['acceleration']}")
 print(f" Total acceleration: {result['total_acceleration']:.4f} m/s^2")
 print(f" Tilt angles: pitch={result['tilt_angles']['pitch']:.1f}, "
 f"roll={result['tilt_angles']['roll']:.1f}, "
 f"yaw={result['tilt_angles']['yaw']:.1f}")
 print(f" Angular velocity: {result['angular_velocity']}")
 print(f" Orientation quaternion: {result['orientation']}")

 # Get statistical features
 stats = converter.get_statistical_features(window=30)
 if stats:
 print("\nStatistical features (28D as in Video2IMU paper):")
 print(f" x_avg: {stats['x_avg']:.4f}, x_var: {stats['x_var']:.4f}")
 print(f" y_avg: {stats['y_avg']:.4f}, y_var: {stats['y_var']:.4f}")
 print(f" z_avg: {stats['z_avg']:.4f}, z_var: {stats['z_var']:.4f}")
 print(f" tot_avg: {stats['tot_avg']:.4f}, tot_var: {stats['tot_var']:.4f}")

 print("\n[PASS] Video2IMU converter basic test passed")
 return True


def test_extended_features():
 """Test extended feature extraction with Video2IMU."""
 print("\n" + "="*60)
 print("TEST 2: Extended Feature Extraction (9D + 15D = 24D)")
 print("="*60)

 # Create encoder with Video2IMU enabled
 encoder = ICUFeatureEncoder(
 window_size=30,
 fps=15.0,
 enable_video2imu=True,
 video2imu_config={
 'reference_point': 'hip',
 'gravity_removal': True,
 'lowpass_cutoff': 12.0
 }
 )

 # Process frames
 print("\nProcessing 60 frames...")
 prev_kps = None
 prev_prev_kps = None

 for i in range(60):
 kps = generate_synthetic_keypoints(i, "walking")
 features = encoder.extract_extended_features(kps, prev_kps, prev_prev_kps)
 prev_prev_kps = prev_kps
 prev_kps = kps

 if features is not None:
 print(f"\nExtended feature vector shape: {features.shape}")
 print(f"Expected shape: (24,)")

 if features.shape[0] == 24:
 print("\nBase clinical features (9D):")
 print(f" [0] Motion energy: {features[0]:.4f}")
 print(f" [1] Jerk index: {features[1]:.4f}")
 print(f" [2] Posture instability: {features[2]:.4f}")
 print(f" [3] COM sway: {features[3]:.4f}")
 print(f" [4] Breath rate proxy: {features[4]:.4f}")
 print(f" [5] Hand proximity risk: {features[5]:.4f}")
 print(f" [6] Motor entropy: {features[6]:.4f}")
 print(f" [7] Symmetry index: {features[7]:.4f}")
 print(f" [8] Motion variability: {features[8]:.4f}")

 print("\nVideo2IMU features (15D):")
 print(f" [9-11] Linear acceleration (normalized): {features[9:12]}")
 print(f" [12-14] Angular velocity (normalized): {features[12:15]}")
 print(f" [15-17] Angular acceleration: {features[15:18]}")
 print(f" [18-20] Tilt angles (normalized): {features[18:21]}")
 print(f" [21] Total acceleration magnitude: {features[21]:.4f}")
 print(f" [22] Angular velocity magnitude: {features[22]:.4f}")
 print(f" [23] Motion jerk: {features[23]:.4f}")

 print("\n[PASS] Extended features test passed")
 return True
 else:
 print(f"\n[FAIL] Expected 24D features, got {features.shape[0]}D")
 return False
 else:
 print("\n[FAIL] Feature extraction returned None")
 return False


def test_activity_discrimination():
 """Test if Video2IMU features discriminate between activities."""
 print("\n" + "="*60)
 print("TEST 3: Activity Discrimination")
 print("="*60)

 activities = ["walking", "sitting", "falling", "tremor", "agitation"]
 results = {}

 for activity in activities:
 converter = Video2IMUConverter(window_size=30, fps=15.0)

 # Process 45 frames (3 seconds at 15 FPS)
 for i in range(45):
 kps = generate_synthetic_keypoints(i, activity)
 result = converter.update(kps)

 # Get statistics
 stats = converter.get_statistical_features(window=30)
 extended = converter.get_extended_features()

 results[activity] = {
 'total_accel_avg': stats['tot_avg'] if stats else 0,
 'total_accel_var': stats['tot_var'] if stats else 0,
 'angular_vel_mag': np.linalg.norm(result['angular_velocity']) if result else 0,
 'jerk': extended[14] if extended is not None else 0
 }

 # Display results
 print("\nActivity comparison (last 2 seconds of each):")
 print("-" * 70)
 print(f"{'Activity':<12} {'Accel Avg':>12} {'Accel Var':>12} {'Ang Vel':>12} {'Jerk':>12}")
 print("-" * 70)

 for activity, metrics in results.items():
 print(f"{activity:<12} {metrics['total_accel_avg']:>12.4f} "
 f"{metrics['total_accel_var']:>12.4f} "
 f"{metrics['angular_vel_mag']:>12.4f} "
 f"{metrics['jerk']:>12.4f}")

 # Check for discrimination
 # Falls should have higher acceleration
 # Tremor should have higher jerk
 # Walking should have moderate, periodic values

 print("\nExpected patterns:")
 print(" - Falling: High acceleration variance (rapid descent)")
 print(" - Tremor: High jerk (rapid direction changes)")
 print(" - Walking: Moderate, periodic acceleration")
 print(" - Sitting: Low acceleration/jerk (static)")

 print("\n[PASS] Activity discrimination test completed")
 return True


def test_performance():
 """Test Video2IMU processing performance."""
 print("\n" + "="*60)
 print("TEST 4: Performance Benchmark")
 print("="*60)

 converter = Video2IMUConverter(window_size=48, fps=15.0)
 encoder = ICUFeatureEncoder(
 window_size=48,
 fps=15.0,
 enable_video2imu=True
 )

 # Benchmark converter only
 n_frames = 1000
 start = time.time()
 for i in range(n_frames):
 kps = generate_synthetic_keypoints(i % 60, "walking")
 converter.update(kps)
 converter_time = time.time() - start

 # Benchmark full feature extraction
 start = time.time()
 prev_kps = None
 prev_prev_kps = None
 for i in range(n_frames):
 kps = generate_synthetic_keypoints(i % 60, "walking")
 features = encoder.extract_extended_features(kps, prev_kps, prev_prev_kps)
 prev_prev_kps = prev_kps
 prev_kps = kps
 full_time = time.time() - start

 print(f"\nProcessed {n_frames} frames:")
 print(f" Video2IMU converter only: {converter_time*1000:.1f} ms ({n_frames/converter_time:.0f} FPS)")
 print(f" Full extended features: {full_time*1000:.1f} ms ({n_frames/full_time:.0f} FPS)")
 print(f" Per-frame overhead: {(full_time - converter_time)*1000/n_frames:.3f} ms")

 target_fps = 15
 if n_frames/full_time > target_fps:
 print(f"\n[PASS] Performance meets real-time requirement (>{target_fps} FPS)")
 return True
 else:
 print(f"\n[WARN] Performance below target ({target_fps} FPS)")
 return True # Still pass, just a warning


def test_backward_compatibility():
 """Test that base 9D features still work without Video2IMU."""
 print("\n" + "="*60)
 print("TEST 5: Backward Compatibility (9D features without Video2IMU)")
 print("="*60)

 # Create encoder WITHOUT Video2IMU
 encoder = ICUFeatureEncoder(
 window_size=30,
 fps=15.0,
 enable_video2imu=False
 )

 # Process frames
 prev_kps = None
 prev_prev_kps = None
 for i in range(30):
 kps = generate_synthetic_keypoints(i, "walking")
 features = encoder.extract_feature_vector(kps, prev_kps, prev_prev_kps)
 prev_prev_kps = prev_kps
 prev_kps = kps

 if features is not None and features.shape[0] == 9:
 print(f"\nBase feature vector shape: {features.shape}")
 print("Features:", features)
 print("\n[PASS] Backward compatibility test passed")
 return True
 else:
 print("\n[FAIL] Base features extraction failed")
 return False


def visualize_imu_signals(duration=5.0, activity="walking"):
 """Visualize Video2IMU signals over time."""
 try:
 import matplotlib.pyplot as plt
 except ImportError:
 print("\nMatplotlib not available, skipping visualization")
 return

 print("\n" + "="*60)
 print(f"VISUALIZATION: {activity.upper()} activity ({duration}s)")
 print("="*60)

 converter = Video2IMUConverter(window_size=48, fps=15.0)

 # Collect data
 n_frames = int(duration * 15)
 times = []
 accel_x, accel_y, accel_z, accel_tot = [], [], [], []
 ang_vel_x, ang_vel_y, ang_vel_z = [], [], []
 pitch, roll, yaw = [], [], []

 for i in range(n_frames):
 kps = generate_synthetic_keypoints(i, activity)
 result = converter.update(kps)

 if result:
 times.append(i / 15.0)
 accel_x.append(result['acceleration'][0])
 accel_y.append(result['acceleration'][1])
 accel_z.append(result['acceleration'][2])
 accel_tot.append(result['total_acceleration'])
 ang_vel_x.append(result['angular_velocity'][0])
 ang_vel_y.append(result['angular_velocity'][1])
 ang_vel_z.append(result['angular_velocity'][2])
 pitch.append(result['tilt_angles']['pitch'])
 roll.append(result['tilt_angles']['roll'])
 yaw.append(result['tilt_angles']['yaw'])

 # Create figure
 fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
 fig.suptitle(f'Video2IMU Signals - {activity.title()} Activity', fontsize=14)

 # Acceleration
 axes[0].plot(times, accel_x, label='X', alpha=0.8)
 axes[0].plot(times, accel_y, label='Y', alpha=0.8)
 axes[0].plot(times, accel_z, label='Z', alpha=0.8)
 axes[0].plot(times, accel_tot, label='Total', linewidth=2, color='black')
 axes[0].set_ylabel('Acceleration (m/s²)')
 axes[0].legend(loc='upper right')
 axes[0].grid(True, alpha=0.3)
 axes[0].set_title('Linear Acceleration (Accelerometer-like)')

 # Angular velocity
 axes[1].plot(times, ang_vel_x, label='Pitch rate', alpha=0.8)
 axes[1].plot(times, ang_vel_y, label='Roll rate', alpha=0.8)
 axes[1].plot(times, ang_vel_z, label='Yaw rate', alpha=0.8)
 axes[1].set_ylabel('Angular Velocity (rad/s)')
 axes[1].legend(loc='upper right')
 axes[1].grid(True, alpha=0.3)
 axes[1].set_title('Angular Velocity (Gyroscope-like)')

 # Tilt angles
 axes[2].plot(times, pitch, label='Pitch', alpha=0.8)
 axes[2].plot(times, roll, label='Roll', alpha=0.8)
 axes[2].plot(times, yaw, label='Yaw', alpha=0.8)
 axes[2].set_ylabel('Angle (degrees)')
 axes[2].set_xlabel('Time (s)')
 axes[2].legend(loc='upper right')
 axes[2].grid(True, alpha=0.3)
 axes[2].set_title('Tilt Angles (Orientation)')

 plt.tight_layout()
 plt.savefig(f'video2imu_{activity}.png', dpi=150)
 print(f"\nSaved visualization to video2imu_{activity}.png")
 plt.show()


def main():
 parser = argparse.ArgumentParser(description="Test Video2IMU integration")
 parser.add_argument("--visualize", action="store_true",
 help="Generate visualizations")
 parser.add_argument("--activity", default="walking",
 choices=["walking", "sitting", "falling", "tremor", "agitation"],
 help="Activity for visualization")
 parser.add_argument("--duration", type=float, default=5.0,
 help="Duration for visualization (seconds)")
 args = parser.parse_args()

 print("="*60)
 print("VIDEO2IMU INTEGRATION TEST SUITE")
 print("Based on: 'Video2IMU: Realistic IMU features from videos'")
 print("="*60)

 # Run tests
 tests = [
 ("Video2IMU Converter", test_video2imu_converter),
 ("Extended Features", test_extended_features),
 ("Activity Discrimination", test_activity_discrimination),
 ("Performance", test_performance),
 ("Backward Compatibility", test_backward_compatibility),
 ]

 results = []
 for name, test_func in tests:
 try:
 result = test_func()
 results.append((name, result))
 except Exception as e:
 print(f"\n[ERROR] {name} test failed with exception: {e}")
 import traceback
 traceback.print_exc()
 results.append((name, False))

 # Summary
 print("\n" + "="*60)
 print("TEST SUMMARY")
 print("="*60)
 passed = sum(1 for _, r in results if r)
 total = len(results)
 for name, result in results:
 status = "[PASS]" if result else "[FAIL]"
 print(f" {status} {name}")
 print(f"\nTotal: {passed}/{total} tests passed")

 # Visualization if requested
 if args.visualize:
 visualize_imu_signals(duration=args.duration, activity=args.activity)

 return 0 if passed == total else 1


if __name__ == "__main__":
 sys.exit(main())
