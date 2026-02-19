# pipeline/pose/video2imu_converter.py
# Video2IMU: Convert video keypoints to IMU-like signals
# Based on: "Video2IMU: Realistic IMU features and signals from videos" (Lamsa et al.)
#
# This module transforms 2D/3D pose keypoints into acceleration and angular velocity
# signals that mimic IMU sensor data, enabling:
# - Enhanced fall detection (acceleration-based gold standard)
# - Tremor/bradykinesia detection (angular acceleration patterns)
# - Cross-modal validation with real IMU sensors
# - Richer feature space for activity classification

import numpy as np
import math
import logging
from collections import deque
from scipy import signal
from scipy.spatial.transform import Rotation

log = logging.getLogger("video2imu")

# COCO Keypoint Indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Minimum confidence threshold
MIN_CONFIDENCE = 0.3

# Physical constants
GRAVITY = 9.81 # m/s^2


def safe_get_kp(kps, idx, default=(0.0, 0.0, 0.0)):
 """Safely get keypoint with confidence check."""
 if kps is None or idx >= len(kps):
 return default
 kp = kps[idx]
 if len(kp) >= 3 and kp[2] > MIN_CONFIDENCE:
 return kp
 return default


def compute_body_center(kps):
 """
 Compute body center (hip center) - analogous to IMU placement on lower back.

 Args:
 kps: Keypoints array (17, 3) with (x, y, confidence)

 Returns:
 (x, y) tuple of body center position
 """
 lh = safe_get_kp(kps, LEFT_HIP)
 rh = safe_get_kp(kps, RIGHT_HIP)

 if lh[2] > MIN_CONFIDENCE and rh[2] > MIN_CONFIDENCE:
 cx = (lh[0] + rh[0]) / 2.0
 cy = (lh[1] + rh[1]) / 2.0
 return (cx, cy)

 # Fallback to available hip
 if lh[2] > MIN_CONFIDENCE:
 return (lh[0], lh[1])
 if rh[2] > MIN_CONFIDENCE:
 return (rh[0], rh[1])

 # Last resort: use shoulder center
 ls = safe_get_kp(kps, LEFT_SHOULDER)
 rs = safe_get_kp(kps, RIGHT_SHOULDER)
 if ls[2] > MIN_CONFIDENCE and rs[2] > MIN_CONFIDENCE:
 cx = (ls[0] + rs[0]) / 2.0
 cy = (ls[1] + rs[1]) / 2.0
 return (cx, cy)

 return (0.5, 0.5) # Default center


def estimate_depth_from_pose(kps, reference_height=1.7):
 """
 Estimate Z-coordinate (depth) from pose size.

 Larger poses = closer to camera, smaller = further away.
 Uses torso length as reference (shoulder to hip distance).

 Args:
 kps: Keypoints array
 reference_height: Reference person height in meters

 Returns:
 Estimated depth (0-1 normalized, 0=far, 1=close)
 """
 ls = safe_get_kp(kps, LEFT_SHOULDER)
 rs = safe_get_kp(kps, RIGHT_SHOULDER)
 lh = safe_get_kp(kps, LEFT_HIP)
 rh = safe_get_kp(kps, RIGHT_HIP)

 # Compute torso length
 if ls[2] > MIN_CONFIDENCE and lh[2] > MIN_CONFIDENCE:
 torso_left = math.hypot(ls[0] - lh[0], ls[1] - lh[1])
 else:
 torso_left = 0.0

 if rs[2] > MIN_CONFIDENCE and rh[2] > MIN_CONFIDENCE:
 torso_right = math.hypot(rs[0] - rh[0], rs[1] - rh[1])
 else:
 torso_right = 0.0

 # Average torso length (normalized coordinates, so ~0.2-0.4 for typical person)
 torso_length = max(torso_left, torso_right)

 if torso_length < 0.01:
 return 0.5 # Unknown depth

 # Normalize: typical torso is ~0.3 in normalized coords at 2m distance
 # Scale: 0.15 = far (4m), 0.45 = close (1m)
 depth_normalized = np.clip((torso_length - 0.15) / 0.30, 0.0, 1.0)

 return depth_normalized


class Video2IMUConverter:
 """
 Converts video pose keypoints to IMU-like signals.

 Produces:
 - 3D linear acceleration (ax, ay, az) in m/s^2
 - 3D angular velocity (wx, wy, wz) in rad/s
 - 3D angular acceleration (αx, αy, αz) in rad/s^2
 - Tilt angles (pitch, roll, yaw) in degrees
 - Orientation quaternion (qx, qy, qz, qw)

 Reference points:
 - Hip center (body_center): Analogous to waist-mounted IMU
 - Wrist: Analogous to wrist-worn IMU
 - Ankle: Analogous to ankle-mounted IMU
 """

 def __init__(self, window_size=48, fps=15.0, pixel_to_meter=0.01,
 lowpass_cutoff=12.0, gravity_removal=True,
 reference_point="hip"):
 """
 Initialize Video2IMU converter.

 Args:
 window_size: Number of frames for history buffer
 fps: Video frame rate
 pixel_to_meter: Conversion factor (meters per normalized unit)
 lowpass_cutoff: Low-pass filter cutoff frequency (Hz)
 gravity_removal: Whether to remove gravity component from acceleration
 reference_point: IMU reference point ("hip", "wrist", "ankle")
 """
 self.window_size = window_size
 self.fps = fps
 self.dt = 1.0 / fps
 self.pixel_to_meter = pixel_to_meter
 self.lowpass_cutoff = lowpass_cutoff
 self.gravity_removal = gravity_removal
 self.reference_point = reference_point

 # Position history buffers (circular)
 self.position_history = deque(maxlen=window_size)
 self.depth_history = deque(maxlen=window_size)

 # Velocity and acceleration buffers
 self.velocity_history = deque(maxlen=window_size)
 self.acceleration_history = deque(maxlen=window_size)

 # Angular measurements
 self.angle_history = deque(maxlen=window_size)
 self.angular_velocity_history = deque(maxlen=window_size)

 # Joint angle history for angular acceleration
 self.joint_angles_history = deque(maxlen=window_size)

 # Low-pass filter (Butterworth)
 self._init_filters()

 # Previous values for derivative computation
 self.prev_position = None
 self.prev_velocity = None
 self.prev_angles = None
 self.prev_angular_velocity = None

 # Calibration (can be updated with real IMU data)
 self.calibration = {
 'acceleration_scale': 1.0,
 'angular_velocity_scale': 1.0,
 'gravity_offset': np.array([0.0, GRAVITY, 0.0]) # Gravity in y-direction (down)
 }

 def _init_filters(self):
 """Initialize low-pass filters for noise reduction."""
 try:
 # Nyquist frequency
 nyquist = self.fps / 2.0
 if self.lowpass_cutoff >= nyquist:
 self.lowpass_cutoff = nyquist * 0.8 # Stay below Nyquist

 # Design Butterworth low-pass filter
 normalized_cutoff = self.lowpass_cutoff / nyquist
 self.b, self.a = signal.butter(2, normalized_cutoff, btype='low')
 self.filter_initialized = True
 except Exception as e:
 log.warning(f"Filter initialization failed: {e}")
 self.filter_initialized = False

 def _get_reference_position(self, kps):
 """
 Get position of reference point based on configuration.

 Args:
 kps: Keypoints array

 Returns:
 (x, y) position tuple
 """
 if self.reference_point == "hip":
 return compute_body_center(kps)
 elif self.reference_point == "wrist":
 # Average of wrists (or use dominant hand)
 lw = safe_get_kp(kps, LEFT_WRIST)
 rw = safe_get_kp(kps, RIGHT_WRIST)
 if lw[2] > MIN_CONFIDENCE and rw[2] > MIN_CONFIDENCE:
 return ((lw[0] + rw[0]) / 2.0, (lw[1] + rw[1]) / 2.0)
 elif lw[2] > MIN_CONFIDENCE:
 return (lw[0], lw[1])
 elif rw[2] > MIN_CONFIDENCE:
 return (rw[0], rw[1])
 return compute_body_center(kps) # Fallback
 elif self.reference_point == "ankle":
 la = safe_get_kp(kps, LEFT_ANKLE)
 ra = safe_get_kp(kps, RIGHT_ANKLE)
 if la[2] > MIN_CONFIDENCE and ra[2] > MIN_CONFIDENCE:
 return ((la[0] + ra[0]) / 2.0, (la[1] + ra[1]) / 2.0)
 elif la[2] > MIN_CONFIDENCE:
 return (la[0], la[1])
 elif ra[2] > MIN_CONFIDENCE:
 return (ra[0], ra[1])
 return compute_body_center(kps) # Fallback
 else:
 return compute_body_center(kps)

 def _apply_lowpass(self, signal_data):
 """Apply low-pass filter to signal."""
 if not self.filter_initialized or len(signal_data) < 6:
 return signal_data

 try:
 return signal.filtfilt(self.b, self.a, signal_data)
 except Exception:
 return signal_data

 def compute_3d_position(self, kps):
 """
 Compute 3D position from 2D keypoints.

 Args:
 kps: Keypoints array (17, 3)

 Returns:
 np.array([x, y, z]) position in meters (approximate)
 """
 x, y = self._get_reference_position(kps)
 z = estimate_depth_from_pose(kps)

 # Convert to meters (approximate)
 # Assuming frame is ~2m wide at typical viewing distance
 frame_width_meters = 2.0
 x_meters = (x - 0.5) * frame_width_meters
 y_meters = (y - 0.5) * frame_width_meters # Flip for standard coords
 z_meters = z * 3.0 # Map 0-1 to 0-3 meters depth range

 return np.array([x_meters, y_meters, z_meters], dtype=np.float32)

 def compute_linear_velocity(self, position):
 """
 Compute linear velocity from position change.

 Args:
 position: Current 3D position (x, y, z)

 Returns:
 np.array([vx, vy, vz]) velocity in m/s
 """
 if self.prev_position is None:
 self.prev_position = position
 return np.zeros(3, dtype=np.float32)

 velocity = (position - self.prev_position) / self.dt
 self.prev_position = position

 return velocity

 def compute_linear_acceleration(self, velocity):
 """
 Compute linear acceleration from velocity change.

 Args:
 velocity: Current 3D velocity (vx, vy, vz)

 Returns:
 np.array([ax, ay, az]) acceleration in m/s^2
 """
 if self.prev_velocity is None:
 self.prev_velocity = velocity
 return np.zeros(3, dtype=np.float32)

 acceleration = (velocity - self.prev_velocity) / self.dt
 self.prev_velocity = velocity

 # Optionally add gravity component (IMU measures gravity)
 if not self.gravity_removal:
 acceleration += self.calibration['gravity_offset']

 return acceleration * self.calibration['acceleration_scale']

 def compute_body_angles(self, kps):
 """
 Compute body orientation angles from keypoints.

 Returns pitch, roll, yaw angles analogous to IMU orientation.

 Args:
 kps: Keypoints array

 Returns:
 np.array([pitch, roll, yaw]) in radians
 """
 # Get key reference points
 ls = safe_get_kp(kps, LEFT_SHOULDER)
 rs = safe_get_kp(kps, RIGHT_SHOULDER)
 lh = safe_get_kp(kps, LEFT_HIP)
 rh = safe_get_kp(kps, RIGHT_HIP)
 nose = safe_get_kp(kps, NOSE)

 # Pitch: Forward/backward tilt (from spine angle)
 # Angle of spine from vertical
 if ls[2] > MIN_CONFIDENCE and rs[2] > MIN_CONFIDENCE and lh[2] > MIN_CONFIDENCE and rh[2] > MIN_CONFIDENCE:
 shoulder_mid = np.array([(ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0])
 hip_mid = np.array([(lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0])
 spine_vec = shoulder_mid - hip_mid
 # Pitch = angle from vertical (positive = leaning forward)
 pitch = math.atan2(spine_vec[0], -spine_vec[1]) # Negative y is up in image coords
 else:
 pitch = 0.0

 # Roll: Left/right tilt (from shoulder line)
 if ls[2] > MIN_CONFIDENCE and rs[2] > MIN_CONFIDENCE:
 shoulder_vec = np.array([rs[0] - ls[0], rs[1] - ls[1]])
 # Roll = angle from horizontal
 roll = math.atan2(shoulder_vec[1], shoulder_vec[0])
 else:
 roll = 0.0

 # Yaw: Rotation around vertical axis (from shoulder width vs hip width)
 # Approximation: narrower shoulders = turned away
 if ls[2] > MIN_CONFIDENCE and rs[2] > MIN_CONFIDENCE and lh[2] > MIN_CONFIDENCE and rh[2] > MIN_CONFIDENCE:
 shoulder_width = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
 hip_width = math.hypot(rh[0] - lh[0], rh[1] - lh[1])
 if hip_width > 0.01:
 width_ratio = shoulder_width / hip_width
 # Ratio < 1 suggests rotation (perspective narrowing)
 yaw = math.asin(np.clip(1.0 - width_ratio, -1.0, 1.0)) if width_ratio < 1.2 else 0.0
 else:
 yaw = 0.0
 else:
 yaw = 0.0

 return np.array([pitch, roll, yaw], dtype=np.float32)

 def compute_angular_velocity(self, angles):
 """
 Compute angular velocity from angle changes.

 Args:
 angles: Current orientation angles (pitch, roll, yaw) in radians

 Returns:
 np.array([wx, wy, wz]) angular velocity in rad/s
 """
 if self.prev_angles is None:
 self.prev_angles = angles
 return np.zeros(3, dtype=np.float32)

 angular_velocity = (angles - self.prev_angles) / self.dt

 # Handle angle wrapping (for yaw especially)
 angular_velocity = np.where(
 np.abs(angular_velocity) > np.pi / self.dt,
 angular_velocity - np.sign(angular_velocity) * 2 * np.pi / self.dt,
 angular_velocity
 )

 self.prev_angles = angles

 return angular_velocity * self.calibration['angular_velocity_scale']

 def compute_angular_acceleration(self, angular_velocity):
 """
 Compute angular acceleration from angular velocity change.

 Args:
 angular_velocity: Current angular velocity (wx, wy, wz) in rad/s

 Returns:
 np.array([αx, αy, αz]) angular acceleration in rad/s^2
 """
 if self.prev_angular_velocity is None:
 self.prev_angular_velocity = angular_velocity
 return np.zeros(3, dtype=np.float32)

 angular_acceleration = (angular_velocity - self.prev_angular_velocity) / self.dt
 self.prev_angular_velocity = angular_velocity

 return angular_acceleration

 def compute_tilt_angles(self, kps):
 """
 Compute tilt angles in degrees (user-friendly format).

 Args:
 kps: Keypoints array

 Returns:
 dict with pitch, roll, yaw in degrees
 """
 angles_rad = self.compute_body_angles(kps)
 return {
 'pitch': float(np.degrees(angles_rad[0])),
 'roll': float(np.degrees(angles_rad[1])),
 'yaw': float(np.degrees(angles_rad[2]))
 }

 def compute_orientation_quaternion(self, kps):
 """
 Compute orientation as quaternion from body angles.

 Args:
 kps: Keypoints array

 Returns:
 np.array([qx, qy, qz, qw]) unit quaternion
 """
 angles = self.compute_body_angles(kps)

 try:
 # Convert Euler angles (ZYX order) to quaternion
 rot = Rotation.from_euler('zyx', [angles[2], angles[1], angles[0]])
 quat = rot.as_quat() # Returns [x, y, z, w]
 return quat.astype(np.float32)
 except Exception:
 # Fallback to identity quaternion
 return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

 def compute_joint_angles(self, kps):
 """
 Compute joint angles for all major joints.

 Used for detecting joint-specific movements like arm swinging,
 knee bending, etc.

 Args:
 kps: Keypoints array

 Returns:
 dict of joint angles in radians
 """
 def angle_between_points(p1, p2, p3):
 """Compute angle at p2 formed by p1-p2-p3."""
 if any(p[2] < MIN_CONFIDENCE for p in [p1, p2, p3]):
 return 0.0

 v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
 v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

 cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
 return math.acos(np.clip(cos_angle, -1.0, 1.0))

 # Get keypoints
 ls = safe_get_kp(kps, LEFT_SHOULDER)
 rs = safe_get_kp(kps, RIGHT_SHOULDER)
 le = safe_get_kp(kps, LEFT_ELBOW)
 re = safe_get_kp(kps, RIGHT_ELBOW)
 lw = safe_get_kp(kps, LEFT_WRIST)
 rw = safe_get_kp(kps, RIGHT_WRIST)
 lh = safe_get_kp(kps, LEFT_HIP)
 rh = safe_get_kp(kps, RIGHT_HIP)
 lk = safe_get_kp(kps, LEFT_KNEE)
 rk = safe_get_kp(kps, RIGHT_KNEE)
 la = safe_get_kp(kps, LEFT_ANKLE)
 ra = safe_get_kp(kps, RIGHT_ANKLE)

 return {
 'left_elbow': angle_between_points(ls, le, lw),
 'right_elbow': angle_between_points(rs, re, rw),
 'left_shoulder': angle_between_points(lh, ls, le),
 'right_shoulder': angle_between_points(rh, rs, re),
 'left_hip': angle_between_points(ls, lh, lk),
 'right_hip': angle_between_points(rs, rh, rk),
 'left_knee': angle_between_points(lh, lk, la),
 'right_knee': angle_between_points(rh, rk, ra)
 }

 def compute_total_acceleration(self, acceleration):
 """
 Compute total (magnitude) acceleration.

 This is analogous to the 'tot' acceleration in the Video2IMU paper.

 Args:
 acceleration: 3D acceleration vector

 Returns:
 float: Total acceleration magnitude
 """
 return float(np.linalg.norm(acceleration))

 def update(self, kps):
 """
 Update converter with new keypoints frame.

 Call this every frame to update internal state.

 Args:
 kps: Keypoints array (17, 3)

 Returns:
 dict with all IMU-like signals
 """
 if kps is None or len(kps) < 5:
 return None

 # Compute 3D position
 position = self.compute_3d_position(kps)
 self.position_history.append(position)

 # Compute linear kinematics
 velocity = self.compute_linear_velocity(position)
 self.velocity_history.append(velocity)

 acceleration = self.compute_linear_acceleration(velocity)
 self.acceleration_history.append(acceleration)

 # Compute angular kinematics
 angles = self.compute_body_angles(kps)
 self.angle_history.append(angles)

 angular_velocity = self.compute_angular_velocity(angles)
 self.angular_velocity_history.append(angular_velocity)

 angular_acceleration = self.compute_angular_acceleration(angular_velocity)

 # Compute joint angles
 joint_angles = self.compute_joint_angles(kps)
 self.joint_angles_history.append(joint_angles)

 # Compute derived values
 tilt_angles = self.compute_tilt_angles(kps)
 orientation = self.compute_orientation_quaternion(kps)
 total_accel = self.compute_total_acceleration(acceleration)

 return {
 # 3D Linear motion
 'position': position,
 'velocity': velocity,
 'acceleration': acceleration,
 'total_acceleration': total_accel,

 # Angular motion
 'angles': angles, # pitch, roll, yaw in radians
 'tilt_angles': tilt_angles, # pitch, roll, yaw in degrees
 'angular_velocity': angular_velocity,
 'angular_acceleration': angular_acceleration,
 'orientation': orientation, # quaternion

 # Joint angles
 'joint_angles': joint_angles,

 # Metadata
 'reference_point': self.reference_point,
 'fps': self.fps
 }

 def get_imu_frame(self, kps, kps_history=None):
 """
 Get complete IMU-like frame from keypoints.

 Alias for update() for API compatibility.

 Args:
 kps: Current keypoints
 kps_history: Optional keypoint history (unused, uses internal history)

 Returns:
 dict with IMU-like signals
 """
 return self.update(kps)

 def get_filtered_acceleration(self, window=10):
 """
 Get low-pass filtered acceleration signal.

 Args:
 window: Number of frames to use for filtering

 Returns:
 np.array of filtered 3D acceleration
 """
 if len(self.acceleration_history) < 3:
 return np.zeros(3, dtype=np.float32)

 # Get recent accelerations
 recent = list(self.acceleration_history)[-window:]
 accel_array = np.array(recent)

 # Apply filter to each axis
 if self.filter_initialized and len(accel_array) >= 6:
 filtered = np.zeros_like(accel_array)
 for i in range(3):
 filtered[:, i] = self._apply_lowpass(accel_array[:, i])
 return filtered[-1]

 return accel_array[-1]

 def get_statistical_features(self, window=30):
 """
 Compute statistical features from acceleration history.

 These are the 7 features used in the Video2IMU paper:
 avg, median, variance, min, max, lower_quartile, upper_quartile

 Args:
 window: Number of frames for statistics

 Returns:
 dict of statistical features per axis
 """
 if len(self.acceleration_history) < 5:
 return None

 # Get recent accelerations
 recent = list(self.acceleration_history)[-window:]
 accel_array = np.array(recent)

 # Compute total acceleration
 total_accel = np.linalg.norm(accel_array, axis=1)

 features = {}
 axes = {'x': 0, 'y': 1, 'z': 2}

 for axis_name, axis_idx in axes.items():
 axis_data = accel_array[:, axis_idx]
 features[f'{axis_name}_avg'] = float(np.mean(axis_data))
 features[f'{axis_name}_med'] = float(np.median(axis_data))
 features[f'{axis_name}_var'] = float(np.var(axis_data))
 features[f'{axis_name}_min'] = float(np.min(axis_data))
 features[f'{axis_name}_max'] = float(np.max(axis_data))
 features[f'{axis_name}_lq'] = float(np.percentile(axis_data, 25))
 features[f'{axis_name}_uq'] = float(np.percentile(axis_data, 75))

 # Total acceleration features
 features['tot_avg'] = float(np.mean(total_accel))
 features['tot_med'] = float(np.median(total_accel))
 features['tot_var'] = float(np.var(total_accel))
 features['tot_min'] = float(np.min(total_accel))
 features['tot_max'] = float(np.max(total_accel))
 features['tot_lq'] = float(np.percentile(total_accel, 25))
 features['tot_uq'] = float(np.percentile(total_accel, 75))

 return features

 def get_extended_features(self, window=30):
 """
 Get extended feature vector combining acceleration and angular features.

 This produces a feature vector that can be concatenated with the
 existing 9D clinical features.

 Args:
 window: Number of frames for feature computation

 Returns:
 np.array of extended features (15D):
 [0-2]: Linear acceleration (ax, ay, az)
 [3-5]: Angular velocity (wx, wy, wz)
 [6-8]: Angular acceleration (αx, αy, αz)
 [9-11]: Tilt angles normalized (pitch, roll, yaw) / 90
 [12]: Total acceleration magnitude
 [13]: Angular velocity magnitude
 [14]: Motion jerk (acceleration change rate)
 """
 if len(self.acceleration_history) < 3:
 return np.zeros(15, dtype=np.float32)

 # Get latest values
 accel = np.array(self.acceleration_history[-1]) if self.acceleration_history else np.zeros(3)
 ang_vel = np.array(self.angular_velocity_history[-1]) if self.angular_velocity_history else np.zeros(3)
 angles = np.array(self.angle_history[-1]) if self.angle_history else np.zeros(3)

 # Compute angular acceleration from history
 if len(self.angular_velocity_history) >= 2:
 ang_accel = (np.array(self.angular_velocity_history[-1]) -
 np.array(self.angular_velocity_history[-2])) / self.dt
 else:
 ang_accel = np.zeros(3)

 # Compute jerk (acceleration change)
 if len(self.acceleration_history) >= 2:
 jerk = np.linalg.norm(
 np.array(self.acceleration_history[-1]) -
 np.array(self.acceleration_history[-2])
 ) / self.dt
 else:
 jerk = 0.0

 # Build feature vector
 features = np.array([
 # Linear acceleration (normalized by gravity)
 accel[0] / GRAVITY,
 accel[1] / GRAVITY,
 accel[2] / GRAVITY,
 # Angular velocity (normalized)
 ang_vel[0] / np.pi,
 ang_vel[1] / np.pi,
 ang_vel[2] / np.pi,
 # Angular acceleration (normalized)
 ang_accel[0] / np.pi,
 ang_accel[1] / np.pi,
 ang_accel[2] / np.pi,
 # Tilt angles (normalized to -1 to 1)
 np.degrees(angles[0]) / 90.0, # pitch
 np.degrees(angles[1]) / 90.0, # roll
 np.degrees(angles[2]) / 180.0, # yaw
 # Magnitudes
 np.linalg.norm(accel) / GRAVITY,
 np.linalg.norm(ang_vel) / np.pi,
 jerk / GRAVITY
 ], dtype=np.float32)

 # Clip to reasonable range
 features = np.clip(features, -10.0, 10.0)

 return features

 def reset(self):
 """Reset converter state."""
 self.position_history.clear()
 self.depth_history.clear()
 self.velocity_history.clear()
 self.acceleration_history.clear()
 self.angle_history.clear()
 self.angular_velocity_history.clear()
 self.joint_angles_history.clear()

 self.prev_position = None
 self.prev_velocity = None
 self.prev_angles = None
 self.prev_angular_velocity = None

 def calibrate(self, real_imu_data=None):
 """
 Calibrate converter using real IMU data (optional).

 Args:
 real_imu_data: dict with 'acceleration' and 'angular_velocity' arrays
 """
 if real_imu_data is None:
 return

 # Compute scale factors to match real IMU range
 if 'acceleration' in real_imu_data and len(self.acceleration_history) > 10:
 real_scale = np.std(real_imu_data['acceleration'])
 video_scale = np.std(np.array(list(self.acceleration_history)))
 if video_scale > 0.01:
 self.calibration['acceleration_scale'] = real_scale / video_scale

 if 'angular_velocity' in real_imu_data and len(self.angular_velocity_history) > 10:
 real_scale = np.std(real_imu_data['angular_velocity'])
 video_scale = np.std(np.array(list(self.angular_velocity_history)))
 if video_scale > 0.01:
 self.calibration['angular_velocity_scale'] = real_scale / video_scale


# Factory function for easy instantiation
def create_video2imu_converter(config=None):
 """
 Create Video2IMU converter from configuration.

 Args:
 config: dict with configuration options

 Returns:
 Video2IMUConverter instance
 """
 if config is None:
 config = {}

 return Video2IMUConverter(
 window_size=config.get('window_size', 48),
 fps=config.get('fps', 15.0),
 pixel_to_meter=config.get('pixel_to_meter', 0.01),
 lowpass_cutoff=config.get('lowpass_cutoff', 12.0),
 gravity_removal=config.get('gravity_removal', True),
 reference_point=config.get('reference_point', 'hip')
 )


# Global converter instance (singleton pattern)
_converter = None


def get_converter(config=None):
 """Get or create global converter instance."""
 global _converter
 if _converter is None:
 _converter = create_video2imu_converter(config)
 return _converter
