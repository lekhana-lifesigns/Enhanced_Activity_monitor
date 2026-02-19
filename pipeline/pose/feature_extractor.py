# pipeline/pose/feature_extractor.py
# ICU-Grade Agitation Feature Encoder
# 4-Layer Medical Signal Architecture

import numpy as np
import math
import logging
from collections import deque
from scipy import signal
from scipy.stats import entropy

log = logging.getLogger("feature")

# MoveNet/COCO Keypoint Indices
NOSE = 0
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

# Minimum confidence threshold for keypoints
MIN_CONFIDENCE = 0.3


def compute_com(coords):
    """Compute center of mass from keypoint coordinates."""
    xs = [c[0] for c in coords if c is not None]
    ys = [c[1] for c in coords if c is not None]
    if not xs:
        return (0.0, 0.0)
    return (float(np.mean(xs)), float(np.mean(ys)))


def normalize_by_torso(kps):
    """Normalize by torso scale (distance between shoulders)."""
    try:
        if len(kps) > max(LEFT_SHOULDER, RIGHT_SHOULDER):
            ls = kps[LEFT_SHOULDER]
            rs = kps[RIGHT_SHOULDER]
            if ls[2] > MIN_CONFIDENCE and rs[2] > MIN_CONFIDENCE:
                scale = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
                if scale < 1e-3:
                    scale = 1.0
                return scale
    except Exception:
        pass
    return 1.0


def get_keypoint(kps, idx, default=(0.0, 0.0, 0.0)):
    """Safely get keypoint with confidence check."""
    if idx < len(kps) and kps[idx][2] > MIN_CONFIDENCE:
        return kps[idx]
    return default


# ============================================================================
# LAYER 1: KINEMATIC FEATURES (Motion Physics)
# ============================================================================

def compute_velocity(kps, prev_kps, dt=1.0):
    """
    Compute velocity vectors for all keypoints.
    TODO-050: Vectorized NumPy implementation for performance.
    """
    if not prev_kps or len(kps) != len(prev_kps):
        return None

    # Convert to NumPy arrays for vectorization
    kps_arr = np.array(kps, dtype=np.float32)
    prev_arr = np.array(prev_kps, dtype=np.float32)

    # Compute velocities vectorized
    valid_mask = (kps_arr[:, 2] > MIN_CONFIDENCE) & (prev_arr[:, 2] > MIN_CONFIDENCE)
    velocities = np.zeros((len(kps), 2), dtype=np.float32)
    velocities[valid_mask, 0] = (kps_arr[valid_mask, 0] - prev_arr[valid_mask, 0]) / dt
    velocities[valid_mask, 1] = (kps_arr[valid_mask, 1] - prev_arr[valid_mask, 1]) / dt

    return [(float(v[0]), float(v[1])) for v in velocities]


def compute_acceleration(velocities, prev_velocities, dt=1.0):
    """Compute acceleration vectors from velocity changes."""
    if not prev_velocities or len(velocities) != len(prev_velocities):
        return None
    
    accelerations = []
    for i in range(len(velocities)):
        ax = (velocities[i][0] - prev_velocities[i][0]) / dt
        ay = (velocities[i][1] - prev_velocities[i][1]) / dt
        accelerations.append((ax, ay))
    return accelerations


def compute_motion_energy(kps, prev_kps, dt=1.0):
    """
    Compute motion energy: E = 0.5 * m * v² per joint.
    Clamped to [0, 100] to suppress detection flicker outliers.
    """
    if not prev_kps or len(kps) != len(prev_kps):
        return 0.0

    # Convert to NumPy arrays
    kps_arr = np.array(kps, dtype=np.float32)
    prev_arr = np.array(prev_kps, dtype=np.float32)

    # Valid mask (both frames have confident keypoints)
    valid_mask = (kps_arr[:, 2] > MIN_CONFIDENCE) & (prev_arr[:, 2] > MIN_CONFIDENCE)
    if not np.any(valid_mask):
        return 0.0

    # Vectorized velocity computation
    dx = (kps_arr[valid_mask, 0] - prev_arr[valid_mask, 0]) / dt
    dy = (kps_arr[valid_mask, 1] - prev_arr[valid_mask, 1]) / dt

    # E = 0.5 * v² (mass = 1 normalized)
    v_squared = dx * dx + dy * dy
    energy = 0.5 * np.sum(v_squared)
    energy_per_joint = float(energy / max(len(dx), 1))

    # Clamp to reasonable range (removes outliers from detection flickers)
    return float(np.clip(energy_per_joint, 0.0, 100.0))


def compute_jerk_index(kps, prev_kps, prev_prev_kps, dt=1.0):
    """
    Compute jerk index: j = da/dt (acceleration derivative).
    Measures suddenness of movement changes.
    Uses 3-point moving average pre-filter to suppress sensor jitter
    before computing 2nd derivative (prevents noise amplification).
    """
    if not prev_kps or not prev_prev_kps:
        return 0.0
    if len(kps) != len(prev_kps) or len(prev_kps) != len(prev_prev_kps):
        return 0.0

    # Convert to NumPy arrays
    kps_arr = np.array(kps, dtype=np.float32)
    prev_arr = np.array(prev_kps, dtype=np.float32)
    prev_prev_arr = np.array(prev_prev_kps, dtype=np.float32)

    # Valid mask (all three frames have confident keypoints)
    valid_mask = (
        (kps_arr[:, 2] > MIN_CONFIDENCE) &
        (prev_arr[:, 2] > MIN_CONFIDENCE) &
        (prev_prev_arr[:, 2] > MIN_CONFIDENCE)
    )
    if not np.any(valid_mask):
        return 0.0

    # Stack 3-frame position sequence: (3, N_valid, 2)
    positions = np.stack([
        prev_prev_arr[valid_mask, :2],
        prev_arr[valid_mask, :2],
        kps_arr[valid_mask, :2],
    ], axis=0)

    # Apply 3-point moving average along time axis to suppress sensor jitter
    # Middle frame: average of all 3; edge frames: average of 2 neighbors
    smoothed = np.empty_like(positions)
    smoothed[0] = (positions[0] + positions[1]) / 2.0
    smoothed[1] = (positions[0] + positions[1] + positions[2]) / 3.0
    smoothed[2] = (positions[1] + positions[2]) / 2.0

    # Compute velocities from smoothed positions
    v1_x = (smoothed[2, :, 0] - smoothed[1, :, 0]) / dt
    v1_y = (smoothed[2, :, 1] - smoothed[1, :, 1]) / dt
    v0_x = (smoothed[1, :, 0] - smoothed[0, :, 0]) / dt
    v0_y = (smoothed[1, :, 1] - smoothed[0, :, 1]) / dt

    # Acceleration = dv/dt
    ax = (v1_x - v0_x) / dt
    ay = (v1_y - v0_y) / dt

    # Jerk magnitude
    jerk_mag = np.sqrt(ax * ax + ay * ay)
    return float(np.mean(jerk_mag))


# ============================================================================
# LAYER 2: POSTURAL INSTABILITY
# ============================================================================

def compute_spine_angle(kps):
    """Compute spine angle from neck (mid-shoulders) to hip center."""
    try:
        ls = get_keypoint(kps, LEFT_SHOULDER)
        rs = get_keypoint(kps, RIGHT_SHOULDER)
        lh = get_keypoint(kps, LEFT_HIP)
        rh = get_keypoint(kps, RIGHT_HIP)
        
        # Mid-shoulder point
        shoulder_x = (ls[0] + rs[0]) / 2.0
        shoulder_y = (ls[1] + rs[1]) / 2.0
        
        # Mid-hip point
        hip_x = (lh[0] + rh[0]) / 2.0
        hip_y = (lh[1] + rh[1]) / 2.0
        
        # Spine vector
        dx = hip_x - shoulder_x
        dy = hip_y - shoulder_y
        
        # Angle from vertical (0 = upright, 90 = horizontal)
        angle = abs(math.degrees(math.atan2(dx, dy)))
        return angle
    except Exception:
        return 0.0


def compute_neck_angle(kps):
    """
    Compute neck deviation from vertical (0 = upright, 90 = horizontal).

    In image coordinates Y increases downward, so the nose is above the
    shoulders when nose_y < shoulder_y (dy < 0).  Using abs(dy) ensures
    atan2 measures lateral deviation regardless of whether the reference
    point is above or below.
    """
    try:
        nose = get_keypoint(kps, NOSE)
        ls = get_keypoint(kps, LEFT_SHOULDER)
        rs = get_keypoint(kps, RIGHT_SHOULDER)

        shoulder_x = (ls[0] + rs[0]) / 2.0
        shoulder_y = (ls[1] + rs[1]) / 2.0

        dx = nose[0] - shoulder_x
        dy = nose[1] - shoulder_y

        # abs(dy) makes the angle independent of image-coord direction:
        # upright -> dx~0, |dy|>0 -> angle~0
        # tilted  -> |dx|>0       -> angle grows toward 90
        angle = abs(math.degrees(math.atan2(dx, abs(dy))))
        return angle
    except Exception:
        return 0.0


def compute_com_sway(kps_history):
    """Compute center of mass sway over time (instability measure)."""
    if len(kps_history) < 3:
        return 0.0
    
    com_positions = []
    for kps in kps_history:
        if kps:
            coords = [(k[0], k[1]) for k in kps if k[2] > MIN_CONFIDENCE]
            if coords:
                comx, comy = compute_com(coords)
                com_positions.append((comx, comy))
    
    if len(com_positions) < 2:
        return 0.0
    
    # Compute sway as standard deviation of COM positions
    xs = [p[0] for p in com_positions]
    ys = [p[1] for p in com_positions]
    
    sway_x = np.std(xs) if len(xs) > 1 else 0.0
    sway_y = np.std(ys) if len(ys) > 1 else 0.0
    
    # Combined sway magnitude
    sway_score = math.sqrt(sway_x * sway_x + sway_y * sway_y)
    return float(sway_score)


def compute_symmetry_index(kps):
    """
    Compute left-right body symmetry index (0-1).
    1.0 = perfect symmetry, 0.0 = complete asymmetry.
    """
    try:
        # Compare left vs right side keypoints
        left_points = []
        right_points = []
        
        # Shoulders
        ls = get_keypoint(kps, LEFT_SHOULDER)
        rs = get_keypoint(kps, RIGHT_SHOULDER)
        if ls[2] > MIN_CONFIDENCE and rs[2] > MIN_CONFIDENCE:
            left_points.append((ls[0], ls[1]))
            right_points.append((rs[0], rs[1]))
        
        # Hips
        lh = get_keypoint(kps, LEFT_HIP)
        rh = get_keypoint(kps, RIGHT_HIP)
        if lh[2] > MIN_CONFIDENCE and rh[2] > MIN_CONFIDENCE:
            left_points.append((lh[0], lh[1]))
            right_points.append((rh[0], rh[1]))
        
        # Elbows
        le = get_keypoint(kps, LEFT_ELBOW)
        re = get_keypoint(kps, RIGHT_ELBOW)
        if le[2] > MIN_CONFIDENCE and re[2] > MIN_CONFIDENCE:
            left_points.append((le[0], le[1]))
            right_points.append((re[0], re[1]))
        
        # Wrists
        lw = get_keypoint(kps, LEFT_WRIST)
        rw = get_keypoint(kps, RIGHT_WRIST)
        if lw[2] > MIN_CONFIDENCE and rw[2] > MIN_CONFIDENCE:
            left_points.append((lw[0], lw[1]))
            right_points.append((rw[0], rw[1]))
        
        if len(left_points) < 2:
            return 0.5  # Neutral if insufficient data
        
        # Compute symmetry: compare distances from body midline
        # Midline = average x-coordinate of shoulders/hips
        mid_x = (ls[0] + rs[0] + lh[0] + rh[0]) / 4.0
        
        left_distances = [abs(p[0] - mid_x) for p in left_points]
        right_distances = [abs(p[0] - mid_x) for p in right_points]
        
        if len(left_distances) == 0 or len(right_distances) == 0:
            return 0.5
        
        avg_left = np.mean(left_distances)
        avg_right = np.mean(right_distances)
        
        if avg_left + avg_right < 1e-6:
            return 1.0
        
        # Symmetry = 1 - normalized difference
        diff = abs(avg_left - avg_right) / (avg_left + avg_right)
        symmetry = 1.0 - diff
        return float(np.clip(symmetry, 0.0, 1.0))
        
    except Exception:
        return 0.5


# ============================================================================
# LAYER 3: RESPIRATORY PROXY SIGNALS (Vision-Derived)
# ============================================================================

def get_thorax_point(kps):
    """Get thorax (chest) point as midpoint between shoulders and hips."""
    try:
        ls = get_keypoint(kps, LEFT_SHOULDER)
        rs = get_keypoint(kps, RIGHT_SHOULDER)
        lh = get_keypoint(kps, LEFT_HIP)
        rh = get_keypoint(kps, RIGHT_HIP)
        
        # Mid-shoulder
        sx = (ls[0] + rs[0]) / 2.0
        sy = (ls[1] + rs[1]) / 2.0
        
        # Mid-hip
        hx = (lh[0] + rh[0]) / 2.0
        hy = (lh[1] + rh[1]) / 2.0
        
        # Thorax = midpoint
        tx = (sx + hx) / 2.0
        ty = (sy + hy) / 2.0
        
        return (tx, ty)
    except Exception:
        return (0.5, 0.5)


def compute_thorax_expansion(kps_history):
    """
    Compute thorax expansion ratio over time.
    Measures breathing-related chest movement.
    """
    if len(kps_history) < 5:
        return 0.0
    
    thorax_y_positions = []
    for kps in kps_history:
        if kps:
            tx, ty = get_thorax_point(kps)
            thorax_y_positions.append(ty)
    
    if len(thorax_y_positions) < 5:
        return 0.0
    
    # Compute expansion as range of vertical movement
    y_min = min(thorax_y_positions)
    y_max = max(thorax_y_positions)
    expansion = y_max - y_min
    
    return float(expansion)


def compute_breath_rate_proxy(kps_history, fps=15.0):
    """
    Estimate breathing rate from thorax vertical oscillation via FFT.

    Requires at least 150 frames (~10s at 15fps) for meaningful frequency
    resolution.  Applies a minimum-power noise gate so random keypoint
    jitter does not produce phantom breathing readings.

    Clinical range: 6-30 bpm (neonates up to 60 bpm, but ICU adults 12-20).
    """
    # Need enough frames for >= 3 FFT bins in the valid range
    min_frames = max(60, int(fps / 0.05))  # At least 4s, ideally 10s+
    if len(kps_history) < min_frames:
        return 0.0

    thorax_y_positions = []
    for kps in kps_history:
        if kps:
            tx, ty = get_thorax_point(kps)
            thorax_y_positions.append(ty)

    if len(thorax_y_positions) < min_frames:
        return 0.0

    try:
        y_signal = np.array(thorax_y_positions)
        y_detrended = signal.detrend(y_signal)

        # FFT
        N = len(y_detrended)
        fft_vals = np.fft.fft(y_detrended)
        freqs = np.fft.fftfreq(N, 1.0 / fps)

        # Breathing range: 0.1-0.5 Hz (6-30 bpm)
        valid_idx = (freqs > 0.1) & (freqs < 0.5)
        if not np.any(valid_idx):
            return 0.0

        # Need at least 2 bins in valid range for a meaningful peak
        if np.sum(valid_idx) < 2:
            return 0.0

        power = np.abs(fft_vals[valid_idx]) ** 2
        peak_idx = np.argmax(power)
        peak_power = power[peak_idx]

        # Noise gate: peak must be significantly above the median power
        # in the valid band (SNR > 3x median).  This prevents random
        # keypoint jitter from producing phantom breath rates.
        median_power = np.median(power)
        if median_power > 0 and peak_power < 3.0 * median_power:
            return 0.0  # No clear respiratory signal

        # Also require minimum absolute amplitude (chest rise > ~1mm at 720p)
        peak_amplitude = np.sqrt(peak_power) * 2.0 / N  # Approximate amplitude
        if peak_amplitude < 0.001:
            return 0.0

        peak_freq = freqs[valid_idx][peak_idx]
        breath_rate = peak_freq * 60.0
        return float(np.clip(breath_rate, 0.0, 40.0))
    except Exception:
        pass

    return 0.0


def compute_breathing_variability(kps_history):
    """
    Compute breathing irregularity metric.
    Higher values indicate irregular breathing patterns.
    """
    if len(kps_history) < 10:
        return 0.0
    
    thorax_y_positions = []
    for kps in kps_history:
        if kps:
            tx, ty = get_thorax_point(kps)
            thorax_y_positions.append(ty)
    
    if len(thorax_y_positions) < 10:
        return 0.0
    
    try:
        # Compute variability as coefficient of variation
        y_signal = np.array(thorax_y_positions)
        std_dev = np.std(y_signal)
        mean_val = np.mean(y_signal)
        
        if mean_val < 1e-6:
            return 0.0
        
        cv = std_dev / mean_val
        return float(cv)
    except Exception:
        return 0.0


# ============================================================================
# LAYER 4: HAND-INTENT DETECTOR (Self-Harm Warning)
# ============================================================================

def compute_hand_proximity_risk(kps):
    """
    Compute hand proximity risk to face/neck/tubes.
    Returns risk score 0-1 (higher = more risk).
    """
    try:
        # Get hand positions
        lw = get_keypoint(kps, LEFT_WRIST)
        rw = get_keypoint(kps, RIGHT_WRIST)
        
        # Get face/neck positions
        nose = get_keypoint(kps, NOSE)
        ls = get_keypoint(kps, LEFT_SHOULDER)
        rs = get_keypoint(kps, RIGHT_SHOULDER)
        
        # Neck = midpoint between shoulders
        neck_x = (ls[0] + rs[0]) / 2.0
        neck_y = (ls[1] + rs[1]) / 2.0
        
        risks = []
        
        # Check left hand
        if lw[2] > MIN_CONFIDENCE:
            # Distance to face (nose)
            if nose[2] > MIN_CONFIDENCE:
                dist_face = math.hypot(lw[0] - nose[0], lw[1] - nose[1])
                risk_face = max(0.0, 1.0 - dist_face * 5.0)  # Normalized risk
                risks.append(risk_face)
            
            # Distance to neck
            dist_neck = math.hypot(lw[0] - neck_x, lw[1] - neck_y)
            risk_neck = max(0.0, 1.0 - dist_neck * 5.0)
            risks.append(risk_neck)
        
        # Check right hand
        if rw[2] > MIN_CONFIDENCE:
            # Distance to face
            if nose[2] > MIN_CONFIDENCE:
                dist_face = math.hypot(rw[0] - nose[0], rw[1] - nose[1])
                risk_face = max(0.0, 1.0 - dist_face * 5.0)
                risks.append(risk_face)
            
            # Distance to neck
            dist_neck = math.hypot(rw[0] - neck_x, rw[1] - neck_y)
            risk_neck = max(0.0, 1.0 - dist_neck * 5.0)
            risks.append(risk_neck)
        
        if not risks:
            return 0.0
        
        # Return maximum risk
        return float(np.clip(max(risks), 0.0, 1.0))
        
    except Exception:
        return 0.0


# ============================================================================
# LAYER 5: CHAOS MEASURE (Mathematical Disorder)
# ============================================================================

def compute_motion_entropy(kps_history):
    """
    Compute motion entropy (Shannon entropy of motion patterns).
    Higher entropy = more chaotic/unpredictable motion.
    """
    if len(kps_history) < 5:
        return 0.0
    
    try:
        # Compute velocity magnitudes for each frame
        velocities = []
        for i in range(1, len(kps_history)):
            if kps_history[i] and kps_history[i-1]:
                vels = compute_velocity(kps_history[i], kps_history[i-1])
                if vels:
                    v_mags = [math.sqrt(vx*vx + vy*vy) for vx, vy in vels]
                    avg_v = np.mean(v_mags) if v_mags else 0.0
                    velocities.append(avg_v)
        
        if len(velocities) < 3:
            return 0.0
        
        # Discretize velocities into bins for entropy calculation
        v_array = np.array(velocities)
        bins = np.linspace(0, np.max(v_array) + 1e-6, 10)
        hist, _ = np.histogram(v_array, bins=bins)
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) < 2:
            return 0.0
        
        # Normalize to probabilities
        probs = hist / np.sum(hist)
        
        # Compute Shannon entropy
        ent = entropy(probs, base=2)
        
        # Normalize to 0-1 range (max entropy for uniform distribution)
        max_ent = np.log2(len(probs))
        normalized_ent = ent / max_ent if max_ent > 0 else 0.0
        
        return float(np.clip(normalized_ent, 0.0, 1.0))
        
    except Exception:
        return 0.0


def compute_inter_joint_incoherence(kps_history):
    """
    Compute inter-joint incoherence (lack of coordination).
    Measures how independently joints move (higher = less coordinated).
    """
    if len(kps_history) < 5:
        return 0.0
    
    try:
        # Compute velocities for each joint over time
        joint_velocities = []
        
        for i in range(1, len(kps_history)):
            if kps_history[i] and kps_history[i-1]:
                vels = compute_velocity(kps_history[i], kps_history[i-1])
                if vels:
                    joint_velocities.append(vels)
        
        if len(joint_velocities) < 3:
            return 0.0
        
        # Compute correlation between joint velocities
        # Lower correlation = higher incoherence
        n_joints = len(joint_velocities[0])
        correlations = []
        
        for i in range(n_joints):
            for j in range(i + 1, n_joints):
                v_i = [v[i][0] for v in joint_velocities]
                v_j = [v[j][0] for v in joint_velocities]
                
                if len(v_i) > 1 and len(v_j) > 1:
                    corr = np.corrcoef(v_i, v_j)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        if not correlations:
            return 0.5
        
        # Incoherence = 1 - average correlation
        avg_corr = np.mean(correlations)
        incoherence = 1.0 - avg_corr
        
        return float(np.clip(incoherence, 0.0, 1.0))
        
    except Exception:
        return 0.0


def compute_motion_variability(kps_history):
    """
    Compute motion variability (coefficient of variation of movement).
    Higher variability = more unpredictable motion.
    TODO-050: Vectorized NumPy implementation.
    """
    if len(kps_history) < 5:
        return 0.0

    try:
        displacements = []
        for i in range(1, len(kps_history)):
            if kps_history[i] and kps_history[i-1]:
                # Vectorized per-frame displacement
                curr = np.array(kps_history[i], dtype=np.float32)
                prev = np.array(kps_history[i-1], dtype=np.float32)

                if curr.shape[0] != prev.shape[0]:
                    continue

                valid_mask = (curr[:, 2] > MIN_CONFIDENCE) & (prev[:, 2] > MIN_CONFIDENCE)
                if not np.any(valid_mask):
                    continue

                dx = curr[valid_mask, 0] - prev[valid_mask, 0]
                dy = curr[valid_mask, 1] - prev[valid_mask, 1]
                disp = np.sqrt(dx * dx + dy * dy)
                displacements.append(float(np.mean(disp)))

        if len(displacements) < 3:
            return 0.0

        # Coefficient of variation
        disp_array = np.array(displacements, dtype=np.float32)
        mean_disp = np.mean(disp_array)
        std_disp = np.std(disp_array)

        if mean_disp < 1e-6:
            return 0.0

        cv = std_disp / mean_disp
        return float(np.clip(cv, 0.0, 2.0))

    except Exception:
        return 0.0


# ============================================================================
# ONLINE FEATURE NORMALIZATION
# ============================================================================

class RunningStandardizer:
    """
    Online z-score normalizer using Welford's algorithm.
    Numerically stable running mean/variance for each feature dimension.
    """

    def __init__(self, n_features=9, warmup=100):
        """
        Args:
            n_features: Number of feature dimensions
            warmup: Samples before normalization activates (~6.7s at 15fps)
        """
        self.n_features = n_features
        self.warmup = warmup
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)

    @property
    def variance(self):
        if self.n < 2:
            return np.ones(self.n_features, dtype=np.float64)
        return self.M2 / (self.n - 1)

    @property
    def std(self):
        return np.sqrt(np.maximum(self.variance, 1e-8))

    @property
    def is_warm(self):
        return self.n >= self.warmup

    def update(self, x):
        """Update running statistics with a new sample (Welford's algorithm)."""
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def transform(self, x):
        """
        Update stats with x, then return normalized version.
        During warmup period, returns raw features.
        """
        x = np.asarray(x, dtype=np.float32)
        self.update(x)

        if not self.is_warm:
            return x

        normalized = (x - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        return np.clip(normalized, -5.0, 5.0).astype(np.float32)

    def get_state(self):
        """Serialize state for persistence."""
        return {
            "n": self.n,
            "mean": self.mean.copy(),
            "M2": self.M2.copy(),
            "warmup": self.warmup,
            "n_features": self.n_features,
        }

    def load_state(self, state):
        """Restore from serialized state."""
        self.n = state["n"]
        self.mean = state["mean"].copy()
        self.M2 = state["M2"].copy()
        self.warmup = state["warmup"]
        self.n_features = state["n_features"]


# ============================================================================
# MAIN FEATURE EXTRACTION FUNCTION
# ============================================================================

class ICUFeatureEncoder:
    """
    ICU-Grade Agitation Feature Encoder.
    Transforms raw pose keypoints into 9-dimensional clinical feature vector.
    """

    # Minimum frames for respiratory FFT (>=5 breathing cycles at 6 bpm = 50s)
    RESP_BUFFER_SECONDS = 30  # 30s gives 3 cycles at 6 bpm, resolution ~0.033 Hz

    # Maximum plausible keypoint displacement per frame (normalized coords).
    # At 15 fps a fast arm swing covers ~1m in 0.3s ≈ 50px/frame at 720p ≈ 0.07 norm.
    # Anything beyond this in a single frame is a detection glitch.
    MAX_KP_JUMP = 0.12

    def __init__(self, window_size=30, fps=15.0, normalize=True, warmup=100):
        self.window_size = window_size
        self.fps = fps
        self.kps_history = deque(maxlen=window_size)
        self.prev_kps = None
        self.prev_prev_kps = None
        self.prev_velocities = None
        self.standardizer = RunningStandardizer(n_features=9, warmup=warmup) if normalize else None

        # Dedicated respiratory buffer — much longer than GRU window
        resp_frames = int(self.RESP_BUFFER_SECONDS * fps)
        self._resp_history = deque(maxlen=resp_frames)
        self._last_breath_rate = 0.0

    def _reject_outlier_keypoints(self, kps):
        """
        Clamp keypoints that jumped further than MAX_KP_JUMP from prev frame.

        A single-frame pose estimator glitch (e.g. wrist teleports to frame
        corner) would otherwise spike motion_energy and jerk_index.  If the
        jump exceeds physiological limits we reuse the previous position
        but keep the current confidence (which is usually low for glitches).
        """
        if self.prev_kps is None or len(kps) != len(self.prev_kps):
            return kps

        cleaned = list(kps)
        for i in range(len(kps)):
            if kps[i][2] < MIN_CONFIDENCE or self.prev_kps[i][2] < MIN_CONFIDENCE:
                continue
            dx = kps[i][0] - self.prev_kps[i][0]
            dy = kps[i][1] - self.prev_kps[i][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > self.MAX_KP_JUMP:
                # Snap back to previous position, keep current (likely low) confidence
                cleaned[i] = (self.prev_kps[i][0], self.prev_kps[i][1], kps[i][2])
        return cleaned
    
    def extract_feature_vector(self, kps, prev_kps=None, prev_prev_kps=None):
        """
        Extract ICU-grade clinical feature vector.
        
        Returns:
            np.array of shape (9,) with:
            [0] motion_energy
            [1] jerk_index
            [2] posture_instability (spine angle normalized)
            [3] sway_score (COM sway)
            [4] breath_rate_proxy (breaths/min)
            [5] hand_proximity_risk (0-1)
            [6] motor_entropy (0-1)
            [7] symmetry_index (0-1)
            [8] motion_variability (0-2)
        """
        if not kps or len(kps) < 5:
            return None

        # Reject single-frame outlier keypoints before any computation
        kps = self._reject_outlier_keypoints(kps)

        # Update history
        self.kps_history.append(kps)
        self._resp_history.append(kps)

        # LAYER 1: Kinematic Features
        motion_energy = compute_motion_energy(kps, prev_kps or self.prev_kps, dt=1.0/self.fps)
        jerk_index = compute_jerk_index(
            kps,
            prev_kps or self.prev_kps,
            prev_prev_kps or self.prev_prev_kps,
            dt=1.0/self.fps
        )

        # LAYER 2: Postural Instability
        spine_angle = compute_spine_angle(kps)    # 0 = upright, 90 = horizontal
        neck_angle = compute_neck_angle(kps)      # 0 = upright, 90 = horizontal
        # Weighted sum: spine contributes more than neck tilt.
        # Clamp to [0, 1] — both angles are in [0, 90] for valid poses.
        posture_instability = float(np.clip(
            (spine_angle * 0.7 + neck_angle * 0.3) / 90.0, 0.0, 1.0
        ))
        sway_score = compute_com_sway(list(self.kps_history))
        symmetry_index = compute_symmetry_index(kps)

        # LAYER 3: Respiratory Proxy
        # Use dedicated longer buffer for FFT (30s vs 3.2s GRU window)
        thorax_expansion = compute_thorax_expansion(list(self.kps_history))
        breath_rate_proxy = compute_breath_rate_proxy(
            list(self._resp_history), fps=self.fps
        )
        breathing_variability = compute_breathing_variability(list(self._resp_history))
        
        # LAYER 4: Hand-Intent Detector
        hand_proximity_risk = compute_hand_proximity_risk(kps)
        
        # LAYER 5: Chaos Measure
        motor_entropy = compute_motion_entropy(list(self.kps_history))
        inter_joint_incoherence = compute_inter_joint_incoherence(list(self.kps_history))
        motion_variability = compute_motion_variability(list(self.kps_history))
        
        # Build feature vector
        features = np.array([
            motion_energy,              # [0] Motion energy
            jerk_index,                 # [1] Jerk index
            posture_instability,         # [2] Posture instability
            sway_score,                 # [3] COM sway
            breath_rate_proxy / 40.0,   # [4] Breath rate proxy (normalized 0-1)
            hand_proximity_risk,        # [5] Hand proximity risk
            motor_entropy,              # [6] Motor entropy
            symmetry_index,            # [7] Symmetry index
            motion_variability / 2.0,   # [8] Motion variability (normalized 0-1)
        ], dtype=np.float32)
        
        # Update state
        self.prev_prev_kps = self.prev_kps
        self.prev_kps = kps
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

        # Online z-score normalization (Welford's algorithm)
        if self.standardizer is not None:
            features = self.standardizer.transform(features)

        return features


# Global encoder instance (for backward compatibility)
_encoder = None

def get_encoder(window_size=30, fps=15.0):
    """Get or create global encoder instance."""
    global _encoder
    if _encoder is None:
        _encoder = ICUFeatureEncoder(window_size=window_size, fps=fps)
    return _encoder


def extract_feature_vector(kps, prev_kps=None, prev_prev_kps=None, window_size=30, fps=15.0):
    """
    Backward-compatible wrapper for extract_feature_vector.
    Uses global encoder instance.
    """
    encoder = get_encoder(window_size=window_size, fps=fps)
    return encoder.extract_feature_vector(kps, prev_kps, prev_prev_kps)


def lift_pseudo_3d(kps):
    """Pseudo-3D lifting (kept for backward compatibility)."""
    if not kps:
        return None
    coords = [(x, y) for x, y, _ in kps]
    scale = normalize_by_torso(kps)
    torso_len = scale
    if torso_len == 0:
        torso_len = 1.0
    zs = []
    for (x, y), (_, _, c) in zip(coords, kps):
        z = (1.0 - y) * 0.5 + 0.5 * (1.0 / torso_len)
        zs.append(z)
    return [(x, y, z) for (x, y), z in zip(coords, zs)]
