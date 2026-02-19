# analytics/keypoint_utils.py
"""
Shared COCO keypoint constants and utility functions.
Used by activity.py, posture.py, frame_visibility.py, enhanced_activity_classifier.py.
"""
import math
import numpy as np

# COCO-17 Keypoint indices
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

MIN_CONFIDENCE = 0.3


def get_keypoint(kps, idx, default=(0.0, 0.0, 0.0)):
    """Safely get keypoint with confidence check."""
    if idx < len(kps) and kps[idx][2] > MIN_CONFIDENCE:
        return kps[idx]
    return default


def point_distance(p1, p2):
    """Euclidean distance between two 2D points (uses first two coords)."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_joint_center(kps, left_idx, right_idx):
    """Get midpoint between left and right joint keypoints."""
    left = get_keypoint(kps, left_idx)
    right = get_keypoint(kps, right_idx)
    return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)


def compute_vertical_extent(kps):
    """Compute vertical extent of body (head to feet)."""
    try:
        nose = get_keypoint(kps, NOSE)
        lankle = get_keypoint(kps, LEFT_ANKLE)
        rankle = get_keypoint(kps, RIGHT_ANKLE)
        top_y = nose[1]
        bottom_y = max(lankle[1], rankle[1])
        return abs(bottom_y - top_y)
    except Exception:
        return 0.0


def compute_horizontal_extent_full(kps):
    """Compute horizontal extent using shoulders and hips."""
    try:
        lshoulder = get_keypoint(kps, LEFT_SHOULDER)
        rshoulder = get_keypoint(kps, RIGHT_SHOULDER)
        lhip = get_keypoint(kps, LEFT_HIP)
        rhip = get_keypoint(kps, RIGHT_HIP)
        left_x = min(lshoulder[0], lhip[0])
        right_x = max(rshoulder[0], rhip[0])
        return abs(right_x - left_x)
    except Exception:
        return 0.0


def compute_horizontal_extent_shoulders(kps):
    """Compute horizontal extent using shoulders only."""
    try:
        lshoulder = get_keypoint(kps, LEFT_SHOULDER)
        rshoulder = get_keypoint(kps, RIGHT_SHOULDER)
        return abs(rshoulder[0] - lshoulder[0])
    except Exception:
        return 0.0


def compute_knee_angle(kps, side='left'):
    """Compute knee angle (degrees) for walking detection."""
    try:
        if side == 'left':
            hip = get_keypoint(kps, LEFT_HIP)
            knee = get_keypoint(kps, LEFT_KNEE)
            ankle = get_keypoint(kps, LEFT_ANKLE)
        else:
            hip = get_keypoint(kps, RIGHT_HIP)
            knee = get_keypoint(kps, RIGHT_KNEE)
            ankle = get_keypoint(kps, RIGHT_ANKLE)

        v1_x = knee[0] - hip[0]
        v1_y = knee[1] - hip[1]
        v2_x = ankle[0] - knee[0]
        v2_y = ankle[1] - knee[1]

        dot = v1_x * v2_x + v1_y * v2_y
        mag1 = math.sqrt(v1_x * v1_x + v1_y * v1_y)
        mag2 = math.sqrt(v2_x * v2_x + v2_y * v2_y)

        if mag1 < 1e-6 or mag2 < 1e-6:
            return 180.0

        cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))
    except Exception:
        return 180.0
