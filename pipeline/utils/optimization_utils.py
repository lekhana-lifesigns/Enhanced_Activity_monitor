# pipeline/utils/optimization_utils.py
# Optimization utilities for Enhanced Activity Monitor
# Implements data structure and algorithm best practices

import numpy as np
import hashlib
import heapq
from collections import OrderedDict, defaultdict
from typing import List, Dict, Tuple, Optional, Any
import functools


class KeypointDistanceCache:
 """
 Pre-computes and caches pairwise keypoint distances using vectorized operations.

 """

 def __init__(self):
 # Pre-allocated distance matrix (17x17 for COCO keypoints)
 self.distance_matrix = np.zeros((17, 17), dtype=np.float32)
 self.squared_distance_matrix = np.zeros((17, 17), dtype=np.float32)
 self.last_kps_hash = None
 self.cache_valid = False

 def compute_distance_matrix(self, kps: np.ndarray) -> np.ndarray:
 """
 Compute all pairwise distances at once using vectorized operations.

 Args:
 kps: Keypoints array of shape (17, 2) or (17, 3)

 Returns:
 (17, 17) matrix where [i, j] = Euclidean distance between keypoint i and j
 """
 if kps.shape[0] < 17:
 # Pad with zeros if necessary
 padded_kps = np.zeros((17, kps.shape[1]), dtype=np.float32)
 padded_kps[:kps.shape[0]] = kps
 kps = padded_kps

 positions = kps[:, :2].astype(np.float32) # (17, 2)

 # Vectorized pairwise distance using broadcasting
 # positions[:, None, :] -> (17, 1, 2)
 # positions[None, :, :] -> (1, 17, 2)
 # Result: (17, 17, 2) -> (17, 17) after norm
 diff = positions[:, None, :] - positions[None, :, :] # (17, 17, 2)

 # Compute both Euclidean and squared distances
 self.squared_distance_matrix = np.sum(diff ** 2, axis=2) # (17, 17)
 self.distance_matrix = np.sqrt(self.squared_distance_matrix) # (17, 17)

 self.cache_valid = True
 return self.distance_matrix

 def get_distance(self, idx1: int, idx2: int) -> float:
 """Get pre-computed distance between two keypoints."""
 if not self.cache_valid:
 raise ValueError("Distance matrix not computed. Call compute_distance_matrix() first.")
 return float(self.distance_matrix[idx1, idx2])

 def get_squared_distance(self, idx1: int, idx2: int) -> float:
 """Get pre-computed squared distance (faster, no sqrt)."""
 if not self.cache_valid:
 raise ValueError("Distance matrix not computed. Call compute_distance_matrix() first.")
 return float(self.squared_distance_matrix[idx1, idx2])

 def get_distances_from_keypoint(self, idx: int) -> np.ndarray:
 """Get all distances from one keypoint to all others."""
 if not self.cache_valid:
 raise ValueError("Distance matrix not computed. Call compute_distance_matrix() first.")
 return self.distance_matrix[idx, :]

 def get_k_nearest(self, idx: int, k: int = 5) -> List[Tuple[int, float]]:
 """Get k nearest keypoints."""
 if not self.cache_valid:
 raise ValueError("Distance matrix not computed. Call compute_distance_matrix() first.")

 distances = self.distance_matrix[idx, :]
 # Get k smallest distances (excluding self at distance 0)
 indices_distances = [(i, dist) for i, dist in enumerate(distances) if i != idx]
 k_nearest = heapq.nsmallest(k, indices_distances, key=lambda x: x[1])
 return k_nearest


class FrameCache:
 """
 LRU cache for frame-based computations with automatic invalidation.

 """

 def __init__(self, max_size: int = 100):
 self.cache = OrderedDict()
 self.max_size = max_size
 self.current_frame_id = None
 self.hit_count = 0
 self.miss_count = 0

 def clear_frame(self, frame_id: Any):
 """Clear cache when new frame arrives."""
 if frame_id != self.current_frame_id:
 self.cache.clear()
 self.current_frame_id = frame_id
 self.hit_count = 0
 self.miss_count = 0

 def get(self, key: str) -> Optional[Any]:
 """Get cached value, moving to end (most recently used)."""
 if key in self.cache:
 self.cache.move_to_end(key)
 self.hit_count += 1
 return self.cache[key]
 self.miss_count += 1
 return None

 def set(self, key: str, value: Any):
 """Set value, evicting LRU if necessary."""
 if key in self.cache:
 self.cache.move_to_end(key)
 self.cache[key] = value

 # Evict LRU if over capacity
 if len(self.cache) > self.max_size:
 self.cache.popitem(last=False)

 def get_hit_rate(self) -> float:
 """Get cache hit rate for monitoring."""
 total = self.hit_count + self.miss_count
 return self.hit_count / total if total > 0 else 0.0


def memoize_frame_computation(cache_instance: FrameCache):
 """
 Decorator for memoizing computations within a frame.


 Usage:
 @memoize_frame_computation(my_cache)
 def expensive_function(kps, param1, param2):
 ...
 """
 def decorator(func):
 @functools.wraps(func)
 def wrapper(*args, **kwargs):
 # Create cache key from function name and arguments
 key_parts = [func.__name__]

 # Hash arguments using stable hash (MD5) to avoid Python hash randomization
 def stable_hash(obj):
 """Compute stable hash that is consistent across Python sessions."""
 if isinstance(obj, np.ndarray):
 return hashlib.md5(obj.tobytes()).hexdigest()[:8]
 elif isinstance(obj, (list, tuple)):
 # Convert to string representation for stable hashing
 return hashlib.md5(str(obj).encode()).hexdigest()[:8]
 elif isinstance(obj, (int, float, str, bool)):
 return hashlib.md5(str(obj).encode()).hexdigest()[:8]
 else:
 # Fallback: use string representation
 return hashlib.md5(repr(obj).encode()).hexdigest()[:8]

 for arg in args:
 key_parts.append(stable_hash(arg))

 for k, v in sorted(kwargs.items()):
 key_parts.append(f"{k}={stable_hash(v)}")

 cache_key = "_".join(key_parts)

 # Check cache
 cached_value = cache_instance.get(cache_key)
 if cached_value is not None:
 return cached_value

 # Compute and cache
 result = func(*args, **kwargs)
 cache_instance.set(cache_key, result)
 return result

 return wrapper
 return decorator


def select_top_k_detections(detections: List[Dict], k: int = 5,
 score_key: str = 'confidence') -> List[Dict]:
 """
 Select top-k detections using heapq for efficiency.


 Args:
 detections: List of detection dictionaries
 k: Number of top detections to return
 score_key: Key for scoring (e.g., 'confidence', 'score')

 Returns:
 List of top-k detections sorted by score (descending)
 """
 if not detections:
 return []


 return heapq.nlargest(k, detections, key=lambda x: x.get(score_key, 0.0))


def select_best_with_criteria(items: List[Any],
 score_func: callable,
 k: int = 1) -> List[Any]:
 """
 Select best items using custom scoring function with heapq.


 Args:
 items: List of items to score
 score_func: Function to score each item
 k: Number of items to return

 Returns:
 Top-k items by score
 """
 if not items:
 return []

 return heapq.nlargest(k, items, key=score_func)


class CircularBuffer:
 """
 Efficient circular buffer using pre-allocated NumPy array.

 """

 def __init__(self, capacity: int, item_shape: Tuple, dtype=np.float32):
 """
 Args:
 capacity: Maximum number of items
 item_shape: Shape of each item (e.g., (17, 3) for keypoints)
 dtype: NumPy data type
 """
 self.capacity = capacity
 self.item_shape = item_shape
 self.buffer = np.zeros((capacity,) + item_shape, dtype=dtype)
 self.idx = 0
 self.filled = False

 def append(self, item: np.ndarray):
 """Append item to buffer (O(1) operation)."""
 # Ensure item matches shape
 item_array = np.asarray(item, dtype=self.buffer.dtype)

 # Pad or truncate if necessary
 if item_array.shape != self.item_shape:
 padded = np.zeros(self.item_shape, dtype=self.buffer.dtype)
 min_shape = tuple(min(s1, s2) for s1, s2 in zip(item_array.shape, self.item_shape))
 slices = tuple(slice(0, s) for s in min_shape)
 padded[slices] = item_array[slices]
 item_array = padded

 self.buffer[self.idx] = item_array
 self.idx = (self.idx + 1) % self.capacity

 if not self.filled and self.idx == 0:
 self.filled = True

 def get_chronological(self, n: Optional[int] = None) -> np.ndarray:
 """
 Get buffer contents in chronological order.

 Args:
 n: Number of recent items to return (None = all)

 Returns:
 Array of shape (T, *item_shape) where T <= capacity
 """
 if not self.filled:
 # Buffer not full, return available frames
 available = self.buffer[:self.idx]
 if n is not None:
 return available[-n:]
 return available

 # Buffer full, reconstruct chronological order
 if self.idx == 0:
 data = self.buffer
 else:
 # Use np.concatenate (optimized C implementation)
 data = np.concatenate([self.buffer[self.idx:], self.buffer[:self.idx]], axis=0)

 if n is not None:
 return data[-n:]
 return data

 def get_latest(self, n: int = 1) -> np.ndarray:
 """Get n most recent items."""
 chronological = self.get_chronological()
 return chronological[-n:]

 def __len__(self) -> int:
 """Return number of items in buffer."""
 return self.capacity if self.filled else self.idx

 def clear(self):
 """Clear buffer."""
 self.buffer.fill(0)
 self.idx = 0
 self.filled = False


def compute_pairwise_distances_vectorized(points: np.ndarray) -> np.ndarray:
 """
 Compute all pairwise Euclidean distances using vectorized operations.


 Args:
 points: Array of shape (n, d) where n = number of points, d = dimensions

 Returns:
 Distance matrix of shape (n, n)
 """
 # points[:, None, :] -> (n, 1, d)
 # points[None, :, :] -> (1, n, d)
 # diff -> (n, n, d)
 diff = points[:, None, :] - points[None, :, :]
 distances = np.linalg.norm(diff, axis=2)
 return distances


def compute_velocities_vectorized(positions: np.ndarray, dt: float = 1.0) -> np.ndarray:
 """
 Compute velocities from position history using vectorized operations.


 Args:
 positions: Array of shape (T, n, d) where T = timesteps, n = points, d = dimensions
 dt: Time delta

 Returns:
 Velocities of shape (T-1, n, d)
 """
 return np.diff(positions, axis=0) / dt


def compute_accelerations_vectorized(velocities: np.ndarray, dt: float = 1.0) -> np.ndarray:
 """
 Compute accelerations from velocity history using vectorized operations.


 Args:
 velocities: Array of shape (T, n, d)
 dt: Time delta

 Returns:
 Accelerations of shape (T-1, n, d)
 """
 return np.diff(velocities, axis=0) / dt


def compute_motion_energy_vectorized(velocities: np.ndarray) -> float:
 """
 Compute motion energy: E = 0.5 * v² using vectorized operations.


 Args:
 velocities: Array of shape (n, d) where n = keypoints, d = dimensions

 Returns:
 Mean kinetic energy across all keypoints
 """
 # Velocity magnitudes: (n,)
 v_magnitudes = np.linalg.norm(velocities, axis=1)

 # Energy per joint: E = 0.5 * v²
 energies = 0.5 * v_magnitudes ** 2

 # Mean energy
 return float(np.mean(energies))


def batch_normalize(features: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
 """
 Batch normalization using vectorized operations.


 Args:
 features: Feature array
 axis: Normalization axis
 eps: Small constant for numerical stability

 Returns:
 Normalized features
 """
 mean = np.mean(features, axis=axis, keepdims=True)
 std = np.std(features, axis=axis, keepdims=True)
 return (features - mean) / (std + eps)


# HASH-BASED UTILITIES

def compute_keypoint_hash(kps: np.ndarray, precision: int = 3) -> str:
 """
 Compute stable hash for keypoints for caching.


 Args:
 kps: Keypoints array
 precision: Decimal precision for hashing

 Returns:
 MD5 hash string
 """
 # Round to precision for stable hashing
 kps_rounded = np.round(kps, decimals=precision)
 return hashlib.md5(kps_rounded.tobytes()).hexdigest()


def compute_pose_similarity(kps1: np.ndarray, kps2: np.ndarray) -> float:
 """
 Compute similarity between two poses (0-1, higher = more similar).


 Args:
 kps1, kps2: Keypoint arrays of shape (n, 2) or (n, 3)

 Returns:
 Similarity score (0-1)
 """
 # Use only positions (x, y)
 pos1 = kps1[:, :2]
 pos2 = kps2[:, :2]

 # Compute normalized Euclidean distance
 diff = pos1 - pos2
 distance = np.linalg.norm(diff)

 # Convert to similarity (exponential decay)
 similarity = np.exp(-distance)
 return float(similarity)


# PERFORMANCE MONITORING

class PerformanceMonitor:
 """
 Monitor performance metrics for optimizations.
 """

 def __init__(self):
 self.metrics = defaultdict(list)

 def record(self, metric_name: str, value: float):
 """Record a metric value."""
 self.metrics[metric_name].append(value)

 def get_statistics(self, metric_name: str) -> Dict[str, float]:
 """Get statistics for a metric."""
 values = self.metrics.get(metric_name, [])
 if not values:
 return {}

 return {
 'mean': float(np.mean(values)),
 'std': float(np.std(values)),
 'min': float(np.min(values)),
 'max': float(np.max(values)),
 'p50': float(np.percentile(values, 50)),
 'p95': float(np.percentile(values, 95)),
 'p99': float(np.percentile(values, 99)),
 }

 def get_all_statistics(self) -> Dict[str, Dict]:
 """Get statistics for all metrics."""
 return {name: self.get_statistics(name) for name in self.metrics.keys()}

 def print_report(self):
 """Print performance report."""
 print("\n" + "=" * 60)
 print("PERFORMANCE METRICS REPORT")
 print("=" * 60)

 for metric_name, stats in self.get_all_statistics().items():
 print(f"\n{metric_name}:")
 for stat_name, value in stats.items():
 print(f" {stat_name}: {value:.4f}")
