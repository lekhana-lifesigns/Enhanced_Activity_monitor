# pipeline/patient/lsh_reid.py
# Locality-Sensitive Hashing for Person Re-Identification

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional, Dict
import heapq


class LSHReID:
 """
 Locality-Sensitive Hashing for efficient person re-identification.
 Uses random projection LSH for appearance feature matching.

 - LSH enables O(log n) average nearest neighbor search
 - Random projection preserves angular distances (cosine similarity)
 - Multiple hash tables improve recall

 - Insert: O(L * k) where L = num_tables, k = hash_size
 - Query: O(L * k + m) where m = candidates found
 - Space: O(L * 2^k + n * d) where n = features, d = dimensions
 """

 def __init__(self, feature_dim: int = 512, num_tables: int = 8, hash_size: int = 16):
 """
 Initialize LSH re-identification system.

 Args:
 feature_dim: Dimension of appearance features (e.g., from ResNet)
 num_tables: Number of hash tables (more = better recall, slower)
 hash_size: Number of hash bits (more = better precision, more memory)
 """
 self.feature_dim = feature_dim
 self.num_tables = num_tables
 self.hash_size = hash_size

 # Random projection matrices for each hash table
 # Using Gaussian random projection for LSH
 self.projections = []
 np.random.seed(42) # For reproducibility
 for _ in range(num_tables):
 # Random Gaussian projection: (hash_size, feature_dim)
 proj = np.random.randn(hash_size, feature_dim).astype(np.float32)
 # Normalize rows for stable hashing
 proj /= np.linalg.norm(proj, axis=1, keepdims=True)
 self.projections.append(proj)

 # Hash tables: {hash_value: [track_ids]}
 self.hash_tables = [defaultdict(list) for _ in range(num_tables)]

 # Track ID to feature mapping
 self.track_features = {} # {track_id: feature_vector}

 # Statistics
 self.num_inserts = 0
 self.num_queries = 0
 self.total_candidates = 0

 def _compute_hash(self, feature: np.ndarray, table_idx: int) -> int:
 """
 Compute LSH hash for a feature vector using random projection.


 Args:
 feature: Feature vector of shape (feature_dim,)
 table_idx: Index of hash table (0 to num_tables-1)

 Returns:
 Integer hash value (0 to 2^hash_size - 1)
 """
 # Project feature: (hash_size,) = (hash_size, feature_dim) @ (feature_dim,)
 projection = self.projections[table_idx] @ feature

 # Threshold at 0: positive = 1, negative = 0
 # This creates a binary hash preserving angular similarity
 binary_hash = (projection > 0).astype(np.uint8)

 # Convert binary array to integer
 hash_value = 0
 for bit in binary_hash:
 hash_value = (hash_value << 1) | bit

 return hash_value

 def insert(self, track_id: int, feature: np.ndarray):
 """
 Insert a feature vector into LSH hash tables.


 Args:
 track_id: Unique track identifier
 feature: Appearance feature vector of shape (feature_dim,)
 """
 feature = np.asarray(feature, dtype=np.float32)

 if feature.shape[0] != self.feature_dim:
 raise ValueError(f"Feature dim mismatch: {feature.shape[0]} vs {self.feature_dim}")

 # Normalize feature for cosine similarity
 feature_norm = np.linalg.norm(feature)
 if feature_norm < 1e-8:
 return # Skip zero features
 feature = feature / feature_norm

 # Update or insert feature
 if track_id in self.track_features:
 # Remove from old hash buckets before updating
 self.remove(track_id)

 self.track_features[track_id] = feature

 # Insert into all hash tables
 for table_idx in range(self.num_tables):
 hash_value = self._compute_hash(feature, table_idx)
 self.hash_tables[table_idx][hash_value].append(track_id)

 self.num_inserts += 1

 def query(self, feature: np.ndarray, top_k: int = 5,
 similarity_threshold: float = 0.7) -> List[Tuple[int, float]]:
 """
 Query LSH tables for similar features.

 
 - Multiple tables reduce false negatives

 Args:
 feature: Query feature vector of shape (feature_dim,)
 top_k: Return top-k most similar
 similarity_threshold: Minimum cosine similarity (0-1)

 Returns:
 List of (track_id, similarity_score) tuples, sorted descending by similarity
 """
 feature = np.asarray(feature, dtype=np.float32)

 # Normalize for cosine similarity
 feature_norm = np.linalg.norm(feature)
 if feature_norm < 1e-8:
 return []
 feature = feature / feature_norm

 # Phase 1: Collect candidates from all hash tables
 candidates = set()
 for table_idx in range(self.num_tables):
 hash_value = self._compute_hash(feature, table_idx)
 candidates.update(self.hash_tables[table_idx][hash_value])

 self.num_queries += 1
 self.total_candidates += len(candidates)

 # Phase 2: Compute exact similarities for candidates only
 similarities = []
 for track_id in candidates:
 if track_id in self.track_features:
 stored_feature = self.track_features[track_id]

 # Cosine similarity (features are normalized, so just dot product)
 similarity = float(np.dot(feature, stored_feature))

 # Clip to [0, 1] range (numerical stability)
 similarity = np.clip(similarity, 0.0, 1.0)

 if similarity >= similarity_threshold:
 similarities.append((track_id, similarity))

 # Phase 3: Return top-k using heapq
 if not similarities:
 return []

 return heapq.nlargest(top_k, similarities, key=lambda x: x[1])

 def remove(self, track_id: int):
 """
 Remove a track from all hash tables.

 Args:
 track_id: Track ID to remove
 """
 if track_id not in self.track_features:
 return

 feature = self.track_features[track_id]

 # Remove from all hash tables
 for table_idx in range(self.num_tables):
 hash_value = self._compute_hash(feature, table_idx)
 bucket = self.hash_tables[table_idx][hash_value]
 if track_id in bucket:
 bucket.remove(track_id)
 # Clean up empty buckets
 if not bucket:
 del self.hash_tables[table_idx][hash_value]

 # Remove feature
 del self.track_features[track_id]

 def update(self, track_id: int, feature: np.ndarray):
 """
 Update feature for existing track.


 Args:
 track_id: Track ID
 feature: New feature vector
 """
 feature = np.asarray(feature, dtype=np.float32)

 if track_id in self.track_features:
 # Exponential moving average: 0.7 * old + 0.3 * new
 old_feature = self.track_features[track_id]
 updated_feature = 0.7 * old_feature + 0.3 * feature
 # Re-normalize
 updated_feature = updated_feature / np.linalg.norm(updated_feature)

 # Remove old entry and insert updated
 self.remove(track_id)
 self.insert(track_id, updated_feature)
 else:
 # New track
 self.insert(track_id, feature)

 def get_statistics(self) -> Dict[str, float]:
 """
 Get statistics about LSH performance.

 Returns:
 Dictionary with performance metrics
 """
 avg_candidates = self.total_candidates / self.num_queries if self.num_queries > 0 else 0

 # Compute average bucket size
 bucket_sizes = []
 for table in self.hash_tables:
 bucket_sizes.extend([len(bucket) for bucket in table.values()])

 avg_bucket_size = np.mean(bucket_sizes) if bucket_sizes else 0
 max_bucket_size = np.max(bucket_sizes) if bucket_sizes else 0

 return {
 'num_tracks': len(self.track_features),
 'num_inserts': self.num_inserts,
 'num_queries': self.num_queries,
 'avg_candidates_per_query': avg_candidates,
 'avg_bucket_size': float(avg_bucket_size),
 'max_bucket_size': float(max_bucket_size),
 'total_buckets': sum(len(table) for table in self.hash_tables),
 }

 def clear(self):
 """Clear all hash tables and features."""
 for table in self.hash_tables:
 table.clear()
 self.track_features.clear()
 self.num_inserts = 0
 self.num_queries = 0
 self.total_candidates = 0

 def __len__(self) -> int:
 """Return number of tracked features."""
 return len(self.track_features)

 def __contains__(self, track_id: int) -> bool:
 """Check if track ID exists."""
 return track_id in self.track_features


# LSH-BASED RE-IDENTIFICATION MANAGER

class LSHReIDManager:
 """
 High-level manager for LSH-based person re-identification.

 Handles:
 - Feature extraction integration
 - Track lifecycle management
 - Re-identification logic
 """

 def __init__(self, feature_dim: int = 512, max_age: int = 30):
 """
 Args:
 feature_dim: Feature dimension
 max_age: Maximum frames to keep track without update
 """
 self.lsh = LSHReID(feature_dim=feature_dim, num_tables=8, hash_size=16)
 self.max_age = max_age

 # Track age management
 self.track_ages = {} # {track_id: age}
 self.track_last_seen = {} # {track_id: frame_id}

 self.current_frame_id = 0

 def update_frame(self, frame_id: Optional[int] = None):
 """Update frame counter and age tracks."""
 if frame_id is not None:
 self.current_frame_id = frame_id
 else:
 self.current_frame_id += 1

 # Age out old tracks
 to_remove = []
 for track_id, last_seen in self.track_last_seen.items():
 age = self.current_frame_id - last_seen
 if age > self.max_age:
 to_remove.append(track_id)

 for track_id in to_remove:
 self.remove_track(track_id)

 def add_or_update_track(self, track_id: int, feature: np.ndarray):
 """Add or update a track with new feature."""
 self.lsh.update(track_id, feature)
 self.track_last_seen[track_id] = self.current_frame_id
 self.track_ages[track_id] = 0

 def find_matches(self, feature: np.ndarray, top_k: int = 3,
 similarity_threshold: float = 0.75) -> List[Tuple[int, float]]:
 """
 Find matching tracks for a given feature.

 Args:
 feature: Query feature vector
 top_k: Number of top matches to return
 similarity_threshold: Minimum similarity for match

 Returns:
 List of (track_id, similarity) tuples
 """
 return self.lsh.query(feature, top_k=top_k,
 similarity_threshold=similarity_threshold)

 def remove_track(self, track_id: int):
 """Remove a track."""
 self.lsh.remove(track_id)
 self.track_ages.pop(track_id, None)
 self.track_last_seen.pop(track_id, None)

 def get_statistics(self) -> Dict:
 """Get statistics."""
 lsh_stats = self.lsh.get_statistics()
 lsh_stats['active_tracks'] = len(self.track_last_seen)
 return lsh_stats
