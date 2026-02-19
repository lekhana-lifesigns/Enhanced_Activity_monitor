#!/usr/bin/env python3
"""
Test script for optimization implementations.
Verifies correctness and measures performance improvements.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import optimization modules
from pipeline.utils.optimization_utils import (
 KeypointDistanceCache,
 CircularBuffer,
 FrameCache,
 compute_motion_energy_vectorized,
 compute_velocities_vectorized,
 select_top_k_detections,
 memoize_frame_computation,
)
from analytics.activity_decision_tree import (
 ActivityDecisionTree,
 ActivitySequenceOptimizer,
)
from pipeline.patient.lsh_reid import LSHReID, LSHReIDManager
from pipeline.pose.skeleton_graph import SkeletonGraph


class OptimizationTester:
 """
 Comprehensive test suite for optimization implementations.
 """

 def __init__(self):
 self.test_kps = self._generate_test_keypoints()
 self.passed_tests = 0
 self.failed_tests = 0
 self.total_speedup = 1.0

 def _generate_test_keypoints(self, num_frames=100):
 """Generate synthetic test keypoints."""
 np.random.seed(42)
 kps = np.random.rand(num_frames, 17, 3).astype(np.float32)
 kps[:, :, 0] *= 640 # x coordinates (0-640)
 kps[:, :, 1] *= 480 # y coordinates (0-480)
 kps[:, :, 2] = np.clip(kps[:, :, 2], 0.3, 1.0) # Confidence (0.3-1.0)
 return kps

 def test_distance_cache(self):
 """Test KeypointDistanceCache."""
 print("\n[TEST] KeypointDistanceCache")
 print("-" * 60)

 try:
 cache = KeypointDistanceCache()
 kps = self.test_kps[0]

 # Compute distance matrix
 distances = cache.compute_distance_matrix(kps)

 # Verify shape
 assert distances.shape == (17, 17), "Distance matrix shape mismatch"

 # Verify symmetry
 assert np.allclose(distances, distances.T), "Distance matrix not symmetric"

 # Verify diagonal is zero
 assert np.allclose(np.diag(distances), 0.0), "Diagonal should be zero"

 # Test get_distance
 dist_01 = cache.get_distance(0, 1)
 expected_dist = np.linalg.norm(kps[0, :2] - kps[1, :2])
 assert np.isclose(dist_01, expected_dist, atol=1e-5), "Distance computation mismatch"

 # Test k-nearest
 k_nearest = cache.get_k_nearest(0, k=5)
 assert len(k_nearest) == 5, "k-nearest should return 5 items"
 assert all(idx != 0 for idx, _ in k_nearest), "Should exclude self"

 print(" All tests passed")
 self.passed_tests += 1
 return True

 except AssertionError as e:
 print(f" Test failed: {e}")
 self.failed_tests += 1
 return False

 def test_circular_buffer(self):
 """Test CircularBuffer."""
 print("\n[TEST] CircularBuffer")
 print("-" * 60)

 try:
 buffer = CircularBuffer(capacity=10, item_shape=(17, 3), dtype=np.float32)

 # Test append
 for i in range(15): # Append more than capacity
 buffer.append(self.test_kps[i])

 # Verify length
 assert len(buffer) == 10, "Buffer should be at capacity"

 # Get chronological data
 chronological = buffer.get_chronological()
 assert chronological.shape[0] == 10, "Should return 10 frames"

 # Verify order (most recent should be last)
 latest = buffer.get_latest(n=1)
 assert np.allclose(latest[0], self.test_kps[14]), "Latest frame mismatch"

 # Test partial fill
 buffer2 = CircularBuffer(capacity=20, item_shape=(17, 3))
 for i in range(5):
 buffer2.append(self.test_kps[i])
 assert len(buffer2) == 5, "Partial buffer length mismatch"

 print(" All tests passed")
 self.passed_tests += 1
 return True

 except AssertionError as e:
 print(f" Test failed: {e}")
 self.failed_tests += 1
 return False

 def test_heapq_topk(self):
 """Test heapq-based top-k selection."""
 print("\n[TEST] Heapq Top-K Selection")
 print("-" * 60)

 try:
 # Generate test detections
 detections = [
 {'id': i, 'confidence': np.random.rand()}
 for i in range(20)
 ]

 # Select top-5
 top_5 = select_top_k_detections(detections, k=5, score_key='confidence')

 # Verify count
 assert len(top_5) == 5, "Should return 5 detections"

 # Verify sorted descending
 confidences = [d['confidence'] for d in top_5]
 assert confidences == sorted(confidences, reverse=True), "Should be sorted descending"

 # Verify correctness (manual verification)
 all_confidences = sorted([d['confidence'] for d in detections], reverse=True)
 assert all(c in all_confidences[:5] for c in confidences), "Top-5 verification failed"

 print(" All tests passed")
 self.passed_tests += 1
 return True

 except AssertionError as e:
 print(f" Test failed: {e}")
 self.failed_tests += 1
 return False

 def test_lsh_reid(self):
 """Test LSH re-identification."""
 print("\n[TEST] LSH Re-Identification")
 print("-" * 60)

 try:
 lsh = LSHReID(feature_dim=128, num_tables=4, hash_size=8)

 # Generate test features
 num_tracks = 20
 features = np.random.randn(num_tracks, 128).astype(np.float32)

 # Insert features
 for i, feat in enumerate(features):
 lsh.insert(track_id=i, feature=feat)

 assert len(lsh) == num_tracks, "Track count mismatch"

 # Query with one of the inserted features (should match itself)
 query_feat = features[5]
 matches = lsh.query(query_feat, top_k=3, similarity_threshold=0.8)

 # Should find at least the query feature itself
 assert len(matches) > 0, "Should find at least one match"
 assert matches[0][0] == 5, "Top match should be query feature itself"
 assert matches[0][1] > 0.99, "Self-similarity should be ~1.0"

 # Test removal
 lsh.remove(track_id=5)
 assert len(lsh) == num_tracks - 1, "Track should be removed"

 # Test update
 new_feature = np.random.randn(128).astype(np.float32)
 lsh.update(track_id=10, feature=new_feature)

 # Test statistics
 stats = lsh.get_statistics()
 assert 'num_tracks' in stats, "Statistics should include num_tracks"

 print(" All tests passed")
 self.passed_tests += 1
 return True

 except AssertionError as e:
 print(f" Test failed: {e}")
 self.failed_tests += 1
 return False

 def test_skeleton_graph(self):
 """Test SkeletonGraph."""
 print("\n[TEST] Skeleton Graph")
 print("-" * 60)

 try:
 # Create graph with test keypoints
 graph = SkeletonGraph(keypoints=self.test_kps[0])

 # Test neighbors
 nose_neighbors = graph.get_neighbors(0) # NOSE
 assert len(nose_neighbors) > 0, "Nose should have neighbors"

 # Test path length
 path_len = graph.get_path_length(0, 5) # NOSE to LEFT_SHOULDER
 assert path_len == 1, "Direct connection should have path length 1"

 # Test connected joints
 connected = graph.get_connected_joints(0, max_distance=2)
 assert len(connected) > 0, "Should find connected joints"

 # Test joint groups
 torso_joints = graph.get_joint_group('torso')
 assert len(torso_joints) == 4, "Torso should have 4 joints"

 # Test BFS
 bfs_result = graph.bfs(start_joint=0, max_depth=2)
 assert len(bfs_result) > 0, "BFS should return results"

 # Test joint angles
 graph.update_keypoints(self.test_kps[1])
 angles = graph.compute_joint_angles_vectorized()
 assert isinstance(angles, dict), "Should return dict of angles"

 # Test limb vectors
 vectors = graph.compute_limb_vectors()
 assert len(vectors) > 0, "Should compute limb vectors"

 print(" All tests passed")
 self.passed_tests += 1
 return True

 except AssertionError as e:
 print(f" Test failed: {e}")
 self.failed_tests += 1
 return False

 def test_decision_tree(self):
 """Test ActivityDecisionTree."""
 print("\n[TEST] Activity Decision Tree")
 print("-" * 60)

 try:
 tree = ActivityDecisionTree()

 # Test classification
 kps = self.test_kps[0]
 kps_history = list(self.test_kps[:10])

 result = tree.classify_with_early_exit(
 kps=kps,
 kps_history=kps_history,
 context={}
 )

 # Verify result structure
 assert 'activity' in result, "Result should have 'activity'"
 assert 'confidence' in result, "Result should have 'confidence'"
 assert 'priority' in result, "Result should have 'priority'"
 assert 'details' in result, "Result should have 'details'"

 # Verify confidence range
 assert 0.0 <= result['confidence'] <= 1.0, "Confidence should be 0-1"

 # Test with fall detected
 result_fall = tree.classify_with_early_exit(
 kps=kps,
 kps_history=kps_history,
 context={'fall_detected': True}
 )
 assert result_fall['activity'] == 'falling', "Should detect falling"

 print(" All tests passed")
 self.passed_tests += 1
 return True

 except AssertionError as e:
 print(f" Test failed: {e}")
 self.failed_tests += 1
 return False

 def test_sequence_optimizer(self):
 """Test ActivitySequenceOptimizer."""
 print("\n[TEST] Activity Sequence Optimizer")
 print("-" * 60)

 try:
 activities = ['lying', 'sitting', 'standing', 'walking']
 optimizer = ActivitySequenceOptimizer(activities)

 # Test sequence optimization
 raw_sequence = ['lying', 'standing', 'lying', 'sitting', 'lying']
 confidences = [0.8, 0.6, 0.7, 0.85, 0.9]

 optimized = optimizer.optimize_sequence(raw_sequence, confidences)

 # Verify length
 assert len(optimized) == len(raw_sequence), "Length should be preserved"

 # Verify all activities are valid
 assert all(act in activities for act in optimized), "All activities should be valid"

 print(" All tests passed")
 self.passed_tests += 1
 return True

 except AssertionError as e:
 print(f" Test failed: {e}")
 self.failed_tests += 1
 return False

 def test_memoization(self):
 """Test memoization framework."""
 print("\n[TEST] Memoization Framework")
 print("-" * 60)

 try:
 cache = FrameCache(max_size=10)

 # Test cache operations
 cache.set('key1', 'value1')
 value = cache.get('key1')
 assert value == 'value1', "Cache get/set mismatch"

 # Test cache miss
 value_miss = cache.get('nonexistent')
 assert value_miss is None, "Cache miss should return None"

 # Test frame clearing
 cache.clear_frame(frame_id=1)
 value_after_clear = cache.get('key1')
 assert value_after_clear is None, "Cache should be cleared"

 # Test hit rate
 cache.clear_frame(frame_id=2)
 cache.set('a', 1)
 _ = cache.get('a') # Hit
 _ = cache.get('b') # Miss
 hit_rate = cache.get_hit_rate()
 assert hit_rate == 0.5, "Hit rate should be 50%"

 print(" All tests passed")
 self.passed_tests += 1
 return True

 except AssertionError as e:
 print(f" Test failed: {e}")
 self.failed_tests += 1
 return False

 def run_all_tests(self):
 """Run all tests."""
 print("\n" + "=" * 60)
 print("OPTIMIZATION TEST SUITE")
 print("=" * 60)

 # Run tests
 self.test_distance_cache()
 self.test_circular_buffer()
 self.test_heapq_topk()
 self.test_lsh_reid()
 self.test_skeleton_graph()
 self.test_decision_tree()
 self.test_sequence_optimizer()
 self.test_memoization()

 # Print summary
 print("\n" + "=" * 60)
 print("TEST SUMMARY")
 print("=" * 60)
 total_tests = self.passed_tests + self.failed_tests
 print(f"Total Tests: {total_tests}")
 print(f"Passed: {self.passed_tests} ")
 print(f"Failed: {self.failed_tests} ")
 print(f"Success Rate: {self.passed_tests / total_tests * 100:.1f}%")
 print("=" * 60 + "\n")

 return self.failed_tests == 0


if __name__ == '__main__':
 tester = OptimizationTester()
 success = tester.run_all_tests()

 if success:
 print(" All optimization tests passed!")
 sys.exit(0)
 else:
 print(" Some tests failed. Please review errors above.")
 sys.exit(1)
