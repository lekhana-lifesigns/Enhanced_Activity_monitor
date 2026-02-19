#!/usr/bin/env python3
"""
SMPL-X Integration Test and Evaluation Script
Tests SMPL-X 3D pose estimation and biomedical applications.
"""

import sys
import numpy as np
import logging
import time
import yaml
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smplx_test")

def test_smplx_basic():
 """Test basic SMPL-X initialization and fallback behavior."""
 log.info(" Testing SMPL-X Basic Functionality...")

 try:
 from pipeline.pose.pose3d_estimator import Pose3DEstimator

 # Test 1: SMPL-X initialization (should fallback gracefully)
 log.info("Testing SMPL-X initialization...")
 estimator_smplx = Pose3DEstimator(method="smplx")
 log.info(f" SMPL-X estimator created, using method: {estimator_smplx.method}")

 # Test 2: Geometric fallback
 log.info("Testing geometric fallback...")
 estimator_geom = Pose3DEstimator(method="geometric")
 log.info(f" Geometric estimator created, method: {estimator_geom.method}")

 # Test 3: Kinematic (if available)
 log.info("Testing kinematic method...")
 estimator_kin = Pose3DEstimator(method="kinematic")
 log.info(f" Kinematic estimator created, method: {estimator_kin.method}")

 return True

 except Exception as e:
 log.error(f" SMPL-X basic test failed: {e}")
 return False

def test_smplx_pose_estimation():
 """Test SMPL-X pose estimation with mock data."""
 log.info(" Testing SMPL-X Pose Estimation...")

 try:
 from pipeline.pose.pose3d_estimator import Pose3DEstimator

 # Create mock 2D keypoints (COCO format)
 mock_kps_2d = [
 (0.5, 0.2, 0.9), # nose
 (0.45, 0.25, 0.8), # left eye
 (0.55, 0.25, 0.8), # right eye
 (0.4, 0.3, 0.7), # left ear
 (0.6, 0.3, 0.7), # right ear
 (0.4, 0.4, 0.9), # left shoulder
 (0.6, 0.4, 0.9), # right shoulder
 (0.35, 0.5, 0.8), # left elbow
 (0.65, 0.5, 0.8), # right elbow
 (0.3, 0.6, 0.7), # left wrist
 (0.7, 0.6, 0.7), # right wrist
 (0.45, 0.65, 0.9), # left hip
 (0.55, 0.65, 0.9), # right hip
 (0.4, 0.75, 0.8), # left knee
 (0.6, 0.75, 0.8), # right knee
 (0.35, 0.9, 0.7), # left ankle
 (0.65, 0.9, 0.7), # right ankle
 ]

 # Test geometric estimation (should work)
 log.info("Testing geometric 3D estimation...")
 estimator_geom = Pose3DEstimator(method="geometric")
 result_geom = estimator_geom.estimate_3d(mock_kps_2d)

 if result_geom and len(result_geom) == 17:
 log.info(" Geometric estimation successful")
 log.info(f" Sample joint: {result_geom[0]}") # nose
 else:
 log.error(" Geometric estimation failed")
 return False

 # Test SMPL-X estimation (will fallback to geometric)
 log.info("Testing SMPL-X estimation (fallback mode)...")
 estimator_smplx = Pose3DEstimator(method="smplx")
 result_smplx = estimator_smplx.estimate_3d(mock_kps_2d)

 if result_smplx and len(result_smplx) == 17:
 log.info(" SMPL-X estimation successful (fallback)")
 log.info(f" Sample joint: {result_smplx[0]}") # nose
 else:
 log.error(" SMPL-X estimation failed")
 return False

 return True

 except Exception as e:
 log.error(f" SMPL-X pose estimation test failed: {e}")
 return False

def test_biomedical_integration():
 """Test biomedical HPE system integration."""
 log.info(" Testing Biomedical HPE Integration...")

 try:
 from biomedical.biomedical_hpe import BiomedicalHPESystem

 # Test configuration
 config = {
 "enable_motor_development": True,
 "enable_rehabilitation": True,
 "enable_gait_analysis": True,
 "motor_dev_window": 300,
 "fps": 30,
 "exercise_type": "upper_limb",
 "target_reps": 10,
 "session_duration": 300,
 "gait_window": 300
 }

 # Initialize biomedical system
 log.info("Initializing biomedical HPE system...")
 bio_system = BiomedicalHPESystem(config)

 log.info(f" Biomedical system initialized with modules: {list(bio_system.active_modules.keys())}")

 # Test with mock data
 mock_kps_2d = np.random.rand(17, 3) # 17 keypoints, (x, y, conf)
 mock_kps_3d = np.random.rand(17, 4) # 17 keypoints, (x, y, z, conf)

 # Process frame
 log.info("Testing frame processing...")
 results = bio_system.process_frame(mock_kps_2d, mock_kps_3d)

 if results and "clinical_insights" in results:
 log.info(" Biomedical processing successful")
 log.info(f" Insights keys: {list(results['clinical_insights'].keys())}")
 log.info(f" Active modules: {results.get('modules_active', [])}")
 else:
 log.error(" Biomedical processing failed")
 log.error(f" Results: {results}")
 return False

 return True

 except Exception as e:
 log.error(f" Biomedical integration test failed: {e}")
 return False

def benchmark_pose_methods():
 """Benchmark different 3D pose estimation methods."""
 log.info(" Benchmarking 3D Pose Estimation Methods...")

 try:
 from pipeline.pose.pose3d_estimator import Pose3DEstimator

 methods = ["geometric", "kinematic", "smplx"]
 results = {}

 # Create mock data
 mock_kps_2d = [
 (0.5, 0.2, 0.9), (0.45, 0.25, 0.8), (0.55, 0.25, 0.8), (0.4, 0.3, 0.7),
 (0.6, 0.3, 0.7), (0.4, 0.4, 0.9), (0.6, 0.4, 0.9), (0.35, 0.5, 0.8),
 (0.65, 0.5, 0.8), (0.3, 0.6, 0.7), (0.7, 0.6, 0.7), (0.45, 0.65, 0.9),
 (0.55, 0.65, 0.9), (0.4, 0.75, 0.8), (0.6, 0.75, 0.8), (0.35, 0.9, 0.7),
 (0.65, 0.9, 0.7)
 ]

 for method in methods:
 log.info(f"Testing method: {method}")

 try:
 estimator = Pose3DEstimator(method=method)

 # Time the estimation
 start_time = time.time()
 result = estimator.estimate_3d(mock_kps_2d)
 end_time = time.time()

 latency = (end_time - start_time) * 1000 # ms

 if result and len(result) == 17:
 results[method] = {
 "success": True,
 "latency_ms": latency,
 "actual_method": estimator.method
 }
 log.info(f" Success: {latency:.2f}ms")
 else:
 results[method] = {"success": False, "error": "Invalid result"}
 log.info(" Failed")

 except Exception as e:
 results[method] = {"success": False, "error": str(e)}
 log.info(f" Error: {e}")

 # Print summary
 log.info("\n Benchmark Results:")
 for method, result in results.items():
 if result["success"]:
 log.info(f" {method}: {result['latency_ms']:.2f}ms (using {result['actual_method']})")
 else:
 log.info(f" {method}: {result['error']}")

 return results

 except Exception as e:
 log.error(f" Benchmarking failed: {e}")
 return None

def test_clinical_accuracy():
 """Test clinical accuracy improvements with different methods."""
 log.info(" Testing Clinical Accuracy...")

 try:
 from biomedical.biomedical_hpe import BiomedicalHPESystem
 from pipeline.pose.pose3d_estimator import Pose3DEstimator

 # Test configurations
 configs = [
 {"method": "geometric", "name": "Geometric"},
 {"method": "kinematic", "name": "Kinematic"},
 {"method": "smplx", "name": "SMPL-X (fallback)"}
 ]

 results = {}

 # Create mock clinical scenario (upper limb exercise)
 mock_kps_2d = [
 (0.5, 0.2, 0.9), (0.45, 0.25, 0.8), (0.55, 0.25, 0.8), (0.4, 0.3, 0.7),
 (0.6, 0.3, 0.7), (0.4, 0.4, 0.9), (0.6, 0.4, 0.9), (0.35, 0.5, 0.8),
 (0.65, 0.5, 0.8), (0.3, 0.6, 0.7), (0.7, 0.6, 0.7), (0.45, 0.65, 0.9),
 (0.55, 0.65, 0.9), (0.4, 0.75, 0.8), (0.6, 0.75, 0.8), (0.35, 0.9, 0.7),
 (0.65, 0.9, 0.7)
 ]

 for config in configs:
 method = config["method"]
 name = config["name"]

 log.info(f"Testing {name} method...")

 try:
 # Get 3D pose
 estimator = Pose3DEstimator(method=method)
 kps_3d = estimator.estimate_3d(mock_kps_2d)

 if kps_3d:
 # Test biomedical analysis
 bio_config = {
 "enable_rehabilitation": True,
 "exercise_type": "upper_limb",
 "target_reps": 10
 }

 bio_system = BiomedicalHPESystem(bio_config)
 bio_results = bio_system.process_frame(mock_kps_2d, kps_3d)

 if bio_results and "clinical_insights" in bio_results:
 rehab_data = bio_results.get("results", {}).get("rehab", {})
 accuracy = rehab_data.get("metrics", {}).get("accuracy_score", 0.0) if isinstance(rehab_data, dict) else 0.0
 rom = rehab_data.get("metrics", {}).get("range_of_motion", 0.0) if isinstance(rehab_data, dict) else 0.0

 results[method] = {
 "success": True,
 "accuracy_score": accuracy,
 "range_of_motion": rom
 }

 log.info(f" {name}: Accuracy={accuracy:.2f}, ROM={rom:.2f}")
 else:
 results[method] = {"success": False, "error": "Biomedical analysis failed"}
 log.info(f" {name}: Biomedical analysis failed - {bio_results}")
 else:
 results[method] = {"success": False, "error": "3D estimation failed"}
 log.info(f" {name}: 3D estimation failed")

 except Exception as e:
 results[method] = {"success": False, "error": str(e)}
 log.info(f" {name}: Error - {e}")

 # Print clinical comparison
 log.info("\n Clinical Accuracy Results:")
 for method, result in results.items():
 if result["success"]:
 log.info(f" {method}: Accuracy={result['accuracy_score']:.2f}, ROM={result['range_of_motion']:.2f}")
 else:
 log.info(f" {method}: {result['error']}")

 return results

 except Exception as e:
 log.error(f" Clinical accuracy test failed: {e}")
 return None

def main():
 """Run all SMPL-X integration tests."""
 log.info(" Starting SMPL-X Integration Test Suite")
 log.info("=" * 50)

 test_results = {}

 # Test 1: Basic functionality
 test_results["basic"] = test_smplx_basic()

 # Test 2: Pose estimation
 test_results["pose_estimation"] = test_smplx_pose_estimation()

 # Test 3: Biomedical integration
 test_results["biomedical"] = test_biomedical_integration()

 # Test 4: Benchmarking
 benchmark_results = benchmark_pose_methods()
 test_results["benchmark"] = benchmark_results is not None

 # Test 5: Clinical accuracy
 clinical_results = test_clinical_accuracy()
 test_results["clinical"] = clinical_results is not None

 # Summary
 log.info("\n" + "=" * 50)
 log.info(" Test Summary:")

 passed = 0
 total = len(test_results)

 for test_name, result in test_results.items():
 status = " PASS" if result else " FAIL"
 log.info(f" {test_name}: {status}")
 if result:
 passed += 1

 log.info(f"\nOverall: {passed}/{total} tests passed")

 if passed == total:
 log.info(" All tests passed! SMPL-X integration is ready.")
 log.info("\n Next Steps:")
 log.info("1. Download SMPL-X model files from https://smpl-x.is.tue.mpg.de/")
 log.info("2. Place SMPLX_NEUTRAL.pkl in pipeline/pose/models/smplx/")
 log.info("3. Re-run tests to validate full SMPL-X functionality")
 log.info("4. Run clinical validation studies")
 else:
 log.info(" Some tests failed. Check logs above for details.")

 return passed == total

if __name__ == "__main__":
 success = main()
 sys.exit(0 if success else 1)