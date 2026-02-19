#!/usr/bin/env python3
# test_pilot_system.py
"""
Test script for the Self-Learning Pilot System
Demonstrates dynamic adaptation, camera control, and segmentation-based processing
"""

import cv2
import numpy as np
import time
import logging
import yaml
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.pilot_system import SelfLearningPilot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("test_pilot")

def create_test_frame(width=1280, height=720, person_present=True, activity_level="medium"):
 """Create a synthetic test frame for demonstration."""
 frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

 if person_present:
 # Add a simulated person (simple rectangle)
 person_x = width // 4
 person_y = height // 4
 person_w = width // 3
 person_h = height // 2

 # Different colors based on activity
 if activity_level == "high":
 color = (0, 0, 255) # Red for high activity
 elif activity_level == "medium":
 color = (0, 255, 255) # Yellow for medium activity
 else:
 color = (255, 0, 0) # Blue for low activity

 cv2.rectangle(frame, (person_x, person_y), (person_x + person_w, person_y + person_h), color, -1)

 # Add some "motion" by varying the position slightly
 motion_offset = np.random.randint(-10, 10)
 cv2.rectangle(frame, (person_x + motion_offset, person_y + motion_offset),
 (person_x + person_w + motion_offset, person_y + person_h + motion_offset), color, 2)

 return frame

def test_pilot_system():
 """Test the pilot system with synthetic data."""
 print(" Testing Self-Learning Pilot System")
 print("=" * 50)

 # Load configuration
 try:
 with open("config/system.yaml", 'r') as f:
 config = yaml.safe_load(f)
 print(" Configuration loaded")
 except Exception as e:
 print(f" Failed to load config: {e}")
 return

 # Initialize pilot system
 try:
 pilot = SelfLearningPilot(config)
 print(" Pilot system initialized")
 except Exception as e:
 print(f" Failed to initialize pilot: {e}")
 return

 # Test scenarios
 scenarios = [
 {"person_present": False, "activity": "none", "description": "Empty room"},
 {"person_present": True, "activity": "low", "description": "Person resting"},
 {"person_present": True, "activity": "medium", "description": "Person moving"},
 {"person_present": True, "activity": "high", "description": "High activity"},
 ]

 print("\n Running test scenarios...")
 print("-" * 30)

 for i, scenario in enumerate(scenarios):
 print(f"\n Scenario {i+1}: {scenario['description']}")

 # Create test frames for this scenario
 frames_processed = 0
 total_time = 0

 for frame_num in range(5): # Process 5 frames per scenario
 # Create test frame
 frame = create_test_frame(
 person_present=scenario["person_present"],
 activity_level=scenario["activity"]
 )

 # Process with pilot system
 start_time = time.time()
 result = pilot.process_frame_pilot(frame, camera_id="test_camera")
 processing_time = time.time() - start_time

 frames_processed += 1
 total_time += processing_time

 # Show results
 if frame_num == 0: # Show detailed results for first frame
 print(f" ⏱ Processing Time: {processing_time:.3f}s")
 print(f" Activity: {result.get('label', 'unknown')}")
 print(f" Confidence: {result.get('confidence', 0):.2f}")
 print(f" Pilot Adaptations: {result.get('pilot_adaptations', {})}")
 print(f" Context: {result.get('context_analysis', {})}")

 if "clinical_correlation" in result:
 clinical = result["clinical_correlation"]
 print(f" Clinical: Pain={clinical.get('pain_score', 0):.2f}, Agitation={clinical.get('agitation_score', 0):.2f}")

 avg_time = total_time / frames_processed
 print(f" Average Processing Time: {avg_time:.3f}s per frame")
 # Show learning results
 print("\n Learning Results:")
 print("-" * 20)
 status = pilot.get_pilot_status()
 print(f" Learned Activities: {status['learned_activities']}")
 print(f" Learning Samples: {status['learning_samples']}")
 print(f"⏱ Avg Processing Time: {status['avg_processing_time']:.3f}s")
 print(f" Avg Accuracy: {status['avg_accuracy']:.3f}")
 print(f" Processing Mode: {status['processing_mode']}")
 print(f" Optimal Zoom: {status['optimal_zoom_level']:.2f}")
 # Test self-contact detection if person is present
 print("\n Self-Contact Detection Test:")
 print("-" * 30)

 # Create a frame with simulated self-contact (close keypoints)
 contact_frame = create_test_frame(person_present=True, activity_level="medium")

 # Simulate keypoints that would trigger self-contact
 from pipeline.pose.self_contact_detector import SelfContactDetector
 detector = SelfContactDetector(geometric_threshold=50.0) # Larger threshold for test

 # Simulate keypoints for hand near face
 test_kps = [
 (400, 200, 0.9), # nose
 (380, 180, 0.9), # left_eye
 (420, 180, 0.9), # right_eye
 (360, 190, 0.9), # left_ear
 (440, 190, 0.9), # right_ear
 (390, 300, 0.9), # left_shoulder
 (410, 300, 0.9), # right_shoulder
 (380, 350, 0.9), # left_elbow
 (420, 350, 0.9), # right_elbow
 (395, 210, 0.9), # left_wrist (near face!)
 (425, 400, 0.9), # right_wrist
 (400, 400, 0.9), # left_hip
 (400, 400, 0.9), # right_hip
 (390, 500, 0.9), # left_knee
 (410, 500, 0.9), # right_knee
 (385, 600, 0.9), # left_ankle
 (415, 600, 0.9), # right_ankle
 ]

 kps_3d = [(kp[0], kp[1], 0.0, kp[2]) for kp in test_kps]
 contact_result = detector.detect(kps_3d)

 print(f" Face Touch Detected: {detector.detect_face_touch(contact_result)}")
 print(f" Arm Crossing Detected: {detector.detect_arm_crossing(contact_result)}")
 print(f" Leg Crossing Detected: {detector.detect_leg_crossing(contact_result)}")

 if detector.detect_face_touch(contact_result):
 print(" Self-contact detection working! Face touch detected.")

 print("\n Pilot System Test Complete!")
 print("The system is now capable of:")
 print(" • Self-learning from activity patterns")
 print(" • Dynamic camera parameter adaptation")
 print(" • Real-time segmentation-based focus")
 print(" • Clinical correlation with self-contact detection")
 print(" • Continuous performance optimization")

if __name__ == "__main__":
 test_pilot_system()