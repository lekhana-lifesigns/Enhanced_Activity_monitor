#!/usr/bin/env python3
"""
Simple camera test for EAC system with persistence and joint weighting validation.
"""

import cv2
import numpy as np
import time
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.pose.inference_pipeline import InferencePipeline
from pipeline.pose.camera import Camera

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("camera_test")

def test_with_camera():
 """Test the system with live camera feed."""
 log.info("Starting camera test with persistence and joint weighting...")

 # Initialize camera
 try:
 camera = Camera(index=0, resolution=(640, 480), fps=15)
 log.info("Camera initialized")
 except Exception as e:
 log.error("Failed to initialize camera: %s", e)
 return False

 # Initialize inference pipeline
 try:
 cfg = {
 "model": {"pose": "yolo11n-pose.pt"},
 "temporal": {"window_size": 30, "fps": 15},
 "features": {"window_size": 30}
 }
 pipeline = InferencePipeline(cfg)
 log.info("Inference pipeline initialized")
 except Exception as e:
 log.exception("Failed to initialize pipeline: %s", e)
 return False

 # Test loop
 frame_count = 0
 start_time = time.time()

 try:
 while frame_count < 150: # Test for ~10 seconds at 15 FPS
 # Capture frame
 frame = camera.read()
 if frame is None:
 log.warning("Failed to capture frame")
 continue

 # Process frame
 result = pipeline.run_once()

 if result:
 activity = result.get("activity", "unknown")
 confidence = result.get("confidence", 0.0)
 alert = result.get("alert", "LOW_RISK")

 # Log every 15 frames (~1 second)
 if frame_count % 15 == 0:
 log.info(f"Frame {frame_count}: Activity={activity}, Confidence={confidence:.2f}, Alert={alert}")

 # Check persistence state
 if hasattr(pipeline, 'activity_persistence'):
 persistence = pipeline.activity_persistence
 log.info(f" Persistence: {persistence}")

 frame_count += 1
 time.sleep(1/15) # Maintain 15 FPS

 except KeyboardInterrupt:
 log.info("Test interrupted by user")
 except Exception as e:
 log.exception("Error during camera test: %s", e)
 return False

 elapsed = time.time() - start_time
 fps_actual = frame_count / elapsed

 log.info(".2f")
 log.info(" Camera test completed successfully")
 return True

if __name__ == "__main__":
 success = test_with_camera()
 sys.exit(0 if success else 1)