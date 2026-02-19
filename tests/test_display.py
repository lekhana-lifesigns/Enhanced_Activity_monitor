#!/usr/bin/env python3
"""
Simple display test for ICU Monitor
"""

import cv2
import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.display.display import ICUMonitorDisplay

def test_display():
 """Test the display functionality."""
 print("Testing ICU Monitor Display...")

 # Create display
 display = ICUMonitorDisplay("ICU Monitor Test")

 # Create test frame
 frame = np.zeros((720, 1280, 3), dtype=np.uint8)
 cv2.putText(frame, "ICU Monitor Test", (400, 300),
 cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
 cv2.putText(frame, "Press 'q' or ESC to exit", (350, 400),
 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

 print("Display window should be open. Press 'q' or ESC in the window to exit.")

 frame_count = 0
 # Show frame for 30 seconds or until user exits
 start_time = time.time()
 while time.time() - start_time < 30:
 # Update frame with counter
 test_frame = frame.copy()
 cv2.putText(test_frame, f"Frame: {frame_count}", (50, 50),
 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
 cv2.putText(test_frame, f"Time: {int(time.time() - start_time)}s", (50, 100),
 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

 if not display.show(test_frame):
 print("User requested exit from display window")
 break

 frame_count += 1
 time.sleep(0.1) # Small delay to prevent excessive CPU usage

 display.close()
 print(f"Display test completed after {frame_count} frames")

if __name__ == "__main__":
 test_display()