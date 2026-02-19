#!/usr/bin/env python3
"""
Test script to check if display window is responding to keyboard input
"""

import cv2
import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.display.display import ICUMonitorDisplay

def test_keyboard_response():
 """Test if the display window responds to keyboard input."""
 print("Testing keyboard response in ICU Monitor Display...")
 print("Window will show for 10 seconds. Try pressing 'q' or ESC to exit early.")

 # Create display
 display = ICUMonitorDisplay("Keyboard Test - Press Q or ESC")

 # Create test frame
 frame = np.zeros((720, 1280, 3), dtype=np.uint8)
 cv2.putText(frame, "KEYBOARD TEST", (400, 200),
 cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
 cv2.putText(frame, "Press 'q' or ESC to exit this test", (300, 350),
 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 cv2.putText(frame, "If window is unresponsive, close it manually", (250, 400),
 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

 start_time = time.time()
 frame_count = 0
 user_exited = False

 while time.time() - start_time < 10: # 10 second test
 # Update frame with timer
 test_frame = frame.copy()
 elapsed = int(time.time() - start_time)
 cv2.putText(test_frame, f"Time remaining: {10 - elapsed}s", (50, 50),
 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
 cv2.putText(test_frame, f"Frames shown: {frame_count}", (50, 100),
 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

 if not display.show(test_frame):
 user_exited = True
 break

 frame_count += 1
 time.sleep(0.05) # Faster updates for responsiveness test

 display.close()

 if user_exited:
 print(" SUCCESS: Display window responded to keyboard input!")
 print(" The ICU Monitor display should work properly now.")
 else:
 print(" WARNING: Display window did not respond to keyboard input.")
 print(" The window may be unresponsive. Try closing it manually.")
 print(" This is a known issue with OpenCV on Windows.")
 print(" The system will continue running in the background.")

 return user_exited

if __name__ == "__main__":
 success = test_keyboard_response()
 sys.exit(0 if success else 1)