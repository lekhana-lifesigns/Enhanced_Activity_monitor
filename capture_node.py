# capture_node.py - Raspberry Pi Capture Node
"""
Capture node for Raspberry Pi in clustered EAM architecture.
Captures video, performs basic preprocessing, and streams to Jetson inference node.
"""

import cv2
import zmq
import yaml
import logging
import time
import signal
import sys
import os
import argparse
from ultralytics import YOLO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pose.camera import Camera

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("capture_node")

class CaptureNode:
 """
 Raspberry Pi capture node that streams video frames to Jetson inference node.
 """

 def __init__(self, config_path="config/capture.yaml"):
 self.config = self._load_config(config_path)
 self.running = True

 # Setup signal handlers
 signal.signal(signal.SIGINT, self._signal_handler)
 signal.signal(signal.SIGTERM, self._signal_handler)

 # Initialize camera
 cam_cfg = self.config.get("camera", {})
 self.camera = Camera(
 index=cam_cfg.get("index", 0),
 resolution=tuple(cam_cfg.get("resolution", [640, 360])),
 fps=cam_cfg.get("fps", 15)
 )

 # Initialize ZeroMQ publisher
 zmq_cfg = self.config.get("zmq", {})
 self.context = zmq.Context()
 self.socket = self.context.socket(zmq.PUB)

 # Get port from config (support both bind_address and port formats)
 bind_address = zmq_cfg.get("bind_address")
 if not bind_address:
 zmq_port = zmq_cfg.get("port", 5555)
 bind_address = f"tcp://*:{zmq_port}"

 self.socket.bind(bind_address)

 self.node_id = self.config.get("node_id", "pi_01")
 
 # Motion detection configuration
 motion_cfg = self.config.get("motion_detection", {})
 self.enable_motion_detection = motion_cfg.get("enabled", False)
 self.motion_threshold = motion_cfg.get("threshold", 0.02)
 self.min_contour_area = motion_cfg.get("min_area", 500)
 self.blur_size = motion_cfg.get("blur_size", 21)
 self.dilate_iterations = motion_cfg.get("dilate_iterations", 2)
 
 # Feature extraction mode
 feature_cfg = self.config.get("feature_mode", {})
 self.enable_feature_mode = feature_cfg.get("enabled", False)
 self.feature_model_path = feature_cfg.get("model_path", "models/yolo11n-pose.pt")
 self.feature_confidence_threshold = feature_cfg.get("confidence_threshold", 0.5)
 
 # Initialize YOLO model for feature extraction if enabled
 self.pose_model = None
 if self.enable_feature_mode:
 try:
 self.pose_model = YOLO(self.feature_model_path)
 log.info(f"Feature extraction enabled with model: {self.feature_model_path}")
 except Exception as e:
 log.error(f"Failed to load pose model: {e}")
 self.enable_feature_mode = False
 
 self.enable_privacy_blur = self.config.get("enable_privacy_blur", False)
 
 # Initialize face detector for privacy blurring
 if self.enable_privacy_blur:
 self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 log.info("Privacy blurring enabled - faces will be blurred")
 else:
 self.face_cascade = None

 log.info(f"Capture node {self.node_id} initialized")
 log.info(f"Camera: {cam_cfg.get('index', 0)} @ {cam_cfg.get('resolution', [640, 360])}")
 log.info(f"Streaming to: {bind_address}")

 def _load_config(self, config_path):
 """Load configuration from YAML file."""
 try:
 with open(config_path, 'r') as f:
 return yaml.safe_load(f)
 except Exception as e:
 log.error(f"Failed to load config {config_path}: {e}")
 # Return default config
 return {
 "node_id": "pi_01",
 "camera": {
 "index": 0,
 "resolution": [640, 360],
 "fps": 15
 },
 "zmq": {
 "bind_address": "tcp://*:5555"
 },
 "enable_motion_detection": False
 }

 def _signal_handler(self, sig, frame):
 """Handle shutdown signals."""
 log.info(f"Received signal {sig}, shutting down...")
 self.running = False

 def _apply_privacy_blur(self, frame):
 """Apply privacy blurring to detected faces."""
 if not self.enable_privacy_blur or self.face_cascade is None:
 return frame
 
 try:
 # Convert to grayscale for face detection
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
 # Detect faces
 faces = self.face_cascade.detectMultiScale(
 gray, 
 scaleFactor=1.1, 
 minNeighbors=5, 
 minSize=(30, 30)
 )
 
 # Blur each detected face
 for (x, y, w, h) in faces:
 # Add padding around face
 padding = int(w * 0.2)
 x1 = max(0, x - padding)
 y1 = max(0, y - padding)
 x2 = min(frame.shape[1], x + w + padding)
 y2 = min(frame.shape[0], y + h + padding)
 
 # Extract face region
 face_region = frame[y1:y2, x1:x2]
 
 # Apply Gaussian blur
 blurred_face = cv2.GaussianBlur(face_region, (23, 23), 30)
 
 # Replace original face with blurred version
 frame[y1:y2, x1:x2] = blurred_face
 
 return frame
 except Exception as e:
 log.debug(f"Privacy blur failed: {e}")
 return frame

 def _detect_motion(self, frame, prev_frame):
 """
 Advanced motion detection using frame differencing.
 
 Args:
 frame: Current frame
 prev_frame: Previous frame for comparison
 
 Returns:
 bool: True if motion detected above threshold
 """
 if prev_frame is None:
 return True # Always process first frame
 
 try:
 # Convert to grayscale
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
 
 # Apply Gaussian blur to reduce noise
 gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
 prev_gray = cv2.GaussianBlur(prev_gray, (self.blur_size, self.blur_size), 0)
 
 # Calculate frame difference
 frame_diff = cv2.absdiff(gray, prev_gray)
 
 # Apply threshold to get binary image
 _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
 
 # Apply morphological operations to reduce noise
 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
 thresh = cv2.dilate(thresh, kernel, iterations=self.dilate_iterations)
 thresh = cv2.erode(thresh, kernel, iterations=1)
 
 # Find contours
 contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
 # Calculate total motion area
 total_motion_area = sum(cv2.contourArea(contour) for contour in contours 
 if cv2.contourArea(contour) > self.min_contour_area)
 
 # Calculate motion ratio (percentage of frame with motion)
 frame_area = gray.shape[0] * gray.shape[1]
 motion_ratio = total_motion_area / frame_area
 
 # Return True if motion exceeds threshold
 return motion_ratio > self.motion_threshold
 
 except Exception as e:
 log.debug(f"Motion detection failed: {e}")
 return True # Default to processing frame if detection fails

 def _extract_pose_features(self, frame):
 """
 Extract pose keypoints and bounding boxes from frame.
 
 Args:
 frame: Input frame
 
 Returns:
 list: List of detected person features
 """
 if self.pose_model is None:
 return []
 
 try:
 # Run YOLO pose estimation
 results = self.pose_model(frame, conf=self.feature_confidence_threshold, verbose=False)
 
 features = []
 for result in results:
 if result.keypoints is not None and len(result.keypoints.data) > 0:
 for person_idx, keypoints in enumerate(result.keypoints.data):
 # Get bounding box
 if result.boxes is not None and person_idx < len(result.boxes.data):
 bbox = result.boxes.data[person_idx][:4].cpu().numpy().tolist()
 else:
 # Fallback: calculate bbox from keypoints
 kps = keypoints.cpu().numpy()
 valid_kps = kps[kps[:, 2] > 0] # Only visible keypoints
 if len(valid_kps) > 0:
 x_coords = valid_kps[:, 0]
 y_coords = valid_kps[:, 1]
 bbox = [
 float(x_coords.min()),
 float(y_coords.min()), 
 float(x_coords.max()),
 float(y_coords.max())
 ]
 else:
 continue
 
 # Convert keypoints to list format
 keypoints_list = keypoints.cpu().numpy().tolist()
 
 features.append({
 'track_id': person_idx,
 'bbox': bbox,
 'keypoints': keypoints_list,
 'confidence': float(keypoints[:, 2].mean()) if len(keypoints) > 0 else 0.0
 })
 
 return features
 
 except Exception as e:
 log.debug(f"Feature extraction failed: {e}")
 return []

 def run(self):
 """Main capture and streaming loop."""
 log.info(f"Starting capture node {self.node_id}")

 prev_frame = None
 frame_count = 0
 start_time = time.time()

 try:
 while self.running:
 # Read frame
 frame = self.camera.read()
 if frame is None:
 log.warning("Failed to read frame, retrying...")
 time.sleep(0.1)
 continue

 # Motion detection (optional)
 if self.enable_motion_detection:
 if not self._detect_motion(frame, prev_frame):
 time.sleep(0.05) # Brief pause if no motion
 continue
 prev_frame = frame.copy()

 # Apply privacy blurring (optional)
 if self.enable_privacy_blur:
 frame = self._apply_privacy_blur(frame)

 # Feature extraction mode or frame mode
 if self.enable_feature_mode and self.pose_model is not None:
 # Extract pose features instead of sending full frame
 features = self._extract_pose_features(frame)
 
 if len(features) > 0: # Only send if people detected
 message = {
 "node_id": self.node_id,
 "timestamp": time.time(),
 "message_type": "features",
 "features": features,
 "frame_shape": frame.shape
 }
 else:
 # No people detected, skip this frame
 time.sleep(0.05)
 continue
 else:
 # Traditional frame mode
 # Encode frame as JPEG for transmission
 encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
 _, encoded_frame = cv2.imencode('.jpg', frame, encode_param)

 message = {
 "node_id": self.node_id,
 "timestamp": time.time(),
 "message_type": "frame",
 "frame_shape": frame.shape,
 "frame_data": encoded_frame.tobytes()
 }

 # Send via ZeroMQ
 self.socket.send_pyobj(message)

 frame_count += 1

 # Log stats every 100 frames
 if frame_count % 100 == 0:
 elapsed = time.time() - start_time
 fps = frame_count / elapsed
 log.info(f"Frames sent: {frame_count} | FPS: {fps:.1f}")

 except Exception as e:
 log.exception(f"Error in capture loop: {e}")
 finally:
 self.cleanup()

 def cleanup(self):
 """Cleanup resources."""
 log.info("Cleaning up capture node...")
 if hasattr(self, 'socket'):
 self.socket.close()
 if hasattr(self, 'context'):
 self.context.term()
 if hasattr(self, 'camera'):
 self.camera.release()
 log.info("Capture node shutdown complete")

if __name__ == "__main__":
 parser = argparse.ArgumentParser(description="EAM Capture Node")
 parser.add_argument("--config", default="config/capture.yaml", help="Path to config file")
 args = parser.parse_args()

 node = CaptureNode(args.config)
 node.run()
