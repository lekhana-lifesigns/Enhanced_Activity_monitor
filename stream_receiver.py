# stream_receiver.py - Stream Receiver for Jetson Inference Node
"""
Stream receiver for Jetson inference node in clustered EAM architecture.
Receives video frames from multiple Raspberry Pi capture nodes.
"""
import zmq
import cv2
import numpy as np
import yaml
import logging
import time
import threading
from collections import deque
from typing import Dict, Optional, Tuple
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stream_receiver")
class StreamReceiver:
 """
 Receives video streams from multiple Raspberry Pi capture nodes.
 Provides synchronized frame access for multi-camera inference.
 """
 def __init__(self, camera_configs: Dict[str, Dict], buffer_size: int = 30):
 """
 Args:
 camera_configs: Dict of camera_id -> config with 'pi_address' key
 buffer_size: Maximum frames to buffer per camera
 """
 self.camera_configs = camera_configs
 self.buffer_size = buffer_size
 self.running = True
 # Frame buffers: camera_id -> deque of (timestamp, frame_or_features)
 self.frame_buffers: Dict[str, deque] = {}
 self.buffer_locks: Dict[str, threading.Lock] = {}
 # ZeroMQ setup
 self.context = zmq.Context()
 self.sockets: Dict[str, zmq.Socket] = {}
 # Initialize connections
 self._setup_connections()
 # Start receiver threads
 self.threads: Dict[str, threading.Thread] = {}
 self._start_receiver_threads()
 log.info(f"Stream receiver initialized for {len(camera_configs)} cameras")
 def _setup_connections(self):
 """Setup ZeroMQ subscriber connections to Pi nodes."""
 for camera_id, config in self.camera_configs.items():
 pi_address = config.get("pi_address")
 if not pi_address:
 log.error(f"No pi_address specified for camera {camera_id}")
 continue
 # Create subscriber socket
 socket = self.context.socket(zmq.SUB)
 socket.connect(pi_address)
 socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all messages
 self.sockets[camera_id] = socket
 self.frame_buffers[camera_id] = deque(maxlen=self.buffer_size)
 self.buffer_locks[camera_id] = threading.Lock()
 log.info(f"Connected to camera {camera_id} at {pi_address}")
 def _start_receiver_threads(self):
 """Start background threads to receive frames from each camera."""
 for camera_id in self.camera_configs.keys():
 thread = threading.Thread(
 target=self._receiver_loop,
 args=(camera_id,),
 name=f"receiver_{camera_id}",
 daemon=True
 )
 thread.start()
 self.threads[camera_id] = thread
 def _receiver_loop(self, camera_id: str):
 """Background receiver loop for a single camera."""
 socket = self.sockets[camera_id]
 buffer = self.frame_buffers[camera_id]
 lock = self.buffer_locks[camera_id]
 log.info(f"Starting receiver loop for camera {camera_id}")
 
 while self.running:
 try:
 # Receive message (blocking)
 message = socket.recv_pyobj(flags=zmq.NOBLOCK)
 
 timestamp = message["timestamp"]
 message_type = message.get("message_type", "frame")
 
 if message_type == "features":
 # Feature message - store features directly
 features = message["features"]
 with lock:
 buffer.append((timestamp, {"type": "features", "data": features}))
 
 elif message_type == "frame":
 # Traditional frame message - decode JPEG
 frame_data = message["frame_data"]
 frame_shape = message["frame_shape"]
 
 # Decode JPEG
 encoded_frame = np.frombuffer(frame_data, dtype=np.uint8)
 frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
 if frame is None:
 log.warning(f"Failed to decode frame from {camera_id}")
 continue
 
 # Ensure correct shape
 if frame.shape != tuple(frame_shape):
 log.warning(f"Frame shape mismatch for {camera_id}: expected {frame_shape}, got {frame.shape}")
 
 with lock:
 buffer.append((timestamp, {"type": "frame", "data": frame}))
 
 else:
 log.warning(f"Unknown message type from {camera_id}: {message_type}")
 continue
 
 except zmq.Again:
 # No message available, continue
 time.sleep(0.001)
 except Exception as e:
 log.exception(f"Error in receiver loop for {camera_id}: {e}")
 time.sleep(0.1) # Brief pause on error
 def get_latest_frames(self) -> Dict[str, Optional[Dict]]:
 """
 Get the latest frame or features from each camera buffer.
 Returns dict of camera_id -> {"type": "frame"/"features", "data": frame_or_features} (or None if no data available)
 """
 frames = {}
 for camera_id in self.camera_configs.keys():
 buffer = self.frame_buffers[camera_id]
 lock = self.buffer_locks[camera_id]
 with lock:
 if buffer:
 # Get most recent frame/features
 _, data = buffer[-1]
 frames[camera_id] = data.copy() if isinstance(data, dict) else data
 else:
 frames[camera_id] = None
 return frames
 def get_frames_at_time(self, target_time: float, tolerance: float = 0.1) -> Dict[str, Optional[Dict]]:
 """
 Get frames or features from all cameras closest to the target timestamp.
 Useful for synchronized multi-camera processing.
 Args:
 target_time: Target timestamp
 tolerance: Maximum age tolerance in seconds
 Returns:
 Dict of camera_id -> data dict (or None if no suitable data)
 """
 frames = {}
 for camera_id in self.camera_configs.keys():
 buffer = self.frame_buffers[camera_id]
 lock = self.buffer_locks[camera_id]
 with lock:
 if not buffer:
 frames[camera_id] = None
 continue
 # Find frame closest to target time
 best_data = None
 min_diff = float('inf')
 for timestamp, data in buffer:
 diff = abs(timestamp - target_time)
 if diff < min_diff and diff <= tolerance:
 min_diff = diff
 best_data = data
 if best_data is not None:
 frames[camera_id] = best_data.copy() if isinstance(best_data, dict) else best_data
 else:
 frames[camera_id] = None
 return frames
 def get_buffer_stats(self) -> Dict[str, Dict]:
 """Get statistics about frame buffers."""
 stats = {}
 for camera_id in self.camera_configs.keys():
 buffer = self.frame_buffers[camera_id]
 lock = self.buffer_locks[camera_id]
 with lock:
 stats[camera_id] = {
 "buffer_size": len(buffer),
 "max_buffer_size": self.buffer_size,
 "oldest_timestamp": buffer[0][0] if buffer else None,
 "newest_timestamp": buffer[-1][0] if buffer else None
 }
 return stats
 def shutdown(self):
 """Shutdown the stream receiver."""
 log.info("Shutting down stream receiver...")
 self.running = False
 # Wait for threads to finish
 for thread in self.threads.values():
 thread.join(timeout=2.0)
 # Close sockets
 for socket in self.sockets.values():
 socket.close()
 self.context.term()
 log.info("Stream receiver shutdown complete")
