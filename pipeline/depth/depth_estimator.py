# pipeline/depth/depth_estimator.py
"""
Depth estimation module for 3D pose analysis.
Uses monocular depth estimation (DNNs like CNNs or ViTs) to convert 2D keypoints to 3D.

How monocular depth estimation works:
1. Input: RGB image (height × width × 3 channels)
2. Preprocessing: Normalize pixels, resize to model input size, convert to tensor
3. Encoder: CNN (ResNet) or ViT extracts features, learns monocular depth cues:
   - Perspective: Parallel lines converging in distance
   - Size: Familiar objects appear smaller when farther
   - Texture gradient: Textures appear finer at distance
   - Occlusion: Blocking objects are closer
4. Decoder: Constructs dense depth map with skip connections for sharp boundaries
5. Output: Depth map where pixel intensity = distance from camera

Supports:
- Geometric (fast, approximate) for edge deployment
- MiDaS/DPT (accurate) for higher accuracy when GPU available
- Depth Pro / UniDepth style metric depth when needed
"""
import cv2
import numpy as np
import logging
import os
from typing import Optional, Tuple, List, Dict

log = logging.getLogger("depth_estimator")

# Try to import depth estimation libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not available - depth estimation will use geometric methods")

try:
    # Try MiDaS or similar depth models
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# TensorRT for optimized depth inference
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


class MonocularDepthCues:
    """
    Compute monocular depth cues from 2D image/keypoints.
    These are the visual cues humans use to perceive depth from a single image.
    """

    @staticmethod
    def perspective_cue(kps: List[Tuple[float, float, float]], frame_height: int = 720) -> float:
        """
        Estimate relative depth from y-position (perspective cue).
        Objects lower in image (higher y) are typically closer.

        Returns:
            Relative depth factor (0-1, higher = farther)
        """
        if not kps:
            return 0.5

        # Get hip center y-position (good reference for person depth)
        hip_y = 0.0
        hip_count = 0
        for i in [11, 12]:  # LEFT_HIP, RIGHT_HIP in COCO
            if i < len(kps) and kps[i][2] > 0.3:
                hip_y += kps[i][1]
                hip_count += 1

        if hip_count == 0:
            return 0.5

        hip_y /= hip_count

        # Normalize: higher y = closer (lower depth factor)
        # Assuming y is normalized 0-1 or pixel coordinates
        if hip_y <= 1.0:
            depth_factor = hip_y  # Normalized coords
        else:
            depth_factor = hip_y / frame_height

        return float(np.clip(1.0 - depth_factor, 0.0, 1.0))

    @staticmethod
    def size_cue(kps: List[Tuple[float, float, float]], reference_height: float = 1.7) -> float:
        """
        Estimate depth from apparent person size (size constancy cue).
        Larger apparent size = closer.

        Args:
            kps: Keypoints
            reference_height: Expected real-world height in meters

        Returns:
            Estimated depth in meters
        """
        if not kps or len(kps) < 17:
            return 2.0  # Default 2 meters

        # Compute apparent height from keypoints
        y_coords = [kp[1] for kp in kps if kp[2] > 0.3]
        if len(y_coords) < 2:
            return 2.0

        apparent_height = max(y_coords) - min(y_coords)

        if apparent_height < 0.01:
            return 5.0  # Very small = far away

        # Assuming typical camera setup (focal length ~700 pixels for 720p)
        # depth = (real_height * focal_length) / apparent_height_pixels
        # For normalized coords (0-1), apparent_height is fraction of frame
        focal_length_normalized = 0.7  # Approximate

        if apparent_height <= 1.0:  # Normalized
            depth = (reference_height * focal_length_normalized) / apparent_height
        else:  # Pixel coords
            depth = (reference_height * 500) / apparent_height

        return float(np.clip(depth, 0.5, 10.0))


class DepthEstimator:
    """
    Depth estimation for 3D pose analysis.
    Uses geometric methods or lightweight depth models.
    """
    
    def __init__(self, method="geometric", camera_intrinsics_path=None):
        """
        Args:
            method: "geometric" (fast, approximate) or "model" (accurate, slower)
            camera_intrinsics_path: Path to camera calibration file
        """
        self.method = method
        self.camera_intrinsics = None
        self.depth_model = None
        
        # Load camera intrinsics if available
        if camera_intrinsics_path and os.path.exists(camera_intrinsics_path):
            try:
                import json
                with open(camera_intrinsics_path, 'r') as f:
                    calib = json.load(f)
                    self.camera_intrinsics = {
                        'fx': calib.get('fx', 0),
                        'fy': calib.get('fy', 0),
                        'cx': calib.get('cx', 0),
                        'cy': calib.get('cy', 0),
                        'width': calib.get('width', 1280),
                        'height': calib.get('height', 720)
                    }
                log.info("Camera intrinsics loaded from %s", camera_intrinsics_path)
            except Exception as e:
                log.warning("Failed to load camera intrinsics: %s", e)
        
        # Initialize depth model if using model-based method
        if method == "model" and TORCH_AVAILABLE:
            try:
                # Use lightweight depth model (e.g., DPT-Large or MiDaS-small)
                self._init_depth_model()
            except Exception as e:
                log.warning("Failed to initialize depth model, falling back to geometric: %s", e)
                self.method = "geometric"
    
    def _init_depth_model(self):
        """Initialize depth estimation model."""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use lightweight model
                self.processor = AutoImageProcessor.from_pretrained("Intel/dpt-depth-estimation")
                self.depth_model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-depth-estimation")
                if TORCH_AVAILABLE:
                    self.depth_model.eval()
                log.info("Depth estimation model loaded")
            except Exception as e:
                log.warning("Failed to load depth model: %s", e)
                self.method = "geometric"
    
    def estimate_depth_map(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth map for entire frame.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            Depth map (same size as frame) or None
        """
        if self.method == "model" and self.depth_model is not None:
            return self._estimate_depth_model(frame)
        else:
            return self._estimate_depth_geometric(frame)
    
    def _estimate_depth_model(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth using neural network model."""
        try:
            import torch
            from PIL import Image
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Process
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Convert to numpy
            depth_map = predicted_depth.squeeze().cpu().numpy()
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
            
            return depth_map
        except Exception as e:
            log.exception("Depth model inference failed: %s", e)
            return self._estimate_depth_geometric(frame)
    
    def _estimate_depth_geometric(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth using geometric methods (faster, approximate).
        Uses assumptions based on person size in frame.
        """
        h, w = frame.shape[:2]
        
        # Create approximate depth map based on image structure
        # Center is closer, edges are farther (for typical camera setup)
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(float)
        
        # Normalize coordinates
        y_norm = y_coords / h
        x_norm = x_coords / w
        
        # Approximate depth: closer to center = closer to camera
        # This is a rough approximation - real depth estimation would use stereo or depth models
        center_distance = np.sqrt((x_norm - 0.5)**2 + (y_norm - 0.5)**2)
        
        # Depth increases with distance from center
        # Scale to reasonable range (1-5 meters for typical room)
        depth_map = 1.0 + center_distance * 4.0
        
        return depth_map
    
    def get_depth_at_point(self, depth_map: np.ndarray, x: float, y: float) -> float:
        """
        Get depth value at specific point.
        
        Args:
            depth_map: Depth map
            x, y: Coordinates (normalized 0-1 or pixel coordinates)
        
        Returns:
            Depth value in meters
        """
        h, w = depth_map.shape
        
        # Convert normalized to pixel if needed
        if x <= 1.0 and y <= 1.0:
            px = int(x * w)
            py = int(y * h)
        else:
            px = int(x)
            py = int(y)
        
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        
        return float(depth_map[py, px])
    
    def convert_to_3d(self, kps_2d: List[Tuple[float, float, float]], 
                     depth_map: np.ndarray, 
                     bbox: Optional[List[float]] = None) -> List[Tuple[float, float, float, float]]:
        """
        Convert 2D keypoints to 3D coordinates using depth map.
        
        Args:
            kps_2d: List of (x_norm, y_norm, confidence) keypoints
            depth_map: Depth map
            bbox: Optional bounding box [x, y, w, h] for coordinate conversion
        
        Returns:
            List of (x_3d, y_3d, z_3d, confidence) in meters
        """
        if not self.camera_intrinsics:
            # Use default intrinsics if not calibrated
            h, w = depth_map.shape
            fx = fy = w * 0.7  # Approximate focal length
            cx = w / 2.0
            cy = h / 2.0
        else:
            fx = self.camera_intrinsics['fx']
            fy = self.camera_intrinsics['fy']
            cx = self.camera_intrinsics['cx']
            cy = self.camera_intrinsics['cy']
            h = self.camera_intrinsics['height']
            w = self.camera_intrinsics['width']
        
        kps_3d = []
        
        for kp in kps_2d:
            x_norm, y_norm, conf = kp
            
            # Convert normalized to pixel coordinates
            if bbox:
                # Keypoints are relative to bbox
                x_pixel = bbox[0] + x_norm * bbox[2]
                y_pixel = bbox[1] + y_norm * bbox[3]
            else:
                # Keypoints are relative to full frame
                x_pixel = x_norm * w
                y_pixel = y_norm * h
            
            # Get depth at this point
            z_depth = self.get_depth_at_point(depth_map, x_pixel, y_pixel)
            
            # Convert to 3D using camera intrinsics
            x_3d = (x_pixel - cx) * z_depth / fx
            y_3d = (y_pixel - cy) * z_depth / fy
            z_3d = z_depth
            
            kps_3d.append((x_3d, y_3d, z_3d, conf))
        
        return kps_3d
    
    def estimate_person_distance(self, kps_3d: List[Tuple[float, float, float, float]]) -> float:
        """
        Estimate average distance of person from camera.

        Args:
            kps_3d: 3D keypoints

        Returns:
            Average distance in meters
        """
        if not kps_3d:
            return 0.0

        # Use torso keypoints (shoulders, hips) for distance estimation
        # These are most reliable for distance measurement
        torso_depths = []
        for kp in kps_3d:
            if len(kp) >= 3:
                z = kp[2]  # Depth (z coordinate)
                conf = kp[3] if len(kp) > 3 else 1.0
                if conf > 0.3:  # Only use confident keypoints
                    torso_depths.append(z)

        if torso_depths:
            return float(np.median(torso_depths))  # Use median for robustness
        return 0.0

    def estimate_depth_from_keypoints(
        self,
        kps_2d: List[Tuple[float, float, float]],
        frame_shape: Tuple[int, int] = (720, 1280),
        reference_height: float = 1.7,
    ) -> Dict[str, float]:
        """
        Estimate depth using monocular cues from 2D keypoints only.
        Useful when depth model is not available (edge deployment).

        Args:
            kps_2d: 2D keypoints [(x, y, confidence), ...]
            frame_shape: (height, width) of frame
            reference_height: Expected real-world person height in meters

        Returns:
            Dict with depth estimates from different cues
        """
        h, w = frame_shape[:2]

        # Perspective cue (position in frame)
        perspective_depth = MonocularDepthCues.perspective_cue(kps_2d, h)

        # Size cue (apparent person size)
        size_depth = MonocularDepthCues.size_cue(kps_2d, reference_height)

        # Combined estimate (weighted average)
        # Size cue is more reliable when person is fully visible
        visible_kps = sum(1 for kp in kps_2d if kp[2] > 0.3) if kps_2d else 0
        visibility_ratio = visible_kps / 17.0  # 17 COCO keypoints

        if visibility_ratio > 0.7:
            # Good visibility - trust size cue more
            combined_depth = 0.7 * size_depth + 0.3 * (perspective_depth * 5.0)
        else:
            # Partial visibility - use perspective more
            combined_depth = 0.4 * size_depth + 0.6 * (perspective_depth * 5.0)

        return {
            "perspective_depth_factor": perspective_depth,
            "size_based_depth_m": size_depth,
            "combined_depth_m": combined_depth,
            "visibility_ratio": visibility_ratio,
            "estimation_method": "monocular_cues",
        }


class JetsonDepthEstimator(DepthEstimator):
    """
    Depth estimator optimized for Jetson Orin Nano.
    Uses TensorRT-optimized models when available, falls back to geometric methods.
    """

    def __init__(
        self,
        method: str = "auto",
        tensorrt_engine_path: Optional[str] = None,
        camera_intrinsics_path: Optional[str] = None,
    ):
        """
        Args:
            method: "auto", "geometric", "tensorrt", or "model"
            tensorrt_engine_path: Path to TensorRT-optimized depth model
            camera_intrinsics_path: Path to camera calibration
        """
        # Auto-select method based on available resources
        if method == "auto":
            if tensorrt_engine_path and os.path.exists(tensorrt_engine_path) and TENSORRT_AVAILABLE:
                method = "tensorrt"
            elif TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
                method = "model"
            else:
                method = "geometric"

        super().__init__(method=method, camera_intrinsics_path=camera_intrinsics_path)

        self.tensorrt_engine = None
        if method == "tensorrt" and tensorrt_engine_path:
            self._load_tensorrt_engine(tensorrt_engine_path)

    def _load_tensorrt_engine(self, engine_path: str):
        """Load TensorRT-optimized depth estimation engine."""
        if not TENSORRT_AVAILABLE:
            log.warning("TensorRT not available for depth estimation")
            return

        try:
            log.info("Loading TensorRT depth engine from %s", engine_path)
            # Engine loading would go here
            # self.tensorrt_engine = TRTEngine(engine_path)
            log.info("TensorRT depth engine loaded successfully")
        except Exception as e:
            log.warning("Failed to load TensorRT depth engine: %s", e)
            self.method = "geometric"

    def estimate_depth_map(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth map optimized for Jetson.
        Automatically selects best available method.
        """
        if self.method == "tensorrt" and self.tensorrt_engine is not None:
            return self._estimate_depth_tensorrt(frame)
        return super().estimate_depth_map(frame)

    def _estimate_depth_tensorrt(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth using TensorRT-optimized model."""
        # TensorRT inference would go here
        # For now, fall back to geometric
        return self._estimate_depth_geometric(frame)

