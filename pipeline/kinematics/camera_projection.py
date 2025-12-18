# pipeline/kinematics/camera_projection.py
"""
Camera Projection Transformation (Tc)
Differentiable camera projection from 3D to 2D.
Based on AAAI-20 approach.
"""

import numpy as np
import math
import logging
from typing import Optional, Dict, Tuple

log = logging.getLogger("camera_projection")


class CameraProjection:
    """
    Differentiable camera projection transformation.
    Projects 3D joint positions to 2D image coordinates.
    """
    
    def __init__(self, camera_intrinsics: Optional[Dict] = None):
        """
        Initialize camera projection.
        
        Args:
            camera_intrinsics: Dict with keys: fx, fy, cx, cy, width, height
        """
        if camera_intrinsics:
            self.fx = float(camera_intrinsics.get('fx', 0))
            self.fy = float(camera_intrinsics.get('fy', 0))
            self.cx = float(camera_intrinsics.get('cx', 0))
            self.cy = float(camera_intrinsics.get('cy', 0))
            self.width = int(camera_intrinsics.get('width', 1280))
            self.height = int(camera_intrinsics.get('height', 720))
        else:
            # Default intrinsics (approximate)
            self.width, self.height = 1280, 720
            self.fx = self.fy = self.width * 0.7
            self.cx = self.width / 2.0
            self.cy = self.height / 2.0
        
        log.info("Camera projection initialized (fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f)",
                 self.fx, self.fy, self.cx, self.cy)
    
    def project(self, kps_3d: np.ndarray, 
                camera_extrinsics: Optional[Dict] = None) -> np.ndarray:
        """
        Project 3D joint positions to 2D image coordinates.
        
        Args:
            kps_3d: 3D joint positions (N, 3) - (x, y, z) in camera coordinates
            camera_extrinsics: Optional dict with rotation and translation
                             Format: {
                                 'rotation': [rx, ry, rz] or rotation matrix (3,3),
                                 'translation': [tx, ty, tz]
                             }
        
        Returns:
            2D keypoints (N, 2) - (x, y) in pixel coordinates
        """
        if kps_3d is None or len(kps_3d) == 0:
            return np.array([])
        
        # Apply camera extrinsics (rotation + translation)
        if camera_extrinsics:
            kps_3d_transformed = self._apply_extrinsics(kps_3d, camera_extrinsics)
        else:
            kps_3d_transformed = kps_3d
        
        # Project to 2D using perspective projection
        kps_2d = np.zeros((len(kps_3d_transformed), 2), dtype=np.float32)
        
        for i, point_3d in enumerate(kps_3d_transformed):
            x, y, z = point_3d[0], point_3d[1], point_3d[2]
            
            # Avoid division by zero
            if abs(z) < 1e-6:
                z = 1e-6
            
            # Perspective projection
            x_2d = (x * self.fx / z) + self.cx
            y_2d = (y * self.fy / z) + self.cy
            
            kps_2d[i] = [x_2d, y_2d]
        
        return kps_2d
    
    def _apply_extrinsics(self, kps_3d: np.ndarray, 
                         extrinsics: Dict) -> np.ndarray:
        """
        Apply camera extrinsics (rotation and translation).
        
        Args:
            kps_3d: 3D points (N, 3)
            extrinsics: Dict with 'rotation' and 'translation'
        
        Returns:
            Transformed 3D points (N, 3)
        """
        rotation = extrinsics.get('rotation')
        translation = extrinsics.get('translation', [0.0, 0.0, 0.0])
        
        # Handle rotation
        if rotation is not None:
            if isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
                # Rotation matrix
                R = rotation
            elif isinstance(rotation, (list, np.ndarray)) and len(rotation) == 3:
                # Euler angles (rx, ry, rz) - convert to rotation matrix
                R = self._euler_to_rotation_matrix(rotation[0], rotation[1], rotation[2])
            else:
                log.warning("Invalid rotation format, using identity")
                R = np.eye(3)
        else:
            R = np.eye(3)
        
        # Apply rotation
        kps_3d_rotated = (R @ kps_3d.T).T
        
        # Apply translation
        translation = np.array(translation)
        kps_3d_transformed = kps_3d_rotated + translation
        
        return kps_3d_transformed
    
    def _euler_to_rotation_matrix(self, rx: float, ry: float, rz: float) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix.
        Uses ZYX convention (yaw, pitch, roll).
        """
        # Convert to radians
        rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ])
        
        Ry = np.array([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ])
        
        Rz = np.array([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx
        
        return R
    
    def unproject(self, kps_2d: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """
        Unproject 2D points to 3D (inverse projection).
        
        Args:
            kps_2d: 2D keypoints (N, 2) in pixel coordinates
            depths: Depth values (N,) for each keypoint
        
        Returns:
            3D points (N, 3) in camera coordinates
        """
        if len(kps_2d) != len(depths):
            raise ValueError("Number of 2D points must match number of depth values")
        
        kps_3d = np.zeros((len(kps_2d), 3), dtype=np.float32)
        
        for i, (point_2d, depth) in enumerate(zip(kps_2d, depths)):
            x_2d, y_2d = point_2d[0], point_2d[1]
            
            # Unproject
            x_3d = (x_2d - self.cx) * depth / self.fx
            y_3d = (y_2d - self.cy) * depth / self.fy
            z_3d = depth
            
            kps_3d[i] = [x_3d, y_3d, z_3d]
        
        return kps_3d



