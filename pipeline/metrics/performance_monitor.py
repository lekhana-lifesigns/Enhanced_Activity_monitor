# pipeline/metrics/performance_monitor.py
"""
Performance Monitoring and Metrics Collection
Tracks FPS, latency, IoU, ID-switch rate, and other performance metrics.
"""

import time
import logging
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

log = logging.getLogger("perf_monitor")


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    frame_id: int
    timestamp: float
    inference_ms: float
    fps: float
    detection_confidence: float = 0.0
    track_id: Optional[int] = None
    bbox: Optional[List[float]] = None
    posture_iou: Optional[float] = None
    face_iou: Optional[float] = None


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    total_frames: int = 0
    successful_frames: int = 0
    failed_frames: int = 0
    
    # Latency metrics
    latency_mean: float = 0.0
    latency_median: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_std: float = 0.0
    
    # FPS metrics
    fps_mean: float = 0.0
    fps_min: float = 0.0
    fps_max: float = 0.0
    fps_std: float = 0.0
    fps_target: float = 15.0
    fps_meets_target: bool = False
    
    # Detection metrics
    detection_confidence_mean: float = 0.0
    posture_iou_mean: float = 0.0
    face_iou_mean: float = 0.0
    
    # Tracking metrics
    id_switches: int = 0
    id_switch_rate: float = 0.0
    track_consistency: float = 0.0


class PerformanceMonitor:
    """
    Monitor and collect performance metrics for the inference pipeline.
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 fps_target: float = 15.0,
                 enable_iou_tracking: bool = False,
                 enable_id_switch_tracking: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of recent frames to track
            fps_target: Target FPS for performance validation
            enable_iou_tracking: Enable IoU calculation (requires ground truth)
            enable_id_switch_tracking: Enable ID-switch rate tracking
        """
        self.window_size = window_size
        self.fps_target = fps_target
        
        # Frame metrics history
        self.frame_metrics: deque = deque(maxlen=window_size)
        self.latency_history: deque = deque(maxlen=window_size)
        self.fps_history: deque = deque(maxlen=window_size)
        
        # Tracking state for ID-switch detection
        self.enable_id_switch_tracking = enable_id_switch_tracking
        self.track_id_history: Dict[int, List[int]] = {}  # track_id -> [frame_ids]
        self.prev_track_id: Optional[int] = None
        self.id_switches: int = 0
        self.total_track_frames: int = 0
        
        # IoU tracking (requires ground truth)
        self.enable_iou_tracking = enable_iou_tracking
        self.ground_truth: Optional[Dict] = None
        
        # Statistics
        self.stats = PerformanceStats(fps_target=fps_target)
        
        # Timing
        self.start_time = time.time()
        self.frame_count = 0
    
    def set_ground_truth(self, ground_truth: Dict):
        """Set ground truth annotations for IoU calculation."""
        self.ground_truth = ground_truth
    
    def record_frame(self,
                    inference_ms: float,
                    fps: float,
                    detection_confidence: float = 0.0,
                    track_id: Optional[int] = None,
                    bbox: Optional[List[float]] = None,
                    frame_id: Optional[int] = None) -> FrameMetrics:
        """
        Record metrics for a single frame.
        
        Returns:
            FrameMetrics object
        """
        self.frame_count += 1
        if frame_id is None:
            frame_id = self.frame_count
        
        timestamp = time.time()
        
        # Calculate IoU if ground truth available
        posture_iou = None
        face_iou = None
        if self.enable_iou_tracking and self.ground_truth:
            if frame_id in self.ground_truth:
                gt = self.ground_truth[frame_id]
                if bbox:
                    # Calculate posture IoU
                    if 'posture_bbox' in gt:
                        posture_iou = self._calculate_iou(bbox, gt['posture_bbox'])
                    
                    # Calculate face IoU
                    if 'face_bbox' in gt and 'face_bbox' in locals():
                        face_iou = self._calculate_iou(bbox, gt['face_bbox'])
        
        # Track ID-switch
        if self.enable_id_switch_tracking and track_id is not None:
            self._track_id_switch(track_id, frame_id)
        
        # Create metrics
        metrics = FrameMetrics(
            frame_id=frame_id,
            timestamp=timestamp,
            inference_ms=inference_ms,
            fps=fps,
            detection_confidence=detection_confidence,
            track_id=track_id,
            bbox=bbox,
            posture_iou=posture_iou,
            face_iou=face_iou
        )
        
        # Store metrics
        self.frame_metrics.append(metrics)
        self.latency_history.append(inference_ms)
        self.fps_history.append(fps)
        
        return metrics
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        # Normalize to [x1, y1, x2, y2]
        if bbox1[2] < bbox1[0]:
            x1_1, y1_1, w1, h1 = bbox1
            x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        else:
            x1_1, y1_1, x2_1, y2_1 = bbox1
        
        if bbox2[2] < bbox2[0]:
            x1_2, y1_2, w2, h2 = bbox2
            x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        else:
            x1_2, y1_2, x2_2, y2_2 = bbox2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _track_id_switch(self, track_id: int, frame_id: int):
        """Track ID switches for identity persistence."""
        self.total_track_frames += 1
        
        # Check if track ID changed unexpectedly
        if self.prev_track_id is not None and self.prev_track_id != track_id:
            # Potential ID switch - check if it's a valid change
            # (e.g., person left and new person entered)
            # For now, count as switch (can be refined with spatial tracking)
            self.id_switches += 1
        
        # Update history
        if track_id not in self.track_id_history:
            self.track_id_history[track_id] = []
        self.track_id_history[track_id].append(frame_id)
        
        self.prev_track_id = track_id
    
    def get_stats(self) -> PerformanceStats:
        """Calculate and return aggregated statistics."""
        if not self.frame_metrics:
            return self.stats
        
        latencies = list(self.latency_history)
        fps_values = list(self.fps_history)
        
        # Calculate latency stats
        if latencies:
            latencies_arr = np.array(latencies)
            self.stats.latency_mean = float(np.mean(latencies_arr))
            self.stats.latency_median = float(np.median(latencies_arr))
            self.stats.latency_p95 = float(np.percentile(latencies_arr, 95))
            self.stats.latency_p99 = float(np.percentile(latencies_arr, 99))
            self.stats.latency_std = float(np.std(latencies_arr))
        
        # Calculate FPS stats
        if fps_values:
            fps_arr = np.array(fps_values)
            self.stats.fps_mean = float(np.mean(fps_arr))
            self.stats.fps_min = float(np.min(fps_arr))
            self.stats.fps_max = float(np.max(fps_arr))
            self.stats.fps_std = float(np.std(fps_arr))
            self.stats.fps_meets_target = self.stats.fps_mean >= self.fps_target
        
        # Calculate detection stats
        confidences = [m.detection_confidence for m in self.frame_metrics if m.detection_confidence > 0]
        if confidences:
            self.stats.detection_confidence_mean = float(np.mean(confidences))
        
        # Calculate IoU stats
        posture_ious = [m.posture_iou for m in self.frame_metrics if m.posture_iou is not None]
        if posture_ious:
            self.stats.posture_iou_mean = float(np.mean(posture_ious))
        
        face_ious = [m.face_iou for m in self.frame_metrics if m.face_iou is not None]
        if face_ious:
            self.stats.face_iou_mean = float(np.mean(face_ious))
        
        # Calculate ID-switch rate
        if self.total_track_frames > 0:
            self.stats.id_switches = self.id_switches
            # ID-switch rate as percentage of total track frames
            self.stats.id_switch_rate = (self.id_switches / self.total_track_frames) * 100.0
            self.stats.track_consistency = ((self.total_track_frames - self.id_switches) / self.total_track_frames) * 100.0
        
        # Overall stats
        self.stats.total_frames = self.frame_count
        self.stats.successful_frames = len(self.frame_metrics)
        self.stats.failed_frames = self.frame_count - len(self.frame_metrics)
        
        return self.stats
    
    def get_summary(self) -> Dict:
        """Get a summary dictionary of current performance."""
        stats = self.get_stats()
        
        return {
            'fps': {
                'mean': stats.fps_mean,
                'min': stats.fps_min,
                'max': stats.fps_max,
                'std': stats.fps_std,
                'target': stats.fps_target,
                'meets_target': stats.fps_meets_target
            },
            'latency': {
                'mean_ms': stats.latency_mean,
                'median_ms': stats.latency_median,
                'p95_ms': stats.latency_p95,
                'p99_ms': stats.latency_p99,
                'std_ms': stats.latency_std
            },
            'detection': {
                'confidence_mean': stats.detection_confidence_mean,
                'posture_iou_mean': stats.posture_iou_mean,
                'face_iou_mean': stats.face_iou_mean
            },
            'tracking': {
                'id_switches': stats.id_switches,
                'id_switch_rate_percent': stats.id_switch_rate,
                'track_consistency_percent': stats.track_consistency,
                'meets_goal': stats.id_switch_rate <= 5.0
            },
            'overall': {
                'total_frames': stats.total_frames,
                'successful_frames': stats.successful_frames,
                'failed_frames': stats.failed_frames,
                'success_rate': stats.successful_frames / stats.total_frames if stats.total_frames > 0 else 0.0
            }
        }
    
    def reset(self):
        """Reset all metrics."""
        self.frame_metrics.clear()
        self.latency_history.clear()
        self.fps_history.clear()
        self.track_id_history.clear()
        self.id_switches = 0
        self.total_track_frames = 0
        self.prev_track_id = None
        self.frame_count = 0
        self.start_time = time.time()
        self.stats = PerformanceStats(fps_target=self.fps_target)

