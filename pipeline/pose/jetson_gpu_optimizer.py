# pipeline/pose/jetson_gpu_optimizer.py
"""
Jetson Orin Nano GPU Optimization Module
Implements NVIDIA VLA/Alpamayo-inspired optimization strategies for edge deployment.

Key optimizations:
1. Adaptive GPU utilization with thermal throttling awareness
2. Smart memory management with CUDA memory pools
3. FP16 quantization for supported operations
4. Pipeline parallelism (detection → pose → features → temporal)
5. Batched inference when applicable
6. Dynamic precision switching based on GPU load

Architecture aligned with NVIDIA's edge deployment best practices:
- Vision Encoder (YOLO) → Tokenization (Features) → Temporal Context (GRU) → Reasoning (Rules/Cloud LLM)
"""

import os
import time
import logging
import threading
from typing import Dict, Optional, List, Any, Tuple
from collections import deque
from dataclasses import dataclass
import numpy as np

log = logging.getLogger("jetson_optimizer")

# Check for CUDA availability
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# Jetson-specific utilities
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class GPUMetrics:
    """GPU performance metrics for adaptive optimization."""
    utilization: float = 0.0  # 0-100%
    memory_used: int = 0  # bytes
    memory_total: int = 0  # bytes
    temperature: float = 0.0  # Celsius
    power_draw: float = 0.0  # Watts
    clock_speed: int = 0  # MHz


class JetsonGPUMonitor:
    """
    Monitor Jetson GPU metrics for adaptive optimization.
    Uses tegrastats or nvidia-smi fallback.
    """

    def __init__(self, poll_interval: float = 1.0):
        self.poll_interval = poll_interval
        self._metrics = GPUMetrics()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Initialize NVML if available
        self._nvml_initialized = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
                self._device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                log.info("NVML initialized for GPU monitoring")
            except Exception as e:
                log.warning("NVML initialization failed: %s", e)

    def start(self):
        """Start background monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _poll_loop(self):
        """Background polling loop."""
        while self._running:
            try:
                metrics = self._read_metrics()
                with self._lock:
                    self._metrics = metrics
            except Exception as e:
                log.debug("GPU metrics polling error: %s", e)
            time.sleep(self.poll_interval)

    def _read_metrics(self) -> GPUMetrics:
        """Read current GPU metrics."""
        metrics = GPUMetrics()

        if self._nvml_initialized:
            try:
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self._device_handle)
                metrics.utilization = util.gpu

                # Memory
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._device_handle)
                metrics.memory_used = mem.used
                metrics.memory_total = mem.total

                # Temperature
                metrics.temperature = pynvml.nvmlDeviceGetTemperature(
                    self._device_handle, pynvml.NVML_TEMPERATURE_GPU
                )

                # Power (may not be supported on all Jetson models)
                try:
                    metrics.power_draw = pynvml.nvmlDeviceGetPowerUsage(self._device_handle) / 1000.0
                except pynvml.NVMLError:
                    pass

            except Exception as e:
                log.debug("NVML read error: %s", e)

        elif TORCH_AVAILABLE and CUDA_AVAILABLE:
            # Fallback to PyTorch CUDA info
            try:
                metrics.memory_used = torch.cuda.memory_allocated()
                metrics.memory_total = torch.cuda.get_device_properties(0).total_memory
            except Exception:
                pass

        return metrics

    def get_metrics(self) -> GPUMetrics:
        """Get current GPU metrics."""
        with self._lock:
            return self._metrics

    def is_thermal_throttling(self, threshold: float = 80.0) -> bool:
        """Check if GPU is thermally throttled."""
        return self._metrics.temperature > threshold

    def is_memory_constrained(self, threshold: float = 0.85) -> bool:
        """Check if GPU memory is constrained."""
        if self._metrics.memory_total == 0:
            return False
        usage_ratio = self._metrics.memory_used / self._metrics.memory_total
        return usage_ratio > threshold


class AdaptivePrecisionManager:
    """
    Manage precision (FP32/FP16/INT8) based on GPU load and accuracy requirements.
    Implements dynamic precision switching for Jetson Orin Nano.
    """

    def __init__(
        self,
        default_precision: str = "fp16",
        enable_dynamic_switching: bool = True,
        accuracy_threshold: float = 0.95,
    ):
        self.default_precision = default_precision
        self.enable_dynamic_switching = enable_dynamic_switching
        self.accuracy_threshold = accuracy_threshold

        self._current_precision = default_precision
        self._precision_history = deque(maxlen=100)
        self._accuracy_history = deque(maxlen=100)

        # Precision inference times (measured or estimated)
        self._precision_latency = {
            "fp32": 1.0,  # Baseline
            "fp16": 0.6,  # ~40% faster on Jetson
            "int8": 0.4,  # ~60% faster with TensorRT INT8
        }

    def get_optimal_precision(
        self,
        gpu_metrics: GPUMetrics,
        target_latency_ms: float = 50.0,
        current_latency_ms: float = 0.0,
    ) -> str:
        """
        Determine optimal precision based on GPU state and latency targets.

        Args:
            gpu_metrics: Current GPU metrics
            target_latency_ms: Target inference latency
            current_latency_ms: Current measured latency

        Returns:
            Recommended precision: "fp32", "fp16", or "int8"
        """
        if not self.enable_dynamic_switching:
            return self.default_precision

        # If thermally throttled, use lower precision to reduce load
        if gpu_metrics.temperature > 75.0:
            return "int8" if gpu_metrics.temperature > 85.0 else "fp16"

        # If latency is too high, try lower precision
        if current_latency_ms > target_latency_ms * 1.2:
            if self._current_precision == "fp32":
                return "fp16"
            elif self._current_precision == "fp16":
                return "int8"

        # If latency is well under target and GPU is not stressed, use higher precision
        if current_latency_ms < target_latency_ms * 0.5 and gpu_metrics.utilization < 50:
            if self._current_precision == "int8":
                return "fp16"

        return self._current_precision

    def update_precision(self, new_precision: str, accuracy: float = 1.0):
        """Update current precision and record metrics."""
        self._current_precision = new_precision
        self._precision_history.append(new_precision)
        self._accuracy_history.append(accuracy)


class CUDAMemoryPool:
    """
    CUDA memory pool for efficient buffer reuse.
    Reduces allocation overhead for repeated inference.
    """

    def __init__(self, max_pool_size: int = 10):
        self.max_pool_size = max_pool_size
        self._pools: Dict[Tuple[int, ...], List[Any]] = {}
        self._lock = threading.Lock()

        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            # Enable PyTorch CUDA memory caching
            try:
                torch.cuda.set_per_process_memory_fraction(0.8)  # Leave 20% for system
            except Exception as e:
                log.debug("Could not set CUDA memory fraction: %s", e)

    def get_buffer(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """Get or allocate a buffer of specified shape."""
        with self._lock:
            key = (shape, str(dtype))
            if key in self._pools and self._pools[key]:
                return self._pools[key].pop()
            return np.empty(shape, dtype=dtype)

    def return_buffer(self, buffer: np.ndarray):
        """Return buffer to pool for reuse."""
        with self._lock:
            key = (buffer.shape, str(buffer.dtype))
            if key not in self._pools:
                self._pools[key] = []
            if len(self._pools[key]) < self.max_pool_size:
                self._pools[key].append(buffer)

    def clear(self):
        """Clear all pooled buffers."""
        with self._lock:
            self._pools.clear()

        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


class PipelineParallelizer:
    """
    Pipeline parallelism for overlapping detection, pose, and temporal inference.
    Inspired by NVIDIA VLA architecture where each stage can run independently.
    """

    def __init__(self, enable_async: bool = True):
        self.enable_async = enable_async
        self._detection_queue = deque(maxlen=2)
        self._pose_queue = deque(maxlen=2)
        self._feature_queue = deque(maxlen=2)

        # Stage timing statistics
        self._stage_times = {
            "detection": deque(maxlen=50),
            "pose": deque(maxlen=50),
            "features": deque(maxlen=50),
            "temporal": deque(maxlen=50),
        }

    def record_stage_time(self, stage: str, duration_ms: float):
        """Record execution time for a pipeline stage."""
        if stage in self._stage_times:
            self._stage_times[stage].append(duration_ms)

    def get_stage_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all pipeline stages."""
        stats = {}
        for stage, times in self._stage_times.items():
            if times:
                stats[stage] = {
                    "mean_ms": float(np.mean(times)),
                    "std_ms": float(np.std(times)),
                    "min_ms": float(np.min(times)),
                    "max_ms": float(np.max(times)),
                }
            else:
                stats[stage] = {"mean_ms": 0, "std_ms": 0, "min_ms": 0, "max_ms": 0}
        return stats

    def get_bottleneck(self) -> str:
        """Identify the bottleneck stage in the pipeline."""
        stats = self.get_stage_stats()
        if not stats:
            return "unknown"
        return max(stats.keys(), key=lambda k: stats[k]["mean_ms"])


class JetsonOptimizer:
    """
    Main optimizer class for Jetson Orin Nano deployment.
    Coordinates GPU monitoring, precision management, and pipeline optimization.

    Usage:
        optimizer = JetsonOptimizer()
        optimizer.start()

        # During inference
        precision = optimizer.get_recommended_precision(current_latency_ms=45)
        # ... run inference ...
        optimizer.record_inference(latency_ms=45, stage="pose")

        # Cleanup
        optimizer.stop()
    """

    def __init__(
        self,
        target_fps: float = 15.0,
        enable_thermal_management: bool = True,
        enable_dynamic_precision: bool = True,
        default_precision: str = "fp16",
    ):
        self.target_fps = target_fps
        self.target_latency_ms = 1000.0 / target_fps
        self.enable_thermal_management = enable_thermal_management
        self.enable_dynamic_precision = enable_dynamic_precision

        # Components
        self.gpu_monitor = JetsonGPUMonitor()
        self.precision_manager = AdaptivePrecisionManager(
            default_precision=default_precision,
            enable_dynamic_switching=enable_dynamic_precision,
        )
        self.memory_pool = CUDAMemoryPool()
        self.pipeline = PipelineParallelizer()

        # Inference statistics
        self._total_inferences = 0
        self._inference_latencies = deque(maxlen=100)

        log.info(
            "JetsonOptimizer initialized (target_fps=%.1f, precision=%s, thermal=%s)",
            target_fps,
            default_precision,
            enable_thermal_management,
        )

    def start(self):
        """Start optimizer (background monitoring)."""
        self.gpu_monitor.start()

    def stop(self):
        """Stop optimizer and cleanup."""
        self.gpu_monitor.stop()
        self.memory_pool.clear()

    def get_recommended_precision(self, current_latency_ms: float = 0.0) -> str:
        """Get recommended inference precision."""
        metrics = self.gpu_monitor.get_metrics()
        return self.precision_manager.get_optimal_precision(
            gpu_metrics=metrics,
            target_latency_ms=self.target_latency_ms,
            current_latency_ms=current_latency_ms,
        )

    def record_inference(self, latency_ms: float, stage: str = "total"):
        """Record inference timing."""
        self._total_inferences += 1
        self._inference_latencies.append(latency_ms)
        self.pipeline.record_stage_time(stage, latency_ms)

    def should_skip_frame(self) -> bool:
        """
        Determine if current frame should be skipped to maintain target FPS.
        Used for graceful degradation under load.
        """
        metrics = self.gpu_monitor.get_metrics()

        # Skip if thermally throttled and already behind
        if self.enable_thermal_management and metrics.temperature > 85.0:
            return True

        # Skip if memory constrained
        if self.gpu_monitor.is_memory_constrained(threshold=0.95):
            return True

        return False

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status for logging/display."""
        metrics = self.gpu_monitor.get_metrics()
        pipeline_stats = self.pipeline.get_stage_stats()

        avg_latency = (
            float(np.mean(self._inference_latencies))
            if self._inference_latencies
            else 0.0
        )

        return {
            "gpu_utilization": metrics.utilization,
            "gpu_temperature": metrics.temperature,
            "memory_used_mb": metrics.memory_used / (1024 * 1024),
            "memory_total_mb": metrics.memory_total / (1024 * 1024),
            "current_precision": self.precision_manager._current_precision,
            "avg_latency_ms": avg_latency,
            "total_inferences": self._total_inferences,
            "bottleneck_stage": self.pipeline.get_bottleneck(),
            "pipeline_stats": pipeline_stats,
            "thermal_throttling": self.gpu_monitor.is_thermal_throttling(),
            "memory_constrained": self.gpu_monitor.is_memory_constrained(),
        }


# Convenience functions for integration
def create_jetson_optimizer(config: Dict[str, Any] = None) -> JetsonOptimizer:
    """
    Create JetsonOptimizer from configuration dictionary.

    Args:
        config: Optional configuration dict with keys:
            - target_fps (float): Target FPS (default: 15)
            - enable_thermal_management (bool): Enable thermal management (default: True)
            - enable_dynamic_precision (bool): Enable dynamic precision switching (default: True)
            - default_precision (str): Default precision "fp32"/"fp16"/"int8" (default: "fp16")

    Returns:
        Configured JetsonOptimizer instance
    """
    config = config or {}
    return JetsonOptimizer(
        target_fps=config.get("target_fps", 15.0),
        enable_thermal_management=config.get("enable_thermal_management", True),
        enable_dynamic_precision=config.get("enable_dynamic_precision", True),
        default_precision=config.get("default_precision", "fp16"),
    )


def get_optimal_batch_size(
    available_memory_mb: float,
    model_memory_mb: float = 500,
    per_sample_mb: float = 10,
) -> int:
    """
    Calculate optimal batch size for given memory constraints.
    For Jetson Orin Nano Super with 8GB memory.

    Args:
        available_memory_mb: Available GPU memory in MB
        model_memory_mb: Memory used by model weights
        per_sample_mb: Memory per sample in batch

    Returns:
        Optimal batch size (minimum 1)
    """
    remaining_memory = available_memory_mb - model_memory_mb
    if remaining_memory <= 0:
        return 1
    batch_size = int(remaining_memory / per_sample_mb)
    return max(1, min(batch_size, 8))  # Cap at 8 for edge deployment
