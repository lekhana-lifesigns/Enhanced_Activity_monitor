# pipeline/pose/tensorrt_optimizer.py
"""
TensorRT Optimization for Jetson Inference Acceleration
Converts TensorFlow Lite and PyTorch models to TensorRT for maximum performance
"""

import os
import logging
import numpy as np
from typing import Optional, Dict, Any
import time

log = logging.getLogger("tensorrt_optimizer")

# TensorRT imports (Jetson-specific)
try:
 import tensorrt as trt
 import pycuda.driver as cuda
 import pycuda.autoinit
 TENSORRT_AVAILABLE = True
 log.info("TensorRT available for Jetson optimization")
except ImportError:
 TENSORRT_AVAILABLE = False
 log.warning("TensorRT not available - install: pip install tensorrt pycuda")

# ONNX for PyTorch conversion
try:
 import onnx
 import onnxruntime as ort
 ONNX_AVAILABLE = True
except ImportError:
 ONNX_AVAILABLE = False
 log.warning("ONNX not available - install: pip install onnx onnxruntime")

# TensorFlow for TFLite conversion
try:
 import tensorflow as tf
 TENSORFLOW_AVAILABLE = True
except ImportError:
 TENSORFLOW_AVAILABLE = False
 log.warning("TensorFlow not available - install: pip install tensorflow")

# PyTorch for model conversion
try:
 import torch
 PYTORCH_AVAILABLE = True
except ImportError:
 PYTORCH_AVAILABLE = False
 log.warning("PyTorch not available - install: pip install torch torchvision")


class TensorRTOptimizer:
 """
 Converts and optimizes ML models for TensorRT inference on Jetson devices.

 Supports conversion from:
 - TensorFlow Lite models (.tflite)
 - PyTorch models (.pt/.pth) via ONNX intermediate
 - ONNX models directly (.onnx)
 """

 def __init__(self, workspace_size: int = 1 << 30, max_batch_size: int = 1):
 """
 Initialize TensorRT optimizer.

 Args:
 workspace_size: Maximum workspace size for TensorRT (default: 1GB)
 max_batch_size: Maximum batch size for inference
 """
 self.workspace_size = workspace_size
 self.max_batch_size = max_batch_size
 self.logger = trt.Logger(trt.Logger.WARNING) if TENSORRT_AVAILABLE else None
 self.engine_cache = {} # Cache for loaded engines

 def convert_tflite_to_tensorrt(self, tflite_path: str, output_path: str,
 input_shapes: Dict[str, tuple],
 precision: str = "FP16") -> bool:
 """
 Convert TensorFlow Lite model to TensorRT engine.

 Args:
 tflite_path: Path to .tflite model
 output_path: Path to save .engine file
 input_shapes: Dict of input names to shapes (e.g., {"input": (1, 192, 192, 3)})
 precision: Precision mode ("FP32", "FP16", "INT8")

 Returns:
 True if conversion successful
 """
 if not TENSORRT_AVAILABLE or not TENSORFLOW_AVAILABLE:
 log.error("TensorRT or TensorFlow not available for TFLite conversion")
 return False

 try:
 log.info(f"Converting TFLite model {tflite_path} to TensorRT (precision: {precision})")

 # Load TFLite model
 with open(tflite_path, 'rb') as f:
 tflite_model = f.read()

 # Create TensorRT builder
 builder = trt.Builder(self.logger)
 network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
 parser = trt.OnnxParser(network, self.logger)

 # Convert TFLite to ONNX first (TensorRT doesn't directly support TFLite)
 # Use tf.lite.TFLiteConverter to get concrete function, then convert to ONNX
 interpreter = tf.lite.Interpreter(model_content=tflite_model)
 interpreter.allocate_tensors()

 # Get model details
 input_details = interpreter.get_input_details()
 output_details = interpreter.get_output_details()

 log.info(f"TFLite model inputs: {[d['name'] for d in input_details]}")
 log.info(f"TFLite model outputs: {[d['name'] for d in output_details]}")

 # For now, create a simple ONNX conversion
 # This is a simplified approach - in production, you'd want proper TFLite->ONNX conversion
 onnx_path = tflite_path.replace('.tflite', '.onnx')

 if self._convert_tflite_to_onnx_simple(tflite_path, onnx_path, input_shapes):
 return self.convert_onnx_to_tensorrt(onnx_path, output_path, precision)
 else:
 log.error("Failed to convert TFLite to ONNX")
 return False

 except Exception as e:
 log.exception(f"Failed to convert TFLite to TensorRT: {e}")
 return False

 def _convert_tflite_to_onnx_simple(self, tflite_path: str, onnx_path: str,
 input_shapes: Dict[str, tuple]) -> bool:
 """
 Simple TFLite to ONNX conversion using tf2onnx.
 In production, use more robust conversion tools.
 """
 try:
 import tf2onnx

 # Load TFLite model
 with open(tflite_path, 'rb') as f:
 tflite_model = f.read()

 # Convert to SavedModel format first
 interpreter = tf.lite.Interpreter(model_content=tflite_model)
 interpreter.allocate_tensors()

 # Use tf2onnx's built-in TFLite-to-ONNX conversion
 input_names = [d['name'] for d in interpreter.get_input_details()]
 output_names = [d['name'] for d in interpreter.get_output_details()]

 model_proto, _ = tf2onnx.convert.from_tflite(
 tflite_path,
 input_names=input_names,
 output_names=output_names
 )

 import onnx
 onnx.save(model_proto, onnx_path)
 log.info("TFLite -> ONNX conversion successful: %s", onnx_path)
 return True

 except ImportError:
 log.error("tf2onnx not available - install: pip install tf2onnx")
 return False
 except Exception as e:
 log.exception(f"TFLite to ONNX conversion failed: {e}")
 return False

 def convert_pytorch_to_tensorrt(self, pytorch_path: str, output_path: str,
 input_shape: tuple, precision: str = "FP16",
 device: str = "cuda") -> bool:
 """
 Convert PyTorch model to TensorRT via ONNX intermediate.

 Args:
 pytorch_path: Path to .pt/.pth PyTorch model
 output_path: Path to save .engine file
 input_shape: Input tensor shape (batch_size, ...)
 precision: Precision mode ("FP32", "FP16", "INT8")

 Returns:
 True if conversion successful
 """
 if not PYTORCH_AVAILABLE or not ONNX_AVAILABLE:
 log.error("PyTorch or ONNX not available for conversion")
 return False

 try:
 log.info(f"Converting PyTorch model {pytorch_path} to TensorRT (precision: {precision})")

 # Convert PyTorch to ONNX first
 onnx_path = pytorch_path.replace('.pt', '.onnx').replace('.pth', '.onnx')

 if self._convert_pytorch_to_onnx(pytorch_path, onnx_path, input_shape, device):
 return self.convert_onnx_to_tensorrt(onnx_path, output_path, precision)
 else:
 log.error("Failed to convert PyTorch to ONNX")
 return False

 except Exception as e:
 log.exception(f"Failed to convert PyTorch to TensorRT: {e}")
 return False

 def _convert_pytorch_to_onnx(self, pytorch_path: str, onnx_path: str,
 input_shape: tuple, device: str) -> bool:
 """Convert PyTorch model to ONNX format."""
 try:
 # Load PyTorch model
 model = torch.load(pytorch_path, map_location=device)
 model.eval()

 # Create dummy input
 dummy_input = torch.randn(input_shape).to(device)

 # Export to ONNX
 torch.onnx.export(
 model, dummy_input, onnx_path,
 export_params=True,
 opset_version=11,
 do_constant_folding=True,
 input_names=['input'],
 output_names=['output'],
 dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
 )

 log.info(f"Successfully converted PyTorch to ONNX: {onnx_path}")
 return True

 except Exception as e:
 log.exception(f"PyTorch to ONNX conversion failed: {e}")
 return False

 def convert_onnx_to_tensorrt(self, onnx_path: str, output_path: str,
 precision: str = "FP16") -> bool:
 """
 Convert ONNX model to TensorRT engine.

 Args:
 onnx_path: Path to .onnx model
 output_path: Path to save .engine file
 precision: Precision mode ("FP32", "FP16", "INT8")

 Returns:
 True if conversion successful
 """
 if not TENSORRT_AVAILABLE:
 log.error("TensorRT not available for ONNX conversion")
 return False

 try:
 log.info(f"Converting ONNX model {onnx_path} to TensorRT engine (precision: {precision})")

 # Create builder and network
 builder = trt.Builder(self.logger)
 network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
 parser = trt.OnnxParser(network, self.logger)

 # Parse ONNX model
 with open(onnx_path, 'rb') as f:
 if not parser.parse(f.read()):
 for error in range(parser.num_errors):
 log.error(f"ONNX parse error: {parser.get_error(error)}")
 return False

 # Configure builder
 config = builder.create_builder_config()
 config.max_workspace_size = self.workspace_size

 # Set precision
 if precision == "FP16" and builder.platform_has_fast_fp16:
 config.set_flag(trt.BuilderFlag.FP16)
 log.info("Using FP16 precision")
 elif precision == "INT8" and builder.platform_has_fast_int8:
 config.set_flag(trt.BuilderFlag.INT8)
 log.info("Using INT8 precision")
 else:
 log.info("Using FP32 precision")

 # Build engine
 engine = builder.build_engine(network, config)
 if engine is None:
 log.error("Failed to build TensorRT engine")
 return False

 # Save engine
 with open(output_path, 'wb') as f:
 f.write(engine.serialize())

 log.info(f"Successfully created TensorRT engine: {output_path}")
 return True

 except Exception as e:
 log.exception(f"Failed to convert ONNX to TensorRT: {e}")
 return False

 def load_engine(self, engine_path: str) -> Optional[Any]:
 """
 Load TensorRT engine from file.

 Args:
 engine_path: Path to .engine file

 Returns:
 TensorRT engine object or None if failed
 """
 if not TENSORRT_AVAILABLE:
 return None

 try:
 if engine_path in self.engine_cache:
 return self.engine_cache[engine_path]

 with open(engine_path, 'rb') as f:
 engine_data = f.read()

 runtime = trt.Runtime(self.logger)
 engine = runtime.deserialize_cuda_engine(engine_data)

 if engine:
 self.engine_cache[engine_path] = engine
 log.info(f"Loaded TensorRT engine: {engine_path}")
 return engine
 else:
 log.error(f"Failed to deserialize engine: {engine_path}")
 return None

 except Exception as e:
 log.exception(f"Failed to load TensorRT engine: {e}")
 return None

 def create_execution_context(self, engine: Any) -> Optional[Any]:
 """Create execution context for TensorRT engine."""
 if not TENSORRT_AVAILABLE or engine is None:
 return None

 try:
 return engine.create_execution_context()
 except Exception as e:
 log.exception(f"Failed to create execution context: {e}")
 return None

 def optimize_pose_model(self, model_path: str, output_dir: str = "models/tensorrt") -> Dict[str, str]:
 """
 Optimize pose estimation model for TensorRT.

 Args:
 model_path: Path to MoveNet TFLite model
 output_dir: Directory to save optimized models

 Returns:
 Dict mapping model types to engine paths
 """
 os.makedirs(output_dir, exist_ok=True)

 optimized_paths = {}

 # Convert pose estimation model
 if model_path.endswith('.tflite'):
 engine_path = os.path.join(output_dir, "movenet_pose.engine")
 input_shapes = {"input": (1, 192, 192, 3)} # MoveNet input shape

 if self.convert_tflite_to_tensorrt(model_path, engine_path, input_shapes, "FP16"):
 optimized_paths["pose"] = engine_path
 else:
 log.warning("Pose model optimization failed, will use original")

 return optimized_paths

 def optimize_temporal_models(self, model_configs: Dict[str, str],
 output_dir: str = "models/tensorrt") -> Dict[str, str]:
 """
 Optimize temporal models for TensorRT.

 Args:
 model_configs: Dict of model types to paths (e.g., {"temporal": "path/to/model.tflite"})
 output_dir: Directory to save optimized models

 Returns:
 Dict mapping model types to engine paths
 """
 os.makedirs(output_dir, exist_ok=True)

 optimized_paths = {}

 for model_type, model_path in model_configs.items():
 if not os.path.exists(model_path):
 log.warning(f"Model not found: {model_path}")
 continue

 engine_name = f"{model_type}.engine"
 engine_path = os.path.join(output_dir, engine_name)

 if model_path.endswith('.tflite'):
 # TensorFlow Lite model
 input_shapes = {"input": (1, 48, 9)} # Typical temporal input shape
 if self.convert_tflite_to_tensorrt(model_path, engine_path, input_shapes, "FP16"):
 optimized_paths[model_type] = engine_path

 elif model_path.endswith(('.pt', '.pth')):
 # PyTorch model
 input_shape = (1, 48, 9) # batch_size, seq_len, features
 if self.convert_pytorch_to_tensorrt(model_path, engine_path, input_shape, "FP16"):
 optimized_paths[model_type] = engine_path

 return optimized_paths

 def benchmark_model(self, engine_path: str, input_shape: tuple,
 num_runs: int = 100) -> Dict[str, float]:
 """
 Benchmark TensorRT engine performance.

 Args:
 engine_path: Path to .engine file
 input_shape: Input tensor shape
 num_runs: Number of inference runs for benchmarking

 Returns:
 Dict with performance metrics
 """
 if not TENSORRT_AVAILABLE:
 return {"error": "TensorRT not available"}

 try:
 engine = self.load_engine(engine_path)
 if engine is None:
 return {"error": "Failed to load engine"}

 context = self.create_execution_context(engine)
 if context is None:
 return {"error": "Failed to create context"}

 # Allocate device memory
 input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
 output_size = trt.volume(engine.get_binding_shape(1)) * np.dtype(np.float32).itemsize

 d_input = cuda.mem_alloc(input_size)
 d_output = cuda.mem_alloc(output_size)
 bindings = [int(d_input), int(d_output)]

 # Create CUDA stream
 stream = cuda.Stream()

 # Warmup
 dummy_input = np.random.randn(*input_shape).astype(np.float32)
 cuda.memcpy_htod(d_input, dummy_input)
 context.execute_v2(bindings)

 # Benchmark
 latencies = []
 for _ in range(num_runs):
 start_time = time.time()

 # Copy input to device
 cuda.memcpy_htod_async(d_input, dummy_input, stream)

 # Execute inference
 context.execute_v2(bindings)

 # Copy output to host
 output = np.empty(engine.get_binding_shape(1), dtype=np.float32)
 cuda.memcpy_dtoh_async(output, d_output, stream)

 # Synchronize
 stream.synchronize()

 latency = (time.time() - start_time) * 1000 # ms
 latencies.append(latency)

 latencies = np.array(latencies)
 return {
 "mean_latency_ms": float(np.mean(latencies)),
 "std_latency_ms": float(np.std(latencies)),
 "min_latency_ms": float(np.min(latencies)),
 "max_latency_ms": float(np.max(latencies)),
 "throughput_fps": float(1000 / np.mean(latencies))
 }

 except Exception as e:
 log.exception(f"Benchmark failed: {e}")
 return {"error": str(e)}


class TensorRTInferenceEngine:
 """
 High-performance inference engine using TensorRT optimized models.
 Provides GPU-accelerated inference for Jetson Orin Nano deployment.
 """

 def __init__(self, engine_path: str):
 self.engine_path = engine_path
 self.optimizer = TensorRTOptimizer()
 self.engine = None
 self.context = None
 self.stream = None
 # Pre-allocated buffers for zero-copy inference
 self._d_input = None
 self._d_output = None
 self._output_buffer = None

 self._load_engine()

 def _load_engine(self):
 """Load TensorRT engine and create execution context."""
 if not TENSORRT_AVAILABLE:
 log.error("TensorRT not available")
 return

 self.engine = self.optimizer.load_engine(self.engine_path)
 if self.engine:
 self.context = self.optimizer.create_execution_context(self.engine)
 self.stream = cuda.Stream()
 log.info(f"TensorRT inference engine ready: {self.engine_path}")
 else:
 log.error(f"Failed to load TensorRT engine: {self.engine_path}")

 def infer(self, input_data: np.ndarray) -> Optional[np.ndarray]:
 """
 Run inference with TensorRT engine.

 Args:
 input_data: Input tensor

 Returns:
 Output tensor or None if failed
 """
 if not self.context or not self.stream:
 return None

 try:
 # Allocate device memory (reuse if possible)
 input_size = trt.volume(input_data.shape) * input_data.dtype.itemsize
 output_shape = self.engine.get_binding_shape(1)
 output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize

 # Reuse pre-allocated buffers for better performance
 if self._d_input is None:
 self._d_input = cuda.mem_alloc(input_size)
 self._d_output = cuda.mem_alloc(output_size)
 self._output_buffer = np.empty(output_shape, dtype=np.float32)

 bindings = [int(self._d_input), int(self._d_output)]

 # Copy input to device
 cuda.memcpy_htod_async(self._d_input, input_data, self.stream)

 # Execute inference
 self.context.execute_v2(bindings)

 # Copy output to host
 cuda.memcpy_dtoh_async(self._output_buffer, self._d_output, self.stream)

 # Synchronize
 self.stream.synchronize()

 return self._output_buffer.copy()

 except Exception as e:
 log.exception(f"TensorRT inference failed: {e}")
 return None

 def __del__(self):
 """Clean up resources."""
 try:
 if self.stream:
 self.stream.synchronize()
 except Exception:
 pass