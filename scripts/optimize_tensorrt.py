#!/usr/bin/env python3
"""
TensorRT Model Optimization Script for Jetson Deployment
Converts TensorFlow Lite and PyTorch models to TensorRT engines for maximum performance.

Usage:
 python scripts/optimize_tensorrt.py --config config/system.yaml
 python scripts/optimize_tensorrt.py --models pose temporal --precision FP16
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path

# Add pipeline to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.pose.tensorrt_optimizer import TensorRTOptimizer

log = logging.getLogger("tensorrt_optimize")

def load_config(config_path: str) -> dict:
 """Load configuration from YAML file."""
 with open(config_path, 'r') as f:
 return yaml.safe_load(f)

def optimize_models(config: dict, models_to_optimize: list = None,
 precision: str = "FP16", force_rebuild: bool = False):
 """
 Optimize models for TensorRT.

 Args:
 config: System configuration dict
 models_to_optimize: List of model types to optimize (None = all available)
 precision: Precision mode ("FP32", "FP16", "INT8")
 force_rebuild: Force rebuild even if engines exist
 """
 log.info("Starting TensorRT model optimization")

 # Initialize optimizer
 optimizer = TensorRTOptimizer(
 workspace_size=config.get("tensorrt_workspace_size", 1 << 30),
 max_batch_size=config.get("tensorrt_max_batch_size", 1)
 )

 tensorrt_dir = config.get("tensorrt_model_dir", "models/tensorrt")
 os.makedirs(tensorrt_dir, exist_ok=True)

 models = config.get("models", {})
 optimized_count = 0

 # Optimize pose estimation model
 if models_to_optimize is None or "pose" in models_to_optimize:
 pose_model = models.get("pose")
 if pose_model and os.path.exists(pose_model):
 engine_path = os.path.join(tensorrt_dir, "movenet_pose.engine")

 if force_rebuild or not os.path.exists(engine_path):
 log.info(f"Optimizing pose model: {pose_model}")
 success = optimizer.convert_tflite_to_tensorrt(
 pose_model, engine_path,
 input_shapes={"input": (1, 192, 192, 3)},
 precision=precision
 )
 if success:
 optimized_count += 1
 log.info(f" Pose model optimized: {engine_path}")
 else:
 log.error(" Pose model optimization failed")
 else:
 log.info(f"Pose engine already exists: {engine_path}")
 else:
 log.warning(f"Pose model not found: {pose_model}")

 # Optimize temporal models
 temporal_models = {}
 temporal_model = models.get("temporal")
 if temporal_model and os.path.exists(temporal_model):
 temporal_models["temporal"] = temporal_model

 if temporal_models and (models_to_optimize is None or "temporal" in models_to_optimize):
 log.info("Optimizing temporal models")
 temporal_engines = optimizer.optimize_temporal_models(temporal_models, tensorrt_dir)

 for model_type, engine_path in temporal_engines.items():
 if os.path.exists(engine_path):
 optimized_count += 1
 log.info(f" {model_type.capitalize()} model optimized: {engine_path}")
 else:
 log.error(f" {model_type.capitalize()} model optimization failed")

 # Benchmark optimized models
 if optimized_count > 0:
 log.info("Benchmarking optimized models...")
 benchmark_results = {}

 # Benchmark pose model
 pose_engine = os.path.join(tensorrt_dir, "movenet_pose.engine")
 if os.path.exists(pose_engine):
 log.info("Benchmarking pose model...")
 results = optimizer.benchmark_model(pose_engine, (1, 192, 192, 3), num_runs=50)
 benchmark_results["pose"] = results
 if "error" not in results:
 log.info(".2f")
 else:
 log.warning(f"Pose benchmark failed: {results['error']}")

 # Benchmark temporal model
 temporal_engine = os.path.join(tensorrt_dir, "temporal.engine")
 if os.path.exists(temporal_engine):
 log.info("Benchmarking temporal model...")
 results = optimizer.benchmark_model(temporal_engine, (1, 48, 9), num_runs=50)
 benchmark_results["temporal"] = results
 if "error" not in results:
 log.info(".2f")
 else:
 log.warning(f"Temporal benchmark failed: {results['error']}")

 # Save benchmark results
 benchmark_file = os.path.join(tensorrt_dir, "benchmark_results.yaml")
 with open(benchmark_file, 'w') as f:
 yaml.dump(benchmark_results, f, default_flow_style=False)
 log.info(f"Benchmark results saved to: {benchmark_file}")

 log.info(f"TensorRT optimization complete. {optimized_count} models optimized.")

 if optimized_count > 0:
 log.info("To use optimized models, set 'enable_tensorrt: true' in system.yaml")
 log.info("Expected performance improvement: 2-5x faster inference on Jetson")

 return optimized_count > 0

def main():
 parser = argparse.ArgumentParser(description="TensorRT Model Optimization for Jetson")
 parser.add_argument("--config", default="config/system.yaml",
 help="Path to system configuration file")
 parser.add_argument("--models", nargs="+",
 choices=["pose", "temporal", "all"],
 help="Models to optimize (default: all available)")
 parser.add_argument("--precision", choices=["FP32", "FP16", "INT8"], default="FP16",
 help="TensorRT precision mode (default: FP16)")
 parser.add_argument("--force", action="store_true",
 help="Force rebuild even if engines exist")
 parser.add_argument("--verbose", action="store_true",
 help="Enable verbose logging")

 args = parser.parse_args()

 # Setup logging
 level = logging.DEBUG if args.verbose else logging.INFO
 logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

 try:
 # Load configuration
 config = load_config(args.config)
 log.info(f"Loaded configuration from {args.config}")

 # Determine which models to optimize
 models_to_optimize = args.models
 if models_to_optimize and "all" in models_to_optimize:
 models_to_optimize = None # Optimize all available

 # Run optimization
 success = optimize_models(
 config=config,
 models_to_optimize=models_to_optimize,
 precision=args.precision,
 force_rebuild=args.force
 )

 sys.exit(0 if success else 1)

 except Exception as e:
 log.exception(f"Optimization failed: {e}")
 sys.exit(1)

if __name__ == "__main__":
 main()