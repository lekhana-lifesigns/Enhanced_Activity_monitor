#!/usr/bin/env python3
"""
Benchmark Inference Latency
Measures P50, P95, P99 latencies for the inference pipeline
"""

import yaml
import time
import numpy as np
import logging
from pipeline.pose.inference_pipeline import InferencePipeline
from pipeline.pose.camera import Camera

logging.basicConfig(level=logging.WARNING)  # Suppress logs during benchmark
log = logging.getLogger("bench")

def benchmark_pipeline(cfg, num_iterations=100):
    """
    Benchmark inference pipeline latency.
    
    Args:
        cfg: Configuration dictionary
        num_iterations: Number of iterations to run
    
    Returns:
        dict with latency statistics
    """
    log.info("Initializing pipeline for benchmark...")
    pipeline = InferencePipeline(cfg)
    
    latencies = []
    errors = 0
    
    log.info(f"Running {num_iterations} iterations...")
    
    for i in range(num_iterations):
        try:
            start = time.time()
            result = pipeline.run_once()
            latency_ms = (time.time() - start) * 1000.0
            latencies.append(latency_ms)
            
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1}/{num_iterations}: {latency_ms:.1f}ms", end='\r')
        except Exception as e:
            errors += 1
            log.warning(f"Iteration {i+1} failed: {e}")
    
    print()  # New line
    
    if not latencies:
        return None
    
    latencies = np.array(latencies)
    
    stats = {
        "iterations": num_iterations,
        "errors": errors,
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "mean": np.mean(latencies),
        "std": np.std(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies),
    }
    
    return stats

def main():
    print("=" * 60)
    print("🏥 Enhanced Activity Monitor - Inference Benchmark")
    print("=" * 60)
    
    # Load config
    try:
        cfg = yaml.safe_load(open("config/system.yaml"))
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1
    
    print(f"\nConfiguration:")
    print(f"  Device ID: {cfg['device_id']}")
    print(f"  Camera: {cfg['camera_idx']} @ {cfg['camera_resolution']}")
    print(f"  Window Size: {cfg['window_size']}")
    print(f"  EdgeTPU: {cfg.get('use_edgetpu', False)}")
    
    # Test camera availability
    print("\n📷 Testing camera...")
    try:
        camera = Camera(
            index=cfg.get("camera_idx", 0),
            resolution=tuple(cfg.get("camera_resolution", [1280, 720])),
            fps=cfg.get("camera_fps", 15)
        )
        test_frame = camera.read()
        camera.release()
        print(f"  ✅ Camera OK ({test_frame.shape[1]}x{test_frame.shape[0]})")
    except Exception as e:
        print(f"  ❌ Camera test failed: {e}")
        print("  ⚠️  Continuing with benchmark (will fail on actual inference)")
    
    # Run benchmark
    print("\n⏱️  Running benchmark (100 iterations)...")
    stats = benchmark_pipeline(cfg, num_iterations=100)
    
    if stats is None:
        print("❌ Benchmark failed - no successful iterations")
        return 1
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 Benchmark Results")
    print("=" * 60)
    print(f"Iterations: {stats['iterations']}")
    print(f"Errors: {stats['errors']}")
    print(f"\nLatency Statistics (ms):")
    print(f"  Mean:   {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"  P50:    {stats['p50']:.2f}")
    print(f"  P95:    {stats['p95']:.2f}")
    print(f"  P99:    {stats['p99']:.2f}")
    print(f"  Min:    {stats['min']:.2f}")
    print(f"  Max:    {stats['max']:.2f}")
    
    # Performance assessment
    print("\n📈 Performance Assessment:")
    if stats['p95'] < 150:
        print("  ✅ Excellent: P95 < 150ms (target met)")
    elif stats['p95'] < 200:
        print("  ✅ Good: P95 < 200ms (acceptable)")
    else:
        print("  ⚠️  Warning: P95 >= 200ms (consider optimization)")
    
    # Save to CSV
    import csv
    csv_path = "benchmark_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value (ms)'])
        writer.writerow(['Mean', f"{stats['mean']:.2f}"])
        writer.writerow(['Std', f"{stats['std']:.2f}"])
        writer.writerow(['P50', f"{stats['p50']:.2f}"])
        writer.writerow(['P95', f"{stats['p95']:.2f}"])
        writer.writerow(['P99', f"{stats['p99']:.2f}"])
        writer.writerow(['Min', f"{stats['min']:.2f}"])
        writer.writerow(['Max', f"{stats['max']:.2f}"])
    
    print(f"\n💾 Results saved to: {csv_path}")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

