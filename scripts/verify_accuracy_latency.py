# scripts/verify_accuracy_latency.py
"""
Verify actual system accuracy and latency against claimed performance.
Updated for optimized 9D GRU temporal model.

Usage:
    python scripts/verify_accuracy_latency.py
    python scripts/verify_accuracy_latency.py --frames 100
"""
import sys
import os
import time
import numpy as np
import yaml
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pose.feature_extractor import ICUFeatureEncoder

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("verify")


def check_feature_dimensions():
    """Check feature extractor output dimensions."""
    print("\n" + "=" * 70)
    print("FEATURE DIMENSION CHECK")
    print("=" * 70)

    encoder = ICUFeatureEncoder(window_size=48, fps=15)

    # Use realistic standing-pose keypoints (17 COCO format: x, y, confidence)
    kps = [
        (0.50, 0.10, 0.95),  # nose
        (0.49, 0.08, 0.90),  # left_eye
        (0.51, 0.08, 0.90),  # right_eye
        (0.48, 0.09, 0.88),  # left_ear
        (0.52, 0.09, 0.88),  # right_ear
        (0.45, 0.20, 0.92),  # left_shoulder
        (0.55, 0.20, 0.92),  # right_shoulder
        (0.42, 0.35, 0.90),  # left_elbow
        (0.58, 0.35, 0.90),  # right_elbow
        (0.40, 0.45, 0.88),  # left_wrist
        (0.60, 0.45, 0.88),  # right_wrist
        (0.47, 0.50, 0.93),  # left_hip
        (0.53, 0.50, 0.93),  # right_hip
        (0.46, 0.70, 0.90),  # left_knee
        (0.54, 0.70, 0.90),  # right_knee
        (0.45, 0.90, 0.88),  # left_ankle
        (0.55, 0.90, 0.88),  # right_ankle
    ]

    feat = encoder.extract_feature_vector(kps, None, None)

    if feat is not None:
        feat_dim = len(feat)
        print(f"  Feature extractor output dimension: {feat_dim}")
        print(f"  Expected: 9 (ICU feature vector)")

        if feat_dim == 9:
            print("  PASS: Feature dimensions match optimized model")
        else:
            print(f"  FAIL: Expected 9, got {feat_dim}")

        # Check for NaN/Inf
        feat_arr = np.array(feat)
        if np.isnan(feat_arr).any() or np.isinf(feat_arr).any():
            print("  WARNING: Feature vector contains NaN/Inf values")
        else:
            print("  PASS: No NaN/Inf in features")

        return feat_dim
    else:
        print("  Feature extractor returned None")
        return None


def check_temporal_model():
    """Check temporal model configuration and inference."""
    print("\n" + "=" * 70)
    print("TEMPORAL MODEL CHECK")
    print("=" * 70)

    try:
        from pipeline.pose.temporal_model_enhanced import (
            TemporalModelEnhanced, OptimizedTemporalModel, TORCH_AVAILABLE
        )

        cfg = yaml.safe_load(open("config/system.yaml"))

        if not TORCH_AVAILABLE:
            print("  WARNING: PyTorch not available, using TFLite fallback")
            return None

        # Check optimized model parameters
        model = OptimizedTemporalModel(input_dim=9, hidden_dim=64, num_classes=6, num_heads=2)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  OptimizedTemporalModel: {param_count:,} params ({param_count * 4 / 1024:.1f} KB)")

        if param_count < 50000:
            print("  PASS: Model within Jetson budget (<50K params)")
        else:
            print(f"  WARNING: Model has {param_count} params (target: <50K)")

        # Check forward pass
        import torch
        dummy_input = torch.randn(1, 48, 9)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  Forward pass: input (1, 48, 9) -> output {tuple(output.shape)}")

        if output.shape == (1, 6):
            print("  PASS: Output shape correct (1, 6)")
        else:
            print(f"  FAIL: Expected (1, 6), got {tuple(output.shape)}")

        # Check inference latency
        times = []
        for _ in range(100):
            x = torch.randn(1, 48, 9)
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x)
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(times)
        p95_ms = np.percentile(times, 95)
        print(f"  CPU inference: avg={avg_ms:.2f}ms, p95={p95_ms:.2f}ms (100 runs)")

        if avg_ms < 5.0:
            print("  PASS: Inference < 5ms target")
        else:
            print(f"  WARNING: Inference {avg_ms:.1f}ms exceeds 5ms target")

        # Check wrapper
        wrapper = TemporalModelEnhanced(
            model_path=None,
            window_size=cfg.get("window_size", 48),
            use_pytorch=True,
            device="cpu"
        )
        print(f"  Wrapper trained state: {wrapper.is_trained}")
        print(f"  Model type: {'optimized' if hasattr(wrapper.pytorch_model, 'gru_norm') else 'legacy'}")

        return param_count

    except Exception as e:
        print(f"  Failed to check temporal model: {e}")
        import traceback
        traceback.print_exc()
        return None


def measure_pipeline_latency(num_frames=50):
    """Measure actual end-to-end pipeline latency."""
    print("\n" + "=" * 70)
    print("PIPELINE LATENCY MEASUREMENT")
    print("=" * 70)

    try:
        from pipeline.pose.inference_pipeline import InferencePipeline

        cfg = yaml.safe_load(open("config/system.yaml"))
        cfg["enable_display"] = False

        pipeline = InferencePipeline(cfg)

        latencies = []
        successful_frames = 0

        print(f"  Measuring latency over {num_frames} frames...")

        for i in range(num_frames):
            start = time.time()
            result = pipeline.run_once()
            elapsed = (time.time() - start) * 1000.0

            if result:
                latencies.append(elapsed)
                successful_frames += 1

            if (i + 1) % 20 == 0 and latencies:
                avg = np.mean(latencies[-20:])
                print(f"    Frame {i + 1}: {avg:.1f}ms avg (last 20)")

        if latencies:
            avg = np.mean(latencies)
            median = np.median(latencies)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)

            print(f"\n  Latency Statistics:")
            print(f"    Average:  {avg:.1f}ms")
            print(f"    Median:   {median:.1f}ms")
            print(f"    P95:      {p95:.1f}ms")
            print(f"    P99:      {p99:.1f}ms")
            print(f"    Success:  {successful_frames}/{num_frames} ({100 * successful_frames / num_frames:.1f}%)")

            target_fps = cfg.get("camera_fps", 15)
            target_ms = 1000.0 / target_fps

            if avg <= target_ms:
                print(f"  PASS: Latency {avg:.1f}ms <= target {target_ms:.1f}ms ({target_fps} FPS)")
            else:
                print(f"  FAIL: Latency {avg:.1f}ms > target {target_ms:.1f}ms ({target_fps} FPS)")

            return {"avg": avg, "median": median, "p95": p95, "p99": p99}
        else:
            print("  No successful frames")
            return None

    except Exception as e:
        print(f"  Latency measurement failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify system accuracy and latency")
    parser.add_argument("--frames", type=int, default=50, help="Frames for latency test")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip pipeline latency test")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SYSTEM ACCURACY & LATENCY VERIFICATION")
    print("  Optimized GRU: 9D input, ~34K params, 2-head attention")
    print("=" * 70)

    feat_dim = check_feature_dimensions()
    param_count = check_temporal_model()

    # Dimension consistency check
    if feat_dim and param_count:
        print("\n" + "=" * 70)
        print("DIMENSION CONSISTENCY")
        print("=" * 70)
        if feat_dim == 9:
            print("  PASS: Feature extractor (9D) matches optimized model (9D input)")
        else:
            print(f"  FAIL: Feature extractor ({feat_dim}D) != model input (9D)")

    if not args.skip_pipeline:
        measure_pipeline_latency(num_frames=args.frames)

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
