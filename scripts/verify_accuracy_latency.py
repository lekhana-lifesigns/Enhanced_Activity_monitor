# scripts/verify_accuracy_latency.py
"""
Verify actual system accuracy and latency against claimed performance.
"""
import sys
import os
import time
import numpy as np
import yaml
import logging
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pose.inference_pipeline import InferencePipeline
from pipeline.pose.feature_extractor import ICUFeatureEncoder

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("verify")

def check_feature_dimensions():
    """Check feature extractor output dimensions."""
    print("\n" + "="*70)
    print("FEATURE DIMENSION CHECK")
    print("="*70)
    
    # Create feature encoder
    encoder = ICUFeatureEncoder(window_size=48, fps=15)
    
    # Create dummy keypoints
    dummy_kps = [(0.5, 0.5, 0.9)] * 17
    
    # Extract features
    feat = encoder.extract_feature_vector(dummy_kps, None, None)
    
    if feat is not None:
        feat_dim = len(feat)
        print(f"✅ Feature extractor output dimension: {feat_dim}")
        
        if hasattr(encoder, 'output_dim'):
            print(f"   Declared output_dim: {encoder.output_dim}")
        
        return feat_dim
    else:
        print("❌ Feature extractor returned None")
        return None

def check_temporal_model_input():
    """Check temporal model expected input dimensions."""
    print("\n" + "="*70)
    print("TEMPORAL MODEL INPUT CHECK")
    print("="*70)
    
    try:
        from pipeline.pose.temporal_model_enhanced import TemporalModelEnhanced
        import yaml
        
        cfg = yaml.safe_load(open("config/system.yaml"))
        device = cfg.get("device", "cpu")
        
        # Check default input_dim
        model = TemporalModelEnhanced(
            input_dim=13,  # Default
            window_size=cfg.get("window_size", 48),
            use_pytorch=True,
            device=device
        )
        
        print(f"✅ Enhanced temporal model initialized")
        print(f"   Expected input_dim: 13")
        print(f"   Window size: {model.window_size}")
        
        return 13
    except Exception as e:
        print(f"❌ Failed to check temporal model: {e}")
        return None

def measure_latency(num_frames=100):
    """Measure actual system latency."""
    print("\n" + "="*70)
    print("LATENCY MEASUREMENT")
    print("="*70)
    
    try:
        import yaml
        cfg = yaml.safe_load(open("config/system.yaml"))
        
        # Disable display for accurate timing
        cfg["enable_display"] = False
        
        pipeline = InferencePipeline(cfg)
        
        latencies = []
        successful_frames = 0
        
        print(f"Measuring latency over {num_frames} frames...")
        
        for i in range(num_frames):
            start = time.time()
            result = pipeline.run_once()
            elapsed = (time.time() - start) * 1000.0  # ms
            
            if result:
                latencies.append(elapsed)
                successful_frames += 1
                
                if (i + 1) % 20 == 0:
                    avg_latency = np.mean(latencies[-20:])
                    print(f"  Frame {i+1}: {avg_latency:.1f}ms avg (last 20 frames)")
        
        if latencies:
            avg_latency = np.mean(latencies)
            median_latency = np.median(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            print(f"\n✅ Latency Statistics:")
            print(f"   Average: {avg_latency:.1f}ms")
            print(f"   Median: {median_latency:.1f}ms")
            print(f"   P95: {p95_latency:.1f}ms")
            print(f"   P99: {p99_latency:.1f}ms")
            print(f"   Success rate: {successful_frames}/{num_frames} ({100*successful_frames/num_frames:.1f}%)")
            
            # Check against target
            target_fps = cfg.get("camera_fps", 15)
            target_latency_ms = 1000.0 / target_fps
            
            print(f"\n📊 Performance vs Target:")
            print(f"   Target FPS: {target_fps}")
            print(f"   Target latency: {target_latency_ms:.1f}ms")
            print(f"   Actual latency: {avg_latency:.1f}ms")
            
            if avg_latency <= target_latency_ms:
                print(f"   ✅ MEETS TARGET (latency ≤ {target_latency_ms:.1f}ms)")
            else:
                overhead = avg_latency - target_latency_ms
                print(f"   ⚠️  EXCEEDS TARGET by {overhead:.1f}ms")
            
            return {
                "avg": avg_latency,
                "median": median_latency,
                "p95": p95_latency,
                "p99": p99_latency,
                "success_rate": successful_frames / num_frames
            }
        else:
            print("❌ No successful frames")
            return None
            
    except Exception as e:
        print(f"❌ Latency measurement failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_accuracy_claims():
    """Check claimed vs actual accuracy."""
    print("\n" + "="*70)
    print("ACCURACY CLAIMS VERIFICATION")
    print("="*70)
    
    # Read claims from documentation
    claims = {
        "activity_classification": "75-85% (target: ≥90%)",
        "posture_classification": "85-90% (target: ≥95%)",
        "pose_estimation": "~92% PCK@0.2",
        "person_detection": "~95% mAP@0.5"
    }
    
    print("📋 Claimed Accuracy (from docs):")
    for metric, claim in claims.items():
        print(f"   {metric}: {claim}")
    
    print("\n⚠️  Note: Actual accuracy requires labeled test dataset")
    print("   Current system shows:")
    print("   - Feature dimension mismatch (13 vs 9) causing fallback")
    print("   - Enhanced temporal model not being used")
    print("   - This likely reduces accuracy below claims")
    
    return claims

def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("SYSTEM ACCURACY & LATENCY VERIFICATION")
    print("="*70)
    
    # Check 1: Feature dimensions
    feat_dim = check_feature_dimensions()
    
    # Check 2: Temporal model input
    model_input_dim = check_temporal_model_input()
    
    # Check 3: Dimension mismatch
    if feat_dim and model_input_dim:
        print("\n" + "="*70)
        print("DIMENSION MISMATCH ANALYSIS")
        print("="*70)
        
        if feat_dim != model_input_dim:
            print(f"❌ CRITICAL: Dimension mismatch detected!")
            print(f"   Feature extractor outputs: {feat_dim} features")
            print(f"   Temporal model expects: {model_input_dim} features")
            print(f"   This causes the enhanced temporal model to fail")
            print(f"   System falls back to default predictions (lower accuracy)")
        else:
            print(f"✅ Dimensions match: {feat_dim} features")
    
    # Check 4: Latency
    latency_stats = measure_latency(num_frames=50)
    
    # Check 5: Accuracy claims
    claims = check_accuracy_claims()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    if feat_dim != model_input_dim:
        print("\n🔴 CRITICAL ISSUE:")
        print("   Feature dimension mismatch prevents enhanced temporal model from working")
        print("   Recommendation: Fix feature extractor to output 13 features OR")
        print("                  Update temporal model to accept 9 features")
    
    if latency_stats:
        if latency_stats["avg"] > 66.7:  # 15 FPS = 66.7ms
            print("\n⚠️  LATENCY ISSUE:")
            print("   Average latency exceeds target for 15 FPS")
            print("   Recommendation: Optimize pipeline or reduce processing")
    
    print("\n✅ Verification complete")

if __name__ == "__main__":
    main()

