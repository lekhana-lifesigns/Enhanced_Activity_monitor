#!/usr/bin/env python3
"""
FPS Optimization Script
Identifies bottlenecks and suggests optimizations to reach 15-20 FPS target.
"""

import yaml
import time
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pose.inference_pipeline import InferencePipeline


def profile_pipeline(cfg: Dict, num_frames: int = 100) -> Dict:
    """
    Profile the inference pipeline to identify bottlenecks.
    
    Returns:
        Dictionary with timing breakdown
    """
    print("\n" + "="*70)
    print("PIPELINE PROFILING")
    print("="*70)
    
    # Disable display for accurate profiling
    cfg_prof = cfg.copy()
    cfg_prof['enable_display'] = False
    
    try:
        pipeline = InferencePipeline(cfg_prof)
        
        timings = {
            'detection': [],
            'pose': [],
            'feature_extraction': [],
            'temporal_model': [],
            'posture_analysis': [],
            'activity_analysis': [],
            'total': []
        }
        
        print(f"Profiling {num_frames} frames...")
        
        for i in range(num_frames):
            frame_start = time.time()
            
            # Time detection
            det_start = time.time()
            frame = pipeline.camera.read()
            if frame is None:
                continue
            
            dets = pipeline.det.infer(frame)
            det_time = (time.time() - det_start) * 1000.0
            timings['detection'].append(det_time)
            
            if not dets:
                continue
            
            # Time pose estimation
            pose_start = time.time()
            # Get first detection
            det = dets[0]
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = pipeline._parse_bbox(bbox, frame.shape)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    kps = pipeline.pose.estimate(crop)
                else:
                    kps = None
            else:
                kps = None
            pose_time = (time.time() - pose_start) * 1000.0
            timings['pose'].append(pose_time)
            
            if kps is None:
                continue
            
            # Time feature extraction
            feat_start = time.time()
            if hasattr(pipeline.feature_encoder, 'extract_features'):
                feat = pipeline.feature_encoder.extract_features(
                    [kps],
                    prev_kps=pipeline.prev_kps,
                    prev_prev_kps=pipeline.prev_prev_kps
                )
            else:
                feat = pipeline.feature_encoder.extract_feature_vector(
                    kps,
                    prev_kps=pipeline.prev_kps,
                    prev_prev_kps=pipeline.prev_prev_kps
                )
            feat_time = (time.time() - feat_start) * 1000.0
            timings['feature_extraction'].append(feat_time)
            
            # Time temporal model (if enough frames)
            if len(pipeline.window) >= 8:
                temp_start = time.time()
                feat_win = np.stack(pipeline.window[-pipeline.temporal.window_size:])
                label, conf, probs = pipeline.temporal.predict(feat_win)
                temp_time = (time.time() - temp_start) * 1000.0
                timings['temporal_model'].append(temp_time)
            
            # Time posture analysis
            posture_start = time.time()
            try:
                from analytics.posture import analyze_posture, classify_posture_state
                posture_state = classify_posture_state(kps)
                posture_analysis = analyze_posture(kps)
            except:
                pass
            posture_time = (time.time() - posture_start) * 1000.0
            timings['posture_analysis'].append(posture_time)
            
            # Time activity analysis
            activity_start = time.time()
            try:
                from analytics.activity import classify_activity
                activity_state = classify_activity(kps)
            except:
                pass
            activity_time = (time.time() - activity_start) * 1000.0
            timings['activity_analysis'].append(activity_time)
            
            total_time = (time.time() - frame_start) * 1000.0
            timings['total'].append(total_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{num_frames} frames", end='\r')
        
        print()  # New line
        
        # Calculate statistics
        stats = {}
        for component, times in timings.items():
            if times:
                stats[component] = {
                    'mean_ms': np.mean(times),
                    'median_ms': np.median(times),
                    'p95_ms': np.percentile(times, 95),
                    'std_ms': np.std(times),
                    'percentage_of_total': (np.mean(times) / np.mean(timings['total']) * 100) if timings['total'] else 0.0
                }
        
        return {
            'timings': timings,
            'stats': stats,
            'total_frames': num_frames
        }
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def suggest_optimizations(profile: Dict) -> List[Dict]:
    """
    Suggest optimizations based on profiling results.
    
    Returns:
        List of optimization suggestions
    """
    suggestions = []
    stats = profile['stats']
    total_mean = stats['total']['mean_ms'] if 'total' in stats else 0
    target_fps = 15.0
    target_ms = 1000.0 / target_fps  # ~66.7ms
    
    print("\n" + "="*70)
    print("OPTIMIZATION SUGGESTIONS")
    print("="*70)
    
    if total_mean > target_ms:
        overhead = total_mean - target_ms
        print(f"\n⚠️  Current latency ({total_mean:.1f}ms) exceeds target ({target_ms:.1f}ms) by {overhead:.1f}ms")
    else:
        print(f"\n✅ Current latency ({total_mean:.1f}ms) meets target ({target_ms:.1f}ms)")
    
    # Check each component
    for component, stat in stats.items():
        if component == 'total':
            continue
        
        mean_ms = stat['mean_ms']
        percentage = stat['percentage_of_total']
        
        if mean_ms > 20.0:  # Component taking > 20ms
            suggestions.append({
                'component': component,
                'current_time_ms': mean_ms,
                'percentage': percentage,
                'priority': 'HIGH' if percentage > 30 else 'MEDIUM',
                'suggestions': get_component_suggestions(component, mean_ms)
            })
    
    # Sort by priority
    suggestions.sort(key=lambda x: x['percentage'], reverse=True)
    
    return suggestions


def get_component_suggestions(component: str, time_ms: float) -> List[str]:
    """Get optimization suggestions for a specific component."""
    suggestions = []
    
    if component == 'detection':
        if time_ms > 30:
            suggestions.append("Use lighter detection model (yolo11n instead of yolo11s/m)")
            suggestions.append("Reduce input resolution (320x320 instead of 640x640)")
            suggestions.append("Increase detection confidence threshold to reduce false positives")
        if time_ms > 50:
            suggestions.append("Consider using TFLite model with EdgeTPU acceleration")
    
    elif component == 'pose':
        if time_ms > 20:
            suggestions.append("Use lighter pose model (yolo11n-pose instead of yolo11s-pose)")
            suggestions.append("Reduce pose input size (192 instead of 256)")
        if time_ms > 40:
            suggestions.append("Consider skipping pose estimation for some frames (every 2nd frame)")
    
    elif component == 'feature_extraction':
        if time_ms > 10:
            suggestions.append("Disable learned features if using hybrid extractor")
            suggestions.append("Reduce feature window size")
    
    elif component == 'temporal_model':
        if time_ms > 15:
            suggestions.append("Use standard temporal model instead of enhanced (if using)")
            suggestions.append("Reduce temporal window size")
    
    elif component == 'posture_analysis':
        if time_ms > 10:
            suggestions.append("Simplify posture analysis (skip full analysis, use classification only)")
            suggestions.append("Run posture analysis every N frames instead of every frame")
    
    elif component == 'activity_analysis':
        if time_ms > 10:
            suggestions.append("Simplify activity analysis")
            suggestions.append("Run activity analysis every N frames instead of every frame")
    
    return suggestions


def print_profiling_results(profile: Dict, suggestions: List[Dict]):
    """Print profiling results and suggestions."""
    stats = profile['stats']
    
    print("\n" + "="*70)
    print("PROFILING RESULTS")
    print("="*70)
    
    print(f"\n{'Component':<25} {'Mean (ms)':<12} {'P95 (ms)':<12} {'% of Total':<12}")
    print("-" * 70)
    
    for component, stat in sorted(stats.items(), key=lambda x: x[1]['mean_ms'], reverse=True):
        print(f"{component:<25} {stat['mean_ms']:>10.2f}  {stat['p95_ms']:>10.2f}  {stat['percentage_of_total']:>10.1f}%")
    
    if suggestions:
        print("\n" + "="*70)
        print("OPTIMIZATION PRIORITIES")
        print("="*70)
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion['component'].upper()} ({suggestion['priority']} Priority)")
            print(f"   Current: {suggestion['current_time_ms']:.2f}ms ({suggestion['percentage']:.1f}% of total)")
            print("   Suggestions:")
            for opt in suggestion['suggestions']:
                print(f"     - {opt}")


def main():
    parser = argparse.ArgumentParser(description='Profile and optimize FPS performance')
    parser.add_argument('--config', type=str, default='config/system.yaml',
                       help='Path to system configuration file')
    parser.add_argument('--frames', type=int, default=100,
                       help='Number of frames to profile')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save profiling results JSON (optional)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Profile pipeline
    profile = profile_pipeline(cfg, args.frames)
    
    if profile is None:
        return 1
    
    # Generate suggestions
    suggestions = suggest_optimizations(profile)
    
    # Print results
    print_profiling_results(profile, suggestions)
    
    # Save results if requested
    if args.output:
        import json
        results = {
            'profile': profile,
            'suggestions': suggestions
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

