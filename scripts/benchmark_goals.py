#!/usr/bin/env python3
"""
Comprehensive Benchmarking Framework
Measures all three goals: IoU ≥ 0.5, ID-switch rate ≤ 5%, FPS ≥ 15-20
"""

import yaml
import json
import time
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pose.inference_pipeline import InferencePipeline
from pipeline.metrics.performance_monitor import PerformanceMonitor
from scripts.evaluate_iou import evaluate_detection_iou, load_ground_truth as load_iou_gt, load_predictions as load_iou_pred
from scripts.evaluate_id_switch import calculate_id_switch_rate, load_ground_truth as load_tracking_gt, load_predictions as load_tracking_pred


def benchmark_fps(cfg: Dict, num_frames: int = 300, enable_display: bool = False) -> Dict:
    """
    Benchmark FPS performance.
    
    Args:
        cfg: Configuration dictionary
        num_frames: Number of frames to process
        enable_display: Whether to enable display (affects FPS)
    
    Returns:
        Dictionary with FPS metrics
    """
    print(f"\n{'='*70}")
    print("FPS PERFORMANCE BENCHMARK")
    print(f"{'='*70}")
    
    # Disable display for accurate FPS measurement
    cfg_bench = cfg.copy()
    cfg_bench['enable_display'] = enable_display
    
    try:
        pipeline = InferencePipeline(cfg_bench)
        monitor = PerformanceMonitor(window_size=num_frames, fps_target=15.0)
        
        print(f"Processing {num_frames} frames...")
        start_time = time.time()
        
        for i in range(num_frames):
            frame_start = time.time()
            result = pipeline.run_once()
            frame_time = (time.time() - frame_start) * 1000.0
            
            if result:
                fps = result.get('fps', 0.0)
                inference_ms = result.get('inference_ms', 0.0)
                track_id = result.get('track_id')
                bbox = result.get('bbox')
                confidence = result.get('confidence', 0.0)
                
                monitor.record_frame(
                    inference_ms=inference_ms,
                    fps=fps,
                    detection_confidence=confidence,
                    track_id=track_id,
                    bbox=bbox,
                    frame_id=i
                )
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                current_fps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Processed {i+1}/{num_frames} frames ({current_fps:.2f} FPS)", end='\r')
        
        print()  # New line
        
        total_time = time.time() - start_time
        actual_fps = num_frames / total_time if total_time > 0 else 0
        
        stats = monitor.get_stats()
        summary = monitor.get_summary()
        
        results = {
            'total_frames': num_frames,
            'total_time_seconds': total_time,
            'actual_fps': actual_fps,
            'fps_stats': summary['fps'],
            'latency_stats': summary['latency'],
            'meets_target': stats.fps_meets_target,
            'target_fps': 15.0,
            'min_target_fps': 15.0,
            'max_target_fps': 20.0
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_all_goals(
    cfg: Dict,
    num_frames: int = 300,
    iou_ground_truth: Optional[str] = None,
    iou_predictions: Optional[str] = None,
    tracking_ground_truth: Optional[str] = None,
    tracking_predictions: Optional[str] = None,
    enable_display: bool = False
) -> Dict:
    """
    Benchmark all three goals comprehensively.
    
    Returns:
        Dictionary with all goal assessments
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE GOALS BENCHMARK")
    print("="*70)
    
    results = {
        'timestamp': time.time(),
        'configuration': {
            'num_frames': num_frames,
            'enable_display': enable_display
        },
        'goals': {}
    }
    
    # Goal 1: FPS Performance
    print("\n📊 Goal 1: FPS Performance (≥ 15-20 FPS)")
    fps_results = benchmark_fps(cfg, num_frames, enable_display)
    if fps_results:
        results['goals']['fps'] = {
            'status': '✅ MET' if fps_results['meets_target'] else '❌ NOT MET',
            'actual_fps': fps_results['actual_fps'],
            'mean_fps': fps_results['fps_stats']['mean'],
            'min_fps': fps_results['fps_stats']['min'],
            'target_range': f"{fps_results['min_target_fps']}-{fps_results['max_target_fps']} FPS",
            'meets_target': fps_results['meets_target'],
            'details': fps_results
        }
    
    # Goal 2: IoU ≥ 0.5
    print("\n📊 Goal 2: Detection IoU (≥ 0.5)")
    if iou_ground_truth and iou_predictions:
        try:
            gt = load_iou_gt(iou_ground_truth)
            pred = load_iou_pred(iou_predictions)
            iou_results = evaluate_detection_iou(gt, pred, iou_threshold=0.5)
            
            results['goals']['iou'] = {
                'status': '✅ MET' if iou_results['overall_meets_goal'] else '❌ NOT MET',
                'posture_iou_mean': iou_results['posture_detection']['mean_iou'],
                'face_iou_mean': iou_results['face_detection']['mean_iou'],
                'posture_meets_goal': iou_results['posture_detection']['meets_threshold'],
                'face_meets_goal': iou_results['face_detection']['meets_threshold'],
                'overall_meets_goal': iou_results['overall_meets_goal'],
                'details': iou_results
            }
        except Exception as e:
            print(f"⚠️  IoU evaluation failed: {e}")
            results['goals']['iou'] = {
                'status': '⚠️  NOT EVALUATED',
                'error': str(e)
            }
    else:
        print("⚠️  IoU evaluation skipped (no ground truth provided)")
        results['goals']['iou'] = {
            'status': '⚠️  NOT EVALUATED',
            'reason': 'No ground truth data provided'
        }
    
    # Goal 3: ID-switch rate ≤ 5%
    print("\n📊 Goal 3: ID-Switch Rate (≤ 5%)")
    if tracking_ground_truth and tracking_predictions:
        try:
            gt = load_tracking_gt(tracking_ground_truth)
            pred = load_tracking_pred(tracking_predictions)
            tracking_results = calculate_id_switch_rate(gt, pred)
            
            results['goals']['id_switch'] = {
                'status': '✅ MET' if tracking_results['meets_goal'] else '❌ NOT MET',
                'id_switch_rate_percent': tracking_results['id_switch_rate_percent'],
                'id_switches': tracking_results['id_switches'],
                'meets_goal': tracking_results['meets_goal'],
                'details': tracking_results
            }
        except Exception as e:
            print(f"⚠️  ID-switch evaluation failed: {e}")
            results['goals']['id_switch'] = {
                'status': '⚠️  NOT EVALUATED',
                'error': str(e)
            }
    else:
        print("⚠️  ID-switch evaluation skipped (no ground truth provided)")
        results['goals']['id_switch'] = {
            'status': '⚠️  NOT EVALUATED',
            'reason': 'No ground truth data provided'
        }
    
    # Overall assessment
    all_met = all(
        goal.get('meets_goal', False) if 'meets_goal' in goal else False
        for goal in results['goals'].values()
        if 'meets_goal' in goal
    )
    
    results['overall'] = {
        'all_goals_met': all_met,
        'goals_evaluated': len([g for g in results['goals'].values() if 'meets_goal' in g]),
        'goals_met': len([g for g in results['goals'].values() if g.get('meets_goal', False)])
    }
    
    return results


def print_benchmark_results(results: Dict):
    """Print comprehensive benchmark results."""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    
    for goal_name, goal_data in results['goals'].items():
        print(f"\n{goal_name.upper().replace('_', ' ')}:")
        print(f"  Status: {goal_data.get('status', 'UNKNOWN')}")
        if 'meets_goal' in goal_data:
            print(f"  Meets Goal: {'✅ YES' if goal_data['meets_goal'] else '❌ NO'}")
    
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)
    overall = results['overall']
    print(f"  Goals Evaluated: {overall['goals_evaluated']}")
    print(f"  Goals Met: {overall['goals_met']}")
    if overall['all_goals_met']:
        print("  ✅ ALL GOALS MET")
    else:
        print("  ⚠️  SOME GOALS NOT MET")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive goals benchmarking')
    parser.add_argument('--config', type=str, default='config/system.yaml',
                       help='Path to system configuration file')
    parser.add_argument('--frames', type=int, default=300,
                       help='Number of frames to process for FPS benchmark')
    parser.add_argument('--iou_gt', type=str, default=None,
                       help='Path to IoU ground truth JSON file')
    parser.add_argument('--iou_pred', type=str, default=None,
                       help='Path to IoU predictions JSON file')
    parser.add_argument('--tracking_gt', type=str, default=None,
                       help='Path to tracking ground truth JSON file')
    parser.add_argument('--tracking_pred', type=str, default=None,
                       help='Path to tracking predictions JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save benchmark results JSON')
    parser.add_argument('--display', action='store_true',
                       help='Enable display during benchmark (may affect FPS)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Run comprehensive benchmark
    results = benchmark_all_goals(
        cfg=cfg,
        num_frames=args.frames,
        iou_ground_truth=args.iou_gt,
        iou_predictions=args.iou_pred,
        tracking_ground_truth=args.tracking_gt,
        tracking_predictions=args.tracking_pred,
        enable_display=args.display
    )
    
    # Print results
    print_benchmark_results(results)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    # Return exit code
    return 0 if results['overall']['all_goals_met'] else 1


if __name__ == '__main__':
    sys.exit(main())

