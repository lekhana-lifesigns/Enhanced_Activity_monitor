#!/usr/bin/env python3
"""
ID-Switch Rate Evaluation Script
Measures identity tracking consistency and calculates ID-switch rate.

Usage:
    python scripts/evaluate_id_switch.py --ground_truth tracking_gt.json --results tracking_results.json
"""

import json
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_id_switch_rate(
    ground_truth: Dict,
    predictions: Dict,
    max_switch_distance: float = 50.0
) -> Dict:
    """
    Calculate ID-switch rate for identity tracking.
    
    ID-switch occurs when:
    - The same ground truth person is assigned different track IDs across frames
    - A track ID switches from one person to another
    
    Args:
        ground_truth: Dictionary with frame-level ground truth track IDs
        predictions: Dictionary with frame-level predicted track IDs
        max_switch_distance: Maximum distance (pixels) to consider same person
    
    Returns:
        Dictionary with ID-switch metrics
    """
    # Build ground truth track sequences
    gt_tracks = defaultdict(list)  # person_id -> [(frame_id, bbox, track_id)]
    pred_tracks = defaultdict(list)  # track_id -> [(frame_id, bbox, person_id)]
    
    # Process ground truth
    for frame_id, gt_data in ground_truth.items():
        if 'persons' in gt_data:
            for person in gt_data['persons']:
                person_id = person.get('person_id')
                track_id = person.get('track_id', person_id)
                bbox = person.get('bbox', [])
                if person_id is not None:
                    gt_tracks[person_id].append((frame_id, bbox, track_id))
    
    # Process predictions
    for frame_id, pred_data in predictions.items():
        track_id = pred_data.get('track_id')
        bbox = pred_data.get('bbox', [])
        if track_id is not None and track_id >= 0:
            pred_tracks[track_id].append((frame_id, bbox, None))
    
    # Match predictions to ground truth using IoU
    def calculate_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
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
    
    # Match tracks frame by frame
    frame_matches = {}  # frame_id -> {pred_track_id: gt_person_id}
    id_switches = 0
    total_track_frames = 0
    
    # Get all frame IDs
    all_frames = sorted(set(list(ground_truth.keys()) + list(predictions.keys())))
    
    prev_matches = {}  # track_id -> person_id from previous frame
    
    for frame_id in all_frames:
        if frame_id not in predictions:
            continue
        
        pred_data = predictions[frame_id]
        pred_track_id = pred_data.get('track_id')
        pred_bbox = pred_data.get('bbox', [])
        
        if pred_track_id is None or pred_track_id < 0:
            continue
        
        total_track_frames += 1
        
        # Find best matching ground truth person
        best_match = None
        best_iou = 0.0
        
        if frame_id in ground_truth:
            gt_data = ground_truth[frame_id]
            if 'persons' in gt_data:
                for person in gt_data['persons']:
                    gt_bbox = person.get('bbox', [])
                    if gt_bbox:
                        iou = calculate_bbox_iou(pred_bbox, gt_bbox)
                        if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                            best_iou = iou
                            best_match = person.get('person_id')
        
        # Check for ID switch
        if pred_track_id in prev_matches:
            prev_person_id = prev_matches[pred_track_id]
            if best_match is not None and prev_person_id != best_match:
                # ID switch detected
                id_switches += 1
        
        # Update previous matches
        if best_match is not None:
            prev_matches[pred_track_id] = best_match
        elif pred_track_id in prev_matches:
            # Track lost, remove from previous matches
            del prev_matches[pred_track_id]
    
    # Calculate metrics
    num_gt_tracks = len(gt_tracks)
    num_pred_tracks = len(pred_tracks)
    
    # ID-switch rate: number of switches / total number of ground truth tracks
    id_switch_rate = (id_switches / num_gt_tracks * 100.0) if num_gt_tracks > 0 else 0.0
    
    # Alternative: switches per frame
    switches_per_frame = id_switches / len(all_frames) if all_frames else 0.0
    
    results = {
        'total_frames': len(all_frames),
        'ground_truth_tracks': num_gt_tracks,
        'predicted_tracks': num_pred_tracks,
        'total_track_frames': total_track_frames,
        'id_switches': id_switches,
        'id_switch_rate_percent': id_switch_rate,
        'switches_per_frame': switches_per_frame,
        'target_rate_percent': 5.0,
        'meets_goal': id_switch_rate <= 5.0,
        'track_consistency': {
            'frames_with_consistent_id': total_track_frames - id_switches,
            'consistency_rate': (total_track_frames - id_switches) / total_track_frames if total_track_frames > 0 else 0.0
        }
    }
    
    return results


def load_ground_truth(file_path: str) -> Dict:
    """Load ground truth tracking annotations."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_predictions(file_path: str) -> Dict:
    """Load prediction tracking results."""
    with open(file_path, 'r') as f:
        return json.load(f)


def print_results(results: Dict):
    """Print evaluation results."""
    print("\n" + "="*70)
    print("ID-SWITCH RATE EVALUATION")
    print("="*70)
    
    print(f"\nTotal Frames: {results['total_frames']}")
    print(f"Ground Truth Tracks: {results['ground_truth_tracks']}")
    print(f"Predicted Tracks: {results['predicted_tracks']}")
    print(f"Total Track Frames: {results['total_track_frames']}")
    
    print("\n" + "-"*70)
    print("ID-SWITCH METRICS")
    print("-"*70)
    print(f"  Total ID Switches: {results['id_switches']}")
    print(f"  ID-Switch Rate: {results['id_switch_rate_percent']:.2f}%")
    print(f"  Target Rate: ≤ {results['target_rate_percent']:.1f}%")
    print(f"  Switches per Frame: {results['switches_per_frame']:.4f}")
    
    print("\n" + "-"*70)
    print("TRACK CONSISTENCY")
    print("-"*70)
    tc = results['track_consistency']
    print(f"  Frames with Consistent ID: {tc['frames_with_consistent_id']}")
    print(f"  Consistency Rate: {tc['consistency_rate']*100:.2f}%")
    
    print("\n" + "="*70)
    print("GOAL ASSESSMENT")
    print("="*70)
    if results['meets_goal']:
        print(f"✅ GOAL MET: ID-switch rate ({results['id_switch_rate_percent']:.2f}%) ≤ 5%")
    else:
        print(f"❌ GOAL NOT MET: ID-switch rate ({results['id_switch_rate_percent']:.2f}%) > 5%")
        print(f"   Need to reduce by {results['id_switch_rate_percent'] - 5.0:.2f}%")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ID-switch rate for tracking')
    parser.add_argument('--ground_truth', type=str, required=True,
                       help='Path to ground truth tracking JSON file')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to prediction tracking JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save evaluation results JSON (optional)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading ground truth from: {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth)
    
    print(f"Loading predictions from: {args.results}")
    predictions = load_predictions(args.results)
    
    # Evaluate
    print("\nCalculating ID-switch rate...")
    results = calculate_id_switch_rate(ground_truth, predictions)
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    # Return exit code based on goal achievement
    return 0 if results['meets_goal'] else 1


if __name__ == '__main__':
    sys.exit(main())

