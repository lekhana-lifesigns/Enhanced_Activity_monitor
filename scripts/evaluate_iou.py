#!/usr/bin/env python3
"""
IoU Evaluation Script for Posture and Face Detection
Measures Intersection over Union (IoU) for detection regions against ground truth.

Usage:
    python scripts/evaluate_iou.py --ground_truth annotations.json --results results.json
"""

import json
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2] or [x, y, w, h]
        bbox2: [x1, y1, x2, y2] or [x, y, w, h]
    
    Returns:
        IoU value (0.0 to 1.0)
    """
    # Normalize to [x1, y1, x2, y2] format
    if len(bbox1) == 4 and len(bbox2) == 4:
        # Check if format is [x, y, w, h] or [x1, y1, x2, y2]
        if bbox1[2] < bbox1[0] or bbox1[3] < bbox1[1]:
            # Assume [x, y, w, h]
            x1_1, y1_1, w1, h1 = bbox1
            x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        else:
            # Assume [x1, y1, x2, y2]
            x1_1, y1_1, x2_1, y2_1 = bbox1
        
        if bbox2[2] < bbox2[0] or bbox2[3] < bbox2[1]:
            # Assume [x, y, w, h]
            x1_2, y1_2, w2, h2 = bbox2
            x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        else:
            # Assume [x1, y1, x2, y2]
            x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    return 0.0


def extract_posture_bbox(kps: List[Dict], frame_shape: Tuple[int, int]) -> Optional[List[float]]:
    """
    Extract bounding box for posture region from keypoints.
    
    Args:
        kps: List of keypoint dictionaries with 'x', 'y', 'confidence'
        frame_shape: (height, width) of frame
    
    Returns:
        [x1, y1, x2, y2] bounding box or None
    """
    if not kps:
        return None
    
    valid_kps = [kp for kp in kps if kp.get('confidence', 0) > 0.3]
    if not valid_kps:
        return None
    
    xs = [kp['x'] for kp in valid_kps]
    ys = [kp['y'] for kp in valid_kps]
    
    x1 = max(0, min(xs) - 20)  # Add padding
    y1 = max(0, min(ys) - 20)
    x2 = min(frame_shape[1], max(xs) + 20)
    y2 = min(frame_shape[0], max(ys) + 20)
    
    return [x1, y1, x2, y2]


def evaluate_detection_iou(
    ground_truth: Dict,
    predictions: Dict,
    iou_threshold: float = 0.5
) -> Dict:
    """
    Evaluate IoU for posture and face detection regions.
    
    Args:
        ground_truth: Dictionary with frame-level ground truth annotations
        predictions: Dictionary with frame-level predictions
        iou_threshold: Minimum IoU threshold for success (default: 0.5)
    
    Returns:
        Dictionary with evaluation metrics
    """
    posture_ious = []
    face_ious = []
    posture_matches = 0
    face_matches = 0
    total_frames = 0
    
    # Process each frame
    for frame_id in ground_truth.keys():
        if frame_id not in predictions:
            continue
        
        total_frames += 1
        gt = ground_truth[frame_id]
        pred = predictions[frame_id]
        
        # Evaluate posture detection
        if 'posture_bbox' in gt and 'bbox' in pred:
            gt_posture = gt['posture_bbox']
            pred_posture = pred['bbox']
            
            # If keypoints available, extract posture bbox from keypoints
            if 'kps' in pred and pred['kps']:
                frame_shape = gt.get('frame_shape', [720, 1280])  # Default: height, width
                pred_posture = extract_posture_bbox(pred['kps'], tuple(frame_shape))
            
            if pred_posture:
                iou = calculate_iou(gt_posture, pred_posture)
                posture_ious.append(iou)
                if iou >= iou_threshold:
                    posture_matches += 1
        
        # Evaluate face detection
        if 'face_bbox' in gt and 'face_bbox' in pred:
            gt_face = gt['face_bbox']
            pred_face = pred['face_bbox']
            
            if pred_face:
                iou = calculate_iou(gt_face, pred_face)
                face_ious.append(iou)
                if iou >= iou_threshold:
                    face_matches += 1
    
    # Calculate metrics
    results = {
        'total_frames': total_frames,
        'posture_detection': {
            'total_evaluated': len(posture_ious),
            'matches_above_threshold': posture_matches,
            'match_rate': posture_matches / len(posture_ious) if posture_ious else 0.0,
            'mean_iou': np.mean(posture_ious) if posture_ious else 0.0,
            'median_iou': np.median(posture_ious) if posture_ious else 0.0,
            'min_iou': np.min(posture_ious) if posture_ious else 0.0,
            'max_iou': np.max(posture_ious) if posture_ious else 0.0,
            'std_iou': np.std(posture_ious) if posture_ious else 0.0,
            'meets_threshold': np.mean(posture_ious) >= iou_threshold if posture_ious else False
        },
        'face_detection': {
            'total_evaluated': len(face_ious),
            'matches_above_threshold': face_matches,
            'match_rate': face_matches / len(face_ious) if face_ious else 0.0,
            'mean_iou': np.mean(face_ious) if face_ious else 0.0,
            'median_iou': np.median(face_ious) if face_ious else 0.0,
            'min_iou': np.min(face_ious) if face_ious else 0.0,
            'max_iou': np.max(face_ious) if face_ious else 0.0,
            'std_iou': np.std(face_ious) if face_ious else 0.0,
            'meets_threshold': np.mean(face_ious) >= iou_threshold if face_ious else False
        },
        'iou_threshold': iou_threshold,
        'overall_meets_goal': (
            (np.mean(posture_ious) >= iou_threshold if posture_ious else False) and
            (np.mean(face_ious) >= iou_threshold if face_ious else False)
        )
    }
    
    return results


def load_ground_truth(file_path: str) -> Dict:
    """Load ground truth annotations from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_predictions(file_path: str) -> Dict:
    """Load prediction results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def print_results(results: Dict):
    """Print evaluation results in a readable format."""
    print("\n" + "="*70)
    print("IoU EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nTotal Frames Evaluated: {results['total_frames']}")
    print(f"IoU Threshold: {results['iou_threshold']}")
    
    print("\n" + "-"*70)
    print("POSTURE DETECTION")
    print("-"*70)
    pd = results['posture_detection']
    print(f"  Frames Evaluated: {pd['total_evaluated']}")
    print(f"  Mean IoU: {pd['mean_iou']:.4f}")
    print(f"  Median IoU: {pd['median_iou']:.4f}")
    print(f"  Std IoU: {pd['std_iou']:.4f}")
    print(f"  Min IoU: {pd['min_iou']:.4f}")
    print(f"  Max IoU: {pd['max_iou']:.4f}")
    print(f"  Matches ≥ {results['iou_threshold']}: {pd['matches_above_threshold']}/{pd['total_evaluated']} ({pd['match_rate']*100:.2f}%)")
    print(f"  Meets Goal (IoU ≥ {results['iou_threshold']}): {'✅ YES' if pd['meets_threshold'] else '❌ NO'}")
    
    print("\n" + "-"*70)
    print("FACE DETECTION")
    print("-"*70)
    fd = results['face_detection']
    print(f"  Frames Evaluated: {fd['total_evaluated']}")
    print(f"  Mean IoU: {fd['mean_iou']:.4f}")
    print(f"  Median IoU: {fd['median_iou']:.4f}")
    print(f"  Std IoU: {fd['std_iou']:.4f}")
    print(f"  Min IoU: {fd['min_iou']:.4f}")
    print(f"  Max IoU: {fd['max_iou']:.4f}")
    print(f"  Matches ≥ {results['iou_threshold']}: {fd['matches_above_threshold']}/{fd['total_evaluated']} ({fd['match_rate']*100:.2f}%)")
    print(f"  Meets Goal (IoU ≥ {results['iou_threshold']}): {'✅ YES' if fd['meets_threshold'] else '❌ NO'}")
    
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)
    if results['overall_meets_goal']:
        print("✅ GOAL MET: Both posture and face detection meet IoU ≥ 0.5")
    else:
        print("❌ GOAL NOT MET: One or both detections below IoU ≥ 0.5")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate IoU for posture and face detection')
    parser.add_argument('--ground_truth', type=str, required=True,
                       help='Path to ground truth annotations JSON file')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to prediction results JSON file')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for success (default: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save evaluation results JSON (optional)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading ground truth from: {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth)
    
    print(f"Loading predictions from: {args.results}")
    predictions = load_predictions(args.results)
    
    # Evaluate
    print(f"\nEvaluating IoU with threshold: {args.iou_threshold}")
    results = evaluate_detection_iou(ground_truth, predictions, args.iou_threshold)
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    # Return exit code based on goal achievement
    return 0 if results['overall_meets_goal'] else 1


if __name__ == '__main__':
    sys.exit(main())

