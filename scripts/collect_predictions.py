#!/usr/bin/env python3
"""
Collect Prediction Results for Evaluation
Runs the inference pipeline and collects predictions in the format needed for evaluation.
"""

import yaml
import json
import time
import argparse
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pose.inference_pipeline import InferencePipeline


def collect_predictions(cfg: Dict, num_frames: int = 100, output_file: str = "predictions.json") -> Dict:
    """
    Collect prediction results from the inference pipeline.
    
    Returns:
        Dictionary with frame-level predictions
    """
    print(f"\nCollecting predictions for {num_frames} frames...")
    
    # Disable display for faster collection
    cfg_collect = cfg.copy()
    cfg_collect['enable_display'] = False
    
    predictions = {}
    
    try:
        pipeline = InferencePipeline(cfg_collect)
        
        for i in range(num_frames):
            result = pipeline.run_once()
            
            if result:
                frame_id = f"frame_{i+1}"
                predictions[frame_id] = {
                    "bbox": result.get("bbox", []),
                    "track_id": result.get("track_id"),
                    "kps": result.get("kps", []),
                    "confidence": result.get("confidence", 0.0),
                    "fps": result.get("fps", 0.0),
                    "inference_ms": result.get("inference_ms", 0.0)
                }
                
                # Extract face bbox if available (would need face detection)
                # For now, we'll leave it empty or extract from keypoints
                if "face_bbox" in result:
                    predictions[frame_id]["face_bbox"] = result["face_bbox"]
            
            if (i + 1) % 20 == 0:
                print(f"  Collected {i+1}/{num_frames} frames", end='\r')
        
        print()  # New line
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"✅ Predictions saved to: {output_file}")
        print(f"   Total frames collected: {len(predictions)}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Collect prediction results for evaluation')
    parser.add_argument('--config', type=str, default='config/system.yaml',
                       help='Path to system configuration file')
    parser.add_argument('--frames', type=int, default=100,
                       help='Number of frames to collect')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Output file path for predictions')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Collect predictions
    predictions = collect_predictions(cfg, args.frames, args.output)
    
    if predictions is None:
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

