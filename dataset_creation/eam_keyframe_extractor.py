#!/usr/bin/env python3
"""
EAM Key Frame Extractor
Extracts 5 key frames from videos according to EAM requirements.
"""

import cv2
import os
from pathlib import Path
from typing import List, Optional, Tuple
import logging

log = logging.getLogger("eam_keyframes")


class EAMKeyFrameExtractor:
 """
 Extracts 5 key frames from videos at specific percentages.
 """
 
 def __init__(self, jpeg_quality: int = 90):
 """
 Initialize extractor.
 
 Args:
 jpeg_quality: JPEG quality (0-100) for saved images
 """
 self.jpeg_quality = jpeg_quality
 self.frame_positions = [0, 0.25, 0.50, 0.75, 1.0] # 0%, 25%, 50%, 75%, 100%
 
 def extract_keyframes(self, 
 video_path: str,
 output_dir: Optional[str] = None,
 base_name: Optional[str] = None) -> List[str]:
 """
 Extract 5 key frames from video.
 
 Args:
 video_path: Path to video file
 output_dir: Directory to save frames (uses video directory if None)
 base_name: Base name for output files (uses video stem if None)
 
 Returns:
 List of paths to extracted frame images
 """
 video_path = Path(video_path)
 
 if not video_path.exists():
 raise FileNotFoundError(f"Video not found: {video_path}")
 
 # Set output directory and base name
 if output_dir is None:
 output_dir = video_path.parent
 else:
 output_dir = Path(output_dir)
 output_dir.mkdir(parents=True, exist_ok=True)
 
 if base_name is None:
 base_name = video_path.stem
 
 # Open video
 cap = cv2.VideoCapture(str(video_path))
 if not cap.isOpened():
 raise ValueError(f"Cannot open video: {video_path}")
 
 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
 if total_frames == 0:
 cap.release()
 raise ValueError(f"Video has no frames: {video_path}")
 
 extracted_paths = []
 
 # Extract frames at specified positions
 for idx, position in enumerate(self.frame_positions, 1):
 # Calculate frame number
 frame_num = int(position * (total_frames - 1))
 frame_num = max(0, min(frame_num, total_frames - 1))
 
 # Seek to frame
 cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
 ret, frame = cap.read()
 
 if not ret:
 log.warning(f"Failed to read frame {frame_num} at position {position}")
 continue
 
 # Generate output filename
 # Format: {base_name}_i{video_num}_f{frame_idx}_{distance}_{date}.jpg
 # For now, using simplified format: {base_name}_f{idx}.jpg
 output_filename = f"{base_name}_f{idx}.jpg"
 output_path = output_dir / output_filename
 
 # Save frame
 success = cv2.imwrite(
 str(output_path),
 frame,
 [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
 )
 
 if success:
 extracted_paths.append(str(output_path))
 log.info(f" Extracted frame {idx}/5: {output_filename} (frame {frame_num}/{total_frames-1})")
 else:
 log.error(f" Failed to save frame {idx}: {output_path}")
 
 cap.release()
 
 log.info(f" Extracted {len(extracted_paths)}/5 key frames from {video_path.name}")
 
 return extracted_paths
 
 def extract_keyframes_with_metadata(self,
 video_path: str,
 video_number: int,
 distance: str,
 date: str,
 output_dir: Optional[str] = None) -> List[str]:
 """
 Extract keyframes with full EAM naming convention.
 
 Args:
 video_path: Path to video file
 video_number: Video number (e.g., 1, 2, 3)
 distance: Distance in meters (e.g., "2.8m", "1.8m")
 date: Date string (e.g., "20260112")
 output_dir: Output directory
 
 Returns:
 List of paths to extracted frames
 """
 video_path = Path(video_path)
 
 # Generate base name from video path
 # Extract activity name from video filename
 activity_name = video_path.stem.split("_")[0] if "_" in video_path.stem else video_path.stem
 
 # Format: {activity}_i{###}_f{idx}_{distance}_{date}.jpg
 base_name = f"{activity_name}_i{video_number:03d}"
 
 return self.extract_keyframes(
 video_path=str(video_path),
 output_dir=output_dir,
 base_name=base_name
 )


def extract_keyframes_from_video(video_path: str,
 output_dir: Optional[str] = None,
 jpeg_quality: int = 90) -> List[str]:
 """
 Convenience function to extract keyframes.
 
 Args:
 video_path: Path to video file
 output_dir: Output directory
 jpeg_quality: JPEG quality (0-100)
 
 Returns:
 List of paths to extracted frames
 """
 extractor = EAMKeyFrameExtractor(jpeg_quality=jpeg_quality)
 return extractor.extract_keyframes(video_path, output_dir)


if __name__ == "__main__":
 import argparse
 
 parser = argparse.ArgumentParser(description="Extract 5 key frames from video")
 parser.add_argument("video", type=str, help="Path to video file")
 parser.add_argument("--output_dir", type=str, help="Output directory")
 parser.add_argument("--quality", type=int, default=90, help="JPEG quality (0-100)")
 parser.add_argument("--video_num", type=int, help="Video number for naming")
 parser.add_argument("--distance", type=str, help="Distance (e.g., '2.8m')")
 parser.add_argument("--date", type=str, help="Date (e.g., '20260112')")
 
 args = parser.parse_args()
 
 extractor = EAMKeyFrameExtractor(jpeg_quality=args.quality)
 
 if args.video_num and args.distance and args.date:
 paths = extractor.extract_keyframes_with_metadata(
 video_path=args.video,
 video_number=args.video_num,
 distance=args.distance,
 date=args.date,
 output_dir=args.output_dir
 )
 else:
 paths = extractor.extract_keyframes(
 video_path=args.video,
 output_dir=args.output_dir
 )
 
 print(f"\n Extracted {len(paths)} key frames:")
 for path in paths:
 print(f" • {path}")
