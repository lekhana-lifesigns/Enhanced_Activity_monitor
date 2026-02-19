#!/usr/bin/env python3
"""
EAM Video Processor - Complete Workflow
Integrates validation, keyframe extraction, and file naming for EAM dataset.
"""

import sys
from pathlib import Path
from typing import Optional, Dict
import logging
import json

from eam_video_validator import EAMVideoValidator, validate_eam_video
from eam_keyframe_extractor import EAMKeyFrameExtractor
from eam_file_naming import EAMFileNaming

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eam_processor")


class EAMVideoProcessor:
 """
 Complete EAM video processing workflow.
 """
 
 def __init__(self):
 """Initialize processor."""
 self.validator = EAMVideoValidator()
 self.extractor = EAMKeyFrameExtractor()
 self.naming = EAMFileNaming()
 
 def process_video(self,
 video_path: str,
 activity: str,
 video_number: int,
 distance: str,
 date: Optional[str] = None,
 output_dir: Optional[str] = None,
 extract_keyframes: bool = True,
 rename_file: bool = True) -> Dict:
 """
 Complete video processing workflow.
 
 Args:
 video_path: Path to video file
 activity: Activity name
 video_number: Video number
 distance: Distance (e.g., "2.8m")
 date: Date (YYYYMMDD), uses today if None
 output_dir: Output directory for keyframes
 extract_keyframes: Whether to extract keyframes
 rename_file: Whether to rename file to EAM convention
 
 Returns:
 Processing results dictionary
 """
 video_path = Path(video_path)
 
 log.info(f"\n{'='*60}")
 log.info(f"Processing video: {video_path.name}")
 log.info(f"{'='*60}")
 
 # Step 1: Validate video
 log.info("\n Step 1: Validating video...")
 validation_result = self.validator.validate_video(str(video_path))
 self.validator.print_validation_report(validation_result)
 
 if not validation_result.passed:
 log.warning(" Video validation failed. Continuing anyway...")
 
 # Step 2: Rename file (if requested)
 new_video_path = video_path
 if rename_file:
 log.info("\n Step 2: Renaming file to EAM convention...")
 try:
 new_video_path = self.naming.rename_file(
 str(video_path),
 activity,
 video_number,
 distance,
 date
 )
 log.info(f" File renamed: {new_video_path.name}")
 except Exception as e:
 log.error(f" Failed to rename file: {e}")
 
 # Step 3: Extract keyframes (if requested)
 keyframe_paths = []
 if extract_keyframes:
 log.info("\n Step 3: Extracting keyframes...")
 try:
 from datetime import datetime
 keyframe_paths = self.extractor.extract_keyframes_with_metadata(
 video_path=str(new_video_path),
 video_number=video_number,
 distance=distance,
 date=date or datetime.now().strftime("%Y%m%d"),
 output_dir=output_dir
 )
 log.info(f" Extracted {len(keyframe_paths)} keyframes")
 except Exception as e:
 log.error(f" Failed to extract keyframes: {e}")
 
 # Compile results
 result = {
 "video_path": str(new_video_path),
 "original_path": str(video_path),
 "validation": validation_result.to_dict(),
 "keyframes": keyframe_paths,
 "metadata": {
 "activity": activity,
 "video_number": video_number,
 "distance": distance,
 "date": date
 }
 }
 
 log.info(f"\n Processing complete!")
 log.info(f"Video: {new_video_path.name}")
 log.info(f"Keyframes: {len(keyframe_paths)}")
 log.info(f"Grade: {validation_result.grade}")
 
 return result
 
 def process_batch(self,
 video_dir: str,
 activity_mapping: Dict[str, Dict],
 output_dir: Optional[str] = None) -> Dict:
 """
 Process multiple videos in batch.
 
 Args:
 video_dir: Directory containing videos
 activity_mapping: Dict mapping video filenames to {
 "activity": str,
 "video_number": int,
 "distance": str,
 "date": Optional[str]
 }
 output_dir: Output directory
 
 Returns:
 Batch processing results
 """
 video_dir = Path(video_dir)
 results = []
 
 # Find all video files
 video_files = list(video_dir.glob("*.mp4")) + \
 list(video_dir.glob("*.avi")) + \
 list(video_dir.glob("*.mov"))
 
 log.info(f"Found {len(video_files)} videos to process")
 
 for video_file in video_files:
 # Get metadata from mapping or use defaults
 metadata = activity_mapping.get(
 video_file.name,
 {
 "activity": video_file.stem,
 "video_number": 1,
 "distance": "unknown",
 "date": None
 }
 )
 
 try:
 result = self.process_video(
 video_path=str(video_file),
 activity=metadata["activity"],
 video_number=metadata["video_number"],
 distance=metadata["distance"],
 date=metadata.get("date"),
 output_dir=output_dir
 )
 results.append(result)
 except Exception as e:
 log.error(f"Failed to process {video_file.name}: {e}")
 results.append({
 "video": str(video_file),
 "error": str(e)
 })
 
 # Save batch summary
 summary = {
 "total_videos": len(video_files),
 "successful": sum(1 for r in results if "error" not in r),
 "failed": sum(1 for r in results if "error" in r),
 "results": results
 }
 
 if output_dir:
 output_dir = Path(output_dir)
 output_dir.mkdir(parents=True, exist_ok=True)
 summary_path = output_dir / "batch_processing_summary.json"
 with open(summary_path, 'w') as f:
 json.dump(summary, f, indent=2, default=str)
 log.info(f"\n Batch summary saved to: {summary_path}")
 
 return summary


if __name__ == "__main__":
 import argparse
 
 parser = argparse.ArgumentParser(description="EAM Video Processor - Complete Workflow")
 parser.add_argument("--video", type=str, help="Path to video file")
 parser.add_argument("--video_dir", type=str, help="Directory containing videos (batch mode)")
 parser.add_argument("--activity", type=str, help="Activity name")
 parser.add_argument("--video_num", type=int, default=1, help="Video number")
 parser.add_argument("--distance", type=str, help="Distance (e.g., '2.8m')")
 parser.add_argument("--date", type=str, help="Date (YYYYMMDD)")
 parser.add_argument("--output_dir", type=str, help="Output directory")
 parser.add_argument("--no-keyframes", action="store_true", help="Skip keyframe extraction")
 parser.add_argument("--no-rename", action="store_true", help="Skip file renaming")
 
 args = parser.parse_args()
 
 processor = EAMVideoProcessor()
 
 if args.video_dir:
 # Batch mode - requires activity mapping file or manual specification
 log.warning("Batch mode requires activity mapping. Using defaults.")
 activity_mapping = {}
 summary = processor.process_batch(
 video_dir=args.video_dir,
 activity_mapping=activity_mapping,
 output_dir=args.output_dir
 )
 print(f"\n Batch processing complete: {summary['successful']}/{summary['total_videos']} successful")
 elif args.video:
 # Single video mode
 if not args.activity or not args.distance:
 parser.error("--activity and --distance are required for single video mode")
 
 result = processor.process_video(
 video_path=args.video,
 activity=args.activity,
 video_number=args.video_num,
 distance=args.distance,
 date=args.date,
 output_dir=args.output_dir,
 extract_keyframes=not args.no_keyframes,
 rename_file=not args.no_rename
 )
 
 print(f"\n Processing complete!")
 print(f"Grade: {result['validation']['grade']}")
 print(f"Keyframes: {len(result['keyframes'])}")
 else:
 parser.print_help()
