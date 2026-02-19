#!/usr/bin/env python3
"""
EAM File Naming Utility
Handles EAM dataset file naming conventions.
"""

import re
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import logging

log = logging.getLogger("eam_naming")


class EAMFileNaming:
 """
 Handles EAM dataset file naming conventions.
 
 Format:
 - Videos: {activity}_v{###}_{distance}_{date}.mp4
 - Images: {activity}_i{###}_f{idx}_{distance}_{date}.jpg
 
 Examples:
 - mask_removal_attempt_v001_2.8m_20260112.mp4
 - mask_removal_attempt_i001_f1_2.8m_20260112.jpg
 """
 
 @staticmethod
 def normalize_activity_name(activity: str) -> str:
 """
 Normalize activity name for filename.
 
 Args:
 activity: Activity name (e.g., "REMOVAL OF MASK", "mask removal attempt")
 
 Returns:
 Normalized name (e.g., "mask_removal_attempt")
 """
 # Convert to lowercase
 normalized = activity.lower()
 
 # Replace spaces and special characters with underscores
 normalized = re.sub(r'[^\w\s-]', '', normalized)
 normalized = re.sub(r'[\s_-]+', '_', normalized)
 
 # Remove leading/trailing underscores
 normalized = normalized.strip('_')
 
 return normalized
 
 @staticmethod
 def format_video_filename(activity: str,
 video_number: int,
 distance: str,
 date: Optional[str] = None,
 extension: str = ".mp4") -> str:
 """
 Format video filename according to EAM convention.
 
 Args:
 activity: Activity name
 video_number: Video number (1, 2, 3, ...)
 distance: Distance (e.g., "2.8m", "1.8m")
 date: Date string (YYYYMMDD), uses today if None
 extension: File extension (default: .mp4)
 
 Returns:
 Formatted filename
 """
 activity_norm = EAMFileNaming.normalize_activity_name(activity)
 
 if date is None:
 date = datetime.now().strftime("%Y%m%d")
 
 # Normalize distance format
 distance_norm = distance.lower().replace(" ", "").replace("m", "m")
 if not distance_norm.endswith("m"):
 distance_norm += "m"
 
 filename = f"{activity_norm}_v{video_number:03d}_{distance_norm}_{date}{extension}"
 
 return filename
 
 @staticmethod
 def format_image_filename(activity: str,
 image_number: int,
 frame_index: int,
 distance: str,
 date: Optional[str] = None,
 extension: str = ".jpg") -> str:
 """
 Format image filename according to EAM convention.
 
 Args:
 activity: Activity name
 image_number: Image/video number (1, 2, 3, ...)
 frame_index: Frame index (1-5)
 distance: Distance (e.g., "2.8m", "1.8m")
 date: Date string (YYYYMMDD), uses today if None
 extension: File extension (default: .jpg)
 
 Returns:
 Formatted filename
 """
 activity_norm = EAMFileNaming.normalize_activity_name(activity)
 
 if date is None:
 date = datetime.now().strftime("%Y%m%d")
 
 # Normalize distance format
 distance_norm = distance.lower().replace(" ", "").replace("m", "m")
 if not distance_norm.endswith("m"):
 distance_norm += "m"
 
 filename = f"{activity_norm}_i{image_number:03d}_f{frame_index}_{distance_norm}_{date}{extension}"
 
 return filename
 
 @staticmethod
 def parse_filename(filename: str) -> Optional[dict]:
 """
 Parse EAM filename to extract components.
 
 Args:
 filename: EAM-formatted filename
 
 Returns:
 Dictionary with parsed components or None if invalid
 """
 filename = Path(filename).stem # Remove extension
 
 # Pattern: {activity}_v{###}_{distance}_{date}
 # or: {activity}_i{###}_f{idx}_{distance}_{date}
 
 # Try video pattern
 video_pattern = r'^(.+?)_v(\d+)_(.+?)_(\d{8})$'
 match = re.match(video_pattern, filename)
 
 if match:
 return {
 "type": "video",
 "activity": match.group(1),
 "number": int(match.group(2)),
 "distance": match.group(3),
 "date": match.group(4)
 }
 
 # Try image pattern
 image_pattern = r'^(.+?)_i(\d+)_f(\d+)_(.+?)_(\d{8})$'
 match = re.match(image_pattern, filename)
 
 if match:
 return {
 "type": "image",
 "activity": match.group(1),
 "number": int(match.group(2)),
 "frame_index": int(match.group(3)),
 "distance": match.group(4),
 "date": match.group(5)
 }
 
 return None
 
 @staticmethod
 def rename_file(old_path: str,
 activity: str,
 video_number: int,
 distance: str,
 date: Optional[str] = None) -> Path:
 """
 Rename file to EAM convention.
 
 Args:
 old_path: Current file path
 activity: Activity name
 video_number: Video number
 distance: Distance
 date: Date (uses file modification date if None)
 
 Returns:
 New Path object
 """
 old_path = Path(old_path)
 
 if date is None:
 # Use file modification date
 mtime = old_path.stat().st_mtime
 date = datetime.fromtimestamp(mtime).strftime("%Y%m%d")
 
 # Determine if it's a video or image
 if old_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
 new_name = EAMFileNaming.format_video_filename(
 activity, video_number, distance, date, old_path.suffix
 )
 elif old_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
 # For images, we need frame index - extract from old name or use 1
 frame_idx = 1
 parsed = EAMFileNaming.parse_filename(old_path.name)
 if parsed and parsed.get("type") == "image":
 frame_idx = parsed.get("frame_index", 1)
 
 new_name = EAMFileNaming.format_image_filename(
 activity, video_number, frame_idx, distance, date, old_path.suffix
 )
 else:
 # Unknown type, use video format
 new_name = EAMFileNaming.format_video_filename(
 activity, video_number, distance, date, old_path.suffix
 )
 
 new_path = old_path.parent / new_name
 
 if old_path.exists() and old_path != new_path:
 old_path.rename(new_path)
 log.info(f"Renamed: {old_path.name} → {new_name}")
 
 return new_path


if __name__ == "__main__":
 import argparse
 
 parser = argparse.ArgumentParser(description="EAM file naming utility")
 parser.add_argument("--activity", type=str, help="Activity name")
 parser.add_argument("--video_num", type=int, help="Video number")
 parser.add_argument("--distance", type=str, help="Distance (e.g., '2.8m')")
 parser.add_argument("--date", type=str, help="Date (YYYYMMDD)")
 parser.add_argument("--rename", type=str, help="File to rename")
 
 args = parser.parse_args()
 
 if args.rename:
 new_path = EAMFileNaming.rename_file(
 args.rename,
 args.activity or "unknown_activity",
 args.video_num or 1,
 args.distance or "unknown",
 args.date
 )
 print(f"Renamed to: {new_path}")
 else:
 # Generate example filenames
 activity = args.activity or "mask_removal_attempt"
 video_num = args.video_num or 1
 distance = args.distance or "2.8m"
 date = args.date or datetime.now().strftime("%Y%m%d")
 
 video_name = EAMFileNaming.format_video_filename(activity, video_num, distance, date)
 print(f"Video filename: {video_name}")
 
 for frame_idx in range(1, 6):
 image_name = EAMFileNaming.format_image_filename(
 activity, video_num, frame_idx, distance, date
 )
 print(f"Image {frame_idx}: {image_name}")
