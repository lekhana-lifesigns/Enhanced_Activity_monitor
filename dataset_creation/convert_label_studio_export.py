#!/usr/bin/env python3
"""
Convert Label Studio JSON export → EAM temporal labels JSONL format.

Label Studio exports annotations as JSON with regions on video timeline.
This script converts those regions to the JSONL segment format expected
by temporal_annotation_converter.py.

Usage:
 python convert_label_studio_export.py \
 --input label_studio_export.json \
 --output annotations/ \
 --fps 15
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ls_converter")

CLASS_NAME_TO_ID = {
 "lying_still": 0,
 "lying_restless": 1,
 "turning_in_bed": 2,
 "sitting_stable": 3,
 "sitting_unstable": 4,
 "bed_exit": 5,
 "standing": 6,
 "walking": 7,
 "agitated": 8,
 "convulsive": 9,
 "fallen": 10,
 "occluded": 11,
}


def convert_label_studio_export(input_path: str, output_dir: str, fps: int = 15):
 """
 Convert Label Studio JSON export to per-video JSONL files.

 Label Studio video annotation export format (simplified):
 [
 {
 "id": 1,
 "data": {"video_url": "/data/video.mp4"},
 "annotations": [
 {
 "result": [
 {
 "type": "videorectangle",
 "value": {
 "start": 0, # start frame
 "end": 45, # end frame
 "labels": ["lying_still"],
 "sequence": [...]
 }
 },
 ...
 ]
 }
 ]
 }
 ]
 """
 with open(input_path, 'r') as f:
 export_data = json.load(f)

 output_path = Path(output_dir)
 output_path.mkdir(parents=True, exist_ok=True)

 if not isinstance(export_data, list):
 export_data = [export_data]

 for task in export_data:
 task_id = task.get('id', 'unknown')
 video_url = task.get('data', {}).get('video_url', '')
 video_id = Path(video_url).stem.replace(" ", "_").lower() if video_url else f"task_{task_id}"

 annotations = task.get('annotations', [])
 if not annotations:
 log.warning("Task %s has no annotations, skipping", task_id)
 continue

 # Take the first annotation (or latest)
 annotation = annotations[0]
 annotator_id = str(annotation.get('completed_by', 'unknown'))
 results = annotation.get('result', [])

 segments = []

 for result in results:
 result_type = result.get('type', '')

 # Handle video timeline annotations
 if result_type in ('videorectangle', 'timeserieslabels', 'labels'):
 value = result.get('value', {})
 labels = value.get('labels', [])

 if not labels:
 continue

 label_name = labels[0] # Take first label (mutually exclusive)
 if label_name not in CLASS_NAME_TO_ID:
 log.warning("Unknown label '%s' in task %s, skipping", label_name, task_id)
 continue

 # Get timing - Label Studio uses different formats
 start_frame = value.get('start', value.get('startFrame', 0))
 end_frame = value.get('end', value.get('endFrame', start_frame + fps * 3))

 # Some exports use seconds directly
 if 'start' in value and value['start'] > 1000:
 # Likely milliseconds
 start_sec = value['start'] / 1000.0
 end_sec = value['end'] / 1000.0
 start_frame = int(start_sec * fps)
 end_frame = int(end_sec * fps)
 else:
 start_sec = start_frame / fps
 end_sec = end_frame / fps

 # Get confidence from choices (if annotated)
 confidence = 0.9 # Default
 # Look for associated choices
 for other_result in results:
 if other_result.get('type') == 'choices' and other_result.get('id') == result.get('id'):
 choices = other_result.get('value', {}).get('choices', [])
 if 'high_confidence' in choices:
 confidence = 0.95
 elif 'low_confidence' in choices:
 confidence = 0.6
 elif 'medium_confidence' in choices:
 confidence = 0.8

 # Get notes
 notes = ""
 for other_result in results:
 if other_result.get('type') == 'textarea' and other_result.get('id') == result.get('id'):
 text_values = other_result.get('value', {}).get('text', [])
 if text_values:
 notes = text_values[0]

 segments.append({
 "video_id": video_id,
 "start_sec": round(start_sec, 3),
 "end_sec": round(end_sec, 3),
 "start_frame": start_frame,
 "end_frame": end_frame,
 "label": label_name,
 "label_id": CLASS_NAME_TO_ID[label_name],
 "confidence": confidence,
 "annotator_id": annotator_id,
 "notes": notes,
 })

 if not segments:
 log.warning("No valid segments for task %s (%s)", task_id, video_id)
 continue

 # Sort by start time
 segments.sort(key=lambda x: x['start_sec'])

 # Validate: check for gaps and overlaps
 validate_segments(segments, video_id)

 # Write JSONL
 output_file = output_path / f"{video_id}_temporal_labels.jsonl"
 with open(output_file, 'w') as f:
 for seg in segments:
 f.write(json.dumps(seg) + '\n')

 log.info("Exported %d segments for %s → %s", len(segments), video_id, output_file)

 # Print summary
 label_counts = {}
 for seg in segments:
 label_counts[seg['label']] = label_counts.get(seg['label'], 0) + 1
 for label, count in sorted(label_counts.items()):
 duration = sum(s['end_sec'] - s['start_sec'] for s in segments if s['label'] == label)
 log.info(" %-18s: %3d segments, %.1fs total", label, count, duration)


def validate_segments(segments: List[Dict], video_id: str):
 """Validate segment consistency."""
 for i in range(1, len(segments)):
 gap = segments[i]['start_sec'] - segments[i-1]['end_sec']
 if gap > 0.2:
 log.warning("[%s] Gap of %.2fs at %.2fs (between %s and %s)",
 video_id, gap, segments[i-1]['end_sec'],
 segments[i-1]['label'], segments[i]['label'])
 if gap < -0.1:
 log.warning("[%s] Overlap of %.2fs at %.2fs (between %s and %s)",
 video_id, -gap, segments[i]['start_sec'],
 segments[i-1]['label'], segments[i]['label'])

 # Check minimum duration
 for seg in segments:
 duration = seg['end_sec'] - seg['start_sec']
 if duration < 1.5:
 log.warning("[%s] Short segment (%.2fs) at %.2fs: %s",
 video_id, duration, seg['start_sec'], seg['label'])


def convert_cvat_export(input_path: str, output_dir: str, fps: int = 15):
 """
 Convert CVAT Video 1.1 JSON export to EAM JSONL format.
 CVAT uses a different structure with tracks and shapes.
 """
 with open(input_path, 'r') as f:
 cvat_data = json.load(f)

 output_path = Path(output_dir)
 output_path.mkdir(parents=True, exist_ok=True)

 # CVAT structure: annotations → tracks → shapes (with frame numbers)
 annotations = cvat_data.get('annotations', [])

 for ann in annotations:
 video_name = ann.get('name', 'unknown')
 video_id = Path(video_name).stem.replace(" ", "_").lower()
 tracks = ann.get('tracks', [])

 segments = []
 for track in tracks:
 label = track.get('label', '')
 if label not in CLASS_NAME_TO_ID:
 continue

 shapes = track.get('shapes', [])
 if not shapes:
 continue

 # CVAT shapes have frame numbers
 start_frame = min(s.get('frame', 0) for s in shapes)
 end_frame = max(s.get('frame', 0) for s in shapes)

 segments.append({
 "video_id": video_id,
 "start_sec": round(start_frame / fps, 3),
 "end_sec": round(end_frame / fps, 3),
 "start_frame": start_frame,
 "end_frame": end_frame,
 "label": label,
 "label_id": CLASS_NAME_TO_ID[label],
 "confidence": 0.9,
 "annotator_id": "cvat",
 "notes": "",
 })

 if segments:
 segments.sort(key=lambda x: x['start_sec'])
 output_file = output_path / f"{video_id}_temporal_labels.jsonl"
 with open(output_file, 'w') as f:
 for seg in segments:
 f.write(json.dumps(seg) + '\n')
 log.info("CVAT export: %d segments for %s", len(segments), video_id)


def main():
 parser = argparse.ArgumentParser(description="Convert annotation tool exports to EAM JSONL")
 parser.add_argument("--input", required=True, help="Path to export JSON file")
 parser.add_argument("--output", default="./annotations", help="Output directory for JSONL files")
 parser.add_argument("--fps", type=int, default=15, help="Video FPS for frame-to-time conversion")
 parser.add_argument("--format", choices=["label-studio", "cvat"], default="label-studio",
 help="Source annotation tool format")

 args = parser.parse_args()

 if args.format == "label-studio":
 convert_label_studio_export(args.input, args.output, args.fps)
 elif args.format == "cvat":
 convert_cvat_export(args.input, args.output, args.fps)


if __name__ == "__main__":
 main()
