"""
Clinical Dataset Annotation Tool
GUI tool for annotating recorded clinical activity videos.

Features:
- Video playback with frame-by-frame control
- Activity labeling with intensity
- Temporal annotation (start/end markers)
- Clinical tags and notes
- Export annotations in standard format
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

from dataset_creation.recording_protocol import TEMPORAL_CLASSES, CLINICAL_TAGS, TEMPORAL_CLASS_NAMES

log = logging.getLogger("annotation_tool")


class AnnotationTool:
 """
 GUI tool for annotating clinical activity videos.
 """

 def __init__(self, root: tk.Tk):
 self.root = root
 self.root.title("Clinical Dataset Annotation Tool")
 self.root.geometry("1400x900")

 # Video state
 self.video_path = None
 self.cap = None
 self.current_frame = 0
 self.total_frames = 0
 self.fps = 30
 self.is_playing = False

 # Annotation state
 self.annotations = []
 self.current_annotation = None
 self.markers = [] # Temporal markers

 # Metadata
 self.metadata = {}

 # Create UI
 self._create_ui()

 # Bind keyboard shortcuts
 self._bind_shortcuts()

 log.info("Annotation tool initialized")

 def _create_ui(self):
 """Create user interface."""
 # Main layout
 main_frame = ttk.Frame(self.root, padding="10")
 main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

 # Configure grid weights
 self.root.columnconfigure(0, weight=1)
 self.root.rowconfigure(0, weight=1)
 main_frame.columnconfigure(1, weight=1)
 main_frame.rowconfigure(1, weight=1)

 # Top bar - File controls
 self._create_file_controls(main_frame)

 # Left panel - Video player
 self._create_video_player(main_frame)

 # Right panel - Annotation controls
 self._create_annotation_controls(main_frame)

 # Bottom panel - Timeline
 self._create_timeline(main_frame)

 # Status bar
 self._create_status_bar(main_frame)

 def _create_file_controls(self, parent):
 """Create file control buttons."""
 file_frame = ttk.Frame(parent)
 file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

 ttk.Button(file_frame, text="Open Video", command=self.open_video).pack(side=tk.LEFT, padx=5)
 ttk.Button(file_frame, text="Load Metadata", command=self.load_metadata).pack(side=tk.LEFT, padx=5)
 ttk.Button(file_frame, text="Save Annotations", command=self.save_annotations).pack(side=tk.LEFT, padx=5)
 ttk.Button(file_frame, text="Export Dataset", command=self.export_dataset).pack(side=tk.LEFT, padx=5)

 # Video info label
 self.video_info_label = ttk.Label(file_frame, text="No video loaded")
 self.video_info_label.pack(side=tk.RIGHT, padx=5)

 def _create_video_player(self, parent):
 """Create video player panel."""
 player_frame = ttk.LabelFrame(parent, text="Video Player", padding="10")
 player_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

 # Video canvas
 self.video_canvas = tk.Canvas(player_frame, bg='black', width=800, height=600)
 self.video_canvas.pack(fill=tk.BOTH, expand=True)

 # Playback controls
 controls_frame = ttk.Frame(player_frame)
 controls_frame.pack(fill=tk.X, pady=(10, 0))

 self.play_button = ttk.Button(controls_frame, text=" Play", command=self.toggle_play)
 self.play_button.pack(side=tk.LEFT, padx=2)

 ttk.Button(controls_frame, text="⏮ Prev Frame", command=self.prev_frame).pack(side=tk.LEFT, padx=2)
 ttk.Button(controls_frame, text="⏭ Next Frame", command=self.next_frame).pack(side=tk.LEFT, padx=2)
 ttk.Button(controls_frame, text="⏪ -10f", command=lambda: self.skip_frames(-10)).pack(side=tk.LEFT, padx=2)
 ttk.Button(controls_frame, text="⏩ +10f", command=lambda: self.skip_frames(10)).pack(side=tk.LEFT, padx=2)

 # Frame slider
 self.frame_var = tk.IntVar(value=0)
 self.frame_slider = ttk.Scale(controls_frame, from_=0, to=100, orient=tk.HORIZONTAL,
 variable=self.frame_var, command=self.on_slider_change)
 self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

 # Frame label
 self.frame_label = ttk.Label(controls_frame, text="Frame: 0/0")
 self.frame_label.pack(side=tk.LEFT, padx=5)

 def _create_annotation_controls(self, parent):
 """Create annotation control panel with 2-level hierarchy."""
 annot_frame = ttk.LabelFrame(parent, text="Annotations", padding="10")
 annot_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

 # Level 1: Temporal Activity (PRIMARY - required)
 ttk.Label(annot_frame, text="Level 1 - Temporal Activity:", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))

 self.activity_var = tk.StringVar()
 activity_combo = ttk.Combobox(annot_frame, textvariable=self.activity_var, state='readonly', width=40)
 activity_combo['values'] = [
 f"[{tc.hotkey}] {tc.name} ({tc.clinical_urgency})"
 for tc in sorted(TEMPORAL_CLASSES.values(), key=lambda x: x.class_id)
 ]
 activity_combo.pack(fill=tk.X, pady=(0, 5))

 # Urgency indicator
 self.urgency_label = ttk.Label(annot_frame, text="", foreground="gray")
 self.urgency_label.pack(anchor=tk.W, pady=(0, 5))
 self.activity_var.trace('w', self._update_urgency_display)

 # Confidence selection
 ttk.Label(annot_frame, text="Confidence:", font=('', 9)).pack(anchor=tk.W, pady=(5, 3))
 self.confidence_var = tk.StringVar(value="high")
 conf_frame = ttk.Frame(annot_frame)
 conf_frame.pack(fill=tk.X, pady=(0, 10))
 for conf in ["high", "medium", "low"]:
 ttk.Radiobutton(conf_frame, text=conf.capitalize(), variable=self.confidence_var,
 value=conf).pack(side=tk.LEFT, padx=5)

 # Temporal markers
 ttk.Label(annot_frame, text="Temporal Markers:", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))

 marker_frame = ttk.Frame(annot_frame)
 marker_frame.pack(fill=tk.X, pady=(0, 10))

 ttk.Button(marker_frame, text="Mark Start", command=self.mark_start).pack(side=tk.LEFT, padx=2)
 ttk.Button(marker_frame, text="Mark End", command=self.mark_end).pack(side=tk.LEFT, padx=2)
 ttk.Button(marker_frame, text="Clear Markers", command=self.clear_markers).pack(side=tk.LEFT, padx=2)

 self.marker_label = ttk.Label(annot_frame, text="Start: - | End: -")
 self.marker_label.pack(anchor=tk.W, pady=(0, 10))

 # Level 2: Clinical Tags (OPTIONAL - secondary metadata)
 ttk.Label(annot_frame, text="Level 2 - Clinical Tags (optional):", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))

 tags_frame = ttk.Frame(annot_frame)
 tags_frame.pack(fill=tk.X, pady=(0, 10))

 self.tag_vars = {}
 # Show most common clinical tags as checkboxes
 common_clinical_tags = [
 "cough_mild", "cough_severe", "breathing_labored",
 "self_contact", "hand_wringing", "pain_guarding",
 "iv_manipulation", "tube_pulling", "mask_removal",
 ]

 for tag in common_clinical_tags:
 tag_info = CLINICAL_TAGS.get(tag)
 display_name = tag_info.tag_name if tag_info else tag.replace('_', ' ').title()
 var = tk.BooleanVar()
 self.tag_vars[tag] = var
 ttk.Checkbutton(tags_frame, text=display_name, variable=var).pack(anchor=tk.W)

 # Notes
 ttk.Label(annot_frame, text="Notes:", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))

 self.notes_text = tk.Text(annot_frame, height=3, width=40)
 self.notes_text.pack(fill=tk.X, pady=(0, 10))

 # Add annotation button
 ttk.Button(annot_frame, text="Add Segment", command=self.add_annotation,
 style='Accent.TButton').pack(fill=tk.X, pady=(10, 5))

 # Annotations list
 ttk.Label(annot_frame, text="Segments:", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))

 list_frame = ttk.Frame(annot_frame)
 list_frame.pack(fill=tk.BOTH, expand=True)

 scrollbar = ttk.Scrollbar(list_frame)
 scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

 self.annotations_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=10)
 self.annotations_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
 scrollbar.config(command=self.annotations_listbox.yview)

 self.annotations_listbox.bind('<Double-Button-1>', self.on_annotation_select)

 # Delete + Export buttons
 btn_frame = ttk.Frame(annot_frame)
 btn_frame.pack(fill=tk.X, pady=(5, 0))
 ttk.Button(btn_frame, text="Delete Selected", command=self.delete_annotation).pack(side=tk.LEFT, padx=2)
 ttk.Button(btn_frame, text="Export JSONL", command=self.export_jsonl).pack(side=tk.RIGHT, padx=2)

 def _update_urgency_display(self, *args):
 """Update urgency indicator when activity changes."""
 activity_str = self.activity_var.get()
 if not activity_str:
 self.urgency_label.config(text="", foreground="gray")
 return
 # Extract class name from "[hotkey] name (urgency)" format
 parts = activity_str.split('] ')
 if len(parts) > 1:
 name = parts[1].split(' (')[0]
 tc = TEMPORAL_CLASSES.get(name)
 if tc:
 colors = {"low": "green", "medium": "orange", "high": "red", "critical": "darkred"}
 self.urgency_label.config(
 text=f"Urgency: {tc.clinical_urgency.upper()} | {tc.description}",
 foreground=colors.get(tc.clinical_urgency, "gray")
 )

 def _create_timeline(self, parent):
 """Create timeline visualization."""
 timeline_frame = ttk.LabelFrame(parent, text="Timeline", padding="5")
 timeline_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

 self.timeline_canvas = tk.Canvas(timeline_frame, height=80, bg='white')
 self.timeline_canvas.pack(fill=tk.X)

 def _create_status_bar(self, parent):
 """Create status bar."""
 self.status_label = ttk.Label(parent, text="Ready", relief=tk.SUNKEN)
 self.status_label.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

 def _bind_shortcuts(self):
 """Bind keyboard shortcuts."""
 self.root.bind('<space>', lambda e: self.toggle_play())
 self.root.bind('<Left>', lambda e: self.prev_frame())
 self.root.bind('<Right>', lambda e: self.next_frame())
 self.root.bind('<Control-s>', lambda e: self.save_annotations())
 self.root.bind('<Control-o>', lambda e: self.open_video())
 self.root.bind('<Control-e>', lambda e: self.export_jsonl())
 self.root.bind('s', lambda e: self.mark_start())
 self.root.bind('e', lambda e: self.mark_end())
 self.root.bind('a', lambda e: self.add_annotation())

 # Level 1 hotkeys (0-9, f, o) for quick activity selection
 for tc in TEMPORAL_CLASSES.values():
 hotkey = tc.hotkey
 name = tc.name
 self.root.bind(hotkey, lambda e, n=name: self._set_activity_by_name(n))

 def _set_activity_by_name(self, name: str):
 """Set activity combobox by class name."""
 tc = TEMPORAL_CLASSES.get(name)
 if tc:
 display = f"[{tc.hotkey}] {tc.name} ({tc.clinical_urgency})"
 self.activity_var.set(display)

 # Video controls
 def open_video(self):
 """Open video file."""
 file_path = filedialog.askopenfilename(
 title="Select Video File",
 filetypes=[("Video files", "*.avi *.mp4 *.mov"), ("All files", "*.*")]
 )

 if file_path:
 self.load_video(file_path)

 def load_video(self, video_path: str):
 """Load video file."""
 self.video_path = Path(video_path)

 if self.cap:
 self.cap.release()

 self.cap = cv2.VideoCapture(str(video_path))

 if not self.cap.isOpened():
 messagebox.showerror("Error", "Failed to open video file")
 return

 self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
 self.fps = self.cap.get(cv2.CAP_PROP_FPS)
 self.current_frame = 0

 self.frame_slider.config(to=self.total_frames - 1)
 self.video_info_label.config(text=f"{self.video_path.name} | {self.total_frames} frames @ {self.fps:.1f} fps")

 self.show_frame(0)
 self.update_status(f"Loaded: {self.video_path.name}")

 log.info(f"Loaded video: {video_path} ({self.total_frames} frames)")

 def show_frame(self, frame_number: int):
 """Display specific frame."""
 if not self.cap:
 return

 self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
 ret, frame = self.cap.read()

 if ret:
 self.current_frame = frame_number

 # Convert to RGB for display
 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

 # Resize to fit canvas
 canvas_width = self.video_canvas.winfo_width()
 canvas_height = self.video_canvas.winfo_height()

 if canvas_width > 1 and canvas_height > 1:
 frame_rgb = cv2.resize(frame_rgb, (canvas_width, canvas_height))

 # Convert to PhotoImage
 img = Image.fromarray(frame_rgb)
 photo = ImageTk.PhotoImage(img)

 self.video_canvas.delete("all")
 self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
 self.video_canvas.image = photo # Keep reference

 # Update labels
 self.frame_var.set(frame_number)
 timestamp = frame_number / self.fps
 self.frame_label.config(text=f"Frame: {frame_number}/{self.total_frames} ({timestamp:.2f}s)")

 # Draw timeline
 self.draw_timeline()

 def toggle_play(self):
 """Toggle play/pause."""
 self.is_playing = not self.is_playing

 if self.is_playing:
 self.play_button.config(text="⏸ Pause")
 self.play_video()
 else:
 self.play_button.config(text=" Play")

 def play_video(self):
 """Play video."""
 if not self.is_playing or not self.cap:
 return

 if self.current_frame < self.total_frames - 1:
 self.show_frame(self.current_frame + 1)
 delay = int(1000 / self.fps)
 self.root.after(delay, self.play_video)
 else:
 self.is_playing = False
 self.play_button.config(text=" Play")

 def prev_frame(self):
 """Show previous frame."""
 if self.current_frame > 0:
 self.show_frame(self.current_frame - 1)

 def next_frame(self):
 """Show next frame."""
 if self.current_frame < self.total_frames - 1:
 self.show_frame(self.current_frame + 1)

 def skip_frames(self, delta: int):
 """Skip frames forward or backward."""
 new_frame = max(0, min(self.total_frames - 1, self.current_frame + delta))
 self.show_frame(new_frame)

 def on_slider_change(self, value):
 """Handle slider change."""
 frame = int(float(value))
 if frame != self.current_frame:
 self.show_frame(frame)

 # Annotation controls
 def mark_start(self):
 """Mark start frame."""
 if len(self.markers) == 0:
 self.markers.append(self.current_frame)
 elif len(self.markers) == 1:
 self.markers[0] = self.current_frame
 else:
 self.markers = [self.current_frame, self.markers[1]]

 self.update_marker_label()
 self.update_status(f"Start marked at frame {self.current_frame}")

 def mark_end(self):
 """Mark end frame."""
 if len(self.markers) == 0:
 self.markers = [0, self.current_frame]
 elif len(self.markers) == 1:
 self.markers.append(self.current_frame)
 else:
 self.markers[1] = self.current_frame

 # Ensure start < end
 if self.markers[0] > self.markers[1]:
 self.markers = [self.markers[1], self.markers[0]]

 self.update_marker_label()
 self.update_status(f"End marked at frame {self.current_frame}")

 def clear_markers(self):
 """Clear temporal markers."""
 self.markers = []
 self.update_marker_label()
 self.draw_timeline()

 def update_marker_label(self):
 """Update marker label display."""
 start_text = f"{self.markers[0]}" if len(self.markers) > 0 else "-"
 end_text = f"{self.markers[1]}" if len(self.markers) > 1 else "-"
 self.marker_label.config(text=f"Start: {start_text} | End: {end_text}")

 def add_annotation(self):
 """Add temporal segment annotation."""
 activity_str = self.activity_var.get()
 if not activity_str:
 messagebox.showwarning("Warning", "Please select a Level 1 activity")
 return

 # Parse class name from display format "[hotkey] name (urgency)"
 parts = activity_str.split('] ')
 if len(parts) > 1:
 class_name = parts[1].split(' (')[0]
 else:
 class_name = activity_str

 tc = TEMPORAL_CLASSES.get(class_name)
 if not tc:
 messagebox.showerror("Error", f"Unknown temporal class: {class_name}")
 return

 # Determine frame range
 if len(self.markers) >= 2:
 start_frame = self.markers[0]
 end_frame = self.markers[1]
 else:
 messagebox.showwarning("Warning", "Mark start and end frames first (press 's' and 'e')")
 return

 # Validate minimum segment duration (1.5s)
 duration = (end_frame - start_frame) / self.fps
 if duration < 1.5:
 messagebox.showwarning("Warning", f"Segment too short ({duration:.1f}s). Minimum is 1.5s.")
 return

 # Collect Level 2 clinical tags
 clinical_tags = [tag for tag, var in self.tag_vars.items() if var.get()]

 # Confidence mapping
 conf_map = {"high": 0.95, "medium": 0.8, "low": 0.6}
 confidence = conf_map.get(self.confidence_var.get(), 0.9)

 # Create annotation (compatible with temporal_annotation_converter.py JSONL format)
 annotation = {
 'video_id': self.video_path.stem if self.video_path else 'unknown',
 'label': class_name,
 'label_id': tc.class_id,
 'start_frame': start_frame,
 'end_frame': end_frame,
 'start_sec': round(start_frame / self.fps, 3),
 'end_sec': round(end_frame / self.fps, 3),
 'confidence': confidence,
 'clinical_tags': clinical_tags,
 'annotator_id': 'local_tool',
 'notes': self.notes_text.get('1.0', tk.END).strip(),
 }

 self.annotations.append(annotation)

 # Update listbox with color-coded display
 display = f"{tc.short_code} {class_name} [{start_frame}-{end_frame}] ({duration:.1f}s)"
 self.annotations_listbox.insert(tk.END, display)

 # Color the listbox entry based on urgency
 idx = self.annotations_listbox.size() - 1
 urgency_colors = {"low": "#e8f5e9", "medium": "#fff3e0", "high": "#ffebee", "critical": "#ffcdd2"}
 self.annotations_listbox.itemconfig(idx, bg=urgency_colors.get(tc.clinical_urgency, "white"))

 # Clear markers and tags
 self.clear_markers()
 for var in self.tag_vars.values():
 var.set(False)
 self.notes_text.delete('1.0', tk.END)

 self.update_status(f"Added: {class_name} ({duration:.1f}s)")
 self.draw_timeline()

 log.info(f"Added segment: {annotation}")

 def delete_annotation(self):
 """Delete selected annotation."""
 selection = self.annotations_listbox.curselection()
 if selection:
 idx = selection[0]
 self.annotations.pop(idx)
 self.annotations_listbox.delete(idx)
 self.draw_timeline()
 self.update_status("Annotation deleted")

 def on_annotation_select(self, event):
 """Jump to selected annotation."""
 selection = self.annotations_listbox.curselection()
 if selection:
 idx = selection[0]
 annotation = self.annotations[idx]
 self.show_frame(annotation['start_frame'])

 def draw_timeline(self):
 """Draw timeline with annotations using TEMPORAL_CLASSES colors."""
 self.timeline_canvas.delete("all")

 if not self.total_frames:
 return

 width = self.timeline_canvas.winfo_width()
 height = self.timeline_canvas.winfo_height()

 if width <= 1:
 return

 # Draw background
 self.timeline_canvas.create_rectangle(0, 0, width, height, fill='white')

 # Draw annotations as colored bars using TEMPORAL_CLASSES colors
 for annotation in self.annotations:
 x1 = (annotation['start_frame'] / self.total_frames) * width
 x2 = (annotation['end_frame'] / self.total_frames) * width

 # Get color from TEMPORAL_CLASSES
 tc = TEMPORAL_CLASSES.get(annotation.get('label', ''))
 fill_color = tc.color if tc else '#95e1d3'

 self.timeline_canvas.create_rectangle(x1, 10, x2, height - 10,
 fill=fill_color, outline='black')

 # Draw label text if segment is wide enough
 seg_width = x2 - x1
 if seg_width > 30 and tc:
 self.timeline_canvas.create_text(
 (x1 + x2) / 2, height / 2,
 text=tc.short_code, font=('', 7), fill='white'
 )

 # Draw current position
 current_x = (self.current_frame / self.total_frames) * width
 self.timeline_canvas.create_line(current_x, 0, current_x, height, fill='red', width=2)

 # Draw markers
 if len(self.markers) > 0:
 marker_x = (self.markers[0] / self.total_frames) * width
 self.timeline_canvas.create_line(marker_x, 0, marker_x, height, fill='green', width=2, dash=(4, 4))

 if len(self.markers) > 1:
 marker_x = (self.markers[1] / self.total_frames) * width
 self.timeline_canvas.create_line(marker_x, 0, marker_x, height, fill='blue', width=2, dash=(4, 4))

 # File operations
 def load_metadata(self):
 """Load metadata JSON file."""
 file_path = filedialog.askopenfilename(
 title="Select Metadata File",
 filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
 )

 if file_path:
 with open(file_path, 'r') as f:
 self.metadata = json.load(f)

 self.update_status(f"Loaded metadata: {Path(file_path).name}")

 def export_jsonl(self):
 """Export annotations as JSONL compatible with temporal_annotation_converter.py."""
 if not self.annotations:
 messagebox.showwarning("Warning", "No annotations to export")
 return

 # Default filename based on video
 default_name = ""
 if self.video_path:
 default_name = f"{self.video_path.stem}_temporal_labels.jsonl"

 file_path = filedialog.asksaveasfilename(
 defaultextension=".jsonl",
 initialfile=default_name,
 filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
 )

 if file_path:
 # Sort annotations by start time
 sorted_anns = sorted(self.annotations, key=lambda a: a['start_sec'])

 with open(file_path, 'w') as f:
 for ann in sorted_anns:
 # Write only the fields expected by temporal_annotation_converter.py
 jsonl_record = {
 'video_id': ann['video_id'],
 'start_sec': ann['start_sec'],
 'end_sec': ann['end_sec'],
 'start_frame': ann['start_frame'],
 'end_frame': ann['end_frame'],
 'label': ann['label'],
 'label_id': ann['label_id'],
 'confidence': ann['confidence'],
 'annotator_id': ann['annotator_id'],
 'notes': ann.get('notes', ''),
 }
 f.write(json.dumps(jsonl_record) + '\n')

 self.update_status(f"Exported {len(sorted_anns)} segments to {Path(file_path).name}")
 messagebox.showinfo("Success", f"Exported {len(sorted_anns)} segments to JSONL")

 def save_annotations(self):
 """Save full annotation session (JSON with metadata + annotations)."""
 if not self.annotations:
 messagebox.showwarning("Warning", "No annotations to save")
 return

 file_path = filedialog.asksaveasfilename(
 defaultextension=".json",
 filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
 )

 if file_path:
 output = {
 'video_path': str(self.video_path) if self.video_path else None,
 'total_frames': self.total_frames,
 'fps': self.fps,
 'metadata': self.metadata,
 'annotations': self.annotations,
 }

 with open(file_path, 'w') as f:
 json.dump(output, f, indent=2)

 self.update_status(f"Saved session: {Path(file_path).name}")
 messagebox.showinfo("Success", f"Session saved to {file_path}")

 def export_dataset(self):
 """Export annotated dataset - delegates to JSONL export."""
 self.export_jsonl()

 def update_status(self, message: str):
 """Update status bar."""
 self.status_label.config(text=message)

 def cleanup(self):
 """Clean up resources."""
 if self.cap:
 self.cap.release()


def launch_annotation_tool():
 """Launch annotation tool GUI."""
 root = tk.Tk()
 app = AnnotationTool(root)

 def on_closing():
 app.cleanup()
 root.destroy()

 root.protocol("WM_DELETE_WINDOW", on_closing)
 root.mainloop()


if __name__ == "__main__":
 launch_annotation_tool()
