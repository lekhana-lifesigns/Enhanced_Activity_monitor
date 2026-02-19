# pipeline/pilot_system.py
"""
Self-Learning Pilot System for Enhanced Activity Monitor
Implements dynamic adaptation, camera control, and segmentation-based processing
"""

import time
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import threading
import json
import os

from .pose.inference_pipeline import InferencePipeline
from .detectors.yolo_segmentation_detector import YOLOSegmentationDetector
from .pose.camera import Camera
from .pose.adaptive_fps import AdaptiveFPSController, ActivityScorer
from analytics.clinical_correlation import ClinicalCorrelationEngine

log = logging.getLogger("pilot_system")


class SelfLearningPilot:
    """
    Self-learning pilot system that dynamically adapts to:
    1. Activity patterns and learns optimal processing strategies
    2. Camera positioning and zoom for optimal context
    3. Segmentation-based focus areas for improved accuracy
    4. Real-time performance optimization
    """

    def __init__(self, config: Dict):
        self.config = config
        self.is_learning = True
        self.learning_data = deque(maxlen=1000)  # Store recent learning samples

        # Core components
        self.inference_pipeline = None
        self.segmentation_detector = None
        self.camera_controller = None
        self.clinical_correlation = None

        # Learning and adaptation
        self.performance_history = deque(maxlen=100)
        self.activity_patterns = {}
        self.camera_adaptation_rules = {}
        self.segmentation_focus_areas = {}

        # Dynamic adaptation state
        self.current_context = "unknown"
        self.optimal_zoom_level = 1.0
        self.focus_regions = []
        self.processing_mode = "standard"

        # Self-learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7
        self.confidence_threshold = 0.8

        # Performance monitoring
        self.frame_times = deque(maxlen=100)
        self.accuracy_scores = deque(maxlen=100)
        self.adaptation_events = []

        # Clinical self-learning state
        self.clinical_patterns = {
            "avg_anomaly_score": 0.0,
            "avg_confidence": 0.0,
            "baseline_state": "collecting",
            "dominant_cluster": -1,
            "cluster_distribution": {},
            "confidence_drops": 0,
            "anomaly_episodes": 0,
            "total_clinical_samples": 0,
        }

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all pilot system components."""
        try:
            # Initialize inference pipeline
            # IMPORTANT: Disable pilot system in the pipeline config to avoid circular dependency
            pipeline_config = self.config.copy()
            pipeline_config['enable_pilot_system'] = False
            self.inference_pipeline = InferencePipeline(pipeline_config)
            log.info("Inference pipeline initialized for pilot system")

            # Initialize segmentation detector
            if self.config.get("use_segmentation", True):
                self.segmentation_detector = YOLOSegmentationDetector(
                    model=self.config.get("yolo_segmentation_model", "yolo11n-seg.pt"),
                    pose_model=self.config.get("yolo_pose_model", "yolo11n-pose.pt"),
                    conf=self.config.get("detector_confidence", 0.5),
                    device=self.config.get("device", "cpu")
                )
                log.info("Segmentation detector initialized (with pose support)")

            # Initialize camera controller
            if self.config.get("enable_camera_control", True):
                # Create virtual camera controller (would connect to physical camera in production)
                self.camera_controller = CameraController(self.config)
                log.info("Camera controller initialized")

            # Initialize clinical correlation
            if self.config.get("enable_clinical_correlation", True):
                self.clinical_correlation = ClinicalCorrelationEngine()
                log.info("Clinical correlation initialized")

            # Load existing learning data if available
            self._load_learning_data()

        except Exception as e:
            log.error(f"Failed to initialize pilot components: {e}")
            raise

    def _load_learning_data(self):
        """Load previously learned patterns and rules."""
        learning_file = "storage/pilot_learning.json"
        if os.path.exists(learning_file):
            try:
                with open(learning_file, 'r') as f:
                    data = json.load(f)
                self.activity_patterns = data.get("activity_patterns", {})
                self.camera_adaptation_rules = data.get("camera_rules", {})
                self.segmentation_focus_areas = data.get("focus_areas", {})
                saved_clinical = data.get("clinical_patterns")
                if saved_clinical:
                    self.clinical_patterns.update(saved_clinical)
                log.info("Loaded existing learning data (clinical patterns: %s)",
                         "yes" if saved_clinical else "no")
            except Exception as e:
                log.warning(f"Failed to load learning data: {e}")

    def _save_learning_data(self):
        """Save learned patterns and rules."""
        learning_file = "storage/pilot_learning.json"
        os.makedirs("storage", exist_ok=True)

        data = {
            "activity_patterns": self.activity_patterns,
            "camera_rules": self.camera_adaptation_rules,
            "focus_areas": self.segmentation_focus_areas,
            "clinical_patterns": self.clinical_patterns,
            "last_updated": time.time()
        }

        try:
            with open(learning_file, 'w') as f:
                json.dump(data, f, indent=2)
            log.debug("Saved learning data (with clinical patterns)")
        except Exception as e:
            log.warning(f"Failed to save learning data: {e}")

    def process_frame_pilot(self, frame: np.ndarray, camera_id: str = None) -> Dict:
        """
        Process frame with self-learning pilot intelligence.

        Args:
            frame: Input frame
            camera_id: Camera identifier

        Returns:
            Processing result with pilot adaptations
        """
        start_time = time.time()

        try:
            # Step 1: Analyze current context and activity
            context_analysis = self._analyze_context(frame)

            # Step 2: Apply segmentation for focus areas
            segmentation_result = self._apply_segmentation_focus(frame)

            # Step 3: Adapt camera parameters based on context
            camera_adaptations = self._adapt_camera_parameters(context_analysis, segmentation_result)

            # Step 4: Process with adapted parameters
            result = self._process_with_adaptations(frame, context_analysis, segmentation_result, camera_id)

            # Check if result is None before proceeding (normal when no person detected)
            if result is None:
                log.debug("Processing returned None (no person detected), skipping learning cycle")
                return None

            # Step 5: Learn from this processing cycle
            self._learn_from_cycle(frame, result, context_analysis, start_time)

            # Step 6: Update pilot state
            self._update_pilot_state(result)

            processing_time = time.time() - start_time
            result["pilot_processing_time"] = processing_time
            result["pilot_adaptations"] = camera_adaptations
            result["context_analysis"] = context_analysis

            return result

        except Exception as e:
            log.error(f"Pilot processing failed: {e}")
            # Fallback to standard processing
            return self.inference_pipeline._process_frame(frame, camera_id)

    def _analyze_context(self, frame: np.ndarray) -> Dict:
        """Analyze current context for intelligent adaptation."""
        context = {
            "activity_level": "low",
            "person_present": False,
            "bed_visible": False,
            "lighting_conditions": "normal",
            "motion_level": "static",
            "focus_regions": [],
            "clinical_relevance": "low"
        }

        try:
            # Quick motion detection
            if hasattr(self, 'prev_frame') and self.prev_frame is not None:
                motion = np.mean(np.abs(frame.astype(np.float32) - self.prev_frame.astype(np.float32)))
                context["motion_level"] = "high" if motion > 20 else "low" if motion < 5 else "medium"

            self.prev_frame = frame.copy()

            # Person detection (quick check)
            # This would use a lightweight detector for context analysis
            person_detected = self._quick_person_detection(frame)
            context["person_present"] = person_detected

            # Lighting analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            context["lighting_conditions"] = "dark" if brightness < 50 else "bright" if brightness > 200 else "normal"

            # Clinical relevance assessment
            if person_detected:
                context["clinical_relevance"] = "high"
                context["activity_level"] = "medium"

            return context

        except Exception as e:
            log.debug(f"Context analysis failed: {e}")
            return context

    def _apply_segmentation_focus(self, frame: np.ndarray) -> Dict:
        """Apply segmentation to identify focus areas."""
        segmentation_result = {
            "masks": [],
            "focus_regions": [],
            "person_mask": None,
            "bed_mask": None,
            "confidence": 0.0
        }

        if not self.segmentation_detector:
            return segmentation_result

        try:
            # Run segmentation
            results = self.segmentation_detector.model(frame, conf=0.3, verbose=False)

            if results and len(results) > 0:
                result = results[0]

                # Extract person and bed masks
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()

                    for i, cls in enumerate(classes):
                        mask = masks[i]
                        if cls == 0:  # person class
                            segmentation_result["person_mask"] = mask
                            # Calculate focus region from mask
                            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                x, y, w, h = cv2.boundingRect(contours[0])
                                segmentation_result["focus_regions"].append((x, y, x+w, y+h))

                segmentation_result["confidence"] = float(result.boxes.conf.mean()) if len(result.boxes) > 0 else 0.0

            return segmentation_result

        except Exception as e:
            log.debug(f"Segmentation focus failed: {e}")
            return segmentation_result

    def _adapt_camera_parameters(self, context: Dict, segmentation: Dict) -> Dict:
        """Adapt camera parameters based on context and segmentation."""
        adaptations = {
            "zoom_level": 1.0,
            "focus_region": None,
            "processing_mode": "standard",
            "fps_adjustment": 1.0
        }

        try:
            # Adapt based on activity level
            if context["activity_level"] == "high":
                adaptations["zoom_level"] = 1.2
                adaptations["processing_mode"] = "high_accuracy"
            elif context["activity_level"] == "low":
                adaptations["zoom_level"] = 0.9
                adaptations["processing_mode"] = "efficient"

            # Use segmentation for focus
            if segmentation["focus_regions"]:
                # Focus on the largest person region
                largest_region = max(segmentation["focus_regions"], key=lambda r: (r[2]-r[0])*(r[3]-r[1]))
                adaptations["focus_region"] = largest_region

                # Adjust zoom based on person size
                frame_area = self.config["camera_resolution"][0] * self.config["camera_resolution"][1]
                person_area = (largest_region[2]-largest_region[0]) * (largest_region[3]-largest_region[1])
                size_ratio = person_area / frame_area

                if size_ratio < 0.1:  # Person too small
                    adaptations["zoom_level"] = min(2.0, adaptations["zoom_level"] * 1.5)
                elif size_ratio > 0.4:  # Person too large
                    adaptations["zoom_level"] = max(0.5, adaptations["zoom_level"] * 0.8)

            # Lighting adaptation
            if context["lighting_conditions"] == "dark":
                adaptations["processing_mode"] = "low_light"
            elif context["lighting_conditions"] == "bright":
                adaptations["processing_mode"] = "high_contrast"

            return adaptations

        except Exception as e:
            log.debug(f"Camera adaptation failed: {e}")
            return adaptations

    def _process_with_adaptations(self, frame: np.ndarray, context: Dict,
                                   segmentation: Dict, camera_id: str) -> Dict:
        """Process frame with applied adaptations."""
        try:
            # Apply segmentation mask for improved processing
            if segmentation.get("person_mask") is not None:
                try:
                    # Use mask to focus processing on person area
                    masked_frame = frame.copy()
                    person_mask = segmentation["person_mask"]

                    # Resize mask to match frame dimensions if needed
                    if person_mask.shape[:2] != frame.shape[:2]:
                        person_mask = cv2.resize(person_mask, (frame.shape[1], frame.shape[0]))

                    # Create 3-channel mask
                    mask_3d = np.stack([person_mask] * 3, axis=2)
                    masked_frame = cv2.bitwise_and(masked_frame, mask_3d.astype(np.uint8) * 255)

                    # Blend with original for context
                    frame = cv2.addWeighted(frame, 0.7, masked_frame, 0.3, 0)
                except Exception as mask_error:
                    log.debug(f"Mask application failed: {mask_error}, skipping")

            # Process with inference pipeline
            result = self.inference_pipeline._process_frame(frame, camera_id)

            # Check if result is None (normal when no person detected)
            if result is None:
                log.debug("Inference pipeline returned None (likely no person in frame)")
                return None

            # Enhance result with segmentation data
            if segmentation.get("person_mask") is not None:
                try:
                    result["segmentation_mask"] = segmentation["person_mask"].tolist()
                    result["focus_regions"] = segmentation["focus_regions"]
                except Exception as seg_error:
                    log.debug(f"Segmentation data enhancement failed: {seg_error}")

            # Add clinical correlation if available
            if self.clinical_correlation and result and "self_contact" in result:
                try:
                    clinical_result = self.clinical_correlation.correlate_clinical_state(
                        posture_state=result.get("posture_state", "unknown"),
                        posture_3d=None,
                        emotions={},
                        movement_features={},
                        frame_visibility={"completeness_score": 1.0},
                        distance=2.0,
                        self_contact=result["self_contact"]
                    )
                    result["clinical_correlation"] = clinical_result
                except Exception as clinical_error:
                    log.debug(f"Clinical correlation failed: {clinical_error}")

            return result

        except Exception as e:
            log.error(f"Adapted processing failed: {e}")
            # Fallback
            return self.inference_pipeline._process_frame(frame, camera_id)

    def _learn_from_cycle(self, frame: np.ndarray, result: Dict,
                          context: Dict, start_time: float):
        """Learn from processing cycle for future adaptation."""
        try:
            # Safety check for None result
            if result is None:
                return

            # Calculate performance metrics
            processing_time = time.time() - start_time
            accuracy_score = result.get("confidence", 0.0)
            clinical_score = 0.0

            if "clinical_correlation" in result:
                clinical = result["clinical_correlation"]
                clinical_score = max(clinical.get("pain_score", 0),
                                     clinical.get("agitation_score", 0))

            # Store learning sample
            learning_sample = {
                "timestamp": time.time(),
                "context": context,
                "processing_time": processing_time,
                "accuracy": accuracy_score,
                "clinical_score": clinical_score,
                "activity_detected": result.get("label", "unknown"),
                "adaptations_applied": result.get("pilot_adaptations", {})
            }

            self.learning_data.append(learning_sample)

            # --- Clinical self-learning ---
            baseline_info = result.get("baseline_info")
            if baseline_info:
                alpha = self.learning_rate
                cp = self.clinical_patterns
                cp["total_clinical_samples"] += 1
                cp["baseline_state"] = baseline_info.get("baseline_state", "collecting")
                anomaly_score = baseline_info.get("anomaly_score", 0.0)
                cp["avg_anomaly_score"] = (1 - alpha) * cp["avg_anomaly_score"] + alpha * anomaly_score
                cp["avg_confidence"] = (1 - alpha) * cp["avg_confidence"] + alpha * accuracy_score

                # Track cluster distribution
                cluster_id = str(baseline_info.get("cluster_id", -1))
                cp["cluster_distribution"][cluster_id] = cp["cluster_distribution"].get(cluster_id, 0) + 1

                # Dominant cluster
                if cp["cluster_distribution"]:
                    cp["dominant_cluster"] = int(max(cp["cluster_distribution"], key=cp["cluster_distribution"].get))

                # Track anomaly episodes (sustained anomalies)
                if baseline_info.get("is_anomaly"):
                    cp["anomaly_episodes"] += 1

                # Track confidence drops (model struggling)
                if accuracy_score < 0.3:
                    cp["confidence_drops"] += 1

            # Update activity patterns
            activity = result.get("label", "unknown")
            if activity not in self.activity_patterns:
                self.activity_patterns[activity] = {
                    "count": 0,
                    "avg_processing_time": 0,
                    "avg_accuracy": 0,
                    "optimal_zoom": 1.0,
                    "optimal_fps": self.config.get("camera_fps", 15)
                }

            pattern = self.activity_patterns[activity]
            pattern["count"] += 1

            # Update running averages
            alpha = self.learning_rate
            pattern["avg_processing_time"] = (1-alpha) * pattern["avg_processing_time"] + alpha * processing_time
            pattern["avg_accuracy"] = (1-alpha) * pattern["avg_accuracy"] + alpha * accuracy_score

            # Learn optimal parameters
            if "pilot_adaptations" in result:
                adaptations = result["pilot_adaptations"]
                if "zoom_level" in adaptations:
                    pattern["optimal_zoom"] = (1-alpha) * pattern["optimal_zoom"] + alpha * adaptations["zoom_level"]

            # Save learning data periodically
            if len(self.learning_data) % 100 == 0:
                self._save_learning_data()

        except Exception as e:
            log.debug(f"Learning cycle failed: {e}")

    def _update_pilot_state(self, result: Dict):
        """Update pilot system state based on results."""
        try:
            # Safety check for None result
            if result is None:
                return

            # Update current context
            self.current_context = result.get("label", "unknown")

            # Update optimal parameters based on learning
            if self.current_context in self.activity_patterns:
                pattern = self.activity_patterns[self.current_context]
                self.optimal_zoom_level = pattern["optimal_zoom"]

            # Update processing mode based on performance
            if len(self.frame_times) > 10:
                avg_time = np.mean(list(self.frame_times))
                target_time = 1.0 / self.config.get("camera_fps", 15)

                if avg_time > target_time * 1.5:
                    self.processing_mode = "efficient"
                elif avg_time < target_time * 0.8:
                    self.processing_mode = "high_accuracy"
                else:
                    self.processing_mode = "standard"

        except Exception as e:
            log.debug(f"State update failed: {e}")

    def _quick_person_detection(self, frame: np.ndarray) -> bool:
        """Quick person detection for context analysis."""
        try:
            # Simple motion-based detection as fallback
            if not hasattr(self, 'background_subtractor'):
                self.background_subtractor = cv2.createBackgroundSubtractorMOG2()

            fg_mask = self.background_subtractor.apply(frame)
            motion_pixels = np.sum(fg_mask > 0)
            total_pixels = frame.shape[0] * frame.shape[1]

            return (motion_pixels / total_pixels) > 0.01  # 1% motion threshold

        except Exception:
            return False

    def get_pilot_status(self) -> Dict:
        """Get current pilot system status and learning metrics."""
        return {
            "is_learning": self.is_learning,
            "current_context": self.current_context,
            "processing_mode": self.processing_mode,
            "optimal_zoom_level": self.optimal_zoom_level,
            "learned_activities": list(self.activity_patterns.keys()),
            "learning_samples": len(self.learning_data),
            "avg_processing_time": np.mean(list(self.frame_times)) if self.frame_times else 0,
            "avg_accuracy": np.mean(list(self.accuracy_scores)) if self.accuracy_scores else 0,
            "clinical_patterns": self.clinical_patterns,
        }

    def reset_learning(self):
        """Reset learning data and patterns."""
        self.learning_data.clear()
        self.activity_patterns.clear()
        self.camera_adaptation_rules.clear()
        self.segmentation_focus_areas.clear()
        self._save_learning_data()
        log.info("Pilot learning data reset")


class CameraController:
    """
    Virtual camera controller for pilot system.
    In production, this would interface with physical camera controls.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.current_zoom = 1.0
        self.current_focus = None
        self.supported_features = ["digital_zoom", "focus_region"]

    def set_zoom(self, zoom_level: float):
        """Set camera zoom level."""
        self.current_zoom = max(0.5, min(3.0, zoom_level))
        log.debug(f"Camera zoom set to: {self.current_zoom}")

    def set_focus_region(self, region: Tuple[int, int, int, int]):
        """Set camera focus region."""
        self.current_focus = region
        log.debug(f"Camera focus set to region: {region}")

    def get_capabilities(self) -> List[str]:
        """Get camera capabilities."""
        return self.supported_features
