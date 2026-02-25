# pipeline/inference_pipeline.py
import time
import logging
import numpy as np
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from .camera import Camera
from pipeline.detectors.detector import Detector
from .pose_estimator import PoseEstimator
from .feature_extractor import ICUFeatureEncoder
from .temporal_model import TemporalModel
from .temporal_model_enhanced import TemporalModelEnhanced
from pipeline.display.display import ICUMonitorDisplay
from .decision_engine import apply_rules

log = logging.getLogger("infpipe")


class InferencePipeline:
    """
    Inference pipeline for enhanced activity monitor.
    Robust handling for missing detections / pose failures, optional display,
    sliding-window temporal prediction and clinical decision engine.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        camres = tuple(cfg.get("camera_resolution", (1280, 720)))
        fps = cfg.get("camera_fps", 15)

        # Resolve device with automatic CUDA fallback
        requested_device = cfg.get("device", "cpu")
        if requested_device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    log.warning("CUDA requested but not available (torch.cuda.is_available()=False). "
                                "Falling back to CPU. Install Jetson-specific PyTorch from "
                                "developer.nvidia.com for GPU acceleration.")
                    cfg["device"] = "cpu"
            except ImportError:
                log.warning("PyTorch not installed, falling back to CPU")
                cfg["device"] = "cpu"

        # camera_url takes priority over camera_idx (RTSP/IP vs USB/CSI)
        camera_url = cfg.get("camera_url")
        self.camera = Camera(
            index=cfg.get("camera_idx", 0), resolution=camres, fps=fps,
            use_gstreamer=cfg.get("use_gstreamer", False),
            flip_method=cfg.get("camera_flip_method", 0),
            url=camera_url,
        )
        models = cfg.get("models", {})
        
        # Detector initialization with segmentation support
        # TODO-002: Auto-switch to detection-only model when segmentation not needed
        detector_type = cfg.get("detector_type", "yolo")
        use_segmentation = cfg.get("use_segmentation", False)
        include_mask_in_result = cfg.get("include_mask_in_result", False)
        enable_display = cfg.get("enable_display", True)
        
        # Auto-detect if segmentation is actually needed
        # Segmentation is only needed if:
        # 1. Explicitly enabled AND
        # 2. Masks are included in result OR display is enabled (masks used for visualization)
        actually_needs_segmentation = use_segmentation and (include_mask_in_result or enable_display)
        
        if detector_type == "yolo":
            if actually_needs_segmentation:
                # Use segmentation detector for instant masks
                from pipeline.detectors.yolo_segmentation_detector import YOLOSegmentationDetector
                self.det = YOLOSegmentationDetector(
                    model=cfg.get("yolo_segmentation_model", "yolo11n-seg.pt"),
                    tracker=cfg.get("tracker_type", "bytetrack.yaml"),
                    conf=cfg.get("detector_confidence", 0.5),
                    device=cfg.get("device", "cpu")
                )
                log.info("Using YOLO segmentation detector (with instant masks)")
            else:
                # Use detection-only model (faster) when segmentation not needed
                # TODO-002: Detection-only model is 30-40% faster than segmentation model
                from pipeline.detectors.yolo_reid_detector import YOLOReIDDetector
                self.det = YOLOReIDDetector(
                    detection_model=cfg.get("yolo_detection_model", "yolo11n.pt"),
                    pose_model=cfg.get("yolo_pose_model", "yolo11n-pose.pt"),
                    tracker=cfg.get("tracker_type", "bytetrack.yaml"),
                    conf=cfg.get("detector_confidence", 0.5),
                    device=cfg.get("device", "cpu"),
                    use_reid=cfg.get("use_reid_tracking", True)
                )
                if use_segmentation and not actually_needs_segmentation:
                    log.info("Using YOLO detection-only model (auto-switched: segmentation not needed, 30-40% faster)")
                else:
                    log.info("Using YOLO ReID detector")
        else:
            # Fallback to basic detector
            self.det = Detector(
                model_path=models.get("detector"),
                input_size=tuple(models.get("det_input_size", (320, 320))),
                use_edgetpu=cfg.get("use_edgetpu", False),
            )
            log.info("Using basic detector")
        
        self.pose = PoseEstimator(model_path=models.get("pose"), input_size=models.get("pose_input_size", 192))
        
        # Shared window size (used by temporal model, feature encoder, keypoint buffer)
        window_size = cfg.get("window_size", 48)

        # Use enhanced temporal model if available
        use_enhanced_temporal = cfg.get("use_enhanced_temporal", True)
        if use_enhanced_temporal:
            try:
                device = cfg.get("device", "cpu")
                use_fp16 = cfg.get("temporal_use_fp16", False)
                feature_dim = cfg.get("feature_dim", 145)
                self.temporal = TemporalModelEnhanced(
                    model_path=models.get("temporal"),
                    window_size=window_size,
                    use_pytorch=True,
                    device=device,
                    use_fp16=use_fp16,
                    feature_dim=feature_dim,
                )
                # Wire temporal stride from config
                self.temporal._prediction_stride = cfg.get("temporal_prediction_stride", 1)
                log.info("Using enhanced temporal model (stride=%d, fp16=%s)",
                         self.temporal._prediction_stride, use_fp16)
            except Exception as e:
                log.warning("Failed to initialize enhanced temporal model: %s, falling back to standard", e)
                self.temporal = TemporalModel(model_path=models.get("temporal"), window_size=window_size)
        else:
            self.temporal = TemporalModel(model_path=models.get("temporal"), window_size=window_size)

        # Training data collection (opt-in, for future GRU training)
        self.training_collector = None
        if cfg.get("collect_training_data", False):
            try:
                from pipeline.pose.temporal_model_enhanced import TrainingDataCollector
                self.training_collector = TrainingDataCollector(
                    save_dir=cfg.get("training_data_dir", "data/training_collection"),
                    min_confidence=cfg.get("training_min_confidence", 0.7),
                )
                self.training_collector.enable()
            except Exception as e:
                log.warning("Failed to initialize training data collector: %s", e)

        # Feature Encoder (handcrafted or learned)
        use_learned_features = cfg.get("use_learned_features", False)
        
        if use_learned_features:
            try:
                from .learned_feature_extractor import LearnedFeatureExtractor, HybridFeatureExtractor
                learned_method = cfg.get("learned_feature_method", "transformer")
                device = cfg.get("device", "cpu")
                
                learned_extractor = LearnedFeatureExtractor(
                    method=learned_method,
                    device=device
                )
                
                # Use hybrid if enabled
                enable_graph = cfg.get("enable_graph_features", True)
                if cfg.get("use_hybrid_features", True):
                    self.feature_encoder = HybridFeatureExtractor(
                        learned_extractor=learned_extractor,
                        handcrafted_extractor=ICUFeatureEncoder(
                            window_size=window_size, fps=fps,
                            enable_graph_features=enable_graph,
                        )
                    )
                    log.info("Using hybrid feature extractor (learned + handcrafted)")
                else:
                    self.feature_encoder = learned_extractor
                    log.info("Using learned feature extractor (method: %s)", learned_method)
            except Exception as e:
                log.warning("Failed to initialize learned features: %s, using handcrafted", e)
                self.feature_encoder = ICUFeatureEncoder(
                    window_size=window_size, fps=fps,
                    enable_graph_features=cfg.get("enable_graph_features", True),
                )
        else:
            self.feature_encoder = ICUFeatureEncoder(
                window_size=window_size, fps=fps,
                enable_graph_features=cfg.get("enable_graph_features", True),
            )
        
        # Initialize keypoint window for learned features
        # TODO-040: Frame buffer management (use deque with maxlen)
        from collections import deque
        self.kps_window = deque(maxlen=window_size)  # Bounded buffer

        self.window = deque(maxlen=window_size)  # list of feat arrays (bounded)
        self.prev_kps = None
        self.prev_prev_kps = None

        # FPS counters
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.fps = 0.0
        
        # TODO-030: Frame Rate Control
        self.target_fps = fps
        self.last_frame_time = time.time()
        self.enable_frame_rate_control = cfg.get("enable_frame_rate_control", True)
        
        # TODO-031: Adaptive Frame Skipping
        self.enable_adaptive_frame_skipping = cfg.get("enable_adaptive_frame_skipping", True)
        self.pose_change_threshold = cfg.get("pose_change_threshold", 0.05)  # 5% change threshold
        self.cached_result = None  # Cache for skipped frames
        self.last_processed_kps = None
        
        # TODO-022: Run activity classification every N frames
        self.activity_classification_frequency = cfg.get("activity_classification_frequency", 1)  # Classify every N frames (1 = every frame)
        self.activity_classification_frame_counter = 0
        self.last_activity_result = None  # Cache for skipped frames
        
        # TODO-008: Cache pose results for static poses
        self.enable_pose_caching = cfg.get("enable_pose_caching", True)
        self.pose_cache_threshold = cfg.get("pose_cache_threshold", 0.03)  # 3% pose change to invalidate cache
        self.last_cached_pose = None
        self.last_cached_pose_hash = None
        # Number of bytes to hash for pose cache (balance between speed and collision risk)
        self.pose_cache_hash_bytes = 1000
        
        # Performance monitoring (optional)
        self.enable_metrics = cfg.get("enable_metrics_collection", False)
        self.performance_monitor = None
        if self.enable_metrics:
            try:
                from pipeline.metrics.performance_monitor import PerformanceMonitor
                self.performance_monitor = PerformanceMonitor(
                    window_size=cfg.get("metrics_window_size", 100),
                    fps_target=fps,
                    enable_iou_tracking=cfg.get("enable_iou_tracking", False),
                    enable_id_switch_tracking=cfg.get("enable_id_switch_tracking", True)
                )
                log.info("Performance monitoring enabled")
            except Exception as e:
                log.warning("Failed to initialize performance monitor: %s", e)

        # Patient tracking state
        self.patient_track_id = None  # Persistent track ID for patient
        self.track_id_history = []  # History of track IDs (for recovery)
        self.track_id_confidence = {}  # Track ID -> confidence score
        self.patient_onboarded = False
        self.patient_missing_frames = 0
        self.patient_missing_threshold = cfg.get("patient_missing_threshold", 30)
        self.patient_missing_threshold_verified = cfg.get("patient_missing_threshold_verified", 150)
        
        # Patient face recognition with anti-spoofing (if enabled)
        self.face_recognizer = None
        if cfg.get("use_face_recognition", False):
            try:
                from pipeline.patient.face_recognition import PatientFaceRecognizer
                self.face_recognizer = PatientFaceRecognizer(
                    reference_faces_dir=cfg.get("patient_faces_dir", "storage/patient_faces"),
                    model_name=cfg.get("face_model", "VGG-Face"),
                    enable_liveness=cfg.get("enable_liveness_detection", True),
                    enable_rate_limiting=cfg.get("enable_rate_limiting", True)
                )
                if self.face_recognizer.enabled:
                    log.info("Patient face recognition enabled (with anti-spoofing)")
                else:
                    log.warning("Face recognition requested but DeepFace not available")
                    self.face_recognizer = None
            except Exception as e:
                log.warning("Failed to initialize face recognition: %s", e)
                self.face_recognizer = None
        
        # Camera security (access control, encryption, audit logging)
        self.camera_security = None
        if cfg.get("enable_camera_security", True):
            try:
                from pipeline.camera.security import create_camera_security
                security_config = cfg.get("camera_security", {})
                self.camera_security = create_camera_security(security_config)
                if self.camera_security:
                    log.info("Camera security enabled (encryption: %s)", self.camera_security.enable_encryption)
            except Exception as e:
                log.warning("Failed to initialize camera security: %s", e)
                self.camera_security = None

        # Bed detection (for context-aware monitoring and zoom control)
        self.bed_detector = None
        self.enable_bed_detection = cfg.get("enable_bed_detection", True)
        self.enable_auto_zoom = cfg.get("enable_auto_zoom", True)
        self.auto_zoom_target_size = cfg.get("auto_zoom_target_size", 0.4)  # Target person size ratio
        
        if self.enable_bed_detection:
            try:
                from analytics.bed_detection import BedDetector
                self.bed_detector = BedDetector(
                    model_path=cfg.get("bed_detection_model"),
                    conf_threshold=cfg.get("bed_detection_confidence", 0.3)
                )
                log.info("Bed detection enabled")
            except Exception as e:
                log.warning("Failed to initialize bed detector: %s", e)
                self.bed_detector = None
                self.enable_bed_detection = False
        
        # Bed state tracking
        self.current_bed = None
        self.bed_detection_frames = 0
        self.bed_stable_threshold = 5  # Frames before bed is considered stable
        
        # Distance monitoring and feedback
        self.enable_distance_monitoring = cfg.get("enable_distance_monitoring", True)
        self.distance_monitor = None
        if self.enable_distance_monitoring:
            try:
                from analytics.distance_monitor import DistanceMonitor
                self.distance_monitor = DistanceMonitor(
                    optimal_min=cfg.get("optimal_distance_min", 1.5),  # 150cm
                    optimal_max=cfg.get("optimal_distance_max", 3.0),  # 300cm
                    target=cfg.get("optimal_distance_target", 2.0),  # 200cm
                    too_close=cfg.get("too_close_threshold", 1.0),  # 100cm
                    too_far=cfg.get("too_far_threshold", 4.0)  # 400cm
                )
                log.info("Distance monitoring enabled (target: %dm)", cfg.get("optimal_distance_target", 2.0))
            except Exception as e:
                log.warning("Failed to initialize distance monitor: %s", e)
                self.distance_monitor = None

        # Display (optional)
        # TODO-025: Disable display in production (config flag)
        self.enable_display = bool(cfg.get("enable_display", True))
        self.disable_display_in_production = cfg.get("disable_display_in_production", False)
        
        # If production mode, disable display regardless of config
        if self.disable_display_in_production:
            self.enable_display = False
            log.info("Display disabled (production mode)")
        
        self.display = ICUMonitorDisplay(title="ICU Live Monitor") if self.enable_display else None
        self.display_enabled = self.enable_display  # Alias for backward compatibility
        
        # TODO-026: Reduce rendering frequency (every N frames)
        self.display_render_frequency = cfg.get("display_render_frequency", 1)  # Render every N frames (1 = every frame)
        self.display_frame_counter = 0


        # Control flag (display or external stop request)
        self.stop_requested = False

        # Lazy-initialized components (declared here to avoid hasattr checks)
        self.keypoint_smoother = None
        self.self_contact_detector = None
        self.posture_smoother = None
        self._enhanced_activity_classifier = None
        self._emotion_detector = None
        self._clinical_correlation_engine = None
        self._clinical_monitor = None  # Bed exit, fall risk, immobility detection
        self._kps_history = []

        # Parallel executor for classification tasks (Jetson optimization)
        # Uses ThreadPoolExecutor for I/O-bound tasks (analytics can release GIL)
        self.enable_parallel_classification = cfg.get("enable_parallel_classification", True)
        self._classification_executor = None
        if self.enable_parallel_classification:
            # 4 workers: activity, emotion, clinical_correlation, clinical_monitor
            self._classification_executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="classify"
            )
            log.info("Parallel classification enabled (4 workers)")

        # Patient baseline analyzer (DBSCAN + PCA per-patient baseline)
        self.baseline_analyzer = None
        self._baseline_save_counter = 0
        self._baseline_save_interval = cfg.get("baseline_save_interval", 1000)
        if cfg.get("enable_baseline_analyzer", True):
            try:
                from analytics.baseline_analyzer import PatientBaselineAnalyzer
                patient_cfg = cfg.get("patient", {})
                patient_id = patient_cfg.get("id", "unknown")
                device_id = cfg.get("device_id", "bed_01")
                baseline_cfg = cfg.get("baseline", {})
                self.baseline_analyzer = PatientBaselineAnalyzer(
                    patient_id=patient_id,
                    device_id=device_id,
                    buffer_size=baseline_cfg.get("buffer_size", 5000),
                    min_samples_for_fit=baseline_cfg.get("min_samples_for_fit", 300),
                    refit_interval=baseline_cfg.get("refit_interval", 500),
                    dbscan_eps=baseline_cfg.get("dbscan_eps", 0.5),
                    dbscan_min_samples=baseline_cfg.get("dbscan_min_samples", 10),
                    pca_components=baseline_cfg.get("pca_components", 6),
                    pca_reconstruction_threshold=baseline_cfg.get("pca_reconstruction_threshold", 1.5),
                    anomaly_persistence_frames=baseline_cfg.get("anomaly_persistence_frames", 15),
                )
                # Try loading existing baseline from database
                self._load_baseline_from_db(patient_id, device_id)
                log.info("Patient baseline analyzer enabled (patient=%s)", patient_id)
            except Exception as e:
                log.warning("Failed to initialize baseline analyzer: %s", e)
                self.baseline_analyzer = None

        # Activity temporal smoothing (TODO-070)
        try:
            from analytics.activity_smoother import ActivityStateMachine
            activity_threshold = cfg.get("activity_transition_threshold", 8)
            self.activity_smoother = ActivityStateMachine(transition_threshold=activity_threshold)
            log.info("Activity smoother initialized (threshold: %d frames)", activity_threshold)
        except Exception as e:
            log.debug("Activity smoother not available: %s", e)
            self.activity_smoother = None

        # ── Hourly Aggregator (per-frame → hourly summaries for Flink) ──
        self._hourly_aggregator = None
        try:
            from analytics.hourly_aggregator import HourlyAggregator
            device_id = cfg.get("device_id", "bed_01")
            patient_id = cfg.get("patient", {}).get("id", "unknown")
            self._hourly_aggregator = HourlyAggregator(
                device_id=device_id,
                patient_id=patient_id,
                auto_flush=False,  # eac.py controls flush timing
            )
            log.info("Hourly aggregator enabled (device=%s)", device_id)
        except Exception as e:
            log.warning("Failed to initialize hourly aggregator: %s", e)

        # ── Video2IMU converter (keypoints → IMU-like signals) ──────────
        self._video2imu = None
        if cfg.get("enable_video2imu", False):
            try:
                from pipeline.pose.video2imu_converter import Video2IMUConverter
                v2i_cfg = cfg.get("video2imu", {})
                self._video2imu = Video2IMUConverter(
                    window_size=window_size,
                    fps=fps,
                    lowpass_cutoff=v2i_cfg.get("lowpass_cutoff", 12.0),
                    gravity_removal=v2i_cfg.get("gravity_removal", True),
                    reference_point=v2i_cfg.get("reference_point", "hip"),
                )
                log.info("Video2IMU converter enabled (ref=%s)", v2i_cfg.get("reference_point", "hip"))
            except Exception as e:
                log.warning("Failed to initialize Video2IMU: %s", e)

        # ── Depth estimator (for 3D pose upgrade) ──────────────────────
        self.depth_estimator = None
        if cfg.get("use_3d_analysis", False):
            try:
                from pipeline.depth.depth_estimator import DepthEstimator
                depth_method = cfg.get("depth_method", "geometric")
                self.depth_estimator = DepthEstimator(method=depth_method)
                log.info("Depth estimator enabled (method=%s)", depth_method)
            except Exception as e:
                log.warning("Failed to initialize depth estimator: %s", e)

        # ── Jetson GPU Optimizer ──────────────────────────────────────
        self._jetson_optimizer = None
        if cfg.get("enable_jetson_optimizer", False):
            try:
                from pipeline.pose.jetson_gpu_optimizer import create_jetson_optimizer
                self._jetson_optimizer = create_jetson_optimizer({
                    "target_fps": fps,
                    "enable_thermal_management": True,
                    "enable_dynamic_precision": True,
                    "default_precision": "fp16",
                })
                self._jetson_optimizer.start()
                log.info("JetsonOptimizer enabled (target_fps=%d)", fps)
            except Exception as e:
                log.warning("Failed to initialize JetsonOptimizer: %s", e)

        # TODO-033: Model warmup
        self._warmup_models()

    # ------------------------------------------------------------------
    # Baseline persistence helpers
    # ------------------------------------------------------------------

    def _load_baseline_from_db(self, patient_id, device_id):
        """Load a previously saved baseline model from SQLite."""
        try:
            from storage.db import LocalDB
            db = LocalDB()
            record = db.load_baseline(patient_id, device_id)
            db.close()
            if record and self.baseline_analyzer:
                self.baseline_analyzer.load_from_bytes(record["model_blob"])
                log.info("Loaded baseline from DB for patient=%s", patient_id)
        except Exception as e:
            log.debug("No saved baseline found: %s", e)

    def get_hourly_aggregator(self):
        """Return the hourly aggregator (used by eac.py for Flink publishing)."""
        return self._hourly_aggregator

    def _save_baseline_to_db(self):
        """Persist current baseline model to SQLite."""
        if not self.baseline_analyzer or self.baseline_analyzer.state != "ready":
            return
        try:
            from storage.db import LocalDB
            db = LocalDB()
            summary = self.baseline_analyzer.get_summary()
            db.save_baseline(
                patient_id=self.baseline_analyzer.patient_id,
                device_id=self.baseline_analyzer.device_id,
                model_blob=self.baseline_analyzer.serialize(),
                n_samples=summary["total_samples"],
                n_clusters=summary["n_clusters"],
                fit_count=summary["fit_count"],
            )
            db.close()
        except Exception as e:
            log.warning("Failed to save baseline to DB: %s", e)

    def _parse_bbox(self, bbox, frame_shape):
        """
        Accepts either (x,y,w,h) or (x1,y1,x2,y2). Returns safely-clamped ints.
        """
        h, w = frame_shape[0], frame_shape[1]
        try:
            bbox = [int(v) for v in bbox]
        except Exception:
            return 0, 0, w, h

        if len(bbox) == 4:
            a, b, c, d = bbox
            # Detect common formats:
            # if c > a and d > b AND likely (x1,y1,x2,y2)
            if c > a and d > b and (c - a > 0 and d - b > 0) and (c > 1 and d > 1) and (c <= w and d <= h):
                x1, y1, x2, y2 = a, b, c, d
            else:
                # assume (x,y,w,h)
                x1, y1 = a, b
                x2, y2 = a + c, b + d
        else:
            # fallback full frame
            x1, y1, x2, y2 = 0, 0, w, h

        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0, 0, w, h
        return x1, y1, x2, y2
    
    def _compute_pose_change(self, current_kps, prev_kps):
        """
        Compute pose change metric between two keypoint sets.
        TODO-031: Adaptive Frame Skipping helper.
        
        Returns:
            float: Normalized pose change (0-1), higher = more change
        """
        if not prev_kps or not current_kps or len(current_kps) != len(prev_kps):
            return 1.0  # Maximum change if incomparable
        
        try:
            import numpy as np
            # Convert to numpy arrays
            current_array = np.array([[kp[0], kp[1]] for kp in current_kps if len(kp) >= 2], dtype=np.float32)
            prev_array = np.array([[kp[0], kp[1]] for kp in prev_kps if len(kp) >= 2], dtype=np.float32)
            
            if len(current_array) != len(prev_array) or len(current_array) == 0:
                return 1.0
            
            # Compute mean squared displacement
            displacement = np.mean(np.linalg.norm(current_array - prev_array, axis=1))
            
            # Normalize by average keypoint spread (to make threshold scale-invariant)
            current_spread = np.std(current_array, axis=0)
            avg_spread = np.mean(current_spread) if len(current_spread) > 0 else 1.0
            
            if avg_spread < 1e-6:
                return 0.0  # No movement if spread is zero
            
            normalized_change = displacement / (avg_spread + 1e-6)
            return float(normalized_change)
        except Exception as e:
            log.debug("Pose change computation failed: %s", e)
            return 1.0  # Assume maximum change on error

    def _compute_crop_hash(self, crop):
        """Compute MD5 hash of crop bytes for pose caching."""
        try:
            crop_bytes = crop.tobytes()[:self.pose_cache_hash_bytes]
            return hashlib.md5(crop_bytes).hexdigest()
        except Exception:
            return None

    def _get_kps_history(self, n):
        """Get last n keypoint frames from history."""
        if len(self.kps_window) > 0:
            return list(self.kps_window)[-n:]
        elif self.prev_kps:
            return [self.prev_kps]
        return []

    def _log_posture(self, state, confidence=None):
        """Log posture state with timestamp."""
        ts = time.time()
        dt = datetime.fromtimestamp(ts)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        if confidence is not None:
            log.info("POSTURE [%s]: %s (confidence: %.2f)", time_str, state, confidence)
        else:
            log.info("POSTURE [%s]: %s", time_str, state)

    def _extract_face_bbox(self, kps, frame_shape):
        """
        Extract face bounding box from COCO keypoints.
        Handles side and front camera angles typical of ICU bed-side setups.

        Args:
            kps: List of keypoints [(x, y, conf), ...]
            frame_shape: (H, W, C) frame shape

        Returns:
            [x, y, w, h] face bbox in pixel coordinates, or None if insufficient keypoints.
        """
        if not kps or len(kps) < 5:
            return None

        h, w = frame_shape[0], frame_shape[1]
        face_indices = [0, 1, 2, 3, 4]  # nose, L-eye, R-eye, L-ear, R-ear
        min_conf = 0.3

        visible = []
        for idx in face_indices:
            if idx < len(kps) and kps[idx] is not None and len(kps[idx]) >= 3:
                kx, ky, kc = kps[idx][0], kps[idx][1], kps[idx][2]
                if kc > min_conf:
                    # Convert normalized coords to pixel coords if needed
                    if 0.0 <= kx <= 1.0 and 0.0 <= ky <= 1.0:
                        px, py = kx * w, ky * h
                    else:
                        px, py = kx, ky
                    visible.append((px, py))

        if len(visible) < 2:
            return None

        xs = [p[0] for p in visible]
        ys = [p[1] for p in visible]

        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)

        # Pad to ensure full face capture (side views have asymmetric spread)
        pad = max(spread_x, spread_y, 30) * 1.5
        fx1 = int(max(0, cx - pad))
        fy1 = int(max(0, cy - pad))
        fx2 = int(min(w, cx + pad))
        fy2 = int(min(h, cy + pad))

        bw = fx2 - fx1
        bh = fy2 - fy1
        if bw < 10 or bh < 10:
            return None

        return [fx1, fy1, bw, bh]

    def _compute_movement_features(self, kps):
        """
        Compute movement features (motion energy, jerk index) from keypoint history.

        Returns:
            dict with motion_energy (0-1) and jerk_index (0-1).
        """
        self._kps_history.append(kps)
        max_history = 15  # ~1 second at 15 FPS
        if len(self._kps_history) > max_history:
            self._kps_history.pop(0)

        if len(self._kps_history) < 3:
            return {"motion_energy": 0.0, "jerk_index": 0.0}

        # Compute displacements between consecutive frames
        displacements = []
        for i in range(1, len(self._kps_history)):
            curr = self._kps_history[i]
            prev = self._kps_history[i - 1]
            if not curr or not prev or len(curr) != len(prev):
                continue
            disp = 0.0
            count = 0
            for c, p in zip(curr, prev):
                if c is None or p is None or len(c) < 3 or len(p) < 3:
                    continue
                if c[2] > 0.3 and p[2] > 0.3:
                    dx = c[0] - p[0]
                    dy = c[1] - p[1]
                    disp += (dx * dx + dy * dy) ** 0.5
                    count += 1
            if count > 0:
                displacements.append(disp / count)

        if not displacements:
            return {"motion_energy": 0.0, "jerk_index": 0.0}

        # Motion energy: mean displacement (normalized)
        motion_energy = min(1.0, np.mean(displacements) / 0.05)

        # Jerk index: variance of displacement changes (smoothness of motion)
        jerk_index = 0.0
        if len(displacements) >= 3:
            accels = [displacements[i] - displacements[i - 1] for i in range(1, len(displacements))]
            jerk_index = min(1.0, float(np.std(accels)) / 0.02)

        return {
            "motion_energy": float(motion_energy),
            "jerk_index": float(jerk_index),
        }

    # ══════════════════════════════════════════════════════════════════════
    # PARALLEL CLASSIFICATION TASK HELPERS
    # These methods are designed to run in ThreadPoolExecutor
    # ══════════════════════════════════════════════════════════════════════

    def _task_distance_monitoring(self, kps, kps_3d, x1, y1, x2, y2, frame_shape):
        """Task: Distance monitoring (thread-safe)."""
        distance_info = None
        distance_feedback = None
        try:
            distance = 0.0
            if kps_3d:
                distance = self.distance_monitor.estimate_distance_from_3d(kps_3d)
            else:
                distance = self.distance_monitor.estimate_distance_from_keypoints(
                    kps, bbox=[x1, y1, x2, y2], frame_shape=frame_shape
                )

            if distance > 0:
                distance_info = self.distance_monitor.check_distance(distance)
                distance_feedback = self.distance_monitor.get_feedback(distance, force=False)
        except Exception as e:
            log.debug("Distance monitoring task error: %s", e)
        return distance_info, distance_feedback

    def _task_emotion_detection(self, frame, kps):
        """Task: Emotion detection (thread-safe)."""
        emotions = {}
        try:
            if self._emotion_detector is None:
                from analytics.emotion_detector import EmotionDetector
                method = self.cfg.get("emotion_detection_method", "geometric")
                self._emotion_detector = EmotionDetector(method=method)
                log.info("Emotion detection enabled (method: %s)", method)

            if self._emotion_detector is not None:
                face_bbox = self._extract_face_bbox(kps, frame.shape)
                emotions = self._emotion_detector.detect_emotions(frame, face_bbox) or {}
        except Exception as e:
            log.debug("Emotion detection task error: %s", e)
        return emotions

    def _task_frame_visibility(self, kps, x1, y1, x2, y2, frame_shape_2d):
        """Task: Frame visibility analysis (thread-safe)."""
        try:
            from analytics.frame_visibility import analyze_frame_visibility
            return analyze_frame_visibility(
                kps, bbox=[x1, y1, x2, y2], frame_shape=frame_shape_2d
            )
        except Exception as e:
            log.debug("Frame visibility task error: %s", e)
            return {"completeness_score": 1.0, "visibility_type": "full"}

    def _task_clinical_correlation(self, posture_state, kps_3d, emotions,
                                   movement_features, frame_visibility,
                                   distance_info, self_contact_signature):
        """Task: Clinical correlation (pain, agitation, dizziness)."""
        clinical_correlation = None
        try:
            if self._clinical_correlation_engine is None:
                if self.cfg.get("enable_clinical_correlation", True):
                    from analytics.clinical_correlation import ClinicalCorrelationEngine
                    self._clinical_correlation_engine = ClinicalCorrelationEngine()
                    log.info("Clinical correlation engine enabled")

            if self._clinical_correlation_engine is not None:
                dist_meters = 2.0
                if distance_info:
                    dist_meters = distance_info.get(
                        "distance_meters",
                        distance_info.get("distance_cm", 200) / 100.0
                    )

                clinical_correlation = self._clinical_correlation_engine.correlate_clinical_state(
                    posture_state=posture_state,
                    posture_3d=kps_3d,
                    emotions=emotions,
                    movement_features=movement_features,
                    frame_visibility=frame_visibility,
                    distance=dist_meters,
                    self_contact=self_contact_signature,
                )
        except Exception as e:
            log.debug("Clinical correlation task error: %s", e)
        return clinical_correlation

    def _task_clinical_monitor(self, posture_state, posture_confidence,
                               support_surface_info, activity_state,
                               activity_confidence, movement_features, baseline_info):
        """Task: Clinical monitor (bed exit, fall risk, immobility)."""
        clinical_alerts = []
        try:
            if self._clinical_monitor is None:
                if self.cfg.get("enable_clinical_monitor", True):
                    from analytics.clinical_monitor import ClinicalMonitor
                    self._clinical_monitor = ClinicalMonitor(
                        window_size=self.cfg.get("clinical_window_size", 300),
                        alert_cooldown=self.cfg.get("alert_cooldown", 60)
                    )
                    log.info("Clinical monitor enabled (bed exit, fall risk, immobility)")

            if self._clinical_monitor is not None:
                pose_result = {
                    "posture": posture_state,
                    "posture_confidence": posture_confidence,
                    "support_surface": support_surface_info,
                    "support_confidence": support_surface_info.get("confidence", 0.0) if support_surface_info else 0.0,
                    "pelvis_height_norm": movement_features.get("pelvis_height", 0.5) if movement_features else 0.5,
                    "vertical_velocity": movement_features.get("vertical_velocity", 0.0) if movement_features else 0.0,
                }
                activity_result = {
                    "activity": activity_state,
                    "confidence": activity_confidence,
                }

                clinical_alerts = self._clinical_monitor.update_clinical_state(
                    pose_result=pose_result,
                    activity_result=activity_result,
                    baseline_info=baseline_info
                )
        except Exception as e:
            log.debug("Clinical monitor task error: %s", e)
        return clinical_alerts

    def _warmup_models(self, num_warmup_frames=10):
        """
        Warmup models with dummy data to eliminate cold start latency.
        TODO-033: Model warmup
        """
        try:
            log.info("Warming up models (%d frames)...", num_warmup_frames)
            dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            for i in range(num_warmup_frames):
                try:
                    # Warmup detector
                    self.det.infer(dummy_frame)
                    # Warmup pose
                    self.pose.infer(dummy_frame)
                    # Warmup temporal model (if enough frames)
                    if i >= 8:
                        dummy_feat = np.random.randn(self.temporal.window_size, 9).astype(np.float32)
                        self.temporal.predict(dummy_feat)
                except Exception as e:
                    log.debug("Warmup frame %d failed: %s", i, e)
            
            log.info("Model warmup complete")
        except Exception as e:
            log.warning("Model warmup failed: %s", e)

    def run_once(self):
        """
        Run single inference step. Returns result dict or None (skip).
        Non-fatal exceptions are logged and return None.
        """
        if self.stop_requested:
            return None

        # TODO-030: Frame Rate Control
        if self.enable_frame_rate_control:
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            target_frame_time = 1.0 / self.target_fps
            
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
            
            self.last_frame_time = time.time()

        st = time.time()
        _stage_times = {}  # Per-stage profiling

        # Jetson thermal/memory protection — skip frame if overloaded
        if self._jetson_optimizer and self._jetson_optimizer.should_skip_frame():
            log.debug("Frame skipped by JetsonOptimizer (thermal/memory)")
            return None

        try:
            _t0 = time.time()
            frame = self.camera.read()
            _stage_times["camera_read"] = (time.time() - _t0) * 1000.0
            if frame is None:
                log.warning("Camera read returned None")
                return None

            # Bed detection (for context and zoom control)
            bed_info = None
            if self.enable_bed_detection and self.bed_detector:
                try:
                    beds = self.bed_detector.detect_beds(frame)
                    if beds:
                        bed_info = beds[0]  # Use primary bed
                        self.bed_detection_frames += 1
                        
                        # Consider bed stable after threshold frames
                        if self.bed_detection_frames >= self.bed_stable_threshold:
                            self.current_bed = bed_info
                            
                            # Auto-zoom to bed region if enabled (will be overridden if person detected)
                            if self.enable_auto_zoom:
                                bed_region = self.bed_detector.get_bed_region_for_zoom(
                                    bed_bbox=bed_info["bbox"],
                                    frame=frame,
                                    padding=0.2
                                )
                                if bed_region:
                                    self.camera.auto_zoom_to_bed(bed_region, padding=0.2)
                    else:
                        self.bed_detection_frames = 0
                        self.current_bed = None
                except Exception as e:
                    log.debug("Bed detection error: %s", e)

            _t0 = time.time()
            dets = self.det.infer(frame)
            _stage_times["detection"] = (time.time() - _t0) * 1000.0
            if not dets:
                log.debug("No person detected — skipping frame")
                return None

            # Multi-modal patient selection (track_id persistence + face recognition + size)
            det = None
            patient_id = self.cfg.get("patient", {}).get("id")
            
            # Step 1: Check if any detection has patient's persistent track_id
            if self.patient_track_id is not None:
                for candidate_det in dets:
                    if candidate_det.get("track_id") == self.patient_track_id:
                        det = candidate_det
                        log.debug("Patient found via persistent track_id: %d", self.patient_track_id)
                        self.patient_missing_frames = 0  # Reset missing counter
                        break
            
            # Step 2: If no track_id match, try face recognition (if enabled)
            if det is None and len(dets) > 1 and self.face_recognizer and self.face_recognizer.enabled and patient_id:
                best_match = None
                best_confidence = 0.0
                
                for candidate_det in dets:
                    try:
                        verified, conf, metadata = self.face_recognizer.verify_patient(
                            frame, candidate_det.get("bbox", []), patient_id, threshold=0.6,
                            liveness_detector=self.face_recognizer.liveness_detector,
                            rate_limiter=self.face_recognizer.rate_limiter
                        )
                        if verified and conf > best_confidence:
                            best_confidence = conf
                            best_match = candidate_det
                            if metadata.get("liveness_passed"):
                                log.debug("Patient verified with liveness check (confidence: %.2f)", conf)
                    except Exception as e:
                        log.debug("Face verification failed for detection: %s", e)
                        continue
                
                if best_match and best_confidence > 0.7:
                    det = best_match
                    # Onboard patient with this track_id if not already onboarded
                    if not self.patient_onboarded and det.get("track_id") is not None:
                        self.patient_track_id = det.get("track_id")
                        self.patient_onboarded = True
                        log.info("Patient onboarded with track_id: %d (face recognition)", self.patient_track_id)
                    log.debug("Patient selected via face recognition (confidence: %.2f)", best_confidence)
            
            # Step 3: Fallback to size-based selection (if no track_id or face match)
            if det is None:
                def calc_bbox_area(bbox):
                    if len(bbox) < 4:
                        return 0.0
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Likely [x1,y1,x2,y2]
                        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    else:  # Likely [x,y,w,h]
                        return bbox[2] * bbox[3] if len(bbox) >= 4 else 0.0
                
                det = max(dets, key=lambda d: d.get("score", 0.0) * calc_bbox_area(d.get("bbox", [])))
                
                # Onboard patient with this track_id if not already onboarded
                if not self.patient_onboarded and det.get("track_id") is not None:
                    self.patient_track_id = det.get("track_id")
                    self.patient_onboarded = True
                    log.info("Patient onboarded with track_id: %d (size-based selection)", self.patient_track_id)
                
                log.debug("Patient selected via size (face recognition failed or not confident)")
            
            # Step 4: Update track_id history and handle missing patient
            current_track_id = det.get("track_id") if det else None
            
            if current_track_id == self.patient_track_id:
                # Patient found - reset missing counter
                self.patient_missing_frames = 0
            elif self.patient_track_id is not None:
                # Patient track_id not found - increment missing counter
                self.patient_missing_frames += 1
                if self.patient_missing_frames > self.patient_missing_threshold_verified:
                    log.warning("Patient missing for %d frames - may have left bed or occluded", 
                              self.patient_missing_frames)
                    # Consider patient as missing (but don't reset track_id yet)
            
            # Update track_id history (keep last 10 track_ids)
            if current_track_id is not None:
                self.track_id_history.append(current_track_id)
                if len(self.track_id_history) > 10:
                    self.track_id_history.pop(0)
            bbox = det.get("bbox", [0, 0, frame.shape[1], frame.shape[0]])
            x1, y1, x2, y2 = self._parse_bbox(bbox, frame.shape)
            
            # Auto-zoom to person if enabled
            if self.enable_auto_zoom:
                try:
                    self.camera.auto_zoom_to_person(
                        person_bbox=[x1, y1, x2, y2],
                        frame=frame,
                        target_size_ratio=self.auto_zoom_target_size
                    )
                except Exception as e:
                    log.debug("Auto-zoom to person failed: %s", e)
            
            # Check person-bed relationship
            person_on_bed = False
            person_near_bed = False
            if self.bed_detector and bed_info:
                try:
                    person_on_bed = self.bed_detector.is_person_on_bed(
                        person_bbox=[x1, y1, x2, y2],
                        bed_bbox=bed_info.get("bbox"),
                        frame=frame,
                        overlap_threshold=0.3
                    )
                    person_near_bed = self.bed_detector.is_person_near_bed(
                        person_bbox=[x1, y1, x2, y2],
                        bed_bbox=bed_info.get("bbox"),
                        frame=frame,
                        threshold=0.3
                    )
                except Exception as e:
                    log.debug("Person-bed relationship check failed: %s", e)
            
            crop = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else frame
            
            # Validate crop size (edge case: zero-size or too small crop)
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                log.debug("Crop too small (%dx%d), using full frame", crop.shape[1] if crop.size > 0 else 0, crop.shape[0] if crop.size > 0 else 0)
                crop = frame

            # pose inference (pose returns normalized / crop coords)
            # TODO-006: Skip pose when no person (already handled by det check above)
            # TODO-008: Cache pose results for static poses
            _t0 = time.time()
            kps = None
            if self.enable_pose_caching and self.last_cached_pose is not None:
                # Check if crop is identical using MD5 hash
                crop_hash = self._compute_crop_hash(crop)
                if crop_hash and crop_hash == self.last_cached_pose_hash:
                    kps = self.last_cached_pose
                    log.debug("Using cached pose (static crop)")

            if kps is None:
                # Run pose estimation
                kps = self.pose.infer(crop)
                if not kps:
                    log.debug("Pose not detected — skipping frame")
                    return None

                # Check if pose has changed significantly (for caching)
                if self.enable_pose_caching and self.last_processed_kps is not None:
                    pose_change = self._compute_pose_change(kps, self.last_processed_kps)
                    if pose_change < self.pose_cache_threshold:
                        self.last_cached_pose = kps
                        self.last_cached_pose_hash = self._compute_crop_hash(crop)
                        log.debug("Cached pose (change: %.4f < %.4f)", pose_change, self.pose_cache_threshold)
                    else:
                        self.last_cached_pose = None
                        self.last_cached_pose_hash = None
                else:
                    self.last_cached_pose = kps
                    self.last_cached_pose_hash = self._compute_crop_hash(crop)
            
            _stage_times["pose"] = (time.time() - _t0) * 1000.0

            # TODO-060: Keypoint smoothing with SC3D
            if self.keypoint_smoother is None:
                from pipeline.pose.keypoint_smoother import KeypointSmoother
                use_sc3d = self.cfg.get("enable_self_contact_detection", True)
                self.keypoint_smoother = KeypointSmoother(
                    alpha=self.cfg.get("keypoint_smoothing_alpha", 0.7),
                    use_self_contact=use_sc3d
                )
            
            # Smooth keypoints (will use 3D if available)
            kps_smoothed = kps  # Will be updated after 3D estimation
            
            # Validate keypoints (clinical-grade robustness)
            try:
                from pipeline.pose.keypoint_validator import create_keypoint_validator
                validator = create_keypoint_validator()
                kps = validator.validate(kps)
                if not kps:
                    log.debug("Keypoint validation failed — skipping frame")
                    return None
            except Exception as e:
                log.debug("Keypoint validation error: %s", e)
                # Continue with original keypoints if validation fails
            
            # Upgrade to 3D pose if enabled (Phase 2)
            kps_3d = None
            self_contact_signature = None
            use_3d_pose = self.cfg.get("use_3d_pose_estimation", False)
            if use_3d_pose:
                try:
                    from pipeline.pose.pose3d_estimator import upgrade_pose_to_3d
                    from pipeline.depth.depth_estimator import DepthEstimator
                    
                    # Get depth map if available
                    depth_map = None
                    if hasattr(self, 'depth_estimator') and self.depth_estimator:
                        depth_map = self.depth_estimator.estimate_depth_map(frame)
                    
                    # Get camera intrinsics
                    camera_intrinsics = None
                    if hasattr(self, 'camera_intrinsics'):
                        camera_intrinsics = self.camera_intrinsics
                    
                    # Upgrade to 3D
                    pose3d_method = self.cfg.get("pose3d_method", "geometric")
                    use_bone_constraints = self.cfg.get("use_bone_constraints", True)
                    
                    # Create estimator with bone constraints setting
                    from pipeline.pose.pose3d_estimator import Pose3DEstimator
                    estimator = Pose3DEstimator(
                        method=pose3d_method,
                        use_bone_constraints=use_bone_constraints
                    )
                    kps_3d = estimator.estimate_3d(
                        kps,
                        depth_map=depth_map,
                        camera_intrinsics=camera_intrinsics
                    )
                    log.debug("Upgraded to 3D pose (method: %s, bone_constraints: %s)", 
                             pose3d_method, use_bone_constraints)
                    
                    # TODO-060: Apply keypoint smoothing with 3D contact info
                    if self.keypoint_smoother:
                        kps_smoothed = self.keypoint_smoother.smooth(kps, kps_3d)
                        kps = kps_smoothed  # Use smoothed keypoints
                    
                    # TODO-065: Detect self-contact for activity classification
                    self_contact_signature = None
                    if self.cfg.get("enable_self_contact_detection", True):
                        try:
                            from pipeline.pose.self_contact_detector import SelfContactDetector
                            if self.self_contact_detector is None:
                                self.self_contact_detector = SelfContactDetector()
                            self_contact_signature = self.self_contact_detector.detect(kps_3d)
                        except Exception as e:
                            log.debug("Self-contact detection failed: %s", e)
                except Exception as e:
                    log.debug("3D pose upgrade failed: %s", e)
                    # Still apply 2D smoothing if available
                    if self.keypoint_smoother:
                        kps_smoothed = self.keypoint_smoother.smooth(kps, None)
                        kps = kps_smoothed

            # Extract features (handcrafted or learned)
            _t0 = time.time()
            if hasattr(self.feature_encoder, 'extract_features'):
                # Learned or hybrid feature extractor
                # TODO-040: Frame buffer management (deque auto-bounds)
                self.kps_window.append(kps)  # deque automatically manages size
                
                feat = self.feature_encoder.extract_features(
                    self.kps_window,
                    prev_kps=self.prev_kps,
                    prev_prev_kps=self.prev_prev_kps
                )
            else:
                # Handcrafted feature extractor
                feat = self.feature_encoder.extract_feature_vector(kps, prev_kps=self.prev_kps, prev_prev_kps=self.prev_prev_kps)

            # validate features
            if feat is not None:
                feat = np.asarray(feat, dtype=np.float32)
                if np.isnan(feat).any():
                    log.warning("Feature vector contains NaN — skipping append")
                else:
                    # TODO-040: Frame buffer management (deque auto-bounds)
                    self.window.append(feat)  # deque automatically manages size

            _stage_times["features"] = (time.time() - _t0) * 1000.0

            # --- Video2IMU: update converter with keypoints ---
            imu_signals = None
            if self._video2imu:
                try:
                    imu_signals = self._video2imu.update(kps)
                except Exception as e:
                    log.debug("Video2IMU update failed: %s", e)

            # --- Patient baseline: update + analyze ---
            baseline_info = None
            if self.baseline_analyzer and feat is not None and not np.isnan(feat).any():
                try:
                    self.baseline_analyzer.update(feat)
                    baseline_info = self.baseline_analyzer.analyze(feat)

                    # Periodic persistence to SQLite
                    self._baseline_save_counter += 1
                    if self._baseline_save_counter >= self._baseline_save_interval:
                        self._baseline_save_counter = 0
                        self._save_baseline_to_db()
                except Exception as e:
                    log.debug("Baseline analysis failed: %s", e)

            # update kps history
            self.prev_prev_kps = self.prev_kps
            self.prev_kps = kps

            # Temporal model prediction when enough frames
            _t0 = time.time()
            label, conf, probs, uncertainty = ("normal", 1.0, [1.0], 0.0)
            if len(self.window) >= max(8, self.temporal.window_size // 4):
                feat_win = np.stack(self.window[-self.temporal.window_size :])  # (T,F)

                # Validate features (edge case: NaN/Inf values)
                if np.isnan(feat_win).any() or np.isinf(feat_win).any():
                    log.warning("NaN/Inf values in feature window, skipping prediction")
                    label, conf, probs, uncertainty = ("unknown", 0.0, [0.0], 1.0)
                else:
                    label, conf, probs, uncertainty = self.temporal.predict(feat_win)

                # If temporal model is untrained, use neutral defaults so decision tree dominates
                if label == "untrained":
                    label, conf, probs = ("normal", 0.0, [1.0 / len(self.temporal.labels)] * len(self.temporal.labels))
                    log.debug("Temporal model untrained - relying on decision tree")

            _stage_times["temporal"] = (time.time() - _t0) * 1000.0
            inference_ms = (time.time() - st) * 1000.0

            # FPS calculation
            self.fps_frame_count += 1
            elapsed = time.time() - self.last_fps_time
            if elapsed >= 1.0:
                self.fps = self.fps_frame_count / elapsed
                self.fps_frame_count = 0
                self.last_fps_time = time.time()

            # Instant posture classification using new unified PostureSystem
            # (includes support surface detection, temporal hysteresis, proper confidence)
            posture_state = "unknown"
            posture_analysis = None
            posture_confidence = 0.0
            support_surface_info = None

            try:
                # Try new unified posture system first (enterprise-grade)
                from analytics.posture_system import get_posture_system

                if not hasattr(self, '_posture_system') or self._posture_system is None:
                    self._posture_system = get_posture_system(
                        stability_window=self.cfg.get("posture_stability_window", 8),
                        state_persistence_time=self.cfg.get("posture_persistence_time", 0.8),
                    )
                    log.info("Unified PostureSystem initialized (support surface + temporal hysteresis)")

                # Get person ID for tracking (use track_id or generate one)
                person_id = f"person_{current_track_id}" if current_track_id else "person_default"

                # Process frame through unified posture system
                posture_result = self._posture_system.process_frame(
                    person_id=person_id,
                    keypoints=kps,
                    frame_shape=frame.shape[:2],
                    depth_map=None,  # Will use keypoint-based depth estimation
                )

                posture_state = posture_result.get("posture", "UNKNOWN")
                posture_confidence = posture_result.get("confidence", 0.0)
                support_surface_info = {
                    "surface_id": posture_result.get("support_surface_id"),
                    "surface_type": posture_result.get("support_surface_type"),
                    "support": posture_result.get("support"),
                }

                # Build posture analysis dict for backward compatibility
                posture_analysis = {
                    "posture_state": posture_state,
                    "confidence": posture_confidence,
                    "subtype": posture_result.get("subtype"),
                    "temporal_stability": posture_result.get("temporal_stability", 0.0),
                    "time_in_state": posture_result.get("time_in_state", 0.0),
                    "support_surface": support_surface_info,
                    "raw_state": posture_result.get("raw_state"),
                }

                self._log_posture(posture_state, posture_confidence)

            except ImportError:
                # Fallback to legacy posture system if new one not available
                log.debug("New posture system not available, using legacy")
                try:
                    from analytics.posture import analyze_posture, classify_posture_state
                    from analytics.posture_smoother import PostureStateMachine

                    posture_state = classify_posture_state(kps, use_strict_thresholds=True)
                    self._log_posture(posture_state)

                    if self.posture_smoother is None:
                        smoothing_frames = self.cfg.get("posture_smoothing_frames", 10)
                        transition_threshold = self.cfg.get("posture_transition_threshold", 5)
                        self.posture_smoother = PostureStateMachine(
                            transition_threshold=transition_threshold,
                            history_size=smoothing_frames
                        )

                    posture_state = self.posture_smoother.update(posture_state)
                    posture_analysis = analyze_posture(kps, features=feat)
                    posture_state = posture_analysis.get("posture_state", posture_state)
                    posture_confidence = posture_analysis.get("confidence", 0.0)
                    self._log_posture(posture_state, posture_confidence)
                except Exception as e:
                    log.debug("Legacy posture classification error: %s", e)

            except Exception as e:
                log.debug("Posture classification error: %s", e)
                posture_state = "UNKNOWN"
                posture_confidence = 0.0

            # Fall detection (single call after posture is available)
            fall_detected = False
            fall_result = None
            try:
                from analytics.fall_detection import detect_patient_fall
                kps_history_for_fall = self._get_kps_history(5)

                fall_result = detect_patient_fall(
                    kps,
                    kps_history_for_fall,
                    posture_state if posture_state != "unknown" else None,
                    frame.shape
                )
                fall_detected = fall_result.get('fall_detected', False)

                if fall_detected:
                    log.critical("FALL DETECTED! Confidence: %.2f, Indicators: %d",
                               fall_result.get('confidence', 0.0),
                               len(fall_result.get('indicators', [])))
            except Exception as e:
                log.debug("Fall detection failed: %s", e)
            
            # Enhanced Activity classification (all 53 activities)
            activity_state = "unknown"
            activity_confidence = 0.0
            activity_priority = "MEDIUM"
            
            # TODO-022: Run activity classification every N frames
            self.activity_classification_frame_counter += 1
            should_classify_activity = (self.activity_classification_frame_counter % 
                                       self.activity_classification_frequency == 0 or 
                                       self.activity_classification_frequency == 1)
            
            if should_classify_activity:
                try:
                    # Try enhanced classifier first (supports all 53 activities)
                    try:
                        from analytics.enhanced_activity_classifier import EnhancedActivityClassifier
                        if self._enhanced_activity_classifier is None:
                            self._enhanced_activity_classifier = EnhancedActivityClassifier()
                            log.info("Enhanced activity classifier initialized (53 activities supported)")
                        use_enhanced = True
                    except ImportError as e:
                        from analytics.activity import classify_activity
                        use_enhanced = False
                        log.debug("Enhanced classifier not available, using basic: %s", e)

                    kps_history_for_activity = self._get_kps_history(10)

                    if use_enhanced:
                        # Get self-contact signature if available
                        contact_signature = None
                        if self.self_contact_detector and kps_3d:
                            try:
                                contact_signature = self.self_contact_detector.detect(kps_3d)
                            except Exception as e:
                                log.debug("Self-contact detection for activity failed: %s", e)

                        activity_result = self._enhanced_activity_classifier.classify_activity(
                            kps=kps,
                            kps_history=kps_history_for_activity or None,
                            posture_state=posture_state,
                            bed_info=bed_info,
                            person_on_bed=person_on_bed,
                            fall_detected=fall_detected,
                            frame=frame,
                            bbox=[x1, y1, x2, y2],
                            kps_3d=kps_3d,
                            contact_signature=contact_signature
                        )
                        activity_state = activity_result.get("activity", "unknown")
                        activity_confidence = activity_result.get("confidence", 0.0)
                        activity_priority = activity_result.get("priority", "MEDIUM")

                        # TODO-070: Apply temporal smoothing
                        if self.activity_smoother:
                            activity_state = self.activity_smoother.update(activity_state, activity_confidence)
                    else:
                        activity_result = classify_activity(kps, kps_history=kps_history_for_activity or None)
                        activity_state = activity_result.get("activity", "unknown")
                        activity_confidence = activity_result.get("confidence", 0.0)
                        activity_priority = "NORMAL"

                    # Cache result for skipped frames (shared across both branches)
                    self.last_activity_result = {
                        "activity": activity_state,
                        "confidence": activity_confidence,
                        "priority": activity_priority
                    }
                except Exception as e:
                    log.debug("Activity classification error: %s", e)
                    activity_state = "unknown"
                    activity_confidence = 0.0
                    activity_priority = "MEDIUM"
            else:
                # Use cached result from last classification
                if self.last_activity_result:
                    activity_state = self.last_activity_result.get("activity", "unknown")
                    activity_confidence = self.last_activity_result.get("confidence", 0.0)
                    activity_priority = self.last_activity_result.get("priority", "MEDIUM")
                    log.debug("Skipping activity classification (frame %d, frequency: %d)", 
                             self.activity_classification_frame_counter, self.activity_classification_frequency)
                else:
                    # No cached result yet, use defaults
                    activity_state = "unknown"
                    activity_confidence = 0.0
                    activity_priority = "MEDIUM"
            
            # Collect training data for future GRU training (opt-in)
            if (self.training_collector and self.training_collector.enabled
                    and len(self.window) >= self.temporal.window_size
                    and activity_state != "unknown"):
                feat_win_for_collection = np.stack(self.window[-self.temporal.window_size:])
                self.training_collector.collect(
                    feat_window=feat_win_for_collection,
                    label=activity_state,
                    confidence=activity_confidence,
                    metadata={"posture": posture_state},
                )

            # ══════════════════════════════════════════════════════════════════
            # PARALLEL CLASSIFICATION TASKS (Jetson GPU/CPU optimization)
            # These tasks are independent and can run concurrently
            # ══════════════════════════════════════════════════════════════════

            # Prepare shared context for parallel tasks
            parallel_context = {
                "kps": kps,
                "kps_3d": kps_3d,
                "frame": frame,
                "bbox": [x1, y1, x2, y2],
                "posture_state": posture_state,
                "posture_confidence": posture_confidence,
                "support_surface_info": support_surface_info,
                "activity_state": activity_state,
                "activity_confidence": activity_confidence,
                "baseline_info": baseline_info,
                "self_contact_signature": self_contact_signature,
            }

            # Initialize results with defaults
            distance_info = None
            distance_feedback = None
            emotions = {}
            frame_visibility = {"completeness_score": 1.0, "visibility_type": "full"}
            movement_features = {}
            clinical_correlation = None
            clinical_alerts = []

            # Movement features must run first (used by clinical monitor)
            try:
                movement_features = self._compute_movement_features(kps)
            except Exception as e:
                log.debug("Movement features extraction failed: %s", e)

            if self.enable_parallel_classification and self._classification_executor:
                # ── PARALLEL EXECUTION PATH ──────────────────────────────────
                futures = {}

                # Task 1: Distance monitoring
                if self.distance_monitor:
                    futures["distance"] = self._classification_executor.submit(
                        self._task_distance_monitoring, kps, kps_3d, x1, y1, x2, y2, frame.shape
                    )

                # Task 2: Emotion detection
                futures["emotion"] = self._classification_executor.submit(
                    self._task_emotion_detection, frame, kps
                )

                # Task 3: Frame visibility
                futures["visibility"] = self._classification_executor.submit(
                    self._task_frame_visibility, kps, x1, y1, x2, y2, frame.shape[:2]
                )

                # Task 4: Clinical correlation (depends on emotion, visibility, movement)
                # We'll run this after collecting dependent results

                # Collect parallel results with timeout (20ms max wait)
                for task_name, future in futures.items():
                    try:
                        result = future.result(timeout=0.02)
                        if task_name == "distance":
                            distance_info, distance_feedback = result
                        elif task_name == "emotion":
                            emotions = result or {}
                        elif task_name == "visibility":
                            frame_visibility = result or {"completeness_score": 1.0, "visibility_type": "full"}
                    except Exception as e:
                        log.debug("Parallel task %s failed: %s", task_name, e)

                # Clinical correlation (now has dependencies)
                try:
                    clinical_correlation = self._task_clinical_correlation(
                        posture_state, kps_3d, emotions, movement_features,
                        frame_visibility, distance_info, self_contact_signature
                    )
                except Exception as e:
                    log.debug("Clinical correlation failed: %s", e)

                # Clinical monitor (depends on activity_state, movement_features)
                try:
                    clinical_alerts = self._task_clinical_monitor(
                        posture_state, posture_confidence, support_surface_info,
                        activity_state, activity_confidence, movement_features, baseline_info
                    )
                except Exception as e:
                    log.debug("Clinical monitor failed: %s", e)

            else:
                # ── SEQUENTIAL EXECUTION PATH (fallback) ─────────────────────
                # Distance monitoring
                if self.distance_monitor:
                    try:
                        distance_info, distance_feedback = self._task_distance_monitoring(
                            kps, kps_3d, x1, y1, x2, y2, frame.shape
                        )
                    except Exception as e:
                        log.debug("Distance monitoring error: %s", e)

                # Emotion detection
                try:
                    emotions = self._task_emotion_detection(frame, kps) or {}
                except Exception as e:
                    log.debug("Emotion detection failed: %s", e)

                # Frame visibility
                try:
                    frame_visibility = self._task_frame_visibility(kps, x1, y1, x2, y2, frame.shape[:2])
                except Exception as e:
                    log.debug("Frame visibility analysis failed: %s", e)

                # Clinical correlation
                try:
                    clinical_correlation = self._task_clinical_correlation(
                        posture_state, kps_3d, emotions, movement_features,
                        frame_visibility, distance_info, self_contact_signature
                    )
                except Exception as e:
                    log.debug("Clinical correlation failed: %s", e)

                # Clinical monitor
                try:
                    clinical_alerts = self._task_clinical_monitor(
                        posture_state, posture_confidence, support_surface_info,
                        activity_state, activity_confidence, movement_features, baseline_info
                    )
                except Exception as e:
                    log.debug("Clinical monitor failed: %s", e)

            # Log distance feedback if present
            if distance_feedback:
                log.info("Distance feedback: %s", distance_feedback["message"])

            # Log clinical correlation scores
            if clinical_correlation:
                if clinical_correlation.get("pain_score", 0.0) > 0.5:
                    log.info("High pain score: %.2f", clinical_correlation["pain_score"])
                if clinical_correlation.get("agitation_score", 0.0) > 0.5:
                    log.info("High agitation score: %.2f", clinical_correlation["agitation_score"])

            # Log clinical alerts
            for alert in clinical_alerts:
                if alert.get("severity") == "high":
                    log.warning("CLINICAL ALERT [%s]: %s", alert.get("type"), alert.get("message"))
                elif alert.get("type") == "bed_exit":
                    log.info("BED EXIT detected: %s", alert.get("message"))

            # Apply decision engine (blends ML + clinical features)
            _t0 = time.time()
            try:
                decision = apply_rules(
                    label, probs, kps,
                    features=feat,
                    posture_state=posture_state,
                    patient_cfg=self.cfg.get("patient"),
                    person_present=True
                )
            except Exception:
                log.exception("Decision engine error - falling back to ML label")
                decision = {"label": label, "confidence": conf, "posture_state": posture_state}

            _stage_times["decision"] = (time.time() - _t0) * 1000.0

            # ── Hourly aggregator: per-frame update ────────────────────
            if self._hourly_aggregator:
                try:
                    # Detect clinical event flags from alerts
                    _fall_flag = fall_detected
                    _immobility_flag = any(
                        a.get("type") == "immobility" for a in clinical_alerts
                    )
                    _distress_flag = any(
                        a.get("type") == "distress" for a in clinical_alerts
                    )

                    self._hourly_aggregator.update(
                        posture=posture_state,
                        support_surface=(
                            support_surface_info.get("surface_type", "unknown")
                            if support_surface_info else "unknown"
                        ),
                        confidence=posture_confidence,
                        person_present=True,
                        activity=activity_state,
                        clinical_decision=decision,
                        fall_detected=_fall_flag,
                        immobility_detected=_immobility_flag,
                        distress_detected=_distress_flag,
                        keypoint_visibility=float(
                            frame_visibility.get("completeness_score", 1.0)
                        ),
                    )
                except Exception as e:
                    log.debug("Hourly aggregator update failed: %s", e)

            # Get segmentation mask if available
            segmentation_mask = None
            if hasattr(det, 'get') and det.get("mask") is not None:
                segmentation_mask = det.get("mask")
            elif isinstance(det, dict) and "mask" in det:
                segmentation_mask = det["mask"]
            
            # Record metrics if monitoring enabled
            if self.performance_monitor:
                try:
                    self.performance_monitor.record_frame(
                        inference_ms=inference_ms,
                        fps=self.fps,
                        detection_confidence=conf,
                        track_id=current_track_id,
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        frame_id=None  # Will use internal counter
                    )
                except Exception as e:
                    log.debug("Metrics recording failed: %s", e)
            
            # Prepare result
            result = {
                "ts": time.time(),
                "label": decision.get("label", label),
                "confidence": float(decision.get("confidence", conf)),
                "probs": probs,
                "uncertainty": uncertainty,  # Prediction uncertainty [0-1] (Shannon entropy)
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "kps": kps,
                "kps_3d": kps_3d,  # 3D keypoints (Phase 2)
                "inference_ms": float(inference_ms),
                "decision": decision,
                "features": feat.tolist() if feat is not None else None,
                "fps": round(self.fps, 2),
                "posture_state": posture_state,  # Instant posture classification
                "posture_confidence": posture_confidence,  # Posture confidence (decoupled from detector)
                "posture_analysis": posture_analysis,  # Full posture metrics
                "support_surface": support_surface_info,  # Support surface detection info
                "activity_state": activity_state,  # Activity classification (all 53 activities)
                "activity_confidence": activity_confidence,  # Activity confidence
                "activity_priority": activity_priority,  # Activity priority (CRITICAL, HIGH, NORMAL, MEDIUM)
                "segmentation_mask": segmentation_mask,  # Instant segmentation mask
                "track_id": current_track_id,  # Current track ID
                "patient_track_id": self.patient_track_id,  # Persistent patient track ID
                "patient_onboarded": self.patient_onboarded,  # Patient onboarding status
                "person_present": True,  # Person is present (we have detection)
                "fall_detected": fall_detected,  # Fall detection result
                "fall_result": fall_result,  # Detailed fall detection data
                "bed_detected": bed_info is not None,  # Bed detection status
                "bed_info": bed_info,  # Bed detection details (bbox, confidence, etc.)
                "person_on_bed": person_on_bed,  # Person is on bed
                "person_near_bed": person_near_bed,  # Person is near bed
                "zoom_level": self.camera.zoom_level if hasattr(self.camera, 'zoom_level') else 1.0,  # Current zoom level
                "distance_info": distance_info,  # Distance monitoring information
                "distance_feedback": distance_feedback,  # Distance adjustment feedback
                "self_contact": self_contact_signature,  # TODO-065: SC3D self-contact signature
                "emotions": emotions,  # Emotion detection results
                "frame_visibility": frame_visibility,  # Frame visibility analysis
                "movement_features": movement_features,  # Motion energy, jerk index
                "clinical_correlation": clinical_correlation,  # Pain, agitation, dizziness scores
                "imu_signals": imu_signals,  # Video2IMU acceleration/angular velocity
                "clinical_alerts": clinical_alerts,  # Bed exit, fall risk, immobility alerts
                "baseline_info": baseline_info,  # DBSCAN + PCA patient baseline anomaly info
                "stage_timings": _stage_times,  # Per-stage latency breakdown (ms)
            }
            
            # Add performance metrics if available
            if self.performance_monitor:
                try:
                    perf_summary = self.performance_monitor.get_summary()
                    result["performance_metrics"] = perf_summary
                except Exception as e:
                    log.debug("Failed to get performance summary: %s", e)

            # Display overlay if enabled
            # TODO-026: Reduce rendering frequency (skip frames if needed)
            self.display_frame_counter += 1
            should_render = (self.display_enabled and self.display and 
                           (self.display_frame_counter % self.display_render_frequency == 0 or 
                            self.display_render_frequency == 1))
            
            if should_render:
                frame_vis = frame.copy()
                
                # Draw instant segmentation mask first (if available)
                if segmentation_mask is not None:
                    frame_vis = self.display.draw_segmentation(frame_vis, segmentation_mask, alpha=0.3, color=(0, 255, 0))
                
                # Draw proper bounding box with track ID
                track_id = result.get("track_id")
                bbox_for_display = [int(x1), int(y1), int(x2), int(y2)]  # Ensure proper format
                frame_vis = self.display.draw_bbox(
                    frame_vis, 
                    bbox_for_display, 
                    label=decision["label"],
                    track_id=track_id,
                    reid_enabled=self.cfg.get("use_reid_tracking", False)
                )
                
                # Draw skeleton
                frame_vis = self.display.draw_skeleton(frame_vis, kps)
                
                # Draw distance feedback if needed (prominent overlay)
                if distance_feedback:
                    frame_vis = self.display.draw_distance_feedback(frame_vis, distance_feedback)
                
                # Draw posture with timestamp and support surface (prominent display)
                current_timestamp = result.get("ts", time.time())
                display_posture_conf = posture_confidence if posture_confidence > 0 else (
                    posture_analysis.get("confidence", 0.0) if posture_analysis else None
                )
                # Include subtype if available (e.g., "SITTING (leaning_back)")
                display_posture_state = posture_state
                if posture_analysis and posture_analysis.get("subtype"):
                    display_posture_state = f"{posture_state} ({posture_analysis['subtype']})"

                frame_vis = self.display.draw_posture_with_timestamp(
                    frame_vis,
                    posture_state=display_posture_state,
                    timestamp=current_timestamp,
                    posture_confidence=display_posture_conf
                )

                # Draw support surface info if detected
                if support_surface_info and support_surface_info.get("surface_id"):
                    support_text = f"Support: {support_surface_info.get('support', 'detected')}"
                    try:
                        self.display.draw_text(frame_vis, support_text, (10, 120),
                                              color=(0, 200, 255), scale=0.6)
                    except Exception:
                        pass  # Display method may not support this
                
                # Draw metrics including instant posture classification
                metrics = {
                    "FPS": round(self.fps, 1),
                    "Activity": decision["label"],
                    "Posture": display_posture_state,  # Posture with subtype
                    "P.Conf": round(posture_confidence, 2) if posture_confidence > 0 else round(decision.get("confidence", conf), 2),
                    "Latency(ms)": round(inference_ms, 1)
                }
                if track_id is not None:
                    metrics["Track ID"] = track_id
                if support_surface_info and support_surface_info.get("surface_id"):
                    metrics["Support"] = support_surface_info.get("surface_type", "detected")
                if distance_info:
                    metrics["Distance"] = f"{distance_info.get('distance_cm', 0)}cm"
                frame_vis = self.display.draw_metrics(frame_vis, metrics)

                if not self.display.show(frame_vis):
                    self.camera.release()
                    self.display.close()
                    exit(0)

            # Feed stage timings to JetsonOptimizer for bottleneck tracking
            if self._jetson_optimizer:
                for stage_name, stage_ms in _stage_times.items():
                    self._jetson_optimizer.record_inference(stage_ms, stage=stage_name)
                self._jetson_optimizer.record_inference(inference_ms, stage="total")

            # Log per-stage profiling every 30 frames
            if not hasattr(self, '_profile_frame_count'):
                self._profile_frame_count = 0
            self._profile_frame_count += 1
            if self._profile_frame_count % 30 == 0:
                parts = " | ".join(f"{k}={v:.1f}" for k, v in _stage_times.items())
                log.info("PROFILING [%s] total=%.1fms", parts, inference_ms)

            log.debug("FPS: %.2f  Label: %s  Latency: %.1fms", self.fps, result["label"], result["inference_ms"])
            return result

        except Exception as e:
            log.exception("Inference pipeline error (skipping frame): %s", e)
            return None

    def run_once_and_publish(self, publish_fn):
        """
        Runs one cycle and publishes via provided callback.
        publish_fn should accept a single dict (the result). If result is None, nothing is published.
        """
        res = self.run_once()
        if res is None:
            return None
        try:
            publish_fn(res)
        except Exception:
            log.exception("Publish function raised an exception")
        return res
   

