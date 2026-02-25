# eac.py - Enhanced Activity Monitor runner
import yaml
import logging
import time
import signal
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── CLI arguments (for multi-device systemd deployment) ───────────────
parser = argparse.ArgumentParser(description="Enhanced Activity Monitor")
parser.add_argument("--device-id", type=str, help="Device/bed ID (overrides config)")
parser.add_argument("--camera-idx", type=int, help="Camera index (overrides config)")
parser.add_argument("--camera-url", type=str, help="RTSP/IP camera URL (overrides camera_idx)")
parser.add_argument("--config", type=str, default="config/system.yaml", help="System config path")
parser.add_argument("--mqtt-config", type=str, default="config/mqtt.yaml", help="MQTT config path")
parser.add_argument("--mqtt-broker", type=str, help="MQTT broker address (overrides config)")
parser.add_argument("--mqtt-port", type=int, help="MQTT broker port (overrides config)")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()

# Load configuration first (before setting up logging)
try:
    cfg = yaml.safe_load(open(args.config))
    mqtt_cfg = yaml.safe_load(open(args.mqtt_config))
except Exception as e:
    print(f"Failed to load config: {e}")
    sys.exit(1)

# Configure logging from config or CLI with rotation (SSD optimization)
log_level_str = cfg.get("log_level", "INFO").upper()
if args.debug:
    log_level_str = "DEBUG"
log_level = getattr(logging, log_level_str, logging.INFO)

# Set up root logger with console handler
logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("eac")

# Add rotating file handler for persistent logs (SSD-friendly)
try:
    from logging.handlers import RotatingFileHandler
    os.makedirs("logs", exist_ok=True)

    # Rotating file handler: 10MB per file, keep 5 backups (50MB total max)
    file_handler = RotatingFileHandler(
        "logs/eac.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    ))
    logging.getLogger().addHandler(file_handler)
    log.info("Log rotation enabled: logs/eac.log (10MB x 5 backups)")
except Exception as e:
    log.warning("Failed to set up log rotation: %s", e)

from pipeline.pose.inference_pipeline import InferencePipeline
from telemetry.mqtt_client import MqttClient
from storage.db import LocalDB
from storage.reporting import ReportGenerator
from datetime import datetime

# Apply CLI overrides
if args.device_id:
    cfg["device_id"] = args.device_id
if args.camera_idx is not None:
    cfg["camera_idx"] = args.camera_idx
if args.camera_url:
    cfg["camera_url"] = args.camera_url
if args.mqtt_broker:
    mqtt_cfg["broker"] = args.mqtt_broker
if args.mqtt_port:
    mqtt_cfg["port"] = args.mqtt_port

# Initialize pipeline
try:
    PIPE = InferencePipeline(cfg)
    log.info("Inference pipeline initialized")
except Exception as e:
    log.exception("Failed to initialize pipeline: %s", e)
    sys.exit(1)

# Initialize MQTT client
try:
    MQ = MqttClient(mqtt_cfg, cfg["device_id"])
    log.info("MQTT client initialized")
except Exception as e:
    log.warning("MQTT client initialization failed: %s", e)
    MQ = None

# Initialize local database
try:
    DB = LocalDB()
    log.info("Local database initialized")
except Exception as e:
    log.warning("Database initialization failed: %s", e)
    DB = None

# Initialize report generator
REPORT_GEN = ReportGenerator(DB, cfg["device_id"]) if DB else None
last_hourly_report = time.time()
last_monthly_report = None  # Will be set on first run

running = True

# Asynchronous processing for MQTT/database (TODO-032)
from concurrent.futures import ThreadPoolExecutor
import threading
import atexit

# Thread pool for async operations with bounded work queue.
# If the queue is full (MQTT/DB slow), new results are dropped to prevent
# unbounded memory growth.  This is safe because clinical alerts are
# generated continuously — missing one frame's publish is acceptable.
MAX_PENDING_TASKS = 32
_pending_semaphore = threading.Semaphore(MAX_PENDING_TASKS)

executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="eac_async")
executor_lock = threading.Lock()
executor_shutdown = False

def shutdown_executor():
    """Safely shutdown executor on exit."""
    global executor_shutdown, executor
    if not executor_shutdown and executor:
        try:
            log.info("Shutting down async executor...")
            # Python 3.9+ supports cancel_futures parameter
            if sys.version_info >= (3, 9):
                executor.shutdown(wait=True, cancel_futures=True)
            else:
                executor.shutdown(wait=True)
            executor_shutdown = True
        except Exception as e:
            log.warning("Error shutting down executor: %s", e)

# Register shutdown handler for cleanup on exit
atexit.register(shutdown_executor)

def stop(sig, frame):
    """Signal handler for graceful shutdown."""
    global running, executor_shutdown
    log.info("Received stop signal (sig=%d), shutting down...", sig)
    running = False
    # Ensure executor is shut down on signal
    shutdown_executor()

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

# ── Publish throttle state ────────────────────────────────────────────
# Routine MQTT status is sent at most once per second (1 Hz) to reduce
# broker bandwidth at scale (1000 devices × 15 fps → 15K msg/s without
# throttling vs 1K msg/s with).  Alerts bypass the throttle entirely.
_last_status_publish = 0.0
_last_status_result = None  # latest result, published on next 1-sec tick

# Severity mapping for clinical alerts
_SEVERITY_MAP = {
    "high": "HIGH_RISK",
    "medium": "MEDIUM_RISK",
    "low": "LOW_RISK",
    "critical": "CRITICAL",
}


def _wrap_publish(res):
    """Wrapper that releases the backpressure semaphore after publish."""
    try:
        publish_fn(res)
    finally:
        _pending_semaphore.release()


def _process_clinical_alerts(res):
    """Persist clinical alerts to DB and publish immediate MQTT alerts."""
    clinical_alerts = res.get("clinical_alerts", [])
    for alert in clinical_alerts:
        alert_type = alert.get("type", "unknown")
        severity = alert.get("severity", "medium")
        alert_level = _SEVERITY_MAP.get(severity, "MEDIUM_RISK")

        # Persist to local database
        if DB:
            DB.insert_alert(
                device=cfg["device_id"],
                alert_level=alert_level,
                label=alert_type.upper(),
                agitation_score=alert.get("anomaly_score"),
                delirium_risk=None,
                respiratory_distress=None,
                hand_proximity_risk=None,
                payload={
                    **alert,
                    "posture": alert.get("posture"),
                    "activity": alert.get("activity"),
                    "support_surface": alert.get("support_surface_after") or alert.get("support_surface"),
                },
            )

        # Publish IMMEDIATE alert to MQTT (bypasses 1/sec throttle)
        if MQ:
            MQ.publish_alert(
                alert_type=alert_type.upper(),
                alert_level=alert_level,
                confidence=float(alert.get("confidence", 1.0)),
                context={
                    "patient_id": res.get("patient_id"),
                    "posture": alert.get("posture"),
                    "activity": alert.get("activity"),
                    "message": alert.get("message"),
                },
            )

        if severity in ("high", "critical"):
            log.warning("CLINICAL ALERT [%s]: %s", alert_type, alert.get("message"))
        else:
            log.info("Clinical alert [%s]: %s", alert_type, alert.get("message"))


def _process_decision_alerts(res, decision):
    """Handle fall detection, policy violations, and risk-level alerts."""
    # ── Fall detection (CRITICAL — immediate) ───────────────────────
    if res.get("fall_detected"):
        fall_result = res.get("fall_result", {})
        log.critical("FALL DETECTED! Creating CRITICAL alert")
        if DB:
            DB.insert_alert(
                device=cfg["device_id"],
                alert_level="CRITICAL",
                label="FALL_DETECTED",
                agitation_score=None,
                delirium_risk=None,
                respiratory_distress=None,
                hand_proximity_risk=None,
                payload={
                    "fall_detected": True,
                    "fall_confidence": fall_result.get("confidence", 0.0),
                    "fall_indicators": fall_result.get("indicators", []),
                    "timestamp": res.get("ts", time.time()),
                },
            )
        if MQ:
            MQ.publish_alert(
                alert_type="FALL_DETECTED",
                alert_level="CRITICAL",
                confidence=float(fall_result.get("confidence", 0.0)),
                context={"patient_id": res.get("patient_id")},
            )

    # ── Policy violations and risk-level alerts → DB ────────────────
    alert_level = decision.get("alert", "LOW_RISK")
    policy_violation = decision.get("policy_violation", False)
    if policy_violation and alert_level == "LOW_RISK":
        alert_level = "MEDIUM_RISK"

    if (alert_level in ("HIGH_RISK", "MEDIUM_RISK", "CRITICAL") or policy_violation) and DB:
        baseline_info = res.get("baseline_info")
        alert_payload = {
            **decision,
            "policy_violation": policy_violation,
            "violation_type": decision.get("violation_type"),
        }
        if baseline_info:
            alert_payload["anomaly_score"] = baseline_info.get("anomaly_score", 0.0)
            alert_payload["baseline_state"] = baseline_info.get("baseline_state", "collecting")
            alert_payload["cluster_id"] = baseline_info.get("cluster_id", -1)

        DB.insert_alert(
            device=cfg["device_id"],
            alert_level=alert_level if not policy_violation else "MEDIUM_RISK",
            label=decision.get("label", "unknown"),
            agitation_score=decision.get("agitation_score"),
            delirium_risk=decision.get("delirium_risk"),
            respiratory_distress=decision.get("respiratory_distress"),
            hand_proximity_risk=decision.get("hand_proximity_risk"),
            payload=alert_payload,
        )

        # Publish immediate MQTT alert for HIGH_RISK+
        if MQ and alert_level in ("HIGH_RISK", "CRITICAL"):
            MQ.publish_alert(
                alert_type=decision.get("label", "unknown").upper(),
                alert_level=alert_level,
                confidence=float(decision.get("clinical_confidence", 0.0)),
                context={
                    "patient_id": res.get("patient_id"),
                    "agitation_score": decision.get("agitation_score"),
                    "policy_violation": policy_violation,
                    "violation_type": decision.get("violation_type"),
                },
            )
        if policy_violation:
            log.warning("Policy violation alert stored: %s", decision.get("violation_type"))


def publish_fn(res):
    """Process inference result: persist to DB, publish alerts immediately,
    and buffer routine status for 1/sec MQTT publishing."""
    global _last_status_publish, _last_status_result
    if res is None:
        return

    try:
        # ── 1. Process clinical alerts (immediate MQTT + DB) ────────
        _process_clinical_alerts(res)

        # ── 2. Process decision engine alerts (immediate MQTT + DB) ─
        decision = res.get("decision", {})
        if "agitation_score" in decision:
            _process_decision_alerts(res, decision)

        # ── 3. Store every frame in local DB (offline resilience) ───
        if DB:
            DB.insert_event(
                device=cfg["device_id"],
                label=res.get("label", "unknown"),
                confidence=res.get("confidence", 0.0),
                payload=res,
            )

        # ── 4. Throttled routine status to MQTT (1/sec) ────────────
        # Buffer the latest result; it gets published in the main loop
        # at 1 Hz to keep broker bandwidth manageable at scale.
        _last_status_result = res

    except Exception as e:
        log.exception("Error in publish_fn: %s", e)


def _publish_throttled_status():
    """Called from main loop at 1 Hz — sends buffered status to MQTT."""
    global _last_status_publish, _last_status_result
    now = time.time()
    if now - _last_status_publish < 1.0:
        return
    if _last_status_result is None:
        return
    if not MQ:
        return

    try:
        MQ.publish_status(_last_status_result)
        _last_status_publish = now
    except Exception as e:
        log.debug("Throttled status publish failed: %s", e)

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("🏥 Enhanced Activity Monitor - Starting")
    log.info("=" * 60)
    log.info("Device ID: %s", cfg["device_id"])
    camera_source = cfg.get("camera_url", f"index={cfg['camera_idx']}")
    log.info("Camera: %s @ %s", camera_source, cfg["camera_resolution"])
    log.info("MQTT Broker: %s:%s", mqtt_cfg.get("broker"), mqtt_cfg.get("port"))
    log.info("=" * 60)
    
    frame_count = 0
    start_time = time.time()
    last_heartbeat = time.time()
    heartbeat_interval = cfg.get("heartbeat_interval", 30)

    while running:
        try:
            res = PIPE.run_once()
            frame_count += 1

            # Asynchronous publishing (non-blocking MQTT/database)
            # Alerts are published immediately inside publish_fn;
            # routine status is buffered and sent at 1 Hz below.
            if res and not executor_shutdown:
                if _pending_semaphore.acquire(blocking=False):
                    try:
                        executor.submit(_wrap_publish, res)
                    except RuntimeError as e:
                        _pending_semaphore.release()
                        log.warning("Executor closed, publishing synchronously: %s", e)
                        publish_fn(res)
                else:
                    log.debug("Publish queue full (%d tasks), dropping frame", MAX_PENDING_TASKS)

            current_time = time.time()

            # ── Throttled MQTT status (1/sec) ───────────────────────
            _publish_throttled_status()

            # ── Heartbeat (every 30s) ───────────────────────────────
            if MQ and (current_time - last_heartbeat) >= heartbeat_interval:
                try:
                    from pipeline.pose.system_metrics import get_health
                    MQ.publish_heartbeat(get_health())
                    last_heartbeat = current_time
                except Exception as e:
                    log.debug("Heartbeat publish failed: %s", e)

            # ── Hourly report (pre-aggregated for Flink) ────────────
            # The HourlyAggregator in the pipeline auto-flushes to DB.
            # Here we also publish via MQTT for Flink consumption.
            if MQ and (current_time - last_hourly_report) >= 3600:
                try:
                    aggregator = PIPE.get_hourly_aggregator()
                    if aggregator:
                        hourly_metrics = aggregator.flush_hour()
                        metrics_dict = hourly_metrics.to_dict()
                        # Persist to local DB
                        if DB:
                            DB.insert_hourly_aggregate(metrics_dict)
                        # Publish to MQTT for Flink
                        MQ.publish_hourly(metrics_dict)
                        log.info("Hourly summary published (frames=%d)", hourly_metrics.total_frames)
                    elif REPORT_GEN:
                        hourly_report = REPORT_GEN.generate_hourly_report()
                        MQ.publish_hourly(hourly_report)
                        log.info("Hourly report published (legacy)")
                    last_hourly_report = current_time
                except Exception as e:
                    log.exception("Failed to publish hourly report: %s", e)

            # ── Monthly report (1st of month) ───────────────────────
            if REPORT_GEN and MQ:
                now = datetime.now()
                current_month = (now.year, now.month)
                if last_monthly_report != current_month and now.day == 1 and now.hour == 0:
                    try:
                        prev_month = now.month - 1
                        prev_year = now.year
                        if prev_month == 0:
                            prev_month = 12
                            prev_year -= 1

                        monthly_report = REPORT_GEN.generate_monthly_report(prev_year, prev_month)
                        MQ.publish_report(monthly_report, "monthly")
                        log.info("Monthly report published for %s/%s", prev_year, prev_month)
                        last_monthly_report = current_month
                    except Exception as e:
                        log.exception("Failed to generate monthly report: %s", e)

            # ── Console log every ~2s ───────────────────────────────
            if res and frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                person_present = res.get("person_present", True)
                posture_state = res.get("posture_state", "unknown")
                log.info("Frames: %d | FPS: %.1f | Latency: %.0fms | %s (%.2f) | Present: %s | Posture: %s",
                        frame_count, fps, res.get("inference_ms", 0),
                        res.get("label", "unknown"), res.get("confidence", 0),
                        person_present, posture_state)
                
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")
            break
        except Exception as e:
            log.exception("Pipeline error - retrying in 2s")
            time.sleep(2)
    
    log.info("=" * 60)
    log.info("EAC runner stopped")
    log.info("Total frames processed: %d", frame_count)
    log.info("=" * 60)
    
    # Persist baseline model before exit
    if hasattr(PIPE, '_save_baseline_to_db'):
        try:
            PIPE._save_baseline_to_db()
            log.info("Baseline model saved on shutdown")
        except Exception as e:
            log.warning("Failed to save baseline on shutdown: %s", e)

    # Shutdown async executor (atexit will also handle this, but explicit is better)
    shutdown_executor()

    if MQ:
        MQ.shutdown()

    if DB:
        DB.close()

    sys.exit(0)
