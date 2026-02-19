# inference_node.py - Jetson Inference Node
"""
Inference node for Jetson in clustered EAM architecture.
Receives streams from multiple Raspberry Pi capture nodes and performs AI inference.
"""
import yaml
import json
import logging
import time
import signal
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.pose.inference_pipeline import InferencePipeline
from telemetry.mqtt_client import MqttClient
from pipeline.pose.system_metrics import get_health
from storage.db import LocalDB
from storage.reporting import ReportGenerator
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference_node")

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Enhanced Activity Monitor - Clustered Inference Node")
parser.add_argument("--mqtt-broker", type=str, help="MQTT broker address (overrides config)")
parser.add_argument("--mqtt-port", type=int, help="MQTT broker port (overrides config)")
parser.add_argument("--device-id", type=str, help="Device ID (overrides config)")
parser.add_argument("--config", type=str, default="config/system.yaml", help="Path to system config file")
parser.add_argument("--mqtt-config", type=str, default="config/mqtt.yaml", help="Path to MQTT config file")
args = parser.parse_args()

# Load configuration
try:
    cfg = yaml.safe_load(open(args.config))
    mqtt_cfg = yaml.safe_load(open(args.mqtt_config))
except Exception as e:
    log.error("Failed to load config: %s", e)
    sys.exit(1)

# Apply CLI overrides
if args.mqtt_broker:
    mqtt_cfg["broker"] = args.mqtt_broker
    log.info(f"CLI override: MQTT broker = {args.mqtt_broker}")
if args.mqtt_port:
    mqtt_cfg["port"] = args.mqtt_port
    log.info(f"CLI override: MQTT port = {args.mqtt_port}")
if args.device_id:
    cfg["device_id"] = args.device_id
    log.info(f"CLI override: Device ID = {args.device_id}")

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

# Initialize hourly aggregator for efficient reporting
from analytics.hourly_aggregator import get_aggregator_manager, HourlyAggregator

AGGREGATOR_MANAGER = get_aggregator_manager(db=DB)
log.info("Hourly aggregator manager initialized")

# Asynchronous processing for MQTT/database
from concurrent.futures import ThreadPoolExecutor
import threading
import atexit

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference_async")
executor_lock = threading.Lock()
executor_shutdown = False


def shutdown_executor():
    """Safely shutdown executor on exit."""
    global executor_shutdown, executor
    if not executor_shutdown and executor:
        try:
            log.info("Shutting down async executor...")
            try:
                executor.shutdown(wait=True, timeout=5.0)
            except TypeError:
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
    shutdown_executor()


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)


def publish_fn(res):
    """Publish result to MQTT and store locally (synchronous version)."""
    if res is None:
        return

    camera_id = res.get("camera_id")
    device_id = f"{cfg['device_id']}_{camera_id}" if camera_id else cfg["device_id"]

    try:
        decision = res.get("decision", {})
        if "agitation_score" in decision and MQ:
            MQ.publish_clinical_event(decision, features=res.get("features"))

        # Check for fall detection (CRITICAL)
        fall_detected = res.get("fall_detected", False)
        if fall_detected:
            fall_result = res.get("fall_result", {})
            log.critical("FALL DETECTED on %s! Creating CRITICAL alert", camera_id or "default")
            if DB:
                DB.insert_alert(
                    device=device_id,
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
                        "camera_id": camera_id
                    }
                )
            if MQ:
                MQ.client.publish(
                    f"{MQ.cfg.get('topic_prefix')}/{device_id}/alerts/critical",
                    json.dumps({
                        "deviceId": device_id,
                        "alert": "CRITICAL",
                        "type": "FALL_DETECTED",
                        "confidence": fall_result.get("confidence", 0.0),
                        "camera_id": camera_id,
                        "timestamp": res.get("ts", time.time())
                    }),
                    qos=2
                )

        # Store alert if high/medium risk OR policy violation
        alert_level = decision.get("alert", "LOW_RISK")
        policy_violation = decision.get("policy_violation", False)
        if policy_violation and alert_level == "LOW_RISK":
            alert_level = "MEDIUM_RISK"

        if (alert_level in ["HIGH_RISK", "MEDIUM_RISK", "CRITICAL"] or policy_violation) and DB:
            DB.insert_alert(
                device=device_id,
                alert_level=alert_level if not policy_violation else "MEDIUM_RISK",
                label=decision.get("label", "unknown"),
                agitation_score=decision.get("agitation_score"),
                delirium_risk=decision.get("delirium_risk"),
                respiratory_distress=decision.get("respiratory_distress"),
                hand_proximity_risk=decision.get("hand_proximity_risk"),
                payload={
                    **decision,
                    "policy_violation": policy_violation,
                    "violation_type": decision.get("violation_type"),
                    "camera_id": camera_id
                }
            )
            if policy_violation:
                log.warning("Policy violation alert stored for %s: %s",
                           camera_id or "default", decision.get("violation_type"))
        elif MQ:
            payload = {
                "deviceId": device_id,
                "ts": res.get("ts", time.time()),
                "label": res.get("label", "unknown"),
                "confidence": float(res.get("confidence", 0.0)),
                "inference_ms": float(res.get("inference_ms", 0.0)),
                "bbox": res.get("bbox"),
                "person_present": res.get("person_present", True),
                "posture_state": res.get("posture_state", "unknown"),
                "distance_info": res.get("distance_info"),
                "distance_feedback": res.get("distance_feedback"),
                "camera_id": camera_id,
                "system": get_health()
            }
            MQ.publish_event(payload)

        # Handle distance feedback
        distance_feedback = res.get("distance_feedback")
        if distance_feedback:
            log.info("DISTANCE FEEDBACK [%s]: %s", camera_id or "default",
                    distance_feedback.get("message", ""))

        # Store in local database
        if DB:
            DB.insert_event(
                device=device_id,
                label=res.get("label", "unknown"),
                confidence=res.get("confidence", 0.0),
                payload=res
            )

        # Update hourly aggregator (efficient reporting)
        try:
            patient_id = res.get("patient_id", cfg.get("patient_id", "unknown"))
            aggregator = AGGREGATOR_MANAGER.get_aggregator(device_id, patient_id)

            # Extract posture info
            posture_info = res.get("posture_info", {})
            support_surface_info = res.get("support_surface_info", {})

            aggregator.update(
                posture=posture_info.get("posture", res.get("posture_state", "unknown")),
                support_surface=support_surface_info.get("surface_type", "unknown"),
                confidence=res.get("confidence", 0.0),
                person_present=res.get("person_present", True),
                activity=res.get("label", "unknown"),
                clinical_decision=decision,
                fall_detected=fall_detected,
                immobility_detected=decision.get("immobility_detected", False),
                distress_detected=decision.get("respiratory_distress", 0) > 0.7,
                keypoint_visibility=res.get("keypoint_visibility", 1.0),
                timestamp=res.get("ts", time.time()),
            )
        except Exception as agg_err:
            log.debug("Aggregator update failed: %s", agg_err)

    except Exception as e:
        log.exception("Error in publish_fn for %s: %s", camera_id or "default", e)


if __name__ == "__main__":
    log.info("=" * 60)
    log.info(" Enhanced Activity Monitor - Jetson Inference Node")
    log.info("=" * 60)
    log.info("Device ID: %s", cfg["device_id"])

    if cfg.get("clustered_mode"):
        cameras = cfg.get("cameras", {})
        log.info("Clustered Mode: %d cameras", len(cameras))
        for cam_id, cam_config in cameras.items():
            log.info("  - %s: %s", cam_id, cam_config.get("pi_address", "unknown"))
    else:
        log.info("Single Camera Mode")
        log.info("Camera: %s @ %s", cfg.get("camera_idx", 0), cfg.get("camera_resolution", [1280, 720]))

    log.info("MQTT Broker: %s:%s", mqtt_cfg.get("broker"), mqtt_cfg.get("port"))
    log.info("=" * 60)

    frame_count = 0
    start_time = time.time()

    while running:
        try:
            res = PIPE.run_once()
            frame_count += 1

            # Handle clustered mode vs single camera mode
            if isinstance(res, dict) and not any(k in res for k in ["ts", "label", "confidence"]):
                results_to_process = res
            else:
                results_to_process = {"default": res} if res else {}

            # Process each camera's result
            for camera_id, camera_res in results_to_process.items():
                if camera_res is None:
                    continue

                if not executor_shutdown:
                    try:
                        if executor and not executor._shutdown:
                            if not hasattr(executor, '_task_count'):
                                executor._task_count = 0

                            max_pending_tasks = 100
                            if executor._task_count < max_pending_tasks:
                                executor._task_count += 1
                                future = executor.submit(publish_fn, camera_res)

                                def on_complete(f):
                                    executor._task_count -= 1
                                future.add_done_callback(on_complete)
                            else:
                                log.warning("Async queue full (%d tasks), publishing synchronously",
                                           executor._task_count)
                                publish_fn(camera_res)
                        else:
                            log.warning("Executor shutting down, publishing synchronously")
                            publish_fn(camera_res)
                    except RuntimeError as e:
                        log.warning("Executor closed, publishing synchronously: %s", e)
                        publish_fn(camera_res)

            current_time = time.time()

            # Generate hourly report
            if REPORT_GEN and MQ and (current_time - last_hourly_report) >= 3600:
                try:
                    hourly_report = REPORT_GEN.generate_hourly_report()
                    MQ.publish_report(hourly_report, "hourly")
                    log.info("Hourly report generated and published")
                    last_hourly_report = current_time
                except Exception as e:
                    log.exception("Failed to generate hourly report: %s", e)

            # Generate monthly report
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
                        log.info("Monthly report generated and published for %s/%s", prev_year, prev_month)
                        last_monthly_report = current_month
                    except Exception as e:
                        log.exception("Failed to generate monthly report: %s", e)

            # Log every 30 frames
            if results_to_process and frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                active_cameras = [cid for cid, cres in results_to_process.items() if cres]
                log.info("Frames: %d | Avg FPS: %.2f | Active cameras: %d/%d",
                        frame_count, fps, len(active_cameras), len(results_to_process))

                for camera_id, camera_res in results_to_process.items():
                    if camera_res:
                        person_present = camera_res.get("person_present", True)
                        posture_state = camera_res.get("posture_state", "unknown")
                        log.info("  [%s] Latency: %.1fms | Label: %s (%.2f) | Present: %s | Posture: %s",
                                camera_id, camera_res.get("inference_ms", 0),
                                camera_res.get("label", "unknown"), camera_res.get("confidence", 0),
                                person_present, posture_state)

        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")
            break
        except Exception as e:
            log.exception("Pipeline error - retrying in 2s")
            time.sleep(2)

    log.info("=" * 60)
    log.info("Jetson inference node stopped")
    log.info("Total frames processed: %d", frame_count)
    log.info("=" * 60)

    # Flush hourly aggregators before shutdown
    try:
        log.info("Flushing hourly aggregators...")
        final_metrics = AGGREGATOR_MANAGER.flush_all()
        log.info("Flushed %d hourly aggregates", len(final_metrics))
    except Exception as e:
        log.warning("Failed to flush aggregators: %s", e)

    shutdown_executor()
    if MQ:
        MQ.shutdown()
    sys.exit(0)
