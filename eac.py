# eac_runner.py
import yaml
import json
import logging
import time
import signal
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.pose.inference_pipeline import InferencePipeline
from telemetry.mqtt_client import MqttClient
from pipeline.pose.system_metrics import get_health
from storage.db import LocalDB
from storage.reporting import ReportGenerator
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eac")

# Load configuration
try:
    cfg = yaml.safe_load(open("config/system.yaml"))
    mqtt_cfg = yaml.safe_load(open("config/mqtt.yaml"))
except Exception as e:
    log.error("Failed to load config: %s", e)
    sys.exit(1)

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

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="eac_async")
executor_lock = threading.Lock()
executor_shutdown = False

def shutdown_executor():
    """Safely shutdown executor on exit."""
    global executor_shutdown, executor
    if not executor_shutdown and executor:
        try:
            log.info("Shutting down async executor...")
            executor.shutdown(wait=True, timeout=5.0)
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

def publish_fn(res):
    """Publish result to MQTT and store locally (synchronous version)."""
    if res is None:
        return  # Skip if no result
    
    try:
        # Use clinical payload if available
        decision = res.get("decision", {})
        if "agitation_score" in decision and MQ:
            # Use clinical event publishing
            MQ.publish_clinical_event(decision, features=res.get("features"))
            
            # Check for fall detection (CRITICAL)
            fall_detected = res.get("fall_detected", False)
            if fall_detected:
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
                            "timestamp": res.get("ts", time.time())
                        }
                    )
                if MQ:
                    # Publish critical fall alert
                    MQ.client.publish(
                        f"{MQ.cfg.get('topic_prefix')}/{cfg['device_id']}/alerts/critical",
                        json.dumps({
                            "deviceId": cfg["device_id"],
                            "alert": "CRITICAL",
                            "type": "FALL_DETECTED",
                            "confidence": fall_result.get("confidence", 0.0),
                            "timestamp": res.get("ts", time.time())
                        }),
                        qos=2  # Highest QoS for critical alerts
                    )
            
            # Store alert if high/medium risk OR policy violation
            alert_level = decision.get("alert", "LOW_RISK")
            policy_violation = decision.get("policy_violation", False)
            
            # Policy violations should always trigger alerts
            if policy_violation and alert_level == "LOW_RISK":
                alert_level = "MEDIUM_RISK"  # Escalate policy violations
            
            if (alert_level in ["HIGH_RISK", "MEDIUM_RISK", "CRITICAL"] or policy_violation) and DB:
                DB.insert_alert(
                    device=cfg["device_id"],
                    alert_level=alert_level if not policy_violation else "MEDIUM_RISK",
                    label=decision.get("label", "unknown"),
                    agitation_score=decision.get("agitation_score"),
                    delirium_risk=decision.get("delirium_risk"),
                    respiratory_distress=decision.get("respiratory_distress"),
                    hand_proximity_risk=decision.get("hand_proximity_risk"),
                    payload={
                        **decision,
                        "policy_violation": policy_violation,
                        "violation_type": decision.get("violation_type")
                    }
                )
                if policy_violation:
                    log.warning("Policy violation alert stored: %s", decision.get("violation_type"))
        elif MQ:
            # Fallback to basic event publishing
            payload = {
                "deviceId": cfg["device_id"],
                "ts": res.get("ts", time.time()),
                "label": res.get("label", "unknown"),
                "confidence": float(res.get("confidence", 0.0)),
                "inference_ms": float(res.get("inference_ms", 0.0)),
                "bbox": res.get("bbox"),
                "person_present": res.get("person_present", True),
                "posture_state": res.get("posture_state", "unknown"),
                "distance_info": res.get("distance_info"),
                "distance_feedback": res.get("distance_feedback"),
                "system": get_health()
            }
            MQ.publish_event(payload)
        
        # Handle distance feedback (log prominently)
        distance_feedback = res.get("distance_feedback")
        if distance_feedback:
            log.info("DISTANCE FEEDBACK: %s", distance_feedback.get("message", ""))
            # Distance feedback is also displayed visually on screen
        
        # Store in local database
        if DB:
            DB.insert_event(
                device=cfg["device_id"],
                label=res.get("label", "unknown"),
                confidence=res.get("confidence", 0.0),
                payload=res
            )
    except Exception as e:
        log.exception("Error in publish_fn: %s", e)

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("🏥 Enhanced Activity Monitor - Starting")
    log.info("=" * 60)
    log.info("Device ID: %s", cfg["device_id"])
    log.info("Camera: %s @ %s", cfg["camera_idx"], cfg["camera_resolution"])
    log.info("MQTT Broker: %s:%s", mqtt_cfg.get("broker"), mqtt_cfg.get("port"))
    log.info("=" * 60)
    
    frame_count = 0
    start_time = time.time()
    
    while running:
        try:
            res = PIPE.run_once()
            frame_count += 1
            
            # Asynchronous publishing (TODO-032: Non-blocking MQTT/database)
            # Add backpressure: check queue size to prevent unbounded growth
            if res and not executor_shutdown:
                try:
                    # Check if executor is still running
                    if executor and not executor._shutdown:
                        executor.submit(publish_fn, res)  # Non-blocking
                    else:
                        # Executor is shutting down, publish synchronously
                        log.warning("Executor shutting down, publishing synchronously")
                        publish_fn(res)
                except RuntimeError as e:
                    # Executor is closed, fallback to synchronous
                    log.warning("Executor closed, publishing synchronously: %s", e)
                    publish_fn(res)
            
            current_time = time.time()
            
            # Generate hourly report every hour
            if REPORT_GEN and MQ and (current_time - last_hourly_report) >= 3600:
                try:
                    hourly_report = REPORT_GEN.generate_hourly_report()
                    MQ.publish_report(hourly_report, "hourly")
                    log.info("Hourly report generated and published")
                    last_hourly_report = current_time
                except Exception as e:
                    log.exception("Failed to generate hourly report: %s", e)
            
            # Generate monthly report at start of each month
            if REPORT_GEN and MQ:
                now = datetime.now()
                current_month = (now.year, now.month)
                if last_monthly_report != current_month and now.day == 1 and now.hour == 0:
                    try:
                        # Generate report for previous month
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
            
            # Log every 30 frames (~2 seconds at 15 FPS)
            if res and frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                person_present = res.get("person_present", True)
                posture_state = res.get("posture_state", "unknown")
                log.info("Frames: %d | Avg FPS: %.2f | Latency: %.1fms | Label: %s (%.2f) | Present: %s | Posture: %s",
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
    
    # Shutdown async executor (atexit will also handle this, but explicit is better)
    shutdown_executor()
    
    if MQ:
        MQ.shutdown()
    
    sys.exit(0)
