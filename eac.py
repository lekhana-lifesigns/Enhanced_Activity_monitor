# eac_runner.py
import yaml
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

running = True

def stop(sig, frame):
    global running
    log.info("Received stop signal, shutting down...")
    running = False

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

def publish_fn(res):
    """Publish result to MQTT and store locally."""
    try:
        # Use clinical payload if available
        decision = res.get("decision", {})
        if "agitation_score" in decision and MQ:
            # Use clinical event publishing
            MQ.publish_clinical_event(decision, features=res.get("features"))
        elif MQ:
            # Fallback to basic event publishing
            payload = {
                "deviceId": cfg["device_id"],
                "ts": res["ts"],
                "label": res["label"],
                "confidence": float(res["confidence"]),
                "inference_ms": float(res["inference_ms"]),
                "bbox": res.get("bbox"),
                "system": get_health()
            }
            MQ.publish_event(payload)
        
        # Store in local database
        if DB:
            DB.insert_event(
                device=cfg["device_id"],
                label=res["label"],
                confidence=res["confidence"],
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
            res = PIPE.run_once_and_publish(publish_fn)
            frame_count += 1
            
            # Log every 30 frames (~2 seconds at 15 FPS)
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                log.info("Frames: %d | Avg FPS: %.2f | Latency: %.1fms | Label: %s (%.2f)",
                        frame_count, fps, res.get("inference_ms", 0), 
                        res.get("label", "unknown"), res.get("confidence", 0))
                
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
    
    if MQ:
        MQ.shutdown()
    
    sys.exit(0)
