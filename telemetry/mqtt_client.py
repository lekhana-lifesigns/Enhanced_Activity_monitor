# telemetry/mqtt_client.py
import paho.mqtt.client as mqtt
import json
import logging
import time
import gzip
import base64

log = logging.getLogger("mqtt")


def compress_payload(payload_dict):
    """
    Compress JSON payload to reduce size (<1KB requirement).
    Returns base64-encoded compressed string.
    """
    try:
        json_str = json.dumps(payload_dict, separators=(',', ':'))
        compressed = gzip.compress(json_str.encode('utf-8'))
        encoded = base64.b64encode(compressed).decode('utf-8')
        return encoded
    except Exception:
        # Fallback to uncompressed JSON
        return json.dumps(payload_dict, separators=(',', ':'))


class MqttClient:
    def __init__(self, cfg, device_id):
        self.cfg = cfg
        self.device_id = device_id
        self.topic = f"{cfg.get('topic_prefix')}/{device_id}"
        self.alert_topic = f"{cfg.get('topic_prefix')}/{device_id}/alerts"
        self.client = mqtt.Client(client_id=device_id)
        self.connected = False
        self.use_compression = cfg.get("compress_payload", True)
        
        # Try to connect with timeout and retry
        try:
            broker = cfg.get("broker")
            port = cfg.get("port", 1883)
            timeout = cfg.get("connection_timeout", 5)
            
            log.info(f"Connecting to MQTT broker {broker}:{port}...")
            self.client.connect(broker, port, keepalive=60)
            self.client.loop_start()
            
            # Wait for connection with timeout
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.connected:
                log.info(f"MQTT client connected to {broker}:{port}")
            else:
                log.warning(f"MQTT connection timeout after {timeout}s - will retry on publish")
        except Exception as e:
            log.warning(f"MQTT client initialization failed: {e}")
            self.connected = False
        
        # Set up connection callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when MQTT client connects."""
        if rc == 0:
            self.connected = True
            log.info("MQTT client connected successfully")
        else:
            self.connected = False
            log.warning(f"MQTT connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when MQTT client disconnects."""
        self.connected = False
        if rc != 0:
            log.warning(f"MQTT client disconnected unexpectedly (rc={rc})")
        else:
            log.info("MQTT client disconnected")

    def publish_event(self, payload):
        """
        Publish event payload (backward compatible).
        """
        try:
            # Check connection and reconnect if needed
            if not self.connected:
                try:
                    self.client.reconnect()
                    time.sleep(0.1)  # Brief wait for connection
                except:
                    pass  # Will fail gracefully if broker unavailable
            
            json_str = json.dumps(payload, separators=(',', ':'))
            if self.use_compression and len(json_str) > 500:
                # Compress if payload is large
                compressed = compress_payload(payload)
                self.client.publish(
                    self.topic,
                    compressed,
                    qos=self.cfg.get("qos", 1)
                )
            else:
                self.client.publish(
                    self.topic,
                    json_str,
                    qos=self.cfg.get("qos", 1)
                )
        except Exception:
            log.debug("MQTT publish failed (broker may be unavailable)")

    def publish_clinical_event(self, decision_result, features=None):
        """
        Publish clinical event with ICU-grade payload schema.
        
        Args:
            decision_result: Result from decision engine (contains clinical scores)
            features: Optional feature vector
        """
        try:
            # Build clinical payload with patient tracking fields
            payload = {
                "deviceId": self.device_id,
                "ts": decision_result.get("ts", time.time()),
                "label": decision_result.get("label", "unknown"),
                "confidence": decision_result.get("confidence", 0.0),
                "agitation_score": decision_result.get("agitation_score", 0.0),
                "lhs_motor": decision_result.get("lhs_motor", 0.5),
                "rhs_motor": decision_result.get("rhs_motor", 0.5),
                "respiratory_distress": decision_result.get("respiratory_distress", 0.0),
                "delirium_risk": decision_result.get("delirium_risk", 0.0),
                "hand_proximity_risk": decision_result.get("hand_proximity_risk", 0.0),
                "breath_rate_proxy": decision_result.get("breath_rate_proxy", 0.0),
                "motion_entropy": decision_result.get("motion_entropy", 0.0),
                "alert": decision_result.get("alert", "LOW_RISK"),
                "clinical_confidence": decision_result.get("clinical_confidence", 0.0),
                # Patient tracking fields
                "patientId": decision_result.get("patient_id"),
                "posture_state": decision_result.get("posture_state", "unknown"),
                "person_present": decision_result.get("person_present", True),
                "policy_violation": decision_result.get("policy_violation", False),
                "violation_type": decision_result.get("violation_type")
            }
            
            # Add features if available
            if features is not None:
                payload["features"] = features.tolist() if hasattr(features, 'tolist') else features
            
            # Publish to main topic
            json_str = json.dumps(payload, separators=(',', ':'))
            
            if self.use_compression and len(json_str) > 500:
                compressed = compress_payload(payload)
                self.client.publish(self.topic, compressed, qos=self.cfg.get("qos", 1))
            else:
                self.client.publish(self.topic, json_str, qos=self.cfg.get("qos", 1))
            
            # Publish to alert topic if high/medium risk
            if payload["alert"] in ["HIGH_RISK", "MEDIUM_RISK"]:
                alert_payload = {
                    "deviceId": self.device_id,
                    "ts": payload["ts"],
                    "alert": payload["alert"],
                    "agitation_score": payload["agitation_score"],
                    "delirium_risk": payload["delirium_risk"],
                    "respiratory_distress": payload["respiratory_distress"],
                    "hand_proximity_risk": payload["hand_proximity_risk"]
                }
                self.client.publish(
                    self.alert_topic,
                    json.dumps(alert_payload, separators=(',', ':')),
                    qos=self.cfg.get("qos", 1)
                )
                
        except Exception:
            log.exception("MQTT clinical event publish failed")

    def publish_heartbeat(self, system_health=None):
        """
        Publish heartbeat packet with system health.
        """
        try:
            payload = {
                "deviceId": self.device_id,
                "ts": time.time(),
                "type": "heartbeat",
                "system": system_health or {}
            }
            self.client.publish(
                f"{self.topic}/heartbeat",
                json.dumps(payload, separators=(',', ':')),
                qos=0  # QoS 0 for heartbeat
            )
        except Exception:
            log.exception("MQTT heartbeat publish failed")

    def publish_report(self, report_data, report_type="hourly"):
        """
        Publish report to cloud ingest layer.
        
        Args:
            report_data: Report dictionary from ReportGenerator
            report_type: "hourly" or "monthly"
        """
        try:
            # Use reports topic for cloud ingest
            report_topic = f"{self.cfg.get('topic_prefix')}/{self.device_id}/reports/{report_type}"
            
            # Compress report payload (reports can be large)
            json_str = json.dumps(report_data, separators=(',', ':'))
            if len(json_str) > 500:
                compressed = compress_payload(report_data)
                self.client.publish(report_topic, compressed, qos=self.cfg.get("qos", 1))
            else:
                self.client.publish(report_topic, json_str, qos=self.cfg.get("qos", 1))
            
            log.info("Published %s report to %s", report_type, report_topic)
        except Exception:
            log.exception("MQTT report publish failed")

    def shutdown(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except:
            pass
