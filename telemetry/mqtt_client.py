# telemetry/mqtt_client.py
# Scale-ready MQTT client for Flink ingestion
# Supports: TLS, sequence numbers, versioned schema, retry with backoff,
#           alert deduplication, 1/sec routine + immediate alerts
import paho.mqtt.client as mqtt
import json
import logging
import os
import ssl
import threading
import time

log = logging.getLogger("mqtt")

# Schema version — bump when payload structure changes.
# Flink consumers use this for schema evolution.
SCHEMA_VERSION = 1

# Retry constants
_MAX_RETRIES = 3
_BASE_BACKOFF = 0.1  # seconds

# TLS version mapping
_TLS_VERSIONS = {
    "tlsv1.2": ssl.PROTOCOL_TLSv1_2 if hasattr(ssl, "PROTOCOL_TLSv1_2") else ssl.PROTOCOL_TLS,
    "tlsv1.3": ssl.PROTOCOL_TLS,
}


class MqttClient:
    """MQTT client optimised for edge→Flink streaming at scale.

    Design decisions
    ────────────────
    • Every message carries ``v`` (schema version) and ``seq`` (monotonic
      sequence number per device) so that Flink can detect gaps, reorder,
      and evolve deserialisers without downtime.
    • Routine status is published at most 1/sec via ``publish_status()``.
      Alerts are published **immediately** via ``publish_alert()``.
    • TLS is configured from ``mqtt.yaml`` — no code changes needed to
      enable encryption in production.
    • Publish retries with exponential back-off (max 3 attempts) prevent
      transient broker hiccups from silently dropping clinical data.
    """

    def __init__(self, cfg, device_id):
        self.cfg = cfg
        self.device_id = device_id
        self._seq = 0
        self._seq_lock = threading.Lock()

        # Topic layout — flat, Flink-friendly
        prefix = cfg.get("topic_prefix", "eam")
        self._topic_events = f"{prefix}/{device_id}/events"
        self._topic_alerts = f"{prefix}/{device_id}/alerts"
        self._topic_heartbeat = f"{prefix}/{device_id}/heartbeat"
        self._topic_hourly = f"{prefix}/{device_id}/hourly"

        # Alert deduplication: track last alert per type to avoid flooding
        self._last_alert_ts: dict[str, float] = {}
        self._alert_dedup_window = cfg.get("alert_dedup_seconds", 30.0)

        # Offline buffering: queue critical alerts when disconnected
        self._offline_queue: list[dict] = []
        self._offline_queue_max = 1000

        # Build client
        self.client = mqtt.Client(
            client_id=device_id,
            clean_session=cfg.get("broker_retention", {}).get("clean_session", True),
        )
        self.connected = False

        # ── Authentication ──────────────────────────────────────────────
        username = os.environ.get("EAM_MQTT_USERNAME")
        password = os.environ.get("EAM_MQTT_PASSWORD")
        if cfg.get("use_authentication") and username:
            self.client.username_pw_set(username, password)
            log.info("MQTT authentication enabled (user=%s)", username)

        # ── TLS ─────────────────────────────────────────────────────────
        if cfg.get("use_tls"):
            self._configure_tls(cfg.get("tls", {}))

        # ── Callbacks (set before connect) ──────────────────────────────
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        # ── Connect ─────────────────────────────────────────────────────
        broker = cfg.get("broker", "localhost")
        port = cfg.get("port", 8883 if cfg.get("use_tls") else 1883)
        timeout = cfg.get("connection_timeout", 5)
        try:
            log.info("Connecting to MQTT broker %s:%d (TLS=%s)...",
                     broker, port, cfg.get("use_tls", False))
            self.client.connect(broker, port, keepalive=60)
            self.client.loop_start()

            # Wait for on_connect callback
            deadline = time.time() + timeout
            while not self.connected and time.time() < deadline:
                time.sleep(0.1)

            if self.connected:
                log.info("MQTT connected to %s:%d", broker, port)
            else:
                log.warning("MQTT connection timeout after %ds — will retry on publish", timeout)
        except Exception as e:
            log.warning("MQTT connection failed: %s", e)
            self.connected = False

    # ── TLS configuration ───────────────────────────────────────────────

    def _configure_tls(self, tls_cfg):
        """Wire TLS from mqtt.yaml into paho client."""
        ca_cert = tls_cfg.get("ca_cert")
        client_cert = tls_cfg.get("client_cert")
        client_key = tls_cfg.get("client_key")
        tls_version_str = tls_cfg.get("tls_version", "tlsv1.2")
        tls_version = _TLS_VERSIONS.get(tls_version_str, ssl.PROTOCOL_TLS)

        try:
            self.client.tls_set(
                ca_certs=ca_cert,
                certfile=client_cert,
                keyfile=client_key,
                tls_version=tls_version,
            )
            # Skip hostname verification for self-signed certs (configurable)
            if tls_cfg.get("insecure", False):
                self.client.tls_insecure_set(True)
            log.info("MQTT TLS configured (ca=%s, version=%s)", ca_cert, tls_version_str)
        except Exception as e:
            log.error("MQTT TLS configuration failed: %s", e)

    # ── Callbacks ───────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            log.info("MQTT connected (flags=%s)", flags)
        else:
            self.connected = False
            log.warning("MQTT connect failed rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            log.warning("MQTT unexpected disconnect rc=%d", rc)

    # ── Sequence number ─────────────────────────────────────────────────

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    # ── Core publish with retry ─────────────────────────────────────────

    def _publish(self, topic: str, payload: dict, qos: int = 1,
                 is_critical: bool = False):
        """Publish JSON with retry + exponential backoff.

        If all retries fail and the message is critical, it is buffered
        offline and replayed on the next successful publish.
        Non-critical messages are downgraded to QoS 0 after the first
        retry to prevent broker backpressure from blocking inference.
        """
        # Replay offline queue on first successful publish attempt
        if self.connected and self._offline_queue:
            self._replay_offline_queue()

        data = json.dumps(payload, separators=(",", ":"))
        for attempt in range(_MAX_RETRIES):
            try:
                if not self.connected:
                    self.client.reconnect()
                    time.sleep(0.05)
                # QoS downgrade: non-critical messages drop to QoS 0 after
                # first retry to prevent broker slowness blocking inference
                effective_qos = qos if (attempt == 0 or is_critical) else 0
                info = self.client.publish(topic, data, qos=effective_qos)
                if info.rc == mqtt.MQTT_ERR_SUCCESS:
                    return
                log.debug("MQTT publish rc=%d attempt=%d", info.rc, attempt + 1)
            except Exception as e:
                log.debug("MQTT publish error attempt=%d: %s", attempt + 1, e)
            time.sleep(_BASE_BACKOFF * (2 ** attempt))

        # All retries exhausted — buffer critical messages for later replay
        if is_critical and len(self._offline_queue) < self._offline_queue_max:
            self._offline_queue.append({"topic": topic, "payload": payload, "qos": qos})
            log.info("Critical message buffered offline (%d queued)", len(self._offline_queue))
        else:
            log.warning("MQTT publish failed after %d retries (topic=%s)", _MAX_RETRIES, topic)

    def _replay_offline_queue(self):
        """Replay buffered offline messages (oldest first)."""
        replayed = 0
        while self._offline_queue and self.connected:
            msg = self._offline_queue.pop(0)
            try:
                data = json.dumps(msg["payload"], separators=(",", ":"))
                info = self.client.publish(msg["topic"], data, qos=msg["qos"])
                if info.rc == mqtt.MQTT_ERR_SUCCESS:
                    replayed += 1
                else:
                    self._offline_queue.insert(0, msg)
                    break
            except Exception:
                self._offline_queue.insert(0, msg)
                break
        if replayed:
            log.info("Replayed %d offline messages (%d remaining)", replayed, len(self._offline_queue))

    # ── Public API: Versioned, Flink-ready ──────────────────────────────

    def publish_status(self, result: dict):
        """Publish routine status (call at 1/sec from eac.py).

        This is the primary data stream Flink consumes for windowed
        aggregation.  Kept small (~200 bytes) to minimise bandwidth
        at 1000+ devices.
        """
        decision = result.get("decision", {})
        payload = {
            "v": SCHEMA_VERSION,
            "device_id": self.device_id,
            "ts": result.get("ts", time.time()),
            "seq": self._next_seq(),
            "patient_id": result.get("patient_id"),
            "person_present": result.get("person_present", True),
            "posture": result.get("posture_state", "unknown"),
            "activity": result.get("label", "unknown"),
            "activity_confidence": float(result.get("confidence", 0.0)),
            "risk": {
                "agitation": float(decision.get("agitation_score", 0.0)),
                "delirium": float(decision.get("delirium_risk", 0.0)),
                "respiratory": float(decision.get("respiratory_distress", 0.0)),
                "fall": float(decision.get("clinical_confidence", 0.0))
                         if decision.get("label") == "fall" else 0.0,
                "hand_proximity": float(decision.get("hand_proximity_risk", 0.0)),
            },
            "alert_level": decision.get("alert", "LOW_RISK"),
        }
        self._publish(self._topic_events, payload, qos=1)

    def publish_alert(self, alert_type: str, alert_level: str,
                      confidence: float, context: dict = None):
        """Publish a deduplicated clinical alert (immediate).

        Alerts are deduplicated per (alert_type) within a configurable
        window (default 30s) to prevent Flink from processing duplicate
        bursts when the same condition persists across frames.
        """
        now = time.time()

        # Deduplication: skip if same alert type fired within window
        last = self._last_alert_ts.get(alert_type, 0.0)
        if now - last < self._alert_dedup_window:
            log.debug("Alert %s deduplicated (%.1fs since last)", alert_type, now - last)
            return
        self._last_alert_ts[alert_type] = now

        qos = 2 if alert_level == "CRITICAL" else 1
        payload = {
            "v": SCHEMA_VERSION,
            "device_id": self.device_id,
            "ts": now,
            "seq": self._next_seq(),
            "alert_type": alert_type,
            "alert_level": alert_level,
            "confidence": float(confidence),
            "patient_id": None,  # filled by caller via context
            "context": context or {},
        }
        if context and "patient_id" in context:
            payload["patient_id"] = context["patient_id"]

        self._publish(self._topic_alerts, payload, qos=qos, is_critical=True)
        log.info("ALERT published: %s [%s] conf=%.2f", alert_type, alert_level, confidence)

    def publish_heartbeat(self, system_health: dict = None):
        """Publish device heartbeat (every 30s)."""
        payload = {
            "v": SCHEMA_VERSION,
            "device_id": self.device_id,
            "ts": time.time(),
            "seq": self._next_seq(),
            "system": system_health or {},
        }
        self._publish(self._topic_heartbeat, payload, qos=0)

    def publish_hourly(self, hourly_data: dict):
        """Publish pre-aggregated hourly summary.

        This is the most efficient data for Flink server-side reporting —
        one message per hour per device instead of 54K raw events.
        """
        hourly_data["v"] = SCHEMA_VERSION
        hourly_data["device_id"] = self.device_id
        hourly_data["seq"] = self._next_seq()
        self._publish(self._topic_hourly, hourly_data, qos=1)
        log.info("Hourly summary published to %s", self._topic_hourly)

    # ── Backward compatibility (used by eac.py during migration) ────────

    def publish_event(self, payload):
        """Legacy: publish raw event. Wraps in versioned envelope."""
        payload["v"] = SCHEMA_VERSION
        payload["seq"] = self._next_seq()
        payload["device_id"] = self.device_id
        self._publish(self._topic_events, payload, qos=1)

    def publish_clinical_event(self, decision_result, features=None):
        """Legacy: publish clinical event. Routes to new schema."""
        result = {
            "ts": decision_result.get("ts", time.time()),
            "label": decision_result.get("label", "unknown"),
            "confidence": decision_result.get("confidence", 0.0),
            "person_present": decision_result.get("person_present", True),
            "posture_state": decision_result.get("posture_state", "unknown"),
            "patient_id": decision_result.get("patient_id"),
            "decision": decision_result,
        }
        self.publish_status(result)

    def publish_critical_alert(self, alert_type, confidence, timestamp=None):
        """Legacy: publish critical alert. Routes to new dedup alert."""
        self.publish_alert(
            alert_type=alert_type,
            alert_level="CRITICAL",
            confidence=confidence,
            context={"timestamp": timestamp or time.time()},
        )

    def publish_report(self, report_data, report_type="hourly"):
        """Legacy: publish report. Routes to publish_hourly for hourly."""
        if report_type == "hourly":
            self.publish_hourly(report_data)
        else:
            # Monthly reports still go on a dedicated topic
            prefix = self.cfg.get("topic_prefix", "eam")
            topic = f"{prefix}/{self.device_id}/reports/{report_type}"
            report_data["v"] = SCHEMA_VERSION
            report_data["seq"] = self._next_seq()
            self._publish(topic, report_data, qos=1)

    # ── Shutdown ────────────────────────────────────────────────────────

    def shutdown(self):
        """Gracefully disconnect."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            log.info("MQTT client shutdown complete")
        except Exception:
            pass
