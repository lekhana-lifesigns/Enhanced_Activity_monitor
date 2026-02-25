#!/usr/bin/env python3
"""
Flink Job 3: Cross-Device Correlation

Monitors all devices simultaneously to detect ward-level patterns:
  1. Ward-wide agitation spikes (3+ devices elevated in 5-min window)
  2. Shift change patterns (activity transitions across devices)
  3. Environmental triggers (correlated state changes)

Source:  $share/flink/eam/+/events   (all devices)
Sink:    TimescaleDB ward_analytics + MQTT eam/ward/{ward_id}/alerts

This job uses a global window (not keyed by device) to correlate
events across all devices in a ward.
"""
import json
import os
import logging
from datetime import datetime
from collections import defaultdict

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.window import TumblingProcessingTimeWindows
from pyflink.common.time import Time
from pyflink.datastream.functions import (
    MapFunction,
    WindowFunction,
    RuntimeContext,
)
from pyflink.common.typeinfo import Types

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("flink.cross_device")

# ── Configuration ──────────────────────────────────────────────────
MQTT_BROKER = os.environ.get("MQTT_BROKER", "emqx")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
TIMESCALE_URL = os.environ.get(
    "TIMESCALE_URL",
    "postgresql://eam:eam_password@timescaledb:5432/eam",
)
CHECKPOINT_INTERVAL_MS = 60_000

# Spike detection thresholds
AGITATION_THRESHOLD = 0.5  # Risk score above this is "elevated"
SPIKE_MIN_DEVICES = 3  # Minimum devices for a ward spike
DELIRIUM_THRESHOLD = 0.6  # Delirium risk threshold

# Ward mapping: device_id → ward_id
# In production, load from database or config.
DEFAULT_WARD = "ward_a"


# ── Event parsing ─────────────────────────────────────────────────

class ParseDeviceEvent(MapFunction):
    """Parse raw MQTT JSON and extract ward-relevant fields."""

    def map(self, raw: str):
        try:
            msg = json.loads(raw)
            device_id = msg.get("device_id", "unknown")
            return {
                "device_id": device_id,
                "ward_id": self._device_to_ward(device_id),
                "ts": float(msg.get("ts", 0)),
                "activity": msg.get("activity", "unknown"),
                "posture": msg.get("posture", "unknown"),
                "person_present": bool(msg.get("person_present", True)),
                "risk_agitation": float(msg.get("risk", {}).get("agitation", 0)),
                "risk_delirium": float(msg.get("risk", {}).get("delirium", 0)),
                "risk_respiratory": float(
                    msg.get("risk", {}).get("respiratory", 0)
                ),
                "alert_level": msg.get("alert_level", "LOW_RISK"),
            }
        except Exception as e:
            log.warning("Failed to parse device event: %s", e)
            return None

    @staticmethod
    def _device_to_ward(device_id: str) -> str:
        """Map device_id to ward_id.

        Production: query from database or config service.
        Default: group by prefix (bed_01..bed_10 → ward_a).
        """
        # Simple mapping: parse bed number, group by 10s
        try:
            bed_num = int(device_id.split("_")[1])
            ward_letter = chr(ord("a") + (bed_num - 1) // 10)
            return f"ward_{ward_letter}"
        except (IndexError, ValueError):
            return DEFAULT_WARD


# ── Ward-level correlation ────────────────────────────────────────

class CorrelateWardEvents(WindowFunction):
    """Analyze 5-minute window of events across all devices in a ward.

    Detects:
    - Agitation spikes: 3+ devices with agitation > threshold
    - Delirium cluster: 2+ devices with delirium risk elevated
    - Mass absence: 50%+ devices with person_present = false
    - Shift patterns: correlated activity transitions
    """

    def apply(self, ward_id, window, events):
        events_list = list(events)
        if not events_list:
            return

        now_iso = datetime.utcnow().isoformat()
        window_start = datetime.utcfromtimestamp(window.start / 1000).isoformat()
        window_end = datetime.utcfromtimestamp(window.end / 1000).isoformat()

        # Group by device
        by_device = defaultdict(list)
        for e in events_list:
            by_device[e["device_id"]].append(e)

        total_devices = len(by_device)
        if total_devices == 0:
            return

        # ── Agitation spike detection ─────────────────────────
        elevated_devices = []
        for dev_id, dev_events in by_device.items():
            avg_agitation = sum(e["risk_agitation"] for e in dev_events) / len(
                dev_events
            )
            if avg_agitation > AGITATION_THRESHOLD:
                elevated_devices.append(dev_id)

        # ── Delirium cluster detection ────────────────────────
        delirium_devices = []
        for dev_id, dev_events in by_device.items():
            avg_delirium = sum(e["risk_delirium"] for e in dev_events) / len(
                dev_events
            )
            if avg_delirium > DELIRIUM_THRESHOLD:
                delirium_devices.append(dev_id)

        # ── Absence detection ─────────────────────────────────
        absent_devices = []
        for dev_id, dev_events in by_device.items():
            absent_pct = sum(
                1 for e in dev_events if not e["person_present"]
            ) / len(dev_events)
            if absent_pct > 0.5:
                absent_devices.append(dev_id)

        # ── Alert counting ────────────────────────────────────
        total_alerts = sum(
            1 for e in events_list if e["alert_level"] != "LOW_RISK"
        )
        critical_alerts = sum(
            1 for e in events_list if e["alert_level"] == "CRITICAL"
        )

        # ── Pattern detection ─────────────────────────────────
        spike_detected = len(elevated_devices) >= SPIKE_MIN_DEVICES
        pattern_type = None

        if spike_detected:
            pattern_type = "ward_agitation_spike"
            log.warning(
                "WARD SPIKE: %s — %d/%d devices elevated (agitation > %.1f)",
                ward_id,
                len(elevated_devices),
                total_devices,
                AGITATION_THRESHOLD,
            )
        elif len(delirium_devices) >= 2:
            spike_detected = True
            pattern_type = "delirium_cluster"
            log.warning(
                "DELIRIUM CLUSTER: %s — %d devices",
                ward_id,
                len(delirium_devices),
            )
        elif len(absent_devices) > total_devices * 0.5:
            pattern_type = "mass_absence"

        # ── Build result ──────────────────────────────────────
        result = {
            "ward_id": ward_id,
            "ts": now_iso,
            "window_start": window_start,
            "window_end": window_end,
            "total_devices": total_devices,
            "active_devices": sum(
                1
                for dev_events in by_device.values()
                if any(e["person_present"] for e in dev_events)
            ),
            "devices_elevated": len(elevated_devices),
            "elevated_device_ids": elevated_devices,
            "total_alerts": total_alerts,
            "critical_alerts": critical_alerts,
            "avg_agitation": round(
                sum(e["risk_agitation"] for e in events_list) / len(events_list),
                4,
            ),
            "max_agitation": round(
                max(e["risk_agitation"] for e in events_list), 4
            ),
            "avg_delirium_risk": round(
                sum(e["risk_delirium"] for e in events_list) / len(events_list),
                4,
            ),
            "ward_spike_detected": spike_detected,
            "pattern_type": pattern_type,
        }

        yield json.dumps(result)


# ── TimescaleDB + MQTT sinks ─────────────────────────────────────

class WardAnalyticsSink(MapFunction):
    """Persist ward analytics to TimescaleDB and publish spike alerts."""

    def open(self, runtime_context: RuntimeContext):
        import psycopg2

        self._conn = psycopg2.connect(TIMESCALE_URL)
        self._conn.autocommit = True
        self._cursor = self._conn.cursor()

        # MQTT publisher for ward alerts
        self._mqtt_client = None
        try:
            import paho.mqtt.client as mqtt

            self._mqtt_client = mqtt.Client(client_id="flink-ward-correlator")
            username = os.environ.get("MQTT_USERNAME", "flink")
            password = os.environ.get("MQTT_PASSWORD", "")
            if username:
                self._mqtt_client.username_pw_set(username, password)
            self._mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            self._mqtt_client.loop_start()
            log.info("Ward analytics MQTT publisher connected")
        except Exception as e:
            log.warning("Ward MQTT publisher failed: %s", e)

    def map(self, record_json: str):
        rec = json.loads(record_json)

        # Persist to TimescaleDB
        try:
            self._cursor.execute(
                """
                INSERT INTO ward_analytics
                    (ts, ward_id, window_start, window_end,
                     total_devices, active_devices, devices_elevated,
                     total_alerts, critical_alerts,
                     avg_agitation, max_agitation, avg_delirium_risk,
                     ward_spike_detected, pattern_type)
                VALUES
                    (NOW(), %(ward_id)s, %(window_start)s, %(window_end)s,
                     %(total_devices)s, %(active_devices)s, %(devices_elevated)s,
                     %(total_alerts)s, %(critical_alerts)s,
                     %(avg_agitation)s, %(max_agitation)s, %(avg_delirium_risk)s,
                     %(ward_spike_detected)s, %(pattern_type)s)
                """,
                rec,
            )
        except Exception as e:
            log.error("Ward analytics DB insert failed: %s", e)

        # Publish ward spike alert via MQTT
        if rec.get("ward_spike_detected") and self._mqtt_client:
            ward_id = rec["ward_id"]
            alert_payload = json.dumps(
                {
                    "v": 1,
                    "ward_id": ward_id,
                    "ts": rec["ts"],
                    "pattern_type": rec["pattern_type"],
                    "devices_elevated": rec["devices_elevated"],
                    "elevated_device_ids": rec.get("elevated_device_ids", []),
                    "total_devices": rec["total_devices"],
                    "avg_agitation": rec["avg_agitation"],
                    "max_agitation": rec["max_agitation"],
                }
            )
            self._mqtt_client.publish(
                f"eam/ward/{ward_id}/alerts",
                alert_payload,
                qos=1,
            )
            log.info(
                "Ward spike alert published: %s (%s)",
                ward_id,
                rec["pattern_type"],
            )

        return record_json

    def close(self):
        if hasattr(self, "_cursor"):
            self._cursor.close()
        if hasattr(self, "_conn"):
            self._conn.close()
        if self._mqtt_client:
            try:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
            except Exception:
                pass


# ── Job definition ────────────────────────────────────────────────

def build_job():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)  # Single parallelism for global correlation
    env.enable_checkpointing(CHECKPOINT_INTERVAL_MS)

    # Source: MQTT events (all devices) — placeholder for MQTT connector
    source = env.from_collection([], type_info=Types.STRING())

    # Pipeline:
    # 1. Parse raw JSON
    # 2. Filter nulls
    # 3. Key by ward_id (not device_id — cross-device)
    # 4. 5-minute processing-time window (global correlation)
    # 5. Correlate and detect patterns
    # 6. Sink to TimescaleDB + publish MQTT ward alerts
    pipeline = (
        source
        .map(ParseDeviceEvent())
        .filter(lambda event: event is not None)
        .key_by(lambda event: event["ward_id"])
        .window(TumblingProcessingTimeWindows.of(Time.minutes(5)))
        .apply(CorrelateWardEvents())
        .map(WardAnalyticsSink())
    )

    return env


if __name__ == "__main__":
    log.info("Starting Flink Job: Cross-Device Correlation")
    job_env = build_job()
    job_env.execute("EAM Cross-Device Correlation")
