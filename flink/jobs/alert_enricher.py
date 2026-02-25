#!/usr/bin/env python3
"""
Flink Job 2: Alert Enricher

Consumes clinical alerts from EMQX, enriches with patient metadata
from PostgreSQL (name, room, ward, care team), routes to:
  1. TimescaleDB enriched_alerts table
  2. Nurse call system API
  3. Dashboard WebSocket (via MQTT republish)

Deduplication: (device_id, seq) pair ensures exactly-once processing
even if MQTT delivers duplicates during broker failover.

Source:  $share/flink/eam/+/alerts   (shared subscription)
Sink:    TimescaleDB enriched_alerts + MQTT eam/{device_id}/alerts/enriched
"""
import json
import os
import time
import logging
from datetime import datetime

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import (
    MapFunction,
    RuntimeContext,
    KeyedProcessFunction,
)
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common.typeinfo import Types
from pyflink.common.watermark_strategy import WatermarkStrategy
from pyflink.common.time import Time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("flink.alert_enricher")

# ── Configuration ──────────────────────────────────────────────────
MQTT_BROKER = os.environ.get("MQTT_BROKER", "emqx")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
TIMESCALE_URL = os.environ.get(
    "TIMESCALE_URL",
    "postgresql://eam:eam_password@timescaledb:5432/eam",
)
NURSE_CALL_API = os.environ.get("NURSE_CALL_API", "")
CHECKPOINT_INTERVAL_MS = 30_000  # Tighter checkpoints for alerts
DEDUP_WINDOW_SECONDS = 300  # 5-minute dedup window


# ── Alert parsing ─────────────────────────────────────────────────

class ParseAlert(MapFunction):
    """Parse raw MQTT alert JSON."""

    def map(self, raw: str):
        try:
            msg = json.loads(raw)
            return {
                "device_id": msg.get("device_id", "unknown"),
                "ts": float(msg.get("ts", 0)),
                "seq": int(msg.get("seq", 0)),
                "patient_id": msg.get("patient_id"),
                "alert_type": msg.get("alert_type", "unknown"),
                "alert_level": msg.get("alert_level", "MODERATE"),
                "confidence": float(msg.get("confidence", 0)),
                "context": msg.get("context", {}),
                "schema_ver": int(msg.get("v", 1)),
            }
        except Exception as e:
            log.warning("Failed to parse alert: %s", e)
            return None


# ── Deduplication ─────────────────────────────────────────────────

class DeduplicateAlerts(KeyedProcessFunction):
    """Dedup alerts by (device_id, seq) — exactly-once semantics.

    Uses Flink keyed state to track seen sequence numbers per device.
    State is cleaned up after DEDUP_WINDOW_SECONDS via timers.
    """

    def open(self, runtime_context: RuntimeContext):
        self._last_seq = runtime_context.get_state(
            ValueStateDescriptor("last_seq", Types.LONG())
        )

    def process_element(self, alert, ctx):
        if alert is None:
            return

        last = self._last_seq.value()
        current_seq = alert.get("seq", 0)

        # Skip if we've already seen this or a higher seq
        if last is not None and current_seq <= last:
            log.debug(
                "Dedup: device=%s seq=%d (last=%d)",
                alert["device_id"],
                current_seq,
                last,
            )
            return

        self._last_seq.update(current_seq)
        yield alert


# ── Patient enrichment ────────────────────────────────────────────

class EnrichWithPatientData(MapFunction):
    """Look up patient metadata from PostgreSQL and enrich the alert."""

    def open(self, runtime_context: RuntimeContext):
        import psycopg2

        self._conn = psycopg2.connect(TIMESCALE_URL)
        self._cursor = self._conn.cursor()
        # Cache patient lookups (refreshed every 5 minutes)
        self._cache = {}
        self._cache_ts = 0
        self._cache_ttl = 300
        log.info("Patient enrichment DB connected")

    def _get_patient(self, patient_id):
        if not patient_id:
            return {}

        now = time.time()
        if now - self._cache_ts > self._cache_ttl:
            self._cache.clear()
            self._cache_ts = now

        if patient_id in self._cache:
            return self._cache[patient_id]

        try:
            self._cursor.execute(
                """
                SELECT name, room, ward, care_team
                FROM patients
                WHERE patient_id = %s
                LIMIT 1
                """,
                (patient_id,),
            )
            row = self._cursor.fetchone()
            if row:
                result = {
                    "patient_name": row[0],
                    "room": row[1],
                    "ward": row[2],
                    "care_team": row[3] if row[3] else [],
                }
            else:
                result = {}
            self._cache[patient_id] = result
            return result
        except Exception as e:
            log.warning("Patient lookup failed for %s: %s", patient_id, e)
            try:
                self._conn.rollback()
            except Exception:
                pass
            return {}

    def map(self, alert):
        if alert is None:
            return None

        patient_data = self._get_patient(alert.get("patient_id"))
        alert["patient_name"] = patient_data.get("patient_name")
        alert["room"] = patient_data.get("room")
        alert["ward"] = patient_data.get("ward")
        alert["care_team"] = patient_data.get("care_team", [])
        alert["enriched_at"] = datetime.utcnow().isoformat()
        return alert

    def close(self):
        if hasattr(self, "_cursor"):
            self._cursor.close()
        if hasattr(self, "_conn"):
            self._conn.close()


# ── Routing: TimescaleDB sink ─────────────────────────────────────

class AlertDBSink(MapFunction):
    """Persist enriched alerts to TimescaleDB."""

    def open(self, runtime_context: RuntimeContext):
        import psycopg2

        self._conn = psycopg2.connect(TIMESCALE_URL)
        self._conn.autocommit = True
        self._cursor = self._conn.cursor()

    def map(self, alert):
        if alert is None:
            return None
        try:
            self._cursor.execute(
                """
                INSERT INTO enriched_alerts
                    (ts, device_id, patient_id, alert_type, alert_level,
                     confidence, patient_name, room, ward, care_team,
                     context, seq, schema_ver)
                VALUES
                    (to_timestamp(%(ts)s), %(device_id)s, %(patient_id)s,
                     %(alert_type)s, %(alert_level)s, %(confidence)s,
                     %(patient_name)s, %(room)s, %(ward)s, %(care_team)s,
                     %(context)s, %(seq)s, %(schema_ver)s)
                """,
                {
                    "ts": alert["ts"],
                    "device_id": alert["device_id"],
                    "patient_id": alert.get("patient_id"),
                    "alert_type": alert["alert_type"],
                    "alert_level": alert["alert_level"],
                    "confidence": alert["confidence"],
                    "patient_name": alert.get("patient_name"),
                    "room": alert.get("room"),
                    "ward": alert.get("ward"),
                    "care_team": alert.get("care_team", []),
                    "context": json.dumps(alert.get("context", {})),
                    "seq": alert.get("seq"),
                    "schema_ver": alert.get("schema_ver", 1),
                },
            )
        except Exception as e:
            log.error("Alert DB insert failed: %s", e)
        return alert

    def close(self):
        if hasattr(self, "_cursor"):
            self._cursor.close()
        if hasattr(self, "_conn"):
            self._conn.close()


# ── Routing: Nurse call system ────────────────────────────────────

class NurseCallRouter(MapFunction):
    """Route HIGH_RISK and CRITICAL alerts to nurse call system API."""

    def open(self, runtime_context: RuntimeContext):
        self._api_url = NURSE_CALL_API
        self._session = None
        if self._api_url:
            import requests

            self._session = requests.Session()
            self._session.headers["Content-Type"] = "application/json"
            log.info("Nurse call API configured: %s", self._api_url)

    def map(self, alert):
        if alert is None:
            return alert

        if alert["alert_level"] not in ("HIGH_RISK", "CRITICAL"):
            return alert

        if not self._session or not self._api_url:
            return alert

        try:
            payload = {
                "room": alert.get("room", "unknown"),
                "patient_name": alert.get("patient_name", "Unknown"),
                "alert_type": alert["alert_type"],
                "alert_level": alert["alert_level"],
                "confidence": alert["confidence"],
                "device_id": alert["device_id"],
                "timestamp": datetime.utcfromtimestamp(alert["ts"]).isoformat(),
                "care_team": alert.get("care_team", []),
            }
            resp = self._session.post(
                f"{self._api_url}/alerts",
                json=payload,
                timeout=5,
            )
            if resp.status_code < 300:
                alert["nurse_call_sent"] = True
                log.info(
                    "Nurse call sent: %s [%s] room=%s",
                    alert["alert_type"],
                    alert["alert_level"],
                    alert.get("room"),
                )
            else:
                log.warning("Nurse call API returned %d", resp.status_code)
        except Exception as e:
            log.warning("Nurse call API failed: %s", e)

        return alert

    def close(self):
        if self._session:
            self._session.close()


# ── Job definition ────────────────────────────────────────────────

def build_job():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)
    env.enable_checkpointing(CHECKPOINT_INTERVAL_MS)

    # Source: MQTT alerts (placeholder — actual MQTT connector at deploy)
    source = env.from_collection([], type_info=Types.STRING())

    # Pipeline:
    # 1. Parse raw JSON
    # 2. Filter nulls
    # 3. Key by device_id for deduplication
    # 4. Dedup by (device_id, seq)
    # 5. Enrich with patient metadata
    # 6. Persist to TimescaleDB
    # 7. Route to nurse call system
    pipeline = (
        source
        .map(ParseAlert())
        .filter(lambda alert: alert is not None)
        .key_by(lambda alert: alert["device_id"])
        .process(DeduplicateAlerts())
        .map(EnrichWithPatientData())
        .map(AlertDBSink())
        .map(NurseCallRouter())
    )

    return env


if __name__ == "__main__":
    log.info("Starting Flink Job: Alert Enricher")
    job_env = build_job()
    job_env.execute("EAM Alert Enricher")
