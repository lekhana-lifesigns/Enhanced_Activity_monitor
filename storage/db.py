# storage/db.py
import sqlite3
import json
import os
import logging
from datetime import timedelta

log = logging.getLogger("db")

DB_PATH = "storage/events.db"

class LocalDB:
    def __init__(self, path=DB_PATH, enable_wal=True):
        """
        Initialize local SQLite database for event storage.

        Args:
            path: Database file path
            enable_wal: Enable WAL mode for better concurrency (SSD optimized)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.path = path

        # ══════════════════════════════════════════════════════════════════
        # SSD OPTIMIZATION: WAL mode + PRAGMA settings for edge deployment
        # ══════════════════════════════════════════════════════════════════
        if enable_wal:
            try:
                # WAL mode: Better concurrency, reduced write amplification
                self.conn.execute("PRAGMA journal_mode=WAL")
                # Synchronous NORMAL: Balance between safety and performance
                # FULL is safer but slower; OFF is fastest but risky
                self.conn.execute("PRAGMA synchronous=NORMAL")
                # Larger cache for better read performance (10MB)
                self.conn.execute("PRAGMA cache_size=-10000")
                # Memory-mapped I/O for faster reads (64MB)
                self.conn.execute("PRAGMA mmap_size=67108864")
                # Temp store in memory (faster, uses RAM)
                self.conn.execute("PRAGMA temp_store=MEMORY")
                log.info("SQLite WAL mode enabled with SSD optimizations")
            except Exception as e:
                log.warning("Failed to enable WAL mode: %s", e)

        self._create()
        log.info("Local database initialized: %s", path)

    def _create(self):
        """Create all tables from schema if they don't exist."""
        # Events table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            device TEXT NOT NULL,
            label TEXT,
            confidence REAL,
            payload TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # Alerts table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            device TEXT NOT NULL,
            alert_level TEXT NOT NULL,
            label TEXT,
            agitation_score REAL,
            delirium_risk REAL,
            respiratory_distress REAL,
            hand_proximity_risk REAL,
            payload TEXT,
            acknowledged BOOLEAN DEFAULT 0,
            acknowledged_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # System health table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS system_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            device_id TEXT NOT NULL,
            cpu_percent REAL,
            ram_percent REAL,
            temp_celsius REAL,
            disk_percent REAL,
            inference_ms REAL,
            fps REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # Patient sessions table (multi-device patient tracking)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS patient_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            bed_id TEXT,
            start_ts REAL NOT NULL,
            end_ts REAL,
            status TEXT DEFAULT 'active',
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        # Audit log table (HIPAA compliance)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            action TEXT NOT NULL,
            user_id TEXT,
            patient_id TEXT,
            details TEXT,
            ip_address TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        # Patient consent table (GDPR / data governance)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS patient_consent (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            consent_type TEXT NOT NULL,
            granted BOOLEAN DEFAULT 0,
            granted_at DATETIME,
            revoked_at DATETIME,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        # Patient baselines table (DBSCAN + PCA models per patient)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS patient_baselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            baseline_type TEXT NOT NULL DEFAULT 'features',
            model_blob BLOB NOT NULL,
            n_samples INTEGER DEFAULT 0,
            n_clusters INTEGER DEFAULT 0,
            fit_count INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(patient_id, device_id, baseline_type)
        )""")

        # Hourly aggregates table (efficient reporting)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS hourly_aggregates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour_start REAL NOT NULL,
            hour_end REAL NOT NULL,
            device_id TEXT NOT NULL,
            patient_id TEXT,
            total_frames INTEGER DEFAULT 0,
            frames_with_person INTEGER DEFAULT 0,
            total_observation_seconds REAL DEFAULT 0,
            posture_seconds TEXT,
            dominant_posture TEXT,
            support_surface_seconds TEXT,
            time_in_bed_seconds REAL DEFAULT 0,
            time_in_bed_pct REAL DEFAULT 0,
            bed_exit_count INTEGER DEFAULT 0,
            bed_exit_attempts INTEGER DEFAULT 0,
            fall_events INTEGER DEFAULT 0,
            immobility_periods INTEGER DEFAULT 0,
            immobility_total_seconds REAL DEFAULT 0,
            distress_periods INTEGER DEFAULT 0,
            distress_total_seconds REAL DEFAULT 0,
            max_agitation_score REAL DEFAULT 0,
            avg_agitation_score REAL DEFAULT 0,
            max_delirium_risk REAL DEFAULT 0,
            avg_delirium_risk REAL DEFAULT 0,
            max_respiratory_distress REAL DEFAULT 0,
            avg_respiratory_distress REAL DEFAULT 0,
            max_fall_risk REAL DEFAULT 0,
            avg_fall_risk REAL DEFAULT 0,
            activity_seconds TEXT,
            dominant_activity TEXT,
            alert_counts TEXT,
            critical_alerts INTEGER DEFAULT 0,
            high_risk_alerts INTEGER DEFAULT 0,
            avg_pose_confidence REAL DEFAULT 0,
            keypoint_visibility_pct REAL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(device_id, hour_start)
        )""")

        # Create indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_device ON events(device)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_label ON events(label)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_device ON alerts(device)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(alert_level)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_health_ts ON system_health(ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_health_device ON system_health(device_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_patient ON patient_sessions(patient_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_device ON patient_sessions(device_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON patient_sessions(status)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_log(patient_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_consent_patient ON patient_consent(patient_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_baselines_patient ON patient_baselines(patient_id, device_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_hourly_device_time ON hourly_aggregates(device_id, hour_start)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_hourly_patient ON hourly_aggregates(patient_id)")

        self.conn.commit()

    def _build_where_clause(self, device=None, start_ts=None, end_ts=None, extra_filters=None):
        """Build WHERE clause with common filters."""
        clauses = []
        params = []
        if device:
            clauses.append("device = ?")
            params.append(device)
        if start_ts:
            clauses.append("ts >= ?")
            params.append(start_ts)
        if end_ts:
            clauses.append("ts <= ?")
            params.append(end_ts)
        if extra_filters:
            for col, val in extra_filters.items():
                if val is not None:
                    clauses.append(f"{col} = ?")
                    params.append(val)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return where, params

    def insert_event(self, device, label, confidence, payload):
        """
        Insert event into database.
        
        Args:
            device: Device ID
            label: Activity label
            confidence: Confidence score
            payload: Full event payload (dict)
        """
        try:
            ts = payload.get("ts", payload.get("timestamp", 0.0)) if payload else 0.0
            
            # Convert numpy arrays and other non-serializable types to lists
            def convert_to_serializable(obj):
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, (bool, np.bool_)):
                    return bool(obj)  # Ensure boolean is JSON-serializable
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, (type(None), str, int, float)):
                    return obj  # Already JSON-serializable
                else:
                    # Try to convert to string as last resort
                    try:
                        return str(obj)
                    except (TypeError, ValueError):
                        return None
            
            # Clean payload for JSON serialization
            clean_payload = convert_to_serializable(payload)
            
            self.conn.execute(
                "INSERT INTO events (ts, device, label, confidence, payload) VALUES (?,?,?,?,?)",
                (ts, device, label, float(confidence), json.dumps(clean_payload))
            )
            self.conn.commit()
        except sqlite3.OperationalError as e:
            error_msg = str(e).lower()
            if "disk" in error_msg or "full" in error_msg or "space" in error_msg:
                log.critical("Disk full - cannot write to database. Free space required.")
                return False
            log.exception("Database operational error: %s", e)
        except Exception as e:
            log.exception("Failed to insert event: %s", e)

    def query_events(self, device=None, start_ts=None, end_ts=None, limit=1000):
        """
        Query events from database.
        
        Args:
            device: Filter by device ID (optional)
            start_ts: Start timestamp (optional)
            end_ts: End timestamp (optional)
            limit: Maximum number of results
        
        Returns:
            List of event dictionaries
        """
        where, params = self._build_where_clause(device, start_ts, end_ts)
        query = "SELECT ts, device, label, confidence, payload FROM events" + where + " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        
        try:
            cursor = self.conn.execute(query, params)
            events = []
            for row in cursor:
                events.append({
                    "ts": row[0],
                    "device": row[1],
                    "label": row[2],
                    "confidence": row[3],
                    "payload": json.loads(row[4])
                })
            return events
        except Exception as e:
            log.exception("Failed to query events: %s", e)
            return []

    def cleanup_old_events(self, days=30):
        """
        Delete events older than specified days.

        Args:
            days: Number of days to keep
        """
        import time
        cutoff_ts = time.time() - (days * 24 * 60 * 60)
        try:
            cursor = self.conn.execute("DELETE FROM events WHERE ts < ?", (cutoff_ts,))
            deleted = cursor.rowcount
            self.conn.commit()
            log.info("Deleted %d old events (older than %d days)", deleted, days)
            return deleted
        except Exception as e:
            log.exception("Failed to cleanup old events: %s", e)
            return 0

    def run_full_cleanup(self, events_days=30, alerts_days=90, health_days=14, audit_days=365):
        """
        Run full cleanup on all tables with configurable retention periods.

        Args:
            events_days: Retain events for N days (default: 30)
            alerts_days: Retain alerts for N days (default: 90, longer for clinical review)
            health_days: Retain system_health for N days (default: 14)
            audit_days: Retain audit_log for N days (default: 365, HIPAA compliance)

        Returns:
            dict: Deleted counts per table
        """
        import time as _time
        results = {
            "events": 0,
            "alerts": 0,
            "system_health": 0,
            "audit_log": 0,
            "hourly_aggregates": 0,
        }

        now = _time.time()

        try:
            # Events cleanup
            cutoff = now - (events_days * 86400)
            cursor = self.conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
            results["events"] = cursor.rowcount

            # Alerts cleanup (longer retention for clinical review)
            cutoff = now - (alerts_days * 86400)
            cursor = self.conn.execute("DELETE FROM alerts WHERE ts < ?", (cutoff,))
            results["alerts"] = cursor.rowcount

            # System health cleanup
            cutoff = now - (health_days * 86400)
            cursor = self.conn.execute("DELETE FROM system_health WHERE ts < ?", (cutoff,))
            results["system_health"] = cursor.rowcount

            # Audit log cleanup (HIPAA: minimum 6 years, we use 1 year default)
            cutoff = now - (audit_days * 86400)
            cursor = self.conn.execute("DELETE FROM audit_log WHERE ts < ?", (cutoff,))
            results["audit_log"] = cursor.rowcount

            # Hourly aggregates cleanup (keep 90 days for reporting)
            cutoff = now - (90 * 86400)
            cursor = self.conn.execute("DELETE FROM hourly_aggregates WHERE hour_start < ?", (cutoff,))
            results["hourly_aggregates"] = cursor.rowcount

            self.conn.commit()

            # VACUUM to reclaim disk space (SSD optimization)
            total_deleted = sum(results.values())
            if total_deleted > 1000:
                try:
                    self.conn.execute("VACUUM")
                    log.info("Database vacuumed after cleanup (%d rows deleted)", total_deleted)
                except Exception as e:
                    log.debug("VACUUM failed (may be in transaction): %s", e)

            log.info("Full cleanup complete: events=%d, alerts=%d, health=%d, audit=%d, hourly=%d",
                     results["events"], results["alerts"], results["system_health"],
                     results["audit_log"], results["hourly_aggregates"])

            return results

        except Exception as e:
            log.exception("Full cleanup failed: %s", e)
            return results

    def get_database_size(self):
        """Get database file size in bytes."""
        try:
            if os.path.exists(self.path):
                size = os.path.getsize(self.path)
                # Also check WAL file if exists
                wal_path = self.path + "-wal"
                if os.path.exists(wal_path):
                    size += os.path.getsize(wal_path)
                return size
            return 0
        except Exception as e:
            log.debug("Failed to get database size: %s", e)
            return 0

    def insert_alert(self, device, alert_level, label=None, agitation_score=None,
                     delirium_risk=None, respiratory_distress=None, hand_proximity_risk=None, payload=None):
        """Insert alert into database."""
        try:
            ts = payload.get("ts", payload.get("timestamp", 0.0)) if payload else 0.0
            self.conn.execute("""
                INSERT INTO alerts (ts, device, alert_level, label, agitation_score, 
                                  delirium_risk, respiratory_distress, hand_proximity_risk, payload)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (ts, device, alert_level, label, agitation_score, delirium_risk,
                  respiratory_distress, hand_proximity_risk, json.dumps(payload) if payload else None))
            self.conn.commit()
        except Exception as e:
            log.exception("Failed to insert alert: %s", e)

    def query_alerts(self, device=None, start_ts=None, end_ts=None, alert_level=None, limit=1000):
        """Query alerts from database."""
        where, params = self._build_where_clause(device, start_ts, end_ts, {"alert_level": alert_level})
        query = "SELECT id, ts, device, alert_level, label, agitation_score, delirium_risk, respiratory_distress, hand_proximity_risk, payload, acknowledged FROM alerts" + where + " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        
        try:
            cursor = self.conn.execute(query, params)
            alerts = []
            for row in cursor:
                alerts.append({
                    "id": row[0],
                    "ts": row[1],
                    "device": row[2],
                    "alert_level": row[3],
                    "label": row[4],
                    "agitation_score": row[5],
                    "delirium_risk": row[6],
                    "respiratory_distress": row[7],
                    "hand_proximity_risk": row[8],
                    "payload": json.loads(row[9]) if row[9] else None,
                    "acknowledged": bool(row[10]) if len(row) > 10 else False
                })
            return alerts
        except Exception as e:
            log.exception("Failed to query alerts: %s", e)
            return []
    
    def acknowledge_alert(self, alert_id):
        """Acknowledge an alert."""
        try:
            from datetime import datetime
            self.conn.execute(
                "UPDATE alerts SET acknowledged = 1, acknowledged_at = ? WHERE id = ?",
                (datetime.now().isoformat(), alert_id)
            )
            self.conn.commit()
            log.info("Alert %d acknowledged", alert_id)
        except Exception as e:
            log.exception("Failed to acknowledge alert: %s", e)
    
    def query_patient_sessions(self, patient_id=None, device_id=None, status='active'):
        """Query patient sessions."""
        query = "SELECT id, patient_id, device_id, bed_id, start_ts, end_ts, status, metadata FROM patient_sessions WHERE 1=1"
        params = []
        
        if patient_id:
            query += " AND patient_id = ?"
            params.append(patient_id)
        if device_id:
            query += " AND device_id = ?"
            params.append(device_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY start_ts DESC"
        
        try:
            cursor = self.conn.execute(query, params)
            sessions = []
            for row in cursor:
                sessions.append({
                    "id": row[0],
                    "patient_id": row[1],
                    "device_id": row[2],
                    "bed_id": row[3],
                    "start_ts": row[4],
                    "end_ts": row[5],
                    "status": row[6],
                    "metadata": json.loads(row[7]) if row[7] else {}
                })
            return sessions
        except Exception as e:
            log.exception("Failed to query patient sessions: %s", e)
            return []

    def get_event_statistics(self, device=None, start_ts=None, end_ts=None):
        """Get aggregated statistics for events in time range."""
        where, params = self._build_where_clause(device, start_ts, end_ts)
        query = """
            SELECT
                COUNT(*) as total_events,
                COUNT(DISTINCT label) as unique_labels,
                AVG(confidence) as avg_confidence,
                MIN(ts) as first_ts,
                MAX(ts) as last_ts
            FROM events""" + where
        
        try:
            cursor = self.conn.execute(query, params)
            row = cursor.fetchone()
            if row:
                # Handle None values from AVG() when no rows match
                avg_conf = row[2] if row[2] is not None else 0.0
                return {
                    "total_events": row[0] or 0,
                    "unique_labels": row[1] or 0,
                    "avg_confidence": float(avg_conf) if avg_conf is not None else 0.0,
                    "first_ts": row[3],
                    "last_ts": row[4]
                }
            return {"total_events": 0, "unique_labels": 0, "avg_confidence": 0.0}
        except Exception as e:
            log.exception("Failed to get event statistics: %s", e)
            return {}

    def get_alert_statistics(self, device=None, start_ts=None, end_ts=None):
        """Get aggregated statistics for alerts in time range."""
        where, params = self._build_where_clause(device, start_ts, end_ts)
        query = """
            SELECT
                alert_level,
                COUNT(*) as count,
                AVG(agitation_score) as avg_agitation,
                AVG(delirium_risk) as avg_delirium,
                AVG(respiratory_distress) as avg_respiratory,
                AVG(hand_proximity_risk) as avg_hand_proximity
            FROM alerts""" + where + " GROUP BY alert_level"
        
        try:
            cursor = self.conn.execute(query, params)
            stats = {}
            for row in cursor:
                stats[row[0]] = {
                    "count": row[1],
                    "avg_agitation": row[2],
                    "avg_delirium": row[3],
                    "avg_respiratory": row[4],
                    "avg_hand_proximity": row[5]
                }
            return stats
        except Exception as e:
            log.exception("Failed to get alert statistics: %s", e)
            return {}

    def insert_patient_session(self, patient_id, device_id, bed_id=None, metadata=None):
        """Start a new patient session."""
        import time as _time
        try:
            self.conn.execute(
                "INSERT INTO patient_sessions (patient_id, device_id, bed_id, start_ts, status, metadata) VALUES (?,?,?,?,?,?)",
                (patient_id, device_id, bed_id or device_id, _time.time(), 'active',
                 json.dumps(metadata) if metadata else None)
            )
            self.conn.commit()
            log.info("Patient session started: %s on %s", patient_id, device_id)
        except Exception as e:
            log.exception("Failed to insert patient session: %s", e)

    def end_patient_session(self, patient_id, device_id=None):
        """End an active patient session."""
        import time as _time
        try:
            query = "UPDATE patient_sessions SET status = 'ended', end_ts = ? WHERE patient_id = ? AND status = 'active'"
            params = [_time.time(), patient_id]
            if device_id:
                query += " AND device_id = ?"
                params.append(device_id)
            self.conn.execute(query, params)
            self.conn.commit()
            log.info("Patient session ended: %s", patient_id)
        except Exception as e:
            log.exception("Failed to end patient session: %s", e)

    def insert_audit_log(self, action, user_id=None, patient_id=None, details=None, ip_address=None):
        """Insert audit log entry (HIPAA compliance)."""
        import time as _time
        try:
            self.conn.execute(
                "INSERT INTO audit_log (ts, action, user_id, patient_id, details, ip_address) VALUES (?,?,?,?,?,?)",
                (_time.time(), action, user_id, patient_id, details, ip_address)
            )
            self.conn.commit()
        except Exception as e:
            log.exception("Failed to insert audit log: %s", e)

    # ------------------------------------------------------------------
    # Patient baseline persistence
    # ------------------------------------------------------------------

    def save_baseline(self, patient_id, device_id, model_blob, n_samples=0,
                      n_clusters=0, fit_count=0, baseline_type="features"):
        """Save or update a patient baseline model."""
        try:
            from datetime import datetime
            now = datetime.now().isoformat()
            self.conn.execute("""
                INSERT INTO patient_baselines
                    (patient_id, device_id, baseline_type, model_blob,
                     n_samples, n_clusters, fit_count, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(patient_id, device_id, baseline_type)
                DO UPDATE SET
                    model_blob = excluded.model_blob,
                    n_samples = excluded.n_samples,
                    n_clusters = excluded.n_clusters,
                    fit_count = excluded.fit_count,
                    updated_at = excluded.updated_at
            """, (patient_id, device_id, baseline_type, model_blob,
                  n_samples, n_clusters, fit_count, now, now))
            self.conn.commit()
            log.info("Baseline saved: patient=%s device=%s type=%s (%d samples, %d clusters)",
                     patient_id, device_id, baseline_type, n_samples, n_clusters)
        except Exception as e:
            log.exception("Failed to save baseline: %s", e)

    def load_baseline(self, patient_id, device_id, baseline_type="features"):
        """Load a patient baseline model. Returns model_blob bytes or None."""
        try:
            cursor = self.conn.execute(
                "SELECT model_blob, n_samples, n_clusters, fit_count, updated_at "
                "FROM patient_baselines WHERE patient_id = ? AND device_id = ? AND baseline_type = ?",
                (patient_id, device_id, baseline_type)
            )
            row = cursor.fetchone()
            if row:
                log.info("Baseline loaded: patient=%s device=%s (%d samples, updated=%s)",
                         patient_id, device_id, row[1], row[4])
                return {
                    "model_blob": row[0],
                    "n_samples": row[1],
                    "n_clusters": row[2],
                    "fit_count": row[3],
                    "updated_at": row[4]
                }
            return None
        except Exception as e:
            log.exception("Failed to load baseline: %s", e)
            return None

    def delete_baseline(self, patient_id, device_id, baseline_type="features"):
        """Delete a patient baseline model."""
        try:
            self.conn.execute(
                "DELETE FROM patient_baselines WHERE patient_id = ? AND device_id = ? AND baseline_type = ?",
                (patient_id, device_id, baseline_type)
            )
            self.conn.commit()
            log.info("Baseline deleted: patient=%s device=%s type=%s",
                     patient_id, device_id, baseline_type)
        except Exception as e:
            log.exception("Failed to delete baseline: %s", e)

    # ------------------------------------------------------------------
    # Hourly aggregates for efficient reporting
    # ------------------------------------------------------------------

    def insert_hourly_aggregate(self, metrics):
        """
        Insert or update hourly aggregate data.

        Args:
            metrics: HourlyMetrics dataclass or dict with aggregate data
        """
        try:
            # Convert dataclass to dict if needed
            if hasattr(metrics, 'to_dict'):
                data = metrics.to_dict()
            else:
                data = metrics

            self.conn.execute("""
                INSERT INTO hourly_aggregates (
                    hour_start, hour_end, device_id, patient_id,
                    total_frames, frames_with_person, total_observation_seconds,
                    posture_seconds, dominant_posture,
                    support_surface_seconds, time_in_bed_seconds, time_in_bed_pct,
                    bed_exit_count, bed_exit_attempts, fall_events,
                    immobility_periods, immobility_total_seconds,
                    distress_periods, distress_total_seconds,
                    max_agitation_score, avg_agitation_score,
                    max_delirium_risk, avg_delirium_risk,
                    max_respiratory_distress, avg_respiratory_distress,
                    max_fall_risk, avg_fall_risk,
                    activity_seconds, dominant_activity,
                    alert_counts, critical_alerts, high_risk_alerts,
                    avg_pose_confidence, keypoint_visibility_pct
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(device_id, hour_start) DO UPDATE SET
                    total_frames = excluded.total_frames,
                    frames_with_person = excluded.frames_with_person,
                    total_observation_seconds = excluded.total_observation_seconds,
                    posture_seconds = excluded.posture_seconds,
                    dominant_posture = excluded.dominant_posture,
                    support_surface_seconds = excluded.support_surface_seconds,
                    time_in_bed_seconds = excluded.time_in_bed_seconds,
                    time_in_bed_pct = excluded.time_in_bed_pct,
                    bed_exit_count = excluded.bed_exit_count,
                    bed_exit_attempts = excluded.bed_exit_attempts,
                    fall_events = excluded.fall_events,
                    immobility_periods = excluded.immobility_periods,
                    immobility_total_seconds = excluded.immobility_total_seconds,
                    distress_periods = excluded.distress_periods,
                    distress_total_seconds = excluded.distress_total_seconds,
                    max_agitation_score = excluded.max_agitation_score,
                    avg_agitation_score = excluded.avg_agitation_score,
                    max_delirium_risk = excluded.max_delirium_risk,
                    avg_delirium_risk = excluded.avg_delirium_risk,
                    max_respiratory_distress = excluded.max_respiratory_distress,
                    avg_respiratory_distress = excluded.avg_respiratory_distress,
                    max_fall_risk = excluded.max_fall_risk,
                    avg_fall_risk = excluded.avg_fall_risk,
                    activity_seconds = excluded.activity_seconds,
                    dominant_activity = excluded.dominant_activity,
                    alert_counts = excluded.alert_counts,
                    critical_alerts = excluded.critical_alerts,
                    high_risk_alerts = excluded.high_risk_alerts,
                    avg_pose_confidence = excluded.avg_pose_confidence,
                    keypoint_visibility_pct = excluded.keypoint_visibility_pct
            """, (
                data.get("hour_start"),
                data.get("hour_end"),
                data.get("device_id"),
                data.get("patient_id"),
                data.get("total_frames", 0),
                data.get("frames_with_person", 0),
                data.get("total_observation_seconds", 0),
                json.dumps(data.get("posture_seconds", {})),
                data.get("dominant_posture"),
                json.dumps(data.get("support_surface_seconds", {})),
                data.get("time_in_bed_seconds", 0),
                data.get("time_in_bed_pct", 0),
                data.get("bed_exit_count", 0),
                data.get("bed_exit_attempts", 0),
                data.get("fall_events", 0),
                data.get("immobility_periods", 0),
                data.get("immobility_total_seconds", 0),
                data.get("distress_periods", 0),
                data.get("distress_total_seconds", 0),
                data.get("max_agitation_score", 0),
                data.get("avg_agitation_score", 0),
                data.get("max_delirium_risk", 0),
                data.get("avg_delirium_risk", 0),
                data.get("max_respiratory_distress", 0),
                data.get("avg_respiratory_distress", 0),
                data.get("max_fall_risk", 0),
                data.get("avg_fall_risk", 0),
                json.dumps(data.get("activity_seconds", {})),
                data.get("dominant_activity"),
                json.dumps(data.get("alert_counts", {})),
                data.get("critical_alerts", 0),
                data.get("high_risk_alerts", 0),
                data.get("avg_pose_confidence", 0),
                data.get("keypoint_visibility_pct", 0),
            ))
            self.conn.commit()
            log.debug("Hourly aggregate saved: device=%s hour=%s",
                     data.get("device_id"), data.get("hour_start"))
        except Exception as e:
            log.exception("Failed to insert hourly aggregate: %s", e)

    def query_hourly_aggregates(
        self,
        device_id: str = None,
        patient_id: str = None,
        start_ts: float = None,
        end_ts: float = None,
        limit: int = 1000
    ):
        """
        Query hourly aggregates for efficient reporting.

        Args:
            device_id: Filter by device
            patient_id: Filter by patient
            start_ts: Start timestamp
            end_ts: End timestamp
            limit: Maximum results

        Returns:
            List of hourly aggregate dictionaries
        """
        query = "SELECT * FROM hourly_aggregates WHERE 1=1"
        params = []

        if device_id:
            query += " AND device_id = ?"
            params.append(device_id)
        if patient_id:
            query += " AND patient_id = ?"
            params.append(patient_id)
        if start_ts:
            query += " AND hour_start >= ?"
            params.append(start_ts)
        if end_ts:
            query += " AND hour_end <= ?"
            params.append(end_ts)

        query += " ORDER BY hour_start DESC LIMIT ?"
        params.append(limit)

        try:
            cursor = self.conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in cursor:
                record = dict(zip(columns, row))
                # Parse JSON fields
                for field in ["posture_seconds", "support_surface_seconds",
                             "activity_seconds", "alert_counts"]:
                    if record.get(field):
                        try:
                            record[field] = json.loads(record[field])
                        except (json.JSONDecodeError, TypeError):
                            record[field] = {}
                results.append(record)

            return results
        except Exception as e:
            log.exception("Failed to query hourly aggregates: %s", e)
            return []

    def get_daily_summary(self, device_id: str, date_str: str) -> dict:
        """
        Get daily summary from hourly aggregates.

        Args:
            device_id: Device identifier
            date_str: Date string (YYYY-MM-DD)

        Returns:
            Aggregated daily summary
        """
        from datetime import datetime as dt

        try:
            date = dt.strptime(date_str, "%Y-%m-%d")
            start_ts = date.timestamp()
            end_ts = (date + timedelta(days=1)).timestamp()

            hours = self.query_hourly_aggregates(
                device_id=device_id,
                start_ts=start_ts,
                end_ts=end_ts,
                limit=24
            )

            if not hours:
                return {}

            # Aggregate hourly data
            total_frames = sum(h.get("total_frames", 0) for h in hours)
            total_bed_time = sum(h.get("time_in_bed_seconds", 0) for h in hours)
            total_observation = sum(h.get("total_observation_seconds", 0) for h in hours)
            total_bed_exits = sum(h.get("bed_exit_count", 0) for h in hours)
            total_falls = sum(h.get("fall_events", 0) for h in hours)
            total_immobility_periods = sum(h.get("immobility_periods", 0) for h in hours)
            total_immobility_seconds = sum(h.get("immobility_total_seconds", 0) for h in hours)

            # Find max risk scores
            max_agitation = max((h.get("max_agitation_score", 0) for h in hours), default=0)
            max_delirium = max((h.get("max_delirium_risk", 0) for h in hours), default=0)
            max_fall_risk = max((h.get("max_fall_risk", 0) for h in hours), default=0)

            # Aggregate posture distribution
            posture_totals = {}
            for h in hours:
                for posture, seconds in h.get("posture_seconds", {}).items():
                    posture_totals[posture] = posture_totals.get(posture, 0) + seconds

            # Find dominant posture
            dominant_posture = max(posture_totals.items(), key=lambda x: x[1])[0] if posture_totals else "unknown"

            # Total alerts
            critical_total = sum(h.get("critical_alerts", 0) for h in hours)
            high_risk_total = sum(h.get("high_risk_alerts", 0) for h in hours)

            return {
                "date": date_str,
                "device_id": device_id,
                "hours_with_data": len(hours),
                "total_frames": total_frames,
                "total_observation_hours": round(total_observation / 3600, 2),
                "time_in_bed_hours": round(total_bed_time / 3600, 2),
                "time_in_bed_pct": round(total_bed_time / total_observation * 100, 1) if total_observation > 0 else 0,
                "bed_exit_count": total_bed_exits,
                "fall_events": total_falls,
                "immobility_periods": total_immobility_periods,
                "immobility_hours": round(total_immobility_seconds / 3600, 2),
                "max_agitation_score": max_agitation,
                "max_delirium_risk": max_delirium,
                "max_fall_risk": max_fall_risk,
                "dominant_posture": dominant_posture,
                "posture_distribution": posture_totals,
                "critical_alerts": critical_total,
                "high_risk_alerts": high_risk_total,
            }
        except Exception as e:
            log.exception("Failed to get daily summary: %s", e)
            return {}

    def close(self):
        """Close database connection."""
        try:
            self.conn.close()
            log.info("Database connection closed")
        except Exception:
            pass
