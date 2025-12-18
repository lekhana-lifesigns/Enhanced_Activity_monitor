# storage/db.py
import sqlite3
import json
import os
import logging

log = logging.getLogger("db")

DB_PATH = "storage/events.db"

class LocalDB:
    def __init__(self, path=DB_PATH):
        """Initialize local SQLite database for event storage."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.path = path
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
        
        # Create indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_device ON events(device)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_label ON events(label)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_device ON alerts(device)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(alert_level)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_health_ts ON system_health(ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_health_device ON system_health(device_id)")
        
        self.conn.commit()

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
                    except:
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
        query = "SELECT ts, device, label, confidence, payload FROM events WHERE 1=1"
        params = []
        
        if device:
            query += " AND device = ?"
            params.append(device)
        
        if start_ts:
            query += " AND ts >= ?"
            params.append(start_ts)
        
        if end_ts:
            query += " AND ts <= ?"
            params.append(end_ts)
        
        query += " ORDER BY ts DESC LIMIT ?"
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
        query = "SELECT id, ts, device, alert_level, label, agitation_score, delirium_risk, respiratory_distress, hand_proximity_risk, payload, acknowledged FROM alerts WHERE 1=1"
        params = []
        
        if device:
            query += " AND device = ?"
            params.append(device)
        if start_ts:
            query += " AND ts >= ?"
            params.append(start_ts)
        if end_ts:
            query += " AND ts <= ?"
            params.append(end_ts)
        if alert_level:
            query += " AND alert_level = ?"
            params.append(alert_level)
        
        query += " ORDER BY ts DESC LIMIT ?"
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
        query = """
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT label) as unique_labels,
                AVG(confidence) as avg_confidence,
                MIN(ts) as first_ts,
                MAX(ts) as last_ts
            FROM events WHERE 1=1
        """
        params = []
        
        if device:
            query += " AND device = ?"
            params.append(device)
        if start_ts:
            query += " AND ts >= ?"
            params.append(start_ts)
        if end_ts:
            query += " AND ts <= ?"
            params.append(end_ts)
        
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
        query = """
            SELECT 
                alert_level,
                COUNT(*) as count,
                AVG(agitation_score) as avg_agitation,
                AVG(delirium_risk) as avg_delirium,
                AVG(respiratory_distress) as avg_respiratory,
                AVG(hand_proximity_risk) as avg_hand_proximity
            FROM alerts WHERE 1=1
        """
        params = []
        
        if device:
            query += " AND device = ?"
            params.append(device)
        if start_ts:
            query += " AND ts >= ?"
            params.append(start_ts)
        if end_ts:
            query += " AND ts <= ?"
            params.append(end_ts)
        
        query += " GROUP BY alert_level"
        
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

    def close(self):
        """Close database connection."""
        try:
            self.conn.close()
            log.info("Database connection closed")
        except Exception:
            pass
