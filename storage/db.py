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
        """Create events table if it doesn't exist."""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            device TEXT,
            label TEXT,
            confidence REAL,
            payload TEXT
        )""")
        
        # Create index on timestamp for faster queries
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ts ON events(ts)
        """)
        
        # Create index on device for filtering
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_device ON events(device)
        """)
        
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
            ts = payload.get("ts", payload.get("timestamp", 0.0))
            self.conn.execute(
                "INSERT INTO events (ts, device, label, confidence, payload) VALUES (?,?,?,?,?)",
                (ts, device, label, float(confidence), json.dumps(payload))
            )
            self.conn.commit()
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

    def close(self):
        """Close database connection."""
        try:
            self.conn.close()
            log.info("Database connection closed")
        except Exception:
            pass
