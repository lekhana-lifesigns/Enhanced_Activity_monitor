-- Enhanced Activity Monitor - Database Schema
-- SQLite database schema for event storage

-- Events table (main event log)
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    device TEXT NOT NULL,
    label TEXT,
    confidence REAL,
    payload TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_device ON events(device);
CREATE INDEX IF NOT EXISTS idx_events_label ON events(label);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);

-- Alerts table (high-priority events)
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
);

CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);
CREATE INDEX IF NOT EXISTS idx_alerts_device ON alerts(device);
CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(alert_level);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);

-- Patient sessions table (for multi-patient support)
CREATE TABLE IF NOT EXISTS patient_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    device_id TEXT NOT NULL,
    bed_id TEXT,
    start_ts REAL NOT NULL,
    end_ts REAL,
    status TEXT DEFAULT 'active',
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sessions_patient ON patient_sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_sessions_device ON patient_sessions(device_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON patient_sessions(status);

-- System health metrics table
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
);

CREATE INDEX IF NOT EXISTS idx_health_ts ON system_health(ts);
CREATE INDEX IF NOT EXISTS idx_health_device ON system_health(device_id);

-- Feature vectors table (for training data collection)
CREATE TABLE IF NOT EXISTS feature_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    device_id TEXT NOT NULL,
    patient_id TEXT,
    features TEXT NOT NULL,  -- JSON array of 9 features
    label TEXT,
    confidence REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_features_ts ON feature_vectors(ts);
CREATE INDEX IF NOT EXISTS idx_features_device ON feature_vectors(device_id);
CREATE INDEX IF NOT EXISTS idx_features_label ON feature_vectors(label);

-- Retention policy: Delete events older than 90 days
-- (Run via cron or scheduled task)
-- DELETE FROM events WHERE ts < (strftime('%s', 'now') - 90 * 24 * 60 * 60);

