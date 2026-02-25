-- Enhanced Activity Monitor - TimescaleDB Schema
-- Replaces plain PostgreSQL for time-series clinical data at scale.
-- Run on first container startup via docker-entrypoint-initdb.d.

-- ============================================
-- Extensions
-- ============================================
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

SET timezone = 'UTC';

-- ============================================
-- Core hypertable: per-device metrics (1/sec from Flink Job 1)
-- ============================================
CREATE TABLE IF NOT EXISTS device_metrics (
    ts          TIMESTAMPTZ     NOT NULL,
    device_id   TEXT            NOT NULL,
    patient_id  TEXT,
    -- Activity state
    activity        TEXT        NOT NULL DEFAULT 'unknown',
    activity_conf   REAL        NOT NULL DEFAULT 0.0,
    posture         TEXT        NOT NULL DEFAULT 'unknown',
    person_present  BOOLEAN     NOT NULL DEFAULT TRUE,
    -- Risk scores (0.0 - 1.0)
    risk_agitation      REAL    NOT NULL DEFAULT 0.0,
    risk_delirium       REAL    NOT NULL DEFAULT 0.0,
    risk_respiratory    REAL    NOT NULL DEFAULT 0.0,
    risk_fall           REAL    NOT NULL DEFAULT 0.0,
    risk_hand_proximity REAL    NOT NULL DEFAULT 0.0,
    -- Alert level
    alert_level TEXT            NOT NULL DEFAULT 'LOW_RISK',
    -- MQTT metadata
    seq         BIGINT,
    schema_ver  INT             NOT NULL DEFAULT 1
);

SELECT create_hypertable('device_metrics', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_dm_device_ts
    ON device_metrics (device_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_dm_alert
    ON device_metrics (alert_level, ts DESC)
    WHERE alert_level != 'LOW_RISK';
CREATE INDEX IF NOT EXISTS idx_dm_patient
    ON device_metrics (patient_id, ts DESC)
    WHERE patient_id IS NOT NULL;

-- ============================================
-- Enriched alerts (from Flink Job 2)
-- ============================================
CREATE TABLE IF NOT EXISTS enriched_alerts (
    ts              TIMESTAMPTZ     NOT NULL,
    device_id       TEXT            NOT NULL,
    patient_id      TEXT,
    alert_type      TEXT            NOT NULL,
    alert_level     TEXT            NOT NULL,
    confidence      REAL            NOT NULL DEFAULT 0.0,
    -- Enrichment fields (added by Flink)
    patient_name    TEXT,
    room            TEXT,
    ward            TEXT,
    care_team       TEXT[],
    -- Routing status
    nurse_call_sent     BOOLEAN DEFAULT FALSE,
    dashboard_pushed    BOOLEAN DEFAULT FALSE,
    acknowledged        BOOLEAN DEFAULT FALSE,
    acknowledged_by     TEXT,
    acknowledged_at     TIMESTAMPTZ,
    -- Context
    context         JSONB           DEFAULT '{}',
    seq             BIGINT,
    schema_ver      INT             NOT NULL DEFAULT 1
);

SELECT create_hypertable('enriched_alerts', 'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ea_device_ts
    ON enriched_alerts (device_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_ea_level
    ON enriched_alerts (alert_level, ts DESC);
CREATE INDEX IF NOT EXISTS idx_ea_unacked
    ON enriched_alerts (acknowledged, ts DESC)
    WHERE acknowledged = FALSE;

-- ============================================
-- Ward-level analytics (from Flink Job 3)
-- ============================================
CREATE TABLE IF NOT EXISTS ward_analytics (
    ts              TIMESTAMPTZ     NOT NULL,
    ward_id         TEXT            NOT NULL,
    window_start    TIMESTAMPTZ     NOT NULL,
    window_end      TIMESTAMPTZ     NOT NULL,
    -- Aggregated counts
    total_devices       INT         NOT NULL DEFAULT 0,
    active_devices      INT         NOT NULL DEFAULT 0,
    devices_elevated    INT         NOT NULL DEFAULT 0,
    total_alerts        INT         NOT NULL DEFAULT 0,
    critical_alerts     INT         NOT NULL DEFAULT 0,
    -- Risk distribution
    avg_agitation       REAL        NOT NULL DEFAULT 0.0,
    max_agitation       REAL        NOT NULL DEFAULT 0.0,
    avg_delirium_risk   REAL        NOT NULL DEFAULT 0.0,
    -- Pattern detection
    ward_spike_detected BOOLEAN     NOT NULL DEFAULT FALSE,
    pattern_type        TEXT
);

SELECT create_hypertable('ward_analytics', 'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ============================================
-- Hourly device summaries (from edge MQTT hourly topic)
-- ============================================
CREATE TABLE IF NOT EXISTS hourly_summaries (
    ts              TIMESTAMPTZ     NOT NULL,
    device_id       TEXT            NOT NULL,
    patient_id      TEXT,
    hour_start      TIMESTAMPTZ     NOT NULL,
    -- Activity distribution (seconds spent in each state)
    seconds_calm        REAL        DEFAULT 0,
    seconds_agitation   REAL        DEFAULT 0,
    seconds_restlessness REAL       DEFAULT 0,
    seconds_delirium    REAL        DEFAULT 0,
    seconds_convulsion  REAL        DEFAULT 0,
    seconds_pain        REAL        DEFAULT 0,
    -- Posture distribution
    seconds_supine      REAL        DEFAULT 0,
    seconds_prone       REAL        DEFAULT 0,
    seconds_lateral     REAL        DEFAULT 0,
    seconds_sitting     REAL        DEFAULT 0,
    seconds_absent      REAL        DEFAULT 0,
    -- Summary stats
    alert_count         INT         DEFAULT 0,
    critical_count      INT         DEFAULT 0,
    avg_confidence      REAL        DEFAULT 0,
    -- Raw data
    summary_data        JSONB       DEFAULT '{}',
    seq                 BIGINT,
    schema_ver          INT         NOT NULL DEFAULT 1
);

SELECT create_hypertable('hourly_summaries', 'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ============================================
-- Continuous Aggregates (materialized views updated automatically)
-- ============================================

-- Hourly risk summary per device (auto-refreshed)
CREATE MATERIALIZED VIEW IF NOT EXISTS device_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', ts) AS bucket,
    device_id,
    patient_id,
    COUNT(*)                        AS sample_count,
    AVG(risk_agitation)             AS avg_agitation,
    MAX(risk_agitation)             AS max_agitation,
    percentile_agg(risk_agitation)  AS pctl_agitation,
    AVG(risk_delirium)              AS avg_delirium,
    MAX(risk_delirium)              AS max_delirium,
    AVG(risk_respiratory)           AS avg_respiratory,
    AVG(risk_fall)                  AS avg_fall,
    -- Posture distribution
    COUNT(*) FILTER (WHERE posture = 'supine')        AS frames_supine,
    COUNT(*) FILTER (WHERE posture = 'prone')         AS frames_prone,
    COUNT(*) FILTER (WHERE posture = 'sitting')       AS frames_sitting,
    COUNT(*) FILTER (WHERE person_present = FALSE)    AS frames_absent,
    -- Alert counts
    COUNT(*) FILTER (WHERE alert_level = 'HIGH_RISK')     AS high_risk_count,
    COUNT(*) FILTER (WHERE alert_level = 'CRITICAL')      AS critical_count
FROM device_metrics
GROUP BY bucket, device_id, patient_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('device_metrics_hourly',
    start_offset    => INTERVAL '3 hours',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- Daily alert summary
CREATE MATERIALIZED VIEW IF NOT EXISTS alerts_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', ts) AS bucket,
    device_id,
    alert_level,
    alert_type,
    COUNT(*)            AS alert_count,
    AVG(confidence)     AS avg_confidence
FROM enriched_alerts
GROUP BY bucket, device_id, alert_level, alert_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy('alerts_daily',
    start_offset    => INTERVAL '2 days',
    end_offset      => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists   => TRUE
);

-- ============================================
-- Retention Policies (auto-drop old data)
-- ============================================

-- Raw metrics: keep 30 days
SELECT add_retention_policy('device_metrics', INTERVAL '30 days', if_not_exists => TRUE);

-- Alerts: keep 90 days (longer for clinical review)
SELECT add_retention_policy('enriched_alerts', INTERVAL '90 days', if_not_exists => TRUE);

-- Ward analytics: keep 180 days
SELECT add_retention_policy('ward_analytics', INTERVAL '180 days', if_not_exists => TRUE);

-- Hourly summaries: keep 2 years
SELECT add_retention_policy('hourly_summaries', INTERVAL '730 days', if_not_exists => TRUE);

-- ============================================
-- Compression (auto-compress old chunks)
-- ============================================

ALTER TABLE device_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id',
    timescaledb.compress_orderby = 'ts DESC'
);

SELECT add_compression_policy('device_metrics', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE enriched_alerts SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id',
    timescaledb.compress_orderby = 'ts DESC'
);

SELECT add_compression_policy('enriched_alerts', INTERVAL '14 days', if_not_exists => TRUE);

-- ============================================
-- Permissions
-- ============================================
GRANT ALL PRIVILEGES ON DATABASE eam TO eam;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO eam;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO eam;

-- Read-only role for Grafana dashboards
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'eam_readonly') THEN
        CREATE ROLE eam_readonly LOGIN PASSWORD 'readonly_changeme';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE eam TO eam_readonly;
GRANT USAGE ON SCHEMA public TO eam_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO eam_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO eam_readonly;

SELECT 'TimescaleDB schema initialized successfully' AS status;
