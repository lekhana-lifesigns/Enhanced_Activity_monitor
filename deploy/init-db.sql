-- Enhanced Activity Monitor - Database Initialization Script
-- This script runs when PostgreSQL container is first created

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for common queries (after tables are created by SQLAlchemy)
-- These are created as functions to be called after migrations

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE eam TO eam;

-- Create read-only user for analytics (optional)
-- CREATE USER eam_readonly WITH PASSWORD 'readonly_password';
-- GRANT CONNECT ON DATABASE eam TO eam_readonly;
-- GRANT USAGE ON SCHEMA public TO eam_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO eam_readonly;

-- Set timezone
SET timezone = 'UTC';

-- Log that initialization completed
SELECT 'EAM Database initialized successfully' as status;
