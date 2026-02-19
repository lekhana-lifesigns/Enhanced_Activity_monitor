#!/usr/bin/env python3
"""
Manual Database Cleanup Script
Allows manual cleanup of old data with configurable retention periods.
Use this for emergency cleanup or testing retention policies.
"""
import sys
import os
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from storage.db import LocalDB
from storage.disk_monitor import DiskMonitor

logging.basicConfig(
 level=logging.INFO,
 format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("manual_cleanup")


def parse_args():
 """Parse command-line arguments."""
 parser = argparse.ArgumentParser(
 description="Manual database cleanup with configurable retention periods"
 )
 parser.add_argument(
 "--events-days",
 type=int,
 default=30,
 help="Days to keep event data (default: 30)"
 )
 parser.add_argument(
 "--alerts-days",
 type=int,
 default=90,
 help="Days to keep alert data (default: 90)"
 )
 parser.add_argument(
 "--health-days",
 type=int,
 default=14,
 help="Days to keep system health data (default: 14)"
 )
 parser.add_argument(
 "--audit-days",
 type=int,
 default=365,
 help="Days to keep audit logs (default: 365)"
 )
 parser.add_argument(
 "--db-path",
 type=str,
 default="storage/events.db",
 help="Path to database file (default: storage/events.db)"
 )
 parser.add_argument(
 "--vacuum",
 action="store_true",
 default=True,
 help="Run VACUUM after cleanup to reclaim space (default: true)"
 )
 parser.add_argument(
 "--no-vacuum",
 action="store_false",
 dest="vacuum",
 help="Skip VACUUM operation"
 )
 parser.add_argument(
 "--check-only",
 action="store_true",
 help="Only check disk space, don't run cleanup"
 )
 parser.add_argument(
 "--dry-run",
 action="store_true",
 help="Show what would be deleted without actually deleting"
 )
 parser.add_argument(
 "--force",
 action="store_true",
 help="Skip confirmation prompt"
 )

 return parser.parse_args()


def check_disk_space(db_path="storage/events.db"):
 """Check and display disk space information."""
 monitor = DiskMonitor()

 log.info("=" * 60)
 log.info("Disk Space Check")
 log.info("=" * 60)

 # Get disk usage
 disk_usage = monitor.get_disk_usage()
 log.info("Disk Usage:")
 log.info(" Total: %.2f GB", disk_usage["total_gb"])
 log.info(" Used: %.2f GB (%.1f%%)", disk_usage["used_gb"], disk_usage["percent_used"])
 log.info(" Free: %.2f GB (%.1f%%)", disk_usage["free_gb"], 100 - disk_usage["percent_used"])

 # Get database size
 db_size = monitor.get_database_size(db_path)
 if db_size.get("exists"):
 log.info("\nDatabase Size:")
 log.info(" Path: %s", db_size["path"])
 log.info(" Size: %.2f MB", db_size["size_mb"])

 # Check thresholds
 if disk_usage["percent_used"] >= 90:
 log.critical("CRITICAL: Disk usage above 90%% - immediate action required!")
 elif disk_usage["percent_used"] >= 80:
 log.warning("WARNING: Disk usage above 80%% - cleanup recommended")
 else:
 log.info("Disk space within normal limits")

 log.info("=" * 60)


def get_record_counts(db):
 """Get current record counts from all tables."""
 counts = {}
 allowed_tables = {"events", "alerts", "system_health", "audit_log"}

 for table in allowed_tables:
 try:
 cursor = db.conn.execute(f"SELECT COUNT(*) FROM {table}") # table names are from allowlist above
 count = cursor.fetchone()[0]
 counts[table] = count
 except Exception as e:
 log.warning("Failed to count records in %s: %s", table, e)
 counts[table] = 0

 return counts


def estimate_deletions(db, events_days, alerts_days, health_days, audit_days):
 """Estimate how many records would be deleted."""
 import time
 cutoff_times = {
 "events": time.time() - (events_days * 24 * 60 * 60),
 "alerts": time.time() - (alerts_days * 24 * 60 * 60),
 "system_health": time.time() - (health_days * 24 * 60 * 60),
 "audit_log": time.time() - (audit_days * 24 * 60 * 60)
 }

 estimates = {}

 # Count events to delete
 try:
 cursor = db.conn.execute("SELECT COUNT(*) FROM events WHERE ts < ?", (cutoff_times["events"],))
 estimates["events"] = cursor.fetchone()[0]
 except:
 estimates["events"] = 0

 # Count alerts to delete
 try:
 cursor = db.conn.execute("SELECT COUNT(*) FROM alerts WHERE ts < ?", (cutoff_times["alerts"],))
 estimates["alerts"] = cursor.fetchone()[0]
 except:
 estimates["alerts"] = 0

 # Count system_health to delete
 try:
 cursor = db.conn.execute("SELECT COUNT(*) FROM system_health WHERE ts < ?", (cutoff_times["system_health"],))
 estimates["system_health"] = cursor.fetchone()[0]
 except:
 estimates["system_health"] = 0

 # Count audit_log to delete
 try:
 cursor = db.conn.execute("SELECT COUNT(*) FROM audit_log WHERE timestamp < ?", (cutoff_times["audit_log"],))
 estimates["audit_log"] = cursor.fetchone()[0]
 except:
 estimates["audit_log"] = 0

 return estimates


def main():
 """Main cleanup function."""
 args = parse_args()

 # Check disk space only
 if args.check_only:
 check_disk_space(args.db_path)
 return 0

 # Initialize database
 if not os.path.exists(args.db_path):
 log.error("Database not found: %s", args.db_path)
 return 1

 try:
 db = LocalDB(args.db_path)
 except Exception as e:
 log.exception("Failed to open database: %s", e)
 return 1

 log.info("=" * 60)
 log.info("Manual Database Cleanup")
 log.info("=" * 60)
 log.info("Database: %s", args.db_path)
 log.info("Retention periods:")
 log.info(" Events: %d days", args.events_days)
 log.info(" Alerts: %d days", args.alerts_days)
 log.info(" System Health: %d days", args.health_days)
 log.info(" Audit Logs: %d days", args.audit_days)
 log.info("=" * 60)

 # Get current record counts
 log.info("\nCurrent record counts:")
 before_counts = get_record_counts(db)
 for table, count in before_counts.items():
 log.info(" %s: %d records", table, count)

 # Estimate deletions
 log.info("\nEstimating records to delete...")
 estimates = estimate_deletions(db, args.events_days, args.alerts_days,
 args.health_days, args.audit_days)

 total_to_delete = sum(estimates.values())
 log.info("Estimated deletions:")
 for table, count in estimates.items():
 log.info(" %s: %d records", table, count)
 log.info("Total: %d records", total_to_delete)

 if total_to_delete == 0:
 log.info("\nNo records to delete. Database is already clean.")
 return 0

 # Dry run mode
 if args.dry_run:
 log.info("\nDRY RUN MODE - No changes will be made")
 return 0

 # Confirmation prompt
 if not args.force:
 log.info("\n" + "=" * 60)
 response = input(f"Delete {total_to_delete} records? This cannot be undone. (yes/no): ")
 if response.lower() != "yes":
 log.info("Cleanup cancelled by user")
 return 0

 # Run cleanup
 log.info("\nStarting cleanup...")
 start_time = datetime.now()

 try:
 results = db.run_full_cleanup(
 events_days=args.events_days,
 alerts_days=args.alerts_days,
 health_days=args.health_days,
 audit_days=args.audit_days
 )

 # Get final record counts
 log.info("\nFinal record counts:")
 after_counts = get_record_counts(db)
 for table, count in after_counts.items():
 before = before_counts[table]
 deleted = before - count
 log.info(" %s: %d records (-%d)", table, count, deleted)

 # Calculate time taken
 end_time = datetime.now()
 duration = (end_time - start_time).total_seconds()
 log.info("\nCleanup completed in %.2f seconds", duration)

 # Check disk space after cleanup
 log.info("\n")
 check_disk_space(args.db_path)

 log.info("\n" + "=" * 60)
 log.info("Cleanup Summary:")
 log.info(" Total records deleted: %d", results["total_deleted"])
 log.info(" Duration: %.2f seconds", duration)
 log.info(" Status: SUCCESS")
 log.info("=" * 60)

 return 0

 except Exception as e:
 log.exception("Cleanup failed: %s", e)
 return 1
 finally:
 db.close()


if __name__ == "__main__":
 sys.exit(main())
