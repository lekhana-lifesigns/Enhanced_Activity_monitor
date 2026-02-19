# storage/disk_monitor.py
"""
Disk Space Monitoring Module
Tracks disk usage and alerts when thresholds are exceeded.
"""
import logging
import shutil
import os
import time
from typing import Dict, Optional

log = logging.getLogger("disk_monitor")


class DiskMonitor:
 """Monitor disk space and alert on threshold violations."""

 def __init__(self,
 warning_threshold: int = 80,
 critical_threshold: int = 90,
 check_interval_minutes: int = 30):
 """
 Initialize disk monitor.

 Args:
 warning_threshold: Percentage at which to issue warnings (default 80%)
 critical_threshold: Percentage at which to issue critical alerts (default 90%)
 check_interval_minutes: Minutes between checks (default 30)
 """
 self.warning_threshold = warning_threshold
 self.critical_threshold = critical_threshold
 self.check_interval_seconds = check_interval_minutes * 60
 self.last_check_time = 0
 self.last_alert_level = None
 self.alert_cooldown = 300 # 5 minutes between duplicate alerts
 self.last_alert_time = 0

 log.info("Disk monitor initialized: warning=%d%%, critical=%d%%",
 warning_threshold, critical_threshold)

 def get_disk_usage(self, path: str = ".") -> Dict:
 """
 Get disk usage statistics for the given path.

 Args:
 path: Path to check (default: current directory)

 Returns:
 Dictionary with disk usage statistics
 """
 try:
 # Get absolute path to ensure we check the correct disk
 abs_path = os.path.abspath(path)

 # Get disk usage statistics
 usage = shutil.disk_usage(abs_path)

 total_gb = usage.total / (1024**3)
 used_gb = usage.used / (1024**3)
 free_gb = usage.free / (1024**3)
 percent_used = (usage.used / usage.total) * 100

 return {
 "path": abs_path,
 "total_gb": round(total_gb, 2),
 "used_gb": round(used_gb, 2),
 "free_gb": round(free_gb, 2),
 "percent_used": round(percent_used, 2),
 "timestamp": time.time()
 }
 except Exception as e:
 log.exception("Failed to get disk usage: %s", e)
 return {
 "path": path,
 "error": str(e),
 "timestamp": time.time()
 }

 def check_disk_space(self, path: str = ".", force: bool = False) -> Optional[Dict]:
 """
 Check disk space and return alert if thresholds exceeded.

 Args:
 path: Path to check (default: current directory)
 force: Force check even if within interval (default: False)

 Returns:
 Alert dictionary if threshold exceeded, None otherwise
 """
 current_time = time.time()

 # Check if enough time has passed since last check
 if not force and (current_time - self.last_check_time) < self.check_interval_seconds:
 return None

 self.last_check_time = current_time

 # Get disk usage
 usage = self.get_disk_usage(path)
 if "error" in usage:
 return None

 percent_used = usage["percent_used"]
 alert_level = None
 message = None

 # Determine alert level
 if percent_used >= self.critical_threshold:
 alert_level = "CRITICAL"
 message = f"CRITICAL: Disk space at {percent_used}% (threshold: {self.critical_threshold}%)"
 elif percent_used >= self.warning_threshold:
 alert_level = "WARNING"
 message = f"WARNING: Disk space at {percent_used}% (threshold: {self.warning_threshold}%)"

 # Return alert if threshold exceeded
 if alert_level:
 # Apply cooldown to prevent alert spam
 if (alert_level == self.last_alert_level and
 (current_time - self.last_alert_time) < self.alert_cooldown):
 return None # Suppress duplicate alert

 self.last_alert_level = alert_level
 self.last_alert_time = current_time

 log.warning("%s | Free: %.2f GB / %.2f GB",
 message, usage["free_gb"], usage["total_gb"])

 return {
 "alert_level": alert_level,
 "message": message,
 "usage": usage,
 "recommendation": self._get_recommendation(usage)
 }

 # Reset alert level if back to normal
 if self.last_alert_level and percent_used < self.warning_threshold:
 log.info("Disk space back to normal: %.2f%% used", percent_used)
 self.last_alert_level = None

 return None

 def _get_recommendation(self, usage: Dict) -> str:
 """Get recommendation for disk space management."""
 percent_used = usage["percent_used"]

 if percent_used >= self.critical_threshold:
 return (
 "URGENT: Free disk space immediately. "
 "Consider: (1) Running manual cleanup, "
 "(2) Archiving old data, "
 "(3) Reducing retention periods in config/system.yaml"
 )
 elif percent_used >= self.warning_threshold:
 return (
 "Free disk space soon. "
 "Check data_retention settings in config/system.yaml "
 "and ensure automatic cleanup is enabled."
 )
 else:
 return "Disk space within normal limits."

 def get_database_size(self, db_path: str = "storage/events.db") -> Dict:
 """
 Get size of database file.

 Args:
 db_path: Path to database file

 Returns:
 Dictionary with database size information
 """
 try:
 if not os.path.exists(db_path):
 return {"size_mb": 0, "exists": False}

 size_bytes = os.path.getsize(db_path)
 size_mb = size_bytes / (1024**2)

 return {
 "path": db_path,
 "size_mb": round(size_mb, 2),
 "size_bytes": size_bytes,
 "exists": True
 }
 except Exception as e:
 log.exception("Failed to get database size: %s", e)
 return {"error": str(e)}

 def log_storage_summary(self, db_path: str = "storage/events.db"):
 """Log a summary of storage usage."""
 disk_usage = self.get_disk_usage()
 db_size = self.get_database_size(db_path)

 log.info("=" * 60)
 log.info("Storage Summary")
 log.info("=" * 60)
 log.info("Disk: %.2f GB used / %.2f GB total (%.1f%%)",
 disk_usage["used_gb"], disk_usage["total_gb"], disk_usage["percent_used"])
 log.info("Free: %.2f GB (%.1f%%)",
 disk_usage["free_gb"], 100 - disk_usage["percent_used"])

 if db_size.get("exists"):
 log.info("Database: %.2f MB", db_size["size_mb"])

 log.info("=" * 60)


def create_disk_monitor(config: dict) -> Optional[DiskMonitor]:
 """
 Create disk monitor from configuration.

 Args:
 config: System configuration dictionary

 Returns:
 DiskMonitor instance or None if disabled
 """
 try:
 retention_cfg = config.get("data_retention", {})

 if not retention_cfg.get("enabled", False):
 log.info("Disk monitoring disabled in configuration")
 return None

 return DiskMonitor(
 warning_threshold=retention_cfg.get("disk_warning_threshold", 80),
 critical_threshold=retention_cfg.get("disk_critical_threshold", 90),
 check_interval_minutes=retention_cfg.get("disk_check_interval_minutes", 30)
 )
 except Exception as e:
 log.exception("Failed to create disk monitor: %s", e)
 return None
