# storage/reporting.py
"""
Reporting module for hourly and monthly reports.
Generates aggregated summaries and sends via MQTT to cloud ingest layer.
"""
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
from .db import LocalDB

log = logging.getLogger("reporting")


class ReportGenerator:
    """Generate hourly and monthly reports from database."""
    
    def __init__(self, db: LocalDB, device_id: str):
        """
        Args:
            db: LocalDB instance
            device_id: Device identifier
        """
        self.db = db
        self.device_id = device_id
    
    def generate_hourly_report(self, end_time: Optional[float] = None) -> Dict:
        """
        Generate hourly report for the last hour.
        
        Args:
            end_time: End timestamp (defaults to now)
        
        Returns:
            Dictionary with report data
        """
        if end_time is None:
            end_time = time.time()
        start_time = end_time - 3600  # Last hour
        
        log.info("Generating hourly report: %s to %s", 
                datetime.fromtimestamp(start_time), datetime.fromtimestamp(end_time))
        
        # Get event statistics
        event_stats = self.db.get_event_statistics(
            device=self.device_id,
            start_ts=start_time,
            end_ts=end_time
        )
        
        # Get alert statistics
        alert_stats = self.db.get_alert_statistics(
            device=self.device_id,
            start_ts=start_time,
            end_ts=end_time
        )
        
        # Get recent alerts
        recent_alerts = self.db.query_alerts(
            device=self.device_id,
            start_ts=start_time,
            end_ts=end_time,
            limit=100
        )
        
        # Count alerts by level
        alert_counts = {}
        for alert in recent_alerts:
            level = alert.get("alert_level", "UNKNOWN")
            alert_counts[level] = alert_counts.get(level, 0) + 1
        
        # Count events by label
        events = self.db.query_events(
            device=self.device_id,
            start_ts=start_time,
            end_ts=end_time,
            limit=1000
        )
        label_counts = {}
        for event in events:
            label = event.get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
        
        report = {
            "report_type": "hourly",
            "device_id": self.device_id,
            "start_time": start_time,
            "end_time": end_time,
            "start_time_iso": datetime.fromtimestamp(start_time).isoformat(),
            "end_time_iso": datetime.fromtimestamp(end_time).isoformat(),
            "summary": {
                "total_events": event_stats.get("total_events", 0),
                "unique_activity_labels": event_stats.get("unique_labels", 0),
                "avg_confidence": round(event_stats.get("avg_confidence", 0.0), 3),
                "total_alerts": len(recent_alerts),
                "alert_counts": alert_counts,
                "activity_distribution": label_counts
            },
            "alert_statistics": alert_stats,
            "high_priority_alerts": [a for a in recent_alerts if a.get("alert_level") in ["HIGH_RISK", "CRITICAL"]],
            "metadata": {
                "generated_at": time.time(),
                "generated_at_iso": datetime.now().isoformat()
            }
        }
        
        return report
    
    def generate_monthly_report(self, year: int, month: int) -> Dict:
        """
        Generate monthly report for a specific month.
        
        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
        
        Returns:
            Dictionary with report data
        """
        # Calculate start and end timestamps for the month
        start_dt = datetime(year, month, 1)
        if month == 12:
            end_dt = datetime(year + 1, 1, 1)
        else:
            end_dt = datetime(year, month + 1, 1)
        
        start_time = start_dt.timestamp()
        end_time = end_dt.timestamp()
        
        log.info("Generating monthly report: %s/%s (%s to %s)", 
                year, month, start_dt.isoformat(), end_dt.isoformat())
        
        # Get event statistics
        event_stats = self.db.get_event_statistics(
            device=self.device_id,
            start_ts=start_time,
            end_ts=end_time
        )
        
        # Get alert statistics
        alert_stats = self.db.get_alert_statistics(
            device=self.device_id,
            start_ts=start_time,
            end_ts=end_time
        )
        
        # Get all alerts for the month
        all_alerts = self.db.query_alerts(
            device=self.device_id,
            start_ts=start_time,
            end_ts=end_time,
            limit=10000
        )
        
        # Aggregate by day
        daily_stats = {}
        for alert in all_alerts:
            day = datetime.fromtimestamp(alert["ts"]).strftime("%Y-%m-%d")
            if day not in daily_stats:
                daily_stats[day] = {
                    "total_alerts": 0,
                    "alert_levels": {},
                    "avg_scores": {
                        "agitation": [],
                        "delirium": [],
                        "respiratory": [],
                        "hand_proximity": []
                    }
                }
            
            daily_stats[day]["total_alerts"] += 1
            level = alert.get("alert_level", "UNKNOWN")
            daily_stats[day]["alert_levels"][level] = daily_stats[day]["alert_levels"].get(level, 0) + 1
            
            if alert.get("agitation_score"):
                daily_stats[day]["avg_scores"]["agitation"].append(alert["agitation_score"])
            if alert.get("delirium_risk"):
                daily_stats[day]["avg_scores"]["delirium"].append(alert["delirium_risk"])
            if alert.get("respiratory_distress"):
                daily_stats[day]["avg_scores"]["respiratory"].append(alert["respiratory_distress"])
            if alert.get("hand_proximity_risk"):
                daily_stats[day]["avg_scores"]["hand_proximity"].append(alert["hand_proximity_risk"])
        
        # Calculate averages
        for day in daily_stats:
            for score_type in daily_stats[day]["avg_scores"]:
                scores = daily_stats[day]["avg_scores"][score_type]
                daily_stats[day]["avg_scores"][score_type] = round(sum(scores) / len(scores), 3) if scores else 0.0
        
        # Get events for activity distribution
        events = self.db.query_events(
            device=self.device_id,
            start_ts=start_time,
            end_ts=end_time,
            limit=10000
        )
        
        label_distribution = {}
        for event in events:
            label = event.get("label", "unknown")
            label_distribution[label] = label_distribution.get(label, 0) + 1
        
        # Safely round avg_confidence (handle None)
        avg_conf = event_stats.get("avg_confidence")
        if avg_conf is not None:
            avg_conf = round(float(avg_conf), 3)
        else:
            avg_conf = 0.0
        
        report = {
            "report_type": "monthly",
            "device_id": self.device_id,
            "year": year,
            "month": month,
            "start_time": start_time,
            "end_time": end_time,
            "start_time_iso": start_dt.isoformat(),
            "end_time_iso": end_dt.isoformat(),
            "summary": {
                "total_events": event_stats.get("total_events", 0),
                "unique_activity_labels": event_stats.get("unique_labels", 0),
                "avg_confidence": avg_conf,
                "total_alerts": len(all_alerts),
                "activity_distribution": label_distribution
            },
            "alert_statistics": alert_stats,
            "daily_breakdown": daily_stats,
            "metadata": {
                "generated_at": time.time(),
                "generated_at_iso": datetime.now().isoformat()
            }
        }
        
        return report
    
    def get_model_metrics(self, start_ts: float, end_ts: float) -> Dict:
        """
        Extract model performance metrics for retraining pipeline.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
        
        Returns:
            Dictionary with model metrics
        """
        events = self.db.query_events(
            device=self.device_id,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=10000
        )
        
        if not events:
            return {}
        
        # Calculate confidence distribution
        confidences = [e.get("confidence", 0.0) for e in events if e.get("confidence")]
        
        # Label distribution (for class imbalance analysis)
        label_counts = {}
        for event in events:
            label = event.get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
        
        metrics = {
            "total_samples": len(events),
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "min_confidence": round(min(confidences), 3) if confidences else 0.0,
            "max_confidence": round(max(confidences), 3) if confidences else 0.0,
            "label_distribution": label_counts,
            "class_imbalance_ratio": round(max(label_counts.values()) / min(label_counts.values()), 2) if len(label_counts) > 1 and min(label_counts.values()) > 0 else 1.0,
            "time_range": {
                "start": start_ts,
                "end": end_ts
            }
        }
        
        return metrics

