# storage/reporting.py
"""
Reporting module for hourly, daily, and monthly reports.

Architecture:
    Live Stream → HourlyAggregator (in-memory) → hourly_aggregates table → Reports

Key Design Decisions:
    1. Reports are generated from hourly aggregates, NOT raw events
    2. This reduces DB load and enables efficient historical queries
    3. Raw events are only queried for fallback/legacy compatibility
"""
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .db import LocalDB

log = logging.getLogger("reporting")


class ReportGenerator:
    """
    Generate hourly, daily, and monthly reports from hourly aggregates.

    Reports are structured for clinical use:
    - Posture patterns (time distribution)
    - Support surface usage (bed time %)
    - Clinical events (bed exits, falls, immobility)
    - Risk score trends
    - Alert summaries
    """

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
        Generate hourly report from hourly aggregates.

        This method preferentially uses hourly aggregates for efficiency.
        Falls back to raw event queries if aggregates unavailable.

        Args:
            end_time: End timestamp (defaults to now)

        Returns:
            Dictionary with comprehensive hourly report
        """
        if end_time is None:
            end_time = time.time()
        start_time = end_time - 3600  # Last hour

        log.info("Generating hourly report: %s to %s",
                datetime.fromtimestamp(start_time), datetime.fromtimestamp(end_time))

        # Try to get from hourly aggregates first (efficient path)
        aggregates = self.db.query_hourly_aggregates(
            device_id=self.device_id,
            start_ts=start_time,
            end_ts=end_time,
            limit=1
        )

        if aggregates:
            # Use pre-aggregated data (efficient)
            agg = aggregates[0]
            report = self._build_report_from_aggregate(agg, "hourly")
        else:
            # Fallback to raw event queries (legacy compatibility)
            report = self._generate_hourly_from_events(start_time, end_time)

        return report

    def _build_report_from_aggregate(self, agg: Dict, report_type: str) -> Dict:
        """Build report dictionary from hourly aggregate data."""
        return {
            "report_type": report_type,
            "device_id": agg.get("device_id", self.device_id),
            "patient_id": agg.get("patient_id"),
            "start_time": agg.get("hour_start"),
            "end_time": agg.get("hour_end"),
            "start_time_iso": datetime.fromtimestamp(agg.get("hour_start", 0)).isoformat(),
            "end_time_iso": datetime.fromtimestamp(agg.get("hour_end", 0)).isoformat(),
            "summary": {
                "total_frames": agg.get("total_frames", 0),
                "frames_with_person": agg.get("frames_with_person", 0),
                "observation_seconds": agg.get("total_observation_seconds", 0),
                "dominant_posture": agg.get("dominant_posture", "unknown"),
                "dominant_activity": agg.get("dominant_activity", "unknown"),
            },
            "posture_analysis": {
                "distribution_seconds": agg.get("posture_seconds", {}),
                "dominant": agg.get("dominant_posture", "unknown"),
            },
            "support_surface": {
                "distribution_seconds": agg.get("support_surface_seconds", {}),
                "time_in_bed_seconds": agg.get("time_in_bed_seconds", 0),
                "time_in_bed_pct": round(agg.get("time_in_bed_pct", 0) * 100, 1),
            },
            "clinical_events": {
                "bed_exits": agg.get("bed_exit_count", 0),
                "bed_exit_attempts": agg.get("bed_exit_attempts", 0),
                "falls": agg.get("fall_events", 0),
                "immobility_periods": agg.get("immobility_periods", 0),
                "immobility_minutes": round(agg.get("immobility_total_seconds", 0) / 60, 1),
                "distress_periods": agg.get("distress_periods", 0),
                "distress_minutes": round(agg.get("distress_total_seconds", 0) / 60, 1),
            },
            "risk_scores": {
                "max_agitation": agg.get("max_agitation_score", 0),
                "avg_agitation": agg.get("avg_agitation_score", 0),
                "max_delirium": agg.get("max_delirium_risk", 0),
                "avg_delirium": agg.get("avg_delirium_risk", 0),
                "max_respiratory": agg.get("max_respiratory_distress", 0),
                "avg_respiratory": agg.get("avg_respiratory_distress", 0),
                "max_fall_risk": agg.get("max_fall_risk", 0),
                "avg_fall_risk": agg.get("avg_fall_risk", 0),
            },
            "alerts": {
                "counts": agg.get("alert_counts", {}),
                "critical": agg.get("critical_alerts", 0),
                "high_risk": agg.get("high_risk_alerts", 0),
            },
            "quality_metrics": {
                "avg_pose_confidence": agg.get("avg_pose_confidence", 0),
                "keypoint_visibility_pct": round(agg.get("keypoint_visibility_pct", 0) * 100, 1),
            },
            "activity_distribution": agg.get("activity_seconds", {}),
            "metadata": {
                "generated_at": time.time(),
                "generated_at_iso": datetime.now().isoformat(),
                "source": "hourly_aggregate",
            }
        }

    def _generate_hourly_from_events(self, start_time: float, end_time: float) -> Dict:
        """Fallback: Generate hourly report from raw events (legacy)."""
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

        return {
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
                "generated_at_iso": datetime.now().isoformat(),
                "source": "raw_events",
            }
        }
    
    def generate_daily_report(self, date_str: str = None) -> Dict:
        """
        Generate daily report from hourly aggregates.

        Args:
            date_str: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Dictionary with daily report
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        log.info("Generating daily report for %s", date_str)

        # Parse date
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            log.error("Invalid date format: %s", date_str)
            return {}

        start_ts = date.timestamp()
        end_ts = (date + timedelta(days=1)).timestamp()

        # Get all hourly aggregates for the day
        hours = self.db.query_hourly_aggregates(
            device_id=self.device_id,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=24
        )

        if not hours:
            log.warning("No hourly data found for %s", date_str)
            return self._generate_empty_daily_report(date_str)

        # Aggregate metrics across hours
        report = self._aggregate_hours_to_daily(hours, date_str)
        return report

    def _aggregate_hours_to_daily(self, hours: List[Dict], date_str: str) -> Dict:
        """Aggregate hourly records into a daily report."""
        # Initialize accumulators
        total_frames = 0
        total_with_person = 0
        total_observation = 0.0
        posture_totals = {}
        surface_totals = {}
        activity_totals = {}
        total_bed_exits = 0
        total_bed_attempts = 0
        total_falls = 0
        total_immobility_periods = 0
        total_immobility_seconds = 0.0
        total_distress_periods = 0
        total_distress_seconds = 0.0
        all_agitation = []
        all_delirium = []
        all_respiratory = []
        all_fall_risk = []
        alert_totals = {}
        critical_total = 0
        high_risk_total = 0

        hourly_breakdown = []

        for h in hours:
            total_frames += h.get("total_frames", 0)
            total_with_person += h.get("frames_with_person", 0)
            total_observation += h.get("total_observation_seconds", 0)

            # Posture distribution
            for posture, seconds in h.get("posture_seconds", {}).items():
                posture_totals[posture] = posture_totals.get(posture, 0) + seconds

            # Support surface distribution
            for surface, seconds in h.get("support_surface_seconds", {}).items():
                surface_totals[surface] = surface_totals.get(surface, 0) + seconds

            # Activity distribution
            for activity, seconds in h.get("activity_seconds", {}).items():
                activity_totals[activity] = activity_totals.get(activity, 0) + seconds

            # Clinical events
            total_bed_exits += h.get("bed_exit_count", 0)
            total_bed_attempts += h.get("bed_exit_attempts", 0)
            total_falls += h.get("fall_events", 0)
            total_immobility_periods += h.get("immobility_periods", 0)
            total_immobility_seconds += h.get("immobility_total_seconds", 0)
            total_distress_periods += h.get("distress_periods", 0)
            total_distress_seconds += h.get("distress_total_seconds", 0)

            # Risk scores
            if h.get("max_agitation_score", 0) > 0:
                all_agitation.append(h["max_agitation_score"])
            if h.get("max_delirium_risk", 0) > 0:
                all_delirium.append(h["max_delirium_risk"])
            if h.get("max_respiratory_distress", 0) > 0:
                all_respiratory.append(h["max_respiratory_distress"])
            if h.get("max_fall_risk", 0) > 0:
                all_fall_risk.append(h["max_fall_risk"])

            # Alerts
            for level, count in h.get("alert_counts", {}).items():
                alert_totals[level] = alert_totals.get(level, 0) + count
            critical_total += h.get("critical_alerts", 0)
            high_risk_total += h.get("high_risk_alerts", 0)

            # Hourly breakdown for charts
            hour_time = datetime.fromtimestamp(h.get("hour_start", 0))
            hourly_breakdown.append({
                "hour": hour_time.strftime("%H:00"),
                "observation_minutes": round(h.get("total_observation_seconds", 0) / 60, 1),
                "dominant_posture": h.get("dominant_posture", "unknown"),
                "bed_time_pct": round(h.get("time_in_bed_pct", 0) * 100, 1),
                "bed_exits": h.get("bed_exit_count", 0),
                "max_agitation": h.get("max_agitation_score", 0),
                "alerts": h.get("critical_alerts", 0) + h.get("high_risk_alerts", 0),
            })

        # Calculate dominant values
        dominant_posture = max(posture_totals.items(), key=lambda x: x[1])[0] if posture_totals else "unknown"
        dominant_activity = max(activity_totals.items(), key=lambda x: x[1])[0] if activity_totals else "unknown"

        bed_time = surface_totals.get("bed", 0)
        bed_time_pct = bed_time / total_observation * 100 if total_observation > 0 else 0

        return {
            "report_type": "daily",
            "device_id": self.device_id,
            "date": date_str,
            "hours_with_data": len(hours),
            "summary": {
                "total_frames": total_frames,
                "frames_with_person": total_with_person,
                "observation_hours": round(total_observation / 3600, 2),
                "dominant_posture": dominant_posture,
                "dominant_activity": dominant_activity,
            },
            "posture_analysis": {
                "distribution_seconds": posture_totals,
                "distribution_pct": {
                    k: round(v / total_observation * 100, 1) if total_observation > 0 else 0
                    for k, v in posture_totals.items()
                },
            },
            "support_surface": {
                "distribution_seconds": surface_totals,
                "time_in_bed_hours": round(bed_time / 3600, 2),
                "time_in_bed_pct": round(bed_time_pct, 1),
            },
            "clinical_events": {
                "bed_exits": total_bed_exits,
                "bed_exit_attempts": total_bed_attempts,
                "falls": total_falls,
                "immobility_periods": total_immobility_periods,
                "immobility_hours": round(total_immobility_seconds / 3600, 2),
                "distress_periods": total_distress_periods,
                "distress_hours": round(total_distress_seconds / 3600, 2),
            },
            "risk_scores": {
                "max_agitation": max(all_agitation) if all_agitation else 0,
                "avg_agitation": round(sum(all_agitation) / len(all_agitation), 3) if all_agitation else 0,
                "max_delirium": max(all_delirium) if all_delirium else 0,
                "max_respiratory": max(all_respiratory) if all_respiratory else 0,
                "max_fall_risk": max(all_fall_risk) if all_fall_risk else 0,
            },
            "alerts": {
                "counts": alert_totals,
                "critical": critical_total,
                "high_risk": high_risk_total,
                "total": sum(alert_totals.values()),
            },
            "activity_distribution": activity_totals,
            "hourly_breakdown": hourly_breakdown,
            "metadata": {
                "generated_at": time.time(),
                "generated_at_iso": datetime.now().isoformat(),
                "source": "hourly_aggregates",
            }
        }

    def _generate_empty_daily_report(self, date_str: str) -> Dict:
        """Generate empty daily report structure."""
        return {
            "report_type": "daily",
            "device_id": self.device_id,
            "date": date_str,
            "hours_with_data": 0,
            "summary": {},
            "posture_analysis": {},
            "support_surface": {},
            "clinical_events": {},
            "risk_scores": {},
            "alerts": {},
            "activity_distribution": {},
            "hourly_breakdown": [],
            "metadata": {
                "generated_at": time.time(),
                "generated_at_iso": datetime.now().isoformat(),
                "source": "empty",
            }
        }

    def generate_monthly_report(self, year: int, month: int) -> Dict:
        """
        Generate monthly report from hourly aggregates.

        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)

        Returns:
            Dictionary with comprehensive monthly report
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

        # Get all hourly aggregates for the month
        hours = self.db.query_hourly_aggregates(
            device_id=self.device_id,
            start_ts=start_time,
            end_ts=end_time,
            limit=744  # Max hours in a month
        )

        if hours:
            # Use hourly aggregates (efficient path)
            return self._generate_monthly_from_aggregates(hours, year, month, start_dt, end_dt)
        else:
            # Fallback to raw events (legacy)
            return self._generate_monthly_from_events(year, month, start_time, end_time, start_dt, end_dt)

    def _generate_monthly_from_aggregates(
        self,
        hours: List[Dict],
        year: int,
        month: int,
        start_dt: datetime,
        end_dt: datetime
    ) -> Dict:
        """Generate monthly report from hourly aggregates."""
        # Group hours by day
        daily_data = {}
        for h in hours:
            day = datetime.fromtimestamp(h.get("hour_start", 0)).strftime("%Y-%m-%d")
            if day not in daily_data:
                daily_data[day] = []
            daily_data[day].append(h)

        # Generate daily summaries
        daily_breakdown = {}
        for day, day_hours in daily_data.items():
            daily_breakdown[day] = {
                "hours_with_data": len(day_hours),
                "total_observation_hours": round(
                    sum(h.get("total_observation_seconds", 0) for h in day_hours) / 3600, 2
                ),
                "bed_exits": sum(h.get("bed_exit_count", 0) for h in day_hours),
                "falls": sum(h.get("fall_events", 0) for h in day_hours),
                "immobility_hours": round(
                    sum(h.get("immobility_total_seconds", 0) for h in day_hours) / 3600, 2
                ),
                "critical_alerts": sum(h.get("critical_alerts", 0) for h in day_hours),
                "high_risk_alerts": sum(h.get("high_risk_alerts", 0) for h in day_hours),
                "max_agitation": max(
                    (h.get("max_agitation_score", 0) for h in day_hours), default=0
                ),
                "avg_bed_time_pct": round(
                    sum(h.get("time_in_bed_pct", 0) for h in day_hours) / len(day_hours) * 100, 1
                ) if day_hours else 0,
            }

        # Calculate monthly totals
        total_observation = sum(h.get("total_observation_seconds", 0) for h in hours)
        total_bed_time = sum(h.get("time_in_bed_seconds", 0) for h in hours)
        total_bed_exits = sum(h.get("bed_exit_count", 0) for h in hours)
        total_falls = sum(h.get("fall_events", 0) for h in hours)
        total_immobility = sum(h.get("immobility_total_seconds", 0) for h in hours)

        # Aggregate postures
        posture_totals = {}
        for h in hours:
            for posture, seconds in h.get("posture_seconds", {}).items():
                posture_totals[posture] = posture_totals.get(posture, 0) + seconds

        # Risk score trends
        daily_risk_trend = []
        for day in sorted(daily_breakdown.keys()):
            daily_risk_trend.append({
                "date": day,
                "max_agitation": daily_breakdown[day]["max_agitation"],
                "critical_alerts": daily_breakdown[day]["critical_alerts"],
            })

        return {
            "report_type": "monthly",
            "device_id": self.device_id,
            "year": year,
            "month": month,
            "start_time": start_dt.timestamp(),
            "end_time": end_dt.timestamp(),
            "start_time_iso": start_dt.isoformat(),
            "end_time_iso": end_dt.isoformat(),
            "summary": {
                "days_with_data": len(daily_data),
                "total_hours": len(hours),
                "observation_hours": round(total_observation / 3600, 1),
                "avg_daily_observation": round(
                    total_observation / 3600 / len(daily_data), 1
                ) if daily_data else 0,
            },
            "posture_analysis": {
                "distribution_hours": {
                    k: round(v / 3600, 2) for k, v in posture_totals.items()
                },
                "dominant_posture": max(
                    posture_totals.items(), key=lambda x: x[1]
                )[0] if posture_totals else "unknown",
            },
            "support_surface": {
                "time_in_bed_hours": round(total_bed_time / 3600, 1),
                "avg_bed_time_pct": round(
                    total_bed_time / total_observation * 100, 1
                ) if total_observation > 0 else 0,
            },
            "clinical_events": {
                "total_bed_exits": total_bed_exits,
                "avg_daily_bed_exits": round(
                    total_bed_exits / len(daily_data), 2
                ) if daily_data else 0,
                "total_falls": total_falls,
                "total_immobility_hours": round(total_immobility / 3600, 1),
            },
            "risk_summary": {
                "days_with_critical_alerts": sum(
                    1 for d in daily_breakdown.values() if d["critical_alerts"] > 0
                ),
                "max_agitation_day": max(
                    daily_breakdown.items(), key=lambda x: x[1]["max_agitation"]
                )[0] if daily_breakdown else None,
                "max_agitation_score": max(
                    (d["max_agitation"] for d in daily_breakdown.values()), default=0
                ),
            },
            "alerts_summary": {
                "total_critical": sum(d["critical_alerts"] for d in daily_breakdown.values()),
                "total_high_risk": sum(d["high_risk_alerts"] for d in daily_breakdown.values()),
            },
            "daily_breakdown": daily_breakdown,
            "risk_trend": daily_risk_trend,
            "metadata": {
                "generated_at": time.time(),
                "generated_at_iso": datetime.now().isoformat(),
                "source": "hourly_aggregates",
            }
        }

    def _generate_monthly_from_events(
        self,
        year: int,
        month: int,
        start_time: float,
        end_time: float,
        start_dt: datetime,
        end_dt: datetime
    ) -> Dict:
        """Fallback: Generate monthly report from raw events."""
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

        return {
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
                "generated_at_iso": datetime.now().isoformat(),
                "source": "raw_events",
            }
        }
    
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

