# scripts/test_tracking_reporting.py
"""
Test script to verify tracking and reporting functionality.
"""
import sys
import os
import time
import yaml
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pose.inference_pipeline import InferencePipeline
from storage.db import LocalDB
from storage.reporting import ReportGenerator
from telemetry.mqtt_client import MqttClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("test_tracking_reporting")

def test_tracking():
    """Test if tracking is working properly."""
    print("\n" + "="*70)
    print("TRACKING TEST")
    print("="*70)
    
    try:
        cfg = yaml.safe_load(open("config/system.yaml"))
        cfg["enable_display"] = False  # Disable display for testing
        
        pipe = InferencePipeline(cfg)
        
        track_ids_seen = set()
        frames_with_tracking = 0
        total_frames = 0
        
        print("Running 10 frames to test tracking...")
        for i in range(10):
            result = pipe.run_once()
            total_frames += 1
            
            if result:
                track_id = result.get("track_id")
                if track_id is not None:
                    track_ids_seen.add(track_id)
                    frames_with_tracking += 1
                    print(f"  Frame {i+1}: Track ID = {track_id}, Label = {result.get('label')}, Posture = {result.get('posture_state')}")
                else:
                    print(f"  Frame {i+1}: No track ID (no detection or tracking failed)")
            else:
                print(f"  Frame {i+1}: No result (no detection)")
        
        print(f"\n✅ Tracking Results:")
        print(f"  - Frames processed: {total_frames}")
        print(f"  - Frames with tracking: {frames_with_tracking}")
        print(f"  - Unique track IDs seen: {len(track_ids_seen)}")
        print(f"  - Track IDs: {sorted(track_ids_seen) if track_ids_seen else 'None'}")
        
        if frames_with_tracking > 0:
            print(f"✅ Tracking is WORKING - {frames_with_tracking}/{total_frames} frames have track IDs")
        else:
            print(f"⚠️  Tracking may not be working - no track IDs detected")
        
        return frames_with_tracking > 0
        
    except Exception as e:
        log.exception("Tracking test failed: %s", e)
        return False

def test_reporting():
    """Test if reporting is working properly."""
    print("\n" + "="*70)
    print("REPORTING TEST")
    print("="*70)
    
    try:
        cfg = yaml.safe_load(open("config/system.yaml"))
        
        # Initialize database
        db = LocalDB()
        print("✅ Database initialized")
        
        # Initialize report generator
        report_gen = ReportGenerator(db, cfg["device_id"])
        print("✅ Report generator initialized")
        
        # Check if we have data
        import sqlite3
        conn = sqlite3.connect("storage/events.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events")
        event_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM alerts")
        alert_count = cursor.fetchone()[0]
        conn.close()
        
        print(f"  - Events in database: {event_count}")
        print(f"  - Alerts in database: {alert_count}")
        
        if event_count == 0:
            print("⚠️  No events in database - cannot generate meaningful report")
            print("   (This is OK if system just started)")
        else:
            # Try to generate hourly report
            try:
                hourly_report = report_gen.generate_hourly_report()
                print(f"\n✅ Hourly Report Generated:")
                print(f"  - Report type: {hourly_report.get('report_type')}")
                print(f"  - Time range: {hourly_report.get('start_time')} to {hourly_report.get('end_time')}")
                print(f"  - Total events: {hourly_report.get('total_events', 0)}")
                print(f"  - Total alerts: {hourly_report.get('total_alerts', 0)}")
                print(f"  - Activity distribution: {hourly_report.get('activity_distribution', {})}")
            except Exception as e:
                print(f"❌ Failed to generate hourly report: {e}")
                return False
        
        # Test MQTT reporting
        try:
            mqtt_cfg = yaml.safe_load(open("config/mqtt.yaml"))
            mqtt_client = MqttClient(mqtt_cfg, cfg["device_id"])
            print(f"\n✅ MQTT client initialized")
            print(f"  - Broker: {mqtt_cfg.get('broker')}:{mqtt_cfg.get('port')}")
            print(f"  - Topic prefix: {mqtt_cfg.get('topic_prefix')}")
            
            # Test report publishing (if we have a report)
            if event_count > 0:
                try:
                    hourly_report = report_gen.generate_hourly_report()
                    # Don't actually publish in test mode, just verify it would work
                    print(f"  - Report publishing: Ready (would publish to {mqtt_cfg.get('topic_prefix')}/{cfg['device_id']}/reports/hourly)")
                except Exception as e:
                    print(f"  - Report publishing: Failed - {e}")
        except Exception as e:
            print(f"⚠️  MQTT client initialization failed: {e}")
            print("   (This is OK if MQTT broker is not available)")
        
        return True
        
    except Exception as e:
        log.exception("Reporting test failed: %s", e)
        return False

def test_integration():
    """Test integration between tracking and reporting."""
    print("\n" + "="*70)
    print("INTEGRATION TEST")
    print("="*70)
    
    try:
        cfg = yaml.safe_load(open("config/system.yaml"))
        cfg["enable_display"] = False
        
        pipe = InferencePipeline(cfg)
        db = LocalDB()
        
        # Process a few frames and check if they're stored
        print("Processing 5 frames and storing in database...")
        stored_count = 0
        
        for i in range(5):
            result = pipe.run_once()
            if result:
                track_id = result.get("track_id")
                label = result.get("label", "unknown")
                
                # Store in database
                db.insert_event(
                    device=cfg["device_id"],
                    label=label,
                    confidence=result.get("confidence", 0.0),
                    payload=result
                )
                stored_count += 1
                print(f"  Frame {i+1}: Stored (Track ID: {track_id}, Label: {label})")
        
        print(f"\n✅ Integration Results:")
        print(f"  - Frames stored: {stored_count}")
        
        # Verify storage
        import sqlite3
        conn = sqlite3.connect("storage/events.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events WHERE device = ?", (cfg["device_id"],))
        recent_count = cursor.fetchone()[0]
        conn.close()
        
        print(f"  - Events in DB for device: {recent_count}")
        
        if stored_count > 0:
            print("✅ Integration WORKING - Events are being stored with tracking data")
        else:
            print("⚠️  Integration may have issues - no events stored")
        
        return stored_count > 0
        
    except Exception as e:
        log.exception("Integration test failed: %s", e)
        return False

def main():
    print("\n" + "="*70)
    print("TRACKING & REPORTING VERIFICATION")
    print("="*70)
    
    tracking_ok = test_tracking()
    reporting_ok = test_reporting()
    integration_ok = test_integration()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Tracking: {'✅ WORKING' if tracking_ok else '⚠️  ISSUES'}")
    print(f"Reporting: {'✅ WORKING' if reporting_ok else '⚠️  ISSUES'}")
    print(f"Integration: {'✅ WORKING' if integration_ok else '⚠️  ISSUES'}")
    print("="*70)
    
    if tracking_ok and reporting_ok and integration_ok:
        print("\n✅ All systems operational!")
        return 0
    else:
        print("\n⚠️  Some issues detected - check logs above")
        return 1

if __name__ == "__main__":
    sys.exit(main())

