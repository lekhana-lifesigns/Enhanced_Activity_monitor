#!/usr/bin/env python3
"""
Clinical Dashboard Web Application
Real-time patient monitoring dashboard with prominent alerts, multi-patient support,
and predictive analytics visualization.
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import sys
import argparse
import yaml
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.db import LocalDB
from analytics.predictive import PredictiveAnalytics
from analytics.clinical_event_tracker import ClinicalEventTracker, PatientProgress

app = Flask(__name__)

# Global clinical event trackers per patient
clinical_trackers = {}
# Load SECRET_KEY from environment variable (required for security)
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    raise ValueError("SECRET_KEY environment variable must be set. "
                     "Set it with: export SECRET_KEY='your-secret-key-here'")
app.config['SECRET_KEY'] = secret_key
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize database connection
db = LocalDB()
predictive = PredictiveAnalytics(db)

# Store active connections
active_patients = {}
alert_history = []


def _resolve_device_id(patient_id):
    """Look up the device_id for a patient from active sessions."""
    sessions = db.query_patient_sessions(patient_id=patient_id, status='active')
    if sessions:
        return sessions[0].get('device_id')
    # Fallback: treat patient_id as device_id directly
    return patient_id


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/patients')
def get_patients():
    """Get list of all monitored patients."""
    try:
        patients = db.query_patient_sessions(status='active')

        patient_list = []
        for patient in patients:
            latest_alert = db.query_alerts(
                device=patient.get('device_id'),
                limit=1
            )
            risk_score = predictive.get_patient_risk(patient.get('patient_id'))

            patient_list.append({
                'patient_id': patient.get('patient_id'),
                'device_id': patient.get('device_id'),
                'bed_id': patient.get('bed_id'),
                'status': patient.get('status'),
                'latest_alert': latest_alert[0] if latest_alert else None,
                'risk_score': risk_score,
                'start_time': patient.get('start_ts')
            })

        return jsonify({'patients': patient_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/patient/<patient_id>/status')
def get_patient_status(patient_id):
    """Get real-time status for a specific patient."""
    try:
        device_id = _resolve_device_id(patient_id)

        events = db.query_events(device=device_id, limit=1)
        latest_event = events[0] if events else None

        alerts = db.query_alerts(device=device_id, limit=1)
        latest_alert = alerts[0] if alerts else None

        risk_forecast = predictive.forecast_risk(patient_id, hours_ahead=6)

        timeline = db.query_events(
            device=device_id,
            start_ts=time.time() - 3600,
            limit=100
        )

        return jsonify({
            'patient_id': patient_id,
            'device_id': device_id,
            'current_status': {
                'activity': latest_event.get('label') if latest_event else 'unknown',
                'posture': latest_event.get('payload', {}).get('posture_state') if latest_event else 'unknown',
                'confidence': latest_event.get('confidence') if latest_event else 0.0,
                'timestamp': latest_event.get('ts') if latest_event else time.time()
            },
            'latest_alert': latest_alert,
            'risk_forecast': risk_forecast,
            'timeline': timeline[:20]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts')
def get_alerts():
    """Get all active and recent alerts."""
    try:
        unacknowledged = db.query_alerts(limit=50)
        unacknowledged = [a for a in unacknowledged if not a.get('acknowledged', False)]

        recent = db.query_alerts(
            start_ts=time.time() - 86400,
            limit=100
        )

        critical = [a for a in unacknowledged if a.get('alert_level') == 'CRITICAL']
        high_risk = [a for a in unacknowledged if a.get('alert_level') == 'HIGH_RISK']
        medium_risk = [a for a in unacknowledged if a.get('alert_level') == 'MEDIUM_RISK']

        return jsonify({
            'unacknowledged': unacknowledged,
            'recent': recent,
            'by_severity': {
                'critical': critical,
                'high_risk': high_risk,
                'medium_risk': medium_risk
            },
            'total_unacknowledged': len(unacknowledged)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert."""
    try:
        db.acknowledge_alert(alert_id)
        socketio.emit('alert_acknowledged', {'alert_id': alert_id})
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analytics/predictive/<patient_id>')
def get_predictive_analytics(patient_id):
    """Get predictive analytics for a patient."""
    try:
        forecast = predictive.forecast_risk(patient_id, hours_ahead=24)
        trends = predictive.analyze_trends(patient_id, days=7)
        risk_factors = predictive.identify_risk_factors(patient_id)

        return jsonify({
            'forecast': forecast,
            'trends': trends,
            'risk_factors': risk_factors
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analytics/compliance/<patient_id>')
def get_compliance_analytics(patient_id):
    """Get protocol compliance analytics."""
    try:
        from analytics.compliance import ComplianceTracker
        tracker = ComplianceTracker(db)

        compliance = tracker.get_compliance_report(patient_id)
        violations = tracker.get_recent_violations(patient_id, hours=24)

        return jsonify({
            'compliance': compliance,
            'violations': violations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =========================================================================
# CLINICAL EVENT TRACKING ENDPOINTS
# =========================================================================

def get_or_create_tracker(patient_id: str) -> ClinicalEventTracker:
    """Get or create a clinical event tracker for a patient."""
    if patient_id not in clinical_trackers:
        clinical_trackers[patient_id] = ClinicalEventTracker(patient_id)
    return clinical_trackers[patient_id]


@app.route('/api/clinical/<patient_id>/dashboard')
def get_clinical_dashboard(patient_id):
    """Get live clinical dashboard data for a patient."""
    try:
        tracker = get_or_create_tracker(patient_id)
        dashboard_data = tracker.get_live_dashboard_data()
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clinical/<patient_id>/progress')
def get_patient_progress(patient_id):
    """Get patient progress summary for the last 24 hours."""
    try:
        hours = request.args.get('hours', 24, type=int)
        tracker = get_or_create_tracker(patient_id)
        progress = tracker.get_patient_progress(hours)
        return jsonify(progress.__dict__)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clinical/<patient_id>/shift-report')
def get_shift_report(patient_id):
    """Generate end-of-shift nursing report."""
    try:
        tracker = get_or_create_tracker(patient_id)
        report = tracker.generate_shift_report()
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clinical/<patient_id>/sleep')
def get_sleep_status(patient_id):
    """Get current sleep tracking status."""
    try:
        tracker = get_or_create_tracker(patient_id)
        if tracker.current_state.get("sleep"):
            return jsonify({
                'patient_id': patient_id,
                'sleep_data': tracker.current_state["sleep"],
                'summary': tracker.sleep_tracker.get_sleep_summary()
            })
        return jsonify({'patient_id': patient_id, 'sleep_data': None})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clinical/<patient_id>/agitation')
def get_agitation_status(patient_id):
    """Get current agitation detection status."""
    try:
        tracker = get_or_create_tracker(patient_id)
        if tracker.current_state.get("agitation"):
            return jsonify({
                'patient_id': patient_id,
                'agitation_data': tracker.current_state["agitation"],
                'tube_pull_risk': tracker.current_state["agitation"].get("tube_pull_risk", 0)
            })
        return jsonify({'patient_id': patient_id, 'agitation_data': None})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clinical/<patient_id>/interventions')
def get_intervention_history(patient_id):
    """Get medical intervention history."""
    try:
        hours = request.args.get('hours', 24, type=int)
        tracker = get_or_create_tracker(patient_id)

        cutoff = time.time() - (hours * 3600)
        interventions = [
            {
                'timestamp': e.timestamp,
                'care_type': e.care_type.value,
                'activity': e.activity,
                'description': e.description
            }
            for e in tracker.events
            if e.timestamp >= cutoff and e.intervention_detected
        ]

        return jsonify({
            'patient_id': patient_id,
            'intervention_count': len(interventions),
            'interventions': interventions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clinical/<patient_id>/vitals-timeline')
def get_vitals_timeline(patient_id):
    """Get vital signs timeline for graphing."""
    try:
        hours = request.args.get('hours', 1, type=float)
        tracker = get_or_create_tracker(patient_id)

        cutoff = time.time() - (hours * 3600)
        vitals_data = []

        for vital in tracker.vitals_history:
            if vital.timestamp >= cutoff:
                vitals_data.append({
                    'timestamp': vital.timestamp,
                    'breath_rate': vital.breath_rate,
                    'breath_confidence': vital.breath_confidence,
                    'hrv_proxy': vital.hrv_proxy,
                    'motion_energy': vital.motion_energy
                })

        return jsonify({
            'patient_id': patient_id,
            'vitals': vitals_data,
            'count': len(vitals_data)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clinical/<patient_id>/alerts/active')
def get_active_clinical_alerts(patient_id):
    """Get active (unacknowledged) clinical alerts for a patient."""
    try:
        tracker = get_or_create_tracker(patient_id)

        active_alerts = [
            {
                'timestamp': e.timestamp,
                'event_type': e.event_type,
                'severity': e.severity.value,
                'description': e.description,
                'requires_acknowledgment': e.requires_acknowledgment
            }
            for e in tracker.events
            if e.requires_acknowledgment
        ][-20:]

        return jsonify({
            'patient_id': patient_id,
            'active_alerts': active_alerts,
            'count': len(active_alerts)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clinical/overview')
def get_clinical_overview():
    """Get overview of all patients for hospital-wide dashboard."""
    try:
        overview = []

        for patient_id, tracker in clinical_trackers.items():
            state = tracker.current_state
            progress = tracker.get_patient_progress(8)

            patient_summary = {
                'patient_id': patient_id,
                'last_update': state.get('last_update', 0),
                'needs_attention': False,
                'alerts_pending': 0,
                'agitation_level': 'none',
                'sleep_quality': 0,
                'intervention_active': False,
                'trend': progress.agitation_trend
            }

            if state.get('agitation'):
                ag = state['agitation']
                patient_summary['agitation_level'] = ag.get('agitation_level', 'none')
                patient_summary['needs_attention'] = ag.get('requires_intervention', False)
                if ag.get('tube_pull_detected'):
                    patient_summary['needs_attention'] = True
                    patient_summary['tube_pull_alert'] = True

            if state.get('sleep'):
                patient_summary['sleep_quality'] = state['sleep'].get('sleep_quality_score', 0)

            if state.get('intervention'):
                patient_summary['intervention_active'] = state['intervention'].get('intervention_detected', False)

            patient_summary['critical_events'] = progress.critical_events
            patient_summary['concern_indicators'] = progress.concern_indicators

            overview.append(patient_summary)

        priority_order = {'critical': 0, 'severe': 1, 'moderate': 2, 'mild': 3, 'none': 4}
        overview.sort(key=lambda x: (
            not x['needs_attention'],
            priority_order.get(x['agitation_level'], 4)
        ))

        return jsonify({
            'total_patients': len(overview),
            'patients_needing_attention': sum(1 for p in overview if p['needs_attention']),
            'patients': overview
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clinical/<patient_id>/activity-distribution')
def get_activity_distribution(patient_id):
    """Get activity distribution for pie/bar charts."""
    try:
        hours = request.args.get('hours', 24, type=int)
        tracker = get_or_create_tracker(patient_id)

        cutoff = time.time() - (hours * 3600)
        activity_counts = {}

        for event in tracker.events:
            if event.timestamp >= cutoff:
                activity = event.activity
                activity_counts[activity] = activity_counts.get(activity, 0) + 1

        distribution = [
            {'activity': k, 'count': v, 'percentage': 0}
            for k, v in sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        total = sum(d['count'] for d in distribution)
        for d in distribution:
            d['percentage'] = round(d['count'] / total * 100, 1) if total > 0 else 0

        return jsonify({
            'patient_id': patient_id,
            'hours_analyzed': hours,
            'total_events': total,
            'distribution': distribution
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clinical/<patient_id>/improvement-tracking')
def get_improvement_tracking(patient_id):
    """Track patient improvement over time."""
    try:
        tracker = get_or_create_tracker(patient_id)

        progress_8h = tracker.get_patient_progress(8)
        progress_24h = tracker.get_patient_progress(24)
        progress_48h = tracker.get_patient_progress(48)

        improvement_data = {
            'patient_id': patient_id,
            'periods': {
                '8h': {
                    'sleep_quality': progress_8h.sleep_quality_score,
                    'agitation_trend': progress_8h.agitation_trend,
                    'critical_events': progress_8h.critical_events,
                    'interventions': progress_8h.intervention_count
                },
                '24h': {
                    'sleep_quality': progress_24h.sleep_quality_score,
                    'agitation_trend': progress_24h.agitation_trend,
                    'critical_events': progress_24h.critical_events,
                    'interventions': progress_24h.intervention_count
                },
                '48h': {
                    'sleep_quality': progress_48h.sleep_quality_score,
                    'agitation_trend': progress_48h.agitation_trend,
                    'critical_events': progress_48h.critical_events,
                    'interventions': progress_48h.intervention_count
                }
            },
            'overall_trend': 'stable',
            'improvements': progress_24h.improvement_indicators,
            'concerns': progress_24h.concern_indicators
        }

        if progress_24h.critical_events < progress_48h.critical_events:
            improvement_data['overall_trend'] = 'improving'
        elif progress_24h.critical_events > progress_48h.critical_events:
            improvement_data['overall_trend'] = 'worsening'

        return jsonify(improvement_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('connected', {'status': 'connected'})


@socketio.on('subscribe_patient')
def handle_subscribe_patient(data):
    """Subscribe to real-time updates for a patient."""
    patient_id = data.get('patient_id')
    if patient_id:
        active_patients[request.sid] = patient_id
        emit('subscribed', {'patient_id': patient_id})


def background_alert_monitor():
    """Background thread to monitor for new alerts and broadcast them."""
    last_check = time.time()

    while True:
        try:
            new_alerts = db.query_alerts(
                start_ts=last_check,
                limit=100
            )

            for alert in new_alerts:
                if not alert.get('acknowledged', False):
                    socketio.emit('new_alert', {
                        'alert': alert,
                        'priority': 'high' if alert.get('alert_level') in ['CRITICAL', 'HIGH_RISK'] else 'medium'
                    })

            last_check = time.time()
            time.sleep(1)
        except Exception as e:
            time.sleep(5)


monitor_thread = None


def start_monitor_thread():
    """Start the background alert monitoring thread."""
    global monitor_thread
    if monitor_thread is None or not monitor_thread.is_alive():
        monitor_thread = threading.Thread(target=background_alert_monitor, daemon=True)
        monitor_thread.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clinical Dashboard Web Application")
    parser.add_argument("--port", type=int, help="Web server port (overrides config)")
    parser.add_argument("--host", type=str, default='0.0.0.0', help="Web server host")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    parser.add_argument("--config", type=str, default="../config/system.yaml",
                        help="Path to system config file")
    args = parser.parse_args()

    port = args.port
    if port is None:
        try:
            with open(args.config, 'r') as f:
                cfg = yaml.safe_load(f)
            port = cfg.get("dashboard_port", 5000)
        except Exception:
            port = 5000

    start_monitor_thread()

    debug_mode = args.debug
    if debug_mode and args.host not in ('127.0.0.1', 'localhost'):
        print("Warning: Debug mode is only allowed on localhost. Disabling debug mode.")
        debug_mode = False
        host = '127.0.0.1'
    else:
        host = args.host

    print("Starting Clinical Dashboard...")
    print(f"Access at: http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug_mode)
