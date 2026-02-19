# patient_portal.py - Patient Privacy Portal
"""
Patient Privacy Portal for HIPAA/GDPR compliance.
"""

from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import sqlite3
import json
import time
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from storage.db import LocalDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("patient_portal")

class PatientPrivacyPortal:
 """HIPAA/GDPR compliant patient portal."""

 def __init__(self, db_path="storage/eam.db", secret_key=None):
 self.db_path = db_path
 self.secret_key = secret_key or secrets.token_hex(32)
 self.app = Flask(__name__, template_folder='dashboard/templates')
 self.app.secret_key = self.secret_key

 # Setup routes
 self._setup_routes()

 log.info("Patient Privacy Portal initialized")

 def _setup_routes(self):
 """Setup Flask routes for patient portal."""

 @self.app.route('/')
 def index():
 """Patient portal home page."""
 if 'patient_id' not in session:
 return redirect(url_for('login'))
 return render_template('patient_portal.html')

 @self.app.route('/login', methods=['GET', 'POST'])
 def login():
 """Patient login page."""
 if request.method == 'POST':
 patient_id = request.form.get('patient_id')
 access_code = request.form.get('access_code')

 if self._verify_patient_access(patient_id, access_code):
 session['patient_id'] = patient_id
 session['login_time'] = datetime.now().isoformat()
 log.info(f"Patient {patient_id} logged into portal")
 return redirect(url_for('dashboard'))
 else:
 return render_template('login.html', error="Invalid credentials")

 return render_template('login.html')

 @self.app.route('/logout')
 def logout():
 """Patient logout."""
 patient_id = session.get('patient_id')
 session.clear()
 if patient_id:
 log.info(f"Patient {patient_id} logged out of portal")
 return redirect(url_for('login'))

 @self.app.route('/dashboard')
 def dashboard():
 """Patient dashboard with data access."""
 if 'patient_id' not in session:
 return redirect(url_for('login'))

 patient_id = session['patient_id']
 return render_template('patient_dashboard.html', patient_id=patient_id)

 @self.app.route('/api/data')
 def get_patient_data():
 """API endpoint to retrieve patient's activity data."""
 if 'patient_id' not in session:
 return jsonify({"error": "Not authenticated"}), 401

 patient_id = session['patient_id']
 days = int(request.args.get('days', 7)) # Default to 7 days

 try:
 # Get patient's activity data
 data = self._get_patient_activity_data(patient_id, days)

 # Log data access for audit
 self._log_data_access(patient_id, "data_access", f"Retrieved {len(data)} records for {days} days")

 return jsonify({
 "patient_id": patient_id,
 "data": data,
 "date_range": f"Last {days} days"
 })

 except Exception as e:
 log.error(f"Error retrieving data for patient {patient_id}: {e}")
 return jsonify({"error": "Data retrieval failed"}), 500

 @self.app.route('/api/delete', methods=['POST'])
 def request_data_deletion():
 """API endpoint for patient to request data deletion (Right to Erasure)."""
 if 'patient_id' not in session:
 return jsonify({"error": "Not authenticated"}), 401

 patient_id = session['patient_id']
 reason = request.json.get('reason', 'Patient requested deletion')

 try:
 # Log deletion request
 self._log_data_access(patient_id, "deletion_request", reason)

 # Mark data for deletion (don't actually delete immediately for compliance)
 self._mark_data_for_deletion(patient_id, reason)

 log.info(f"Patient {patient_id} requested data deletion: {reason}")
 return jsonify({"message": "Deletion request submitted. Data will be removed within 30 days."})

 except Exception as e:
 log.error(f"Error processing deletion request for patient {patient_id}: {e}")
 return jsonify({"error": "Deletion request failed"}), 500

 @self.app.route('/api/export')
 def export_patient_data():
 """API endpoint for data export (Data Portability)."""
 if 'patient_id' not in session:
 return jsonify({"error": "Not authenticated"}), 401

 patient_id = session['patient_id']
 format_type = request.args.get('format', 'json')

 try:
 # Get all patient data
 data = self._get_patient_activity_data(patient_id, days=365*2) # Last 2 years

 if format_type == 'json':
 # Log export for audit
 self._log_data_access(patient_id, "data_export", f"JSON export of {len(data)} records")

 return jsonify({
 "patient_id": patient_id,
 "export_date": datetime.now().isoformat(),
 "data": data
 })
 else:
 return jsonify({"error": "Unsupported format"}), 400

 except Exception as e:
 log.error(f"Error exporting data for patient {patient_id}: {e}")
 return jsonify({"error": "Data export failed"}), 500

 @self.app.route('/api/consent', methods=['GET', 'POST'])
 def manage_consent():
 """API endpoint for consent management."""
 if 'patient_id' not in session:
 return jsonify({"error": "Not authenticated"}), 401

 patient_id = session['patient_id']

 if request.method == 'GET':
 # Get current consent status
 consent = self._get_patient_consent(patient_id)
 return jsonify(consent)

 elif request.method == 'POST':
 # Update consent preferences
 consent_data = request.json
 self._update_patient_consent(patient_id, consent_data)

 # Log consent change
 self._log_data_access(patient_id, "consent_update", json.dumps(consent_data))

 log.info(f"Patient {patient_id} updated consent preferences")
 return jsonify({"message": "Consent preferences updated"})

 def _verify_patient_access(self, patient_id, access_code):
 """Verify patient access credentials."""
 # In production, this would verify against a secure patient database
 # For demo purposes, we'll use a simple hash check
 try:
 # This is a placeholder - in real implementation, verify against
 # hospital patient management system or secure token
 expected_hash = hashlib.sha256(f"{patient_id}_access_code".encode()).hexdigest()[:8]
 return access_code == expected_hash
 except:
 return False

 def _get_patient_activity_data(self, patient_id, days=7):
 """Retrieve patient's activity data from database."""
 try:
 # Query database for patient's data
 # This is a simplified version - real implementation would filter by patient_id
 conn = sqlite3.connect(self.db_path)
 cursor = conn.cursor()

 # Get events from last N days
 cutoff_time = time.time() - (days * 24 * 60 * 60)

 cursor.execute("""
 SELECT ts, label, confidence, payload
 FROM events
 WHERE device LIKE ?
 AND ts > ?
 ORDER BY ts DESC
 LIMIT 1000
 """, (f"%{patient_id}%", cutoff_time))

 events = []
 for row in cursor.fetchall():
 ts, label, confidence, payload_str = row
 try:
 payload = json.loads(payload_str) if payload_str else {}
 except:
 payload = {}

 events.append({
 "timestamp": ts,
 "activity": label,
 "confidence": confidence,
 "details": payload
 })

 conn.close()
 return events

 except Exception as e:
 log.error(f"Database query failed: {e}")
 return []

 def _mark_data_for_deletion(self, patient_id, reason):
 """Mark patient data for deletion (don't actually delete for audit purposes)."""
 try:
 conn = sqlite3.connect(self.db_path)
 cursor = conn.cursor()

 # Insert deletion request into audit log
 cursor.execute("""
 INSERT INTO audit_log (timestamp, action, patient_id, details)
 VALUES (?, ?, ?, ?)
 """, (time.time(), "deletion_requested", patient_id, reason))

 conn.commit()
 conn.close()

 except Exception as e:
 log.error(f"Failed to mark data for deletion: {e}")

 def _log_data_access(self, patient_id, action, details):
 """Log data access for audit purposes."""
 try:
 conn = sqlite3.connect(self.db_path)
 cursor = conn.cursor()

 cursor.execute("""
 INSERT INTO audit_log (timestamp, action, patient_id, details, ip_address)
 VALUES (?, ?, ?, ?, ?)
 """, (time.time(), action, patient_id, details, request.remote_addr))

 conn.commit()
 conn.close()

 except Exception as e:
 log.error(f"Failed to log data access: {e}")

 def _get_patient_consent(self, patient_id):
 """Get patient's current consent preferences."""
 try:
 conn = sqlite3.connect(self.db_path)
 cursor = conn.cursor()

 cursor.execute("""
 SELECT data_collection, analytics_sharing, emergency_alerts, last_updated
 FROM patient_consent
 WHERE patient_id = ?
 """, (patient_id,))

 row = cursor.fetchone()
 conn.close()

 if row:
 return {
 "data_collection": bool(row[0]),
 "analytics_sharing": bool(row[1]),
 "emergency_alerts": bool(row[2]),
 "last_updated": row[3]
 }
 else:
 # Return defaults if no consent record exists
 return {
 "data_collection": True,
 "analytics_sharing": False,
 "emergency_alerts": True,
 "last_updated": None
 }

 except Exception as e:
 log.error(f"Error getting patient consent: {e}")
 return {
 "data_collection": True,
 "analytics_sharing": False,
 "emergency_alerts": True,
 "last_updated": None
 }

 def _update_patient_consent(self, patient_id, consent_data):
 """Update patient's consent preferences."""
 try:
 conn = sqlite3.connect(self.db_path)
 cursor = conn.cursor()

 # Insert or replace consent record
 cursor.execute("""
 INSERT OR REPLACE INTO patient_consent
 (patient_id, data_collection, analytics_sharing, emergency_alerts, last_updated)
 VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
 """, (
 patient_id,
 consent_data.get('data_collection', True),
 consent_data.get('analytics_sharing', False),
 consent_data.get('emergency_alerts', True)
 ))

 conn.commit()
 conn.close()

 log.info(f"Updated consent for patient {patient_id}")

 except Exception as e:
 log.error(f"Error updating patient consent: {e}")
 raise

 def run(self, host='0.0.0.0', port=5001, debug=False):
 """Run the patient portal web server."""
 print(f"Starting Patient Privacy Portal on {host}:{port}")
 self.app.run(host=host, port=port, debug=debug)

# Standalone runner
if __name__ == "__main__":
 import argparse
 import yaml

 # Parse CLI arguments
 parser = argparse.ArgumentParser(description="Patient Privacy Portal - HIPAA/GDPR Compliant")
 parser.add_argument("--port", type=int, help="Web server port (overrides config)")
 parser.add_argument("--host", type=str, default='0.0.0.0', help="Web server host")
 parser.add_argument("--debug", action='store_true', help="Enable debug mode")
 parser.add_argument("--config", type=str, default="config/system.yaml", help="Path to system config file")
 args = parser.parse_args()

 # Load port from config or use CLI argument
 port = args.port
 if port is None:
 try:
 with open(args.config, 'r') as f:
 cfg = yaml.safe_load(f)
 port = cfg.get("patient_portal_port", 5001)
 except Exception:
 port = 5001 # Fallback to default

 portal = PatientPrivacyPortal()
 portal.run(host=args.host, port=port, debug=args.debug)