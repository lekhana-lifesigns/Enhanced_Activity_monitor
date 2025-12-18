# analytics/compliance.py
"""
Automated Documentation and Protocol Compliance Tracking
Tracks patient adherence to care protocols and automatically documents activities.
"""

import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

log = logging.getLogger("compliance")


class ComplianceTracker:
    """
    Tracks protocol compliance and automated documentation.
    Monitors patient activities against care protocols and generates compliance reports.
    """
    
    def __init__(self, db):
        """Initialize compliance tracker."""
        self.db = db
        
        # Protocol definitions
        self.protocols = {
            'bed_rest': {
                'name': 'Bed Rest Protocol',
                'required_postures': ['supine', 'left_lateral', 'right_lateral'],
                'forbidden_postures': ['sitting', 'standing', 'walking'],
                'check_interval_minutes': 15
            },
            'position_change': {
                'name': 'Position Change Protocol',
                'required_changes_per_hour': 2,
                'min_change_interval_minutes': 30
            },
            'hand_hygiene': {
                'name': 'Hand Hygiene Protocol',
                'required_before_contact': True
            }
        }
    
    def check_compliance(self, patient_id: str, event: Dict) -> Dict:
        """
        Check if an event complies with protocols.
        
        Args:
            patient_id: Patient identifier
            event: Event data (activity, posture, etc.)
        
        Returns:
            Compliance check result
        """
        violations = []
        compliance_score = 1.0
        
        # Get patient configuration
        # For now, use default protocols
        protocols = self.protocols
        
        # Check bed rest protocol
        if 'bed_rest' in protocols:
            posture = event.get('posture_state', 'unknown')
            if posture in protocols['bed_rest']['forbidden_postures']:
                violations.append({
                    'protocol': 'bed_rest',
                    'violation': f'Patient in forbidden posture: {posture}',
                    'severity': 'HIGH',
                    'timestamp': event.get('ts', time.time())
                })
                compliance_score -= 0.3
        
        # Check position change protocol
        if 'position_change' in protocols:
            # This would require tracking position changes over time
            # Simplified check for now
            pass
        
        # Check hand proximity (safety protocol)
        hand_risk = event.get('hand_proximity_risk', 0.0)
        if hand_risk > 0.7:
            violations.append({
                'protocol': 'safety',
                'violation': 'High hand proximity risk detected',
                'severity': 'CRITICAL',
                'timestamp': event.get('ts', time.time())
            })
            compliance_score -= 0.5
        
        compliance_score = max(0.0, compliance_score)
        
        return {
            'compliant': len(violations) == 0,
            'compliance_score': compliance_score,
            'violations': violations,
            'timestamp': time.time()
        }
    
    def get_compliance_report(self, patient_id: str, hours: int = 24) -> Dict:
        """
        Generate compliance report for a patient.
        
        Args:
            patient_id: Patient identifier
            hours: Hours of data to analyze
        
        Returns:
            Compliance report
        """
        try:
            # Get events from last N hours
            start_ts = time.time() - (hours * 3600)
            events = self.db.query_events(
                start_ts=start_ts,
                limit=1000
            )
            
            if not events:
                return {
                    'patient_id': patient_id,
                    'period_hours': hours,
                    'total_events': 0,
                    'compliance_score': 1.0,
                    'violations': [],
                    'protocol_adherence': {}
                }
            
            # Analyze compliance
            total_violations = 0
            protocol_violations = {}
            compliance_scores = []
            
            for event in events:
                payload = event.get('payload', {})
                if isinstance(payload, str):
                    import json
                    payload = json.loads(payload)
                
                compliance = self.check_compliance(patient_id, {
                    'posture_state': payload.get('posture_state'),
                    'activity_state': payload.get('activity_state'),
                    'hand_proximity_risk': payload.get('decision', {}).get('hand_proximity_risk', 0.0),
                    'ts': event.get('ts')
                })
                
                compliance_scores.append(compliance['compliance_score'])
                
                for violation in compliance['violations']:
                    total_violations += 1
                    protocol = violation['protocol']
                    if protocol not in protocol_violations:
                        protocol_violations[protocol] = []
                    protocol_violations[protocol].append(violation)
            
            avg_compliance = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 1.0
            
            # Calculate protocol adherence
            protocol_adherence = {}
            for protocol_name in self.protocols.keys():
                violations_count = len(protocol_violations.get(protocol_name, []))
                adherence = 1.0 - (violations_count / len(events)) if events else 1.0
                protocol_adherence[protocol_name] = {
                    'adherence_rate': max(0.0, adherence),
                    'violations_count': violations_count
                }
            
            return {
                'patient_id': patient_id,
                'period_hours': hours,
                'total_events': len(events),
                'compliance_score': avg_compliance,
                'violations_count': total_violations,
                'violations': list(protocol_violations.values())[:10],  # Top 10
                'protocol_adherence': protocol_adherence
            }
            
        except Exception as e:
            log.exception("Get compliance report failed: %s", e)
            return {
                'patient_id': patient_id,
                'error': str(e)
            }
    
    def get_recent_violations(self, patient_id: str, hours: int = 24) -> List[Dict]:
        """Get recent protocol violations."""
        report = self.get_compliance_report(patient_id, hours)
        violations = []
        
        for protocol_violations in report.get('violations', []):
            violations.extend(protocol_violations)
        
        # Sort by timestamp (most recent first)
        violations.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        return violations[:20]  # Return top 20
    
    def document_activity(self, patient_id: str, activity: Dict) -> Dict:
        """
        Automatically document patient activity.
        
        Args:
            patient_id: Patient identifier
            activity: Activity data
        
        Returns:
            Documentation record
        """
        try:
            # Create documentation entry
            doc = {
                'patient_id': patient_id,
                'timestamp': time.time(),
                'activity': activity.get('label', 'unknown'),
                'posture': activity.get('posture_state', 'unknown'),
                'confidence': activity.get('confidence', 0.0),
                'compliance': self.check_compliance(patient_id, activity),
                'metadata': {
                    'device_id': activity.get('device_id'),
                    'track_id': activity.get('track_id')
                }
            }
            
            # Store in database (would need documentation table)
            # For now, return the document
            
            return doc
            
        except Exception as e:
            log.exception("Document activity failed: %s", e)
            return {'error': str(e)}

