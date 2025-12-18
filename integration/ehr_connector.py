# integration/ehr_connector.py
"""
EHR/HIMS Integration Connector
Provides HL7/FHIR integration for seamless workflow integration.
"""

import json
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime

log = logging.getLogger("ehr")


class EHRConnector:
    """
    Connector for Electronic Health Records (EHR) and Hospital Information Management Systems (HIMS).
    Supports HL7 FHIR R4 standard for interoperability.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize EHR connector.
        
        Args:
            config: Configuration dictionary with EHR settings
        """
        self.config = config
        self.fhir_base_url = config.get('fhir_base_url')
        self.hl7_endpoint = config.get('hl7_endpoint')
        self.api_key = config.get('api_key')
        self.enabled = config.get('enabled', False)
        
    def create_observation(self, patient_id: str, observation_data: Dict) -> Dict:
        """
        Create FHIR Observation resource for patient monitoring data.
        
        Args:
            patient_id: Patient identifier
            observation_data: Observation data (activity, posture, scores, etc.)
        
        Returns:
            FHIR Observation resource
        """
        if not self.enabled:
            return {'error': 'EHR integration disabled'}
        
        try:
            # Create FHIR Observation resource
            observation = {
                'resourceType': 'Observation',
                'status': 'final',
                'category': [{
                    'coding': [{
                        'system': 'http://terminology.hl7.org/CodeSystem/observation-category',
                        'code': 'vital-signs',
                        'display': 'Vital Signs'
                    }]
                }],
                'code': {
                    'coding': [{
                        'system': 'http://loinc.org',
                        'code': '85354-9',  # Patient activity
                        'display': 'Patient Activity'
                    }]
                },
                'subject': {
                    'reference': f'Patient/{patient_id}'
                },
                'effectiveDateTime': datetime.fromtimestamp(
                    observation_data.get('timestamp', time.time())
                ).isoformat(),
                'valueString': observation_data.get('activity', 'unknown'),
                'component': [
                    {
                        'code': {
                            'coding': [{
                                'code': 'posture',
                                'display': 'Patient Posture'
                            }]
                        },
                        'valueString': observation_data.get('posture', 'unknown')
                    },
                    {
                        'code': {
                            'coding': [{
                                'code': 'agitation_score',
                                'display': 'Agitation Score'
                            }]
                        },
                        'valueQuantity': {
                            'value': observation_data.get('agitation_score', 0.0),
                            'unit': 'score',
                            'system': 'http://unitsofmeasure.org'
                        }
                    },
                    {
                        'code': {
                            'coding': [{
                                'code': 'delirium_risk',
                                'display': 'Delirium Risk'
                            }]
                        },
                        'valueQuantity': {
                            'value': observation_data.get('delirium_risk', 0.0),
                            'unit': 'score',
                            'system': 'http://unitsofmeasure.org'
                        }
                    }
                ]
            }
            
            # In production, this would POST to FHIR server
            # For now, return the resource
            log.info("Created FHIR Observation for patient %s", patient_id)
            
            return observation
            
        except Exception as e:
            log.exception("Create observation failed: %s", e)
            return {'error': str(e)}
    
    def create_alert(self, patient_id: str, alert_data: Dict) -> Dict:
        """
        Create FHIR Flag resource for clinical alerts.
        
        Args:
            patient_id: Patient identifier
            alert_data: Alert data
        
        Returns:
            FHIR Flag resource
        """
        if not self.enabled:
            return {'error': 'EHR integration disabled'}
        
        try:
            # Map alert level to FHIR severity
            severity_map = {
                'CRITICAL': 'critical',
                'HIGH_RISK': 'high',
                'MEDIUM_RISK': 'moderate',
                'LOW_RISK': 'low'
            }
            
            severity = severity_map.get(alert_data.get('alert_level', 'LOW_RISK'), 'low')
            
            flag = {
                'resourceType': 'Flag',
                'status': 'active',
                'category': [{
                    'coding': [{
                        'system': 'http://terminology.hl7.org/CodeSystem/flag-category',
                        'code': 'safety',
                        'display': 'Safety'
                    }]
                }],
                'code': {
                    'coding': [{
                        'system': 'http://snomed.info/sct',
                        'code': '248218005',  # Patient monitoring
                        'display': 'Patient Monitoring Alert'
                    }]
                },
                'subject': {
                    'reference': f'Patient/{patient_id}'
                },
                'period': {
                    'start': datetime.fromtimestamp(
                        alert_data.get('timestamp', time.time())
                    ).isoformat()
                },
                'severity': severity
            }
            
            log.info("Created FHIR Flag for patient %s (severity: %s)", patient_id, severity)
            
            return flag
            
        except Exception as e:
            log.exception("Create alert failed: %s", e)
            return {'error': str(e)}
    
    def sync_patient_data(self, patient_id: str, events: List[Dict]) -> Dict:
        """
        Sync patient monitoring data to EHR system.
        
        Args:
            patient_id: Patient identifier
            events: List of events to sync
        
        Returns:
            Sync result
        """
        if not self.enabled:
            return {'error': 'EHR integration disabled', 'synced': 0}
        
        try:
            synced_count = 0
            
            for event in events:
                payload = event.get('payload', {})
                if isinstance(payload, str):
                    payload = json.loads(payload)
                
                # Create observation
                observation_data = {
                    'timestamp': event.get('ts', time.time()),
                    'activity': payload.get('label', 'unknown'),
                    'posture': payload.get('posture_state', 'unknown'),
                    'agitation_score': payload.get('decision', {}).get('agitation_score', 0.0),
                    'delirium_risk': payload.get('decision', {}).get('delirium_risk', 0.0)
                }
                
                self.create_observation(patient_id, observation_data)
                synced_count += 1
            
            return {
                'success': True,
                'synced': synced_count,
                'total': len(events)
            }
            
        except Exception as e:
            log.exception("Sync patient data failed: %s", e)
            return {'error': str(e), 'synced': 0}
    
    def get_patient_info(self, patient_id: str) -> Optional[Dict]:
        """
        Retrieve patient information from EHR system.
        
        Args:
            patient_id: Patient identifier
        
        Returns:
            Patient information
        """
        if not self.enabled:
            return None
        
        try:
            # In production, this would GET from FHIR server
            # For now, return placeholder
            return {
                'patient_id': patient_id,
                'name': 'Retrieved from EHR',
                'dob': None,
                'allergies': [],
                'medications': []
            }
        except Exception as e:
            log.exception("Get patient info failed: %s", e)
            return None

