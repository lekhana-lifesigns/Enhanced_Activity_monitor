# analytics/predictive.py
"""
Predictive Analytics for Health Deterioration Forecasting
Forecasts potential health deterioration hours in advance using pattern analysis.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta

log = logging.getLogger("predictive")


class PredictiveAnalytics:
    """
    Predictive analytics engine for forecasting health deterioration.
    Uses pattern recognition and trend analysis to predict adverse events.
    """
    
    def __init__(self, db, window_hours: int = 24, forecast_hours: int = 6):
        """
        Initialize predictive analytics.
        
        Args:
            db: Database connection
            window_hours: Hours of historical data to analyze
            forecast_hours: Hours ahead to forecast
        """
        self.db = db
        self.window_hours = window_hours
        self.forecast_hours = forecast_hours
        
        # Risk thresholds
        self.agitation_threshold = 0.7
        self.delirium_threshold = 0.6
        self.respiratory_threshold = 0.5
        
    def forecast_risk(self, patient_id: str, hours_ahead: int = 6) -> Dict:
        """
        Forecast risk of adverse events hours in advance.
        
        Args:
            patient_id: Patient identifier
            hours_ahead: Hours to forecast ahead
        
        Returns:
            Dictionary with risk forecast
        """
        try:
            # Get historical data
            end_ts = time.time()
            start_ts = end_ts - (self.window_hours * 3600)
            
            alerts = self.db.query_alerts(
                start_ts=start_ts,
                end_ts=end_ts
            )
            
            if not alerts:
                return {
                    'risk_level': 'LOW',
                    'confidence': 0.0,
                    'forecasted_events': [],
                    'risk_factors': []
                }
            
            # Analyze trends
            agitation_trend = self._analyze_trend(alerts, 'agitation_score')
            delirium_trend = self._analyze_trend(alerts, 'delirium_risk')
            respiratory_trend = self._analyze_trend(alerts, 'respiratory_distress')
            
            # Predict future risk
            predicted_agitation = self._extrapolate_trend(agitation_trend, hours_ahead)
            predicted_delirium = self._extrapolate_trend(delirium_trend, hours_ahead)
            predicted_respiratory = self._extrapolate_trend(respiratory_trend, hours_ahead)
            
            # Calculate overall risk
            max_risk = max(predicted_agitation, predicted_delirium, predicted_respiratory)
            
            if max_risk > 0.8:
                risk_level = 'HIGH'
            elif max_risk > 0.5:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            # Identify forecasted events
            forecasted_events = []
            if predicted_agitation > self.agitation_threshold:
                forecasted_events.append({
                    'type': 'agitation_episode',
                    'probability': predicted_agitation,
                    'time_horizon_hours': hours_ahead
                })
            
            if predicted_delirium > self.delirium_threshold:
                forecasted_events.append({
                    'type': 'delirium_risk',
                    'probability': predicted_delirium,
                    'time_horizon_hours': hours_ahead
                })
            
            if predicted_respiratory > self.respiratory_threshold:
                forecasted_events.append({
                    'type': 'respiratory_distress',
                    'probability': predicted_respiratory,
                    'time_horizon_hours': hours_ahead
                })
            
            # Risk factors
            risk_factors = []
            if agitation_trend.get('slope', 0) > 0.1:
                risk_factors.append('Increasing agitation trend')
            if delirium_trend.get('slope', 0) > 0.1:
                risk_factors.append('Rising delirium risk')
            if respiratory_trend.get('slope', 0) > 0.1:
                risk_factors.append('Deteriorating respiratory status')
            
            return {
                'risk_level': risk_level,
                'confidence': min(max_risk, 1.0),
                'forecasted_events': forecasted_events,
                'risk_factors': risk_factors,
                'predictions': {
                    'agitation': predicted_agitation,
                    'delirium': predicted_delirium,
                    'respiratory': predicted_respiratory
                },
                'forecast_horizon_hours': hours_ahead
            }
            
        except Exception as e:
            log.exception("Forecast risk failed: %s", e)
            return {
                'risk_level': 'UNKNOWN',
                'confidence': 0.0,
                'forecasted_events': [],
                'risk_factors': ['Analysis error']
            }
    
    def _analyze_trend(self, alerts: List[Dict], metric: str) -> Dict:
        """Analyze trend for a specific metric."""
        if not alerts:
            return {'mean': 0.0, 'slope': 0.0, 'volatility': 0.0}
        
        values = []
        timestamps = []
        
        for alert in alerts:
            value = alert.get(metric)
            if value is not None:
                values.append(value)
                timestamps.append(alert.get('ts', 0))
        
        if len(values) < 2:
            return {'mean': np.mean(values) if values else 0.0, 'slope': 0.0, 'volatility': 0.0}
        
        # Calculate slope (trend)
        timestamps = np.array(timestamps)
        values = np.array(values)
        
        # Normalize timestamps
        if timestamps[-1] != timestamps[0]:
            slope = np.polyfit(timestamps - timestamps[0], values, 1)[0]
        else:
            slope = 0.0
        
        # Calculate volatility (standard deviation)
        volatility = np.std(values)
        
        return {
            'mean': float(np.mean(values)),
            'slope': float(slope),
            'volatility': float(volatility),
            'current': float(values[-1]) if values else 0.0
        }
    
    def _extrapolate_trend(self, trend: Dict, hours_ahead: int) -> float:
        """Extrapolate trend to predict future value."""
        current = trend.get('current', 0.0)
        slope = trend.get('slope', 0.0)
        
        # Extrapolate (slope is per second, convert hours to seconds)
        predicted = current + (slope * hours_ahead * 3600)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, predicted))
    
    def analyze_trends(self, patient_id: str, days: int = 7) -> Dict:
        """Analyze trends over multiple days."""
        try:
            end_ts = time.time()
            start_ts = end_ts - (days * 24 * 3600)
            
            alerts = self.db.query_alerts(
                start_ts=start_ts,
                end_ts=end_ts
            )
            
            agitation_trend = self._analyze_trend(alerts, 'agitation_score')
            delirium_trend = self._analyze_trend(alerts, 'delirium_risk')
            respiratory_trend = self._analyze_trend(alerts, 'respiratory_distress')
            
            return {
                'agitation': agitation_trend,
                'delirium': delirium_trend,
                'respiratory': respiratory_trend,
                'period_days': days
            }
        except Exception as e:
            log.exception("Analyze trends failed: %s", e)
            return {}
    
    def identify_risk_factors(self, patient_id: str) -> List[str]:
        """Identify key risk factors for a patient."""
        risk_factors = []
        
        try:
            # Get recent alerts
            alerts = self.db.query_alerts(
                start_ts=time.time() - (24 * 3600),
                limit=100
            )
            
            if not alerts:
                return ['No recent data available']
            
            # Count high-risk alerts
            high_risk_count = sum(1 for a in alerts if a.get('alert_level') == 'HIGH_RISK')
            if high_risk_count > 5:
                risk_factors.append(f'Multiple high-risk alerts ({high_risk_count} in 24h)')
            
            # Check for increasing trends
            trends = self.analyze_trends(patient_id, days=1)
            if trends.get('agitation', {}).get('slope', 0) > 0.1:
                risk_factors.append('Rapidly increasing agitation')
            if trends.get('delirium', {}).get('slope', 0) > 0.1:
                risk_factors.append('Rising delirium risk')
            if trends.get('respiratory', {}).get('slope', 0) > 0.1:
                risk_factors.append('Deteriorating respiratory function')
            
            # Check for sustained high values
            avg_agitation = trends.get('agitation', {}).get('mean', 0.0)
            if avg_agitation > 0.7:
                risk_factors.append('Sustained high agitation levels')
            
            if not risk_factors:
                risk_factors.append('No significant risk factors identified')
            
        except Exception as e:
            log.exception("Identify risk factors failed: %s", e)
            risk_factors = ['Analysis error']
        
        return risk_factors
    
    def get_patient_risk(self, patient_id: str) -> Dict:
        """Get current risk assessment for a patient."""
        forecast = self.forecast_risk(patient_id, hours_ahead=6)
        
        return {
            'current_risk': forecast.get('risk_level', 'UNKNOWN'),
            'confidence': forecast.get('confidence', 0.0),
            'forecasted_events_count': len(forecast.get('forecasted_events', [])),
            'risk_factors_count': len(forecast.get('risk_factors', []))
        }

