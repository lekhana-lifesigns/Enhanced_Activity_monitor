#!/usr/bin/env python3
"""
Test script for ICU Patient Monitoring System
Tests components without requiring camera/MQTT
"""

import sys
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("test")

def test_feature_encoder():
    """Test ICU Feature Encoder"""
    log.info("Testing ICU Feature Encoder...")
    try:
        from pipeline.pose.feature_extractor import ICUFeatureEncoder
        
        encoder = ICUFeatureEncoder(window_size=30, fps=15.0)
        
        # Create mock keypoints (17 keypoints, normalized 0-1)
        kps = [
            (0.5, 0.2, 0.9),  # nose
            (0.45, 0.25, 0.8),  # left eye
            (0.55, 0.25, 0.8),  # right eye
            (0.4, 0.3, 0.7),   # left ear
            (0.6, 0.3, 0.7),   # right ear
            (0.4, 0.4, 0.9),   # left shoulder
            (0.6, 0.4, 0.9),   # right shoulder
            (0.35, 0.5, 0.8),  # left elbow
            (0.65, 0.5, 0.8),  # right elbow
            (0.3, 0.6, 0.7),   # left wrist
            (0.7, 0.6, 0.7),   # right wrist
            (0.45, 0.65, 0.9), # left hip
            (0.55, 0.65, 0.9), # right hip
            (0.4, 0.75, 0.8),  # left knee
            (0.6, 0.75, 0.8),  # right knee
            (0.35, 0.9, 0.7),  # left ankle
            (0.65, 0.9, 0.7),  # right ankle
        ]
        
        prev_kps = [(k[0] + 0.01, k[1] + 0.01, k[2]) for k in kps]
        prev_prev_kps = [(k[0] + 0.02, k[1] + 0.02, k[2]) for k in kps]
        
        # Extract features
        features = encoder.extract_feature_vector(kps, prev_kps, prev_prev_kps)
        
        if features is not None and len(features) == 9:
            log.info("✅ Feature Encoder: PASSED")
            log.info(f"   Features shape: {features.shape}")
            log.info(f"   Feature values: {features}")
            return True
        else:
            log.error("❌ Feature Encoder: FAILED - Wrong feature shape")
            return False
            
    except Exception as e:
        log.exception("❌ Feature Encoder: FAILED - %s", e)
        return False


def test_decision_engine():
    """Test Enhanced Decision Engine"""
    log.info("Testing Decision Engine...")
    try:
        from pipeline.pose.decision_engine import apply_rules
        
        # Mock keypoints
        kps = [
            (0.5, 0.2, 0.9),  # nose
            (0.45, 0.25, 0.8),
            (0.55, 0.25, 0.8),
            (0.4, 0.3, 0.7),
            (0.6, 0.3, 0.7),
            (0.4, 0.4, 0.9),  # left shoulder
            (0.6, 0.4, 0.9),  # right shoulder
            (0.35, 0.5, 0.8),
            (0.65, 0.5, 0.8),
            (0.3, 0.6, 0.7),  # left wrist
            (0.7, 0.6, 0.7),  # right wrist
            (0.45, 0.65, 0.9), # left hip
            (0.55, 0.65, 0.9), # right hip
            (0.4, 0.75, 0.8),
            (0.6, 0.75, 0.8),
            (0.35, 0.9, 0.7),
            (0.65, 0.9, 0.7),
        ]
        
        # Mock features
        features = np.array([0.5, 0.3, 0.2, 0.1, 0.4, 0.1, 0.6, 0.8, 0.3])
        
        # Mock ML prediction
        label = "agitation"
        probs = [0.7, 0.2, 0.05, 0.03, 0.01, 0.01]
        
        # Test decision engine
        decision = apply_rules(label, probs, kps, features)
        
        required_keys = [
            "agitation_score", "delirium_risk", "respiratory_distress",
            "lhs_motor", "rhs_motor", "hand_proximity_risk",
            "alert", "clinical_confidence"
        ]
        
        if all(key in decision for key in required_keys):
            log.info("✅ Decision Engine: PASSED")
            log.info(f"   Agitation Score: {decision['agitation_score']:.2f}")
            log.info(f"   Alert Level: {decision['alert']}")
            return True
        else:
            log.error("❌ Decision Engine: FAILED - Missing keys")
            return False
            
    except Exception as e:
        log.exception("❌ Decision Engine: FAILED - %s", e)
        return False


def test_analytics():
    """Test Analytics Modules"""
    log.info("Testing Analytics Modules...")
    results = []
    
    # Test Activity Classification
    try:
        from analytics.activity import classify_activity
        
        kps = [
            (0.5, 0.2, 0.9), (0.45, 0.25, 0.8), (0.55, 0.25, 0.8),
            (0.4, 0.3, 0.7), (0.6, 0.3, 0.7),
            (0.4, 0.4, 0.9), (0.6, 0.4, 0.9),
            (0.35, 0.5, 0.8), (0.65, 0.5, 0.8),
            (0.3, 0.6, 0.7), (0.7, 0.6, 0.7),
            (0.45, 0.65, 0.9), (0.55, 0.65, 0.9),
            (0.4, 0.75, 0.8), (0.6, 0.75, 0.8),
            (0.35, 0.9, 0.7), (0.65, 0.9, 0.7),
        ]
        
        activity = classify_activity(kps)
        if "activity" in activity and "confidence" in activity:
            log.info("✅ Activity Classification: PASSED")
            log.info(f"   Activity: {activity['activity']}")
            results.append(True)
        else:
            results.append(False)
    except Exception as e:
        log.exception("❌ Activity Classification: FAILED - %s", e)
        results.append(False)
    
    # Test Posture Analysis
    try:
        from analytics.posture import analyze_posture
        
        posture = analyze_posture(kps)
        if "spine_curvature" in posture and "symmetry" in posture:
            log.info("✅ Posture Analysis: PASSED")
            results.append(True)
        else:
            results.append(False)
    except Exception as e:
        log.exception("❌ Posture Analysis: FAILED - %s", e)
        results.append(False)
    
    # Test Vitals
    try:
        from analytics.vitals import estimate_breath_rate
        
        kps_history = [kps] * 30  # Mock history
        vitals = estimate_breath_rate(kps_history, fps=15.0)
        if "breath_rate" in vitals:
            log.info("✅ Vitals Estimation: PASSED")
            results.append(True)
        else:
            results.append(False)
    except Exception as e:
        log.exception("❌ Vitals Estimation: FAILED - %s", e)
        results.append(False)
    
    return all(results)


def test_temporal_model():
    """Test Temporal Model"""
    log.info("Testing Temporal Model...")
    try:
        from pipeline.pose.temporal_model import TemporalModel
        
        model = TemporalModel(window_size=48, labels=[
            "calm", "agitation", "restlessness", "delirium", "convulsion", "pain_response"
        ])
        
        # Mock feature window
        feat_window = np.random.rand(48, 9).astype(np.float32)
        
        # Test prediction (will use fallback since no model loaded)
        label, conf, probs = model.predict(feat_window)
        
        if label in model.labels and len(probs) == len(model.labels):
            log.info("✅ Temporal Model: PASSED")
            log.info(f"   Label: {label}, Confidence: {conf:.2f}")
            return True
        else:
            log.error("❌ Temporal Model: FAILED")
            return False
            
    except Exception as e:
        log.exception("❌ Temporal Model: FAILED - %s", e)
        return False


def main():
    """Run all tests"""
    log.info("=" * 60)
    log.info("ICU Patient Monitoring System - Component Tests")
    log.info("=" * 60)
    
    tests = [
        ("ICU Feature Encoder", test_feature_encoder),
        ("Decision Engine", test_decision_engine),
        ("Analytics Modules", test_analytics),
        ("Temporal Model", test_temporal_model),
    ]
    
    results = []
    for name, test_func in tests:
        log.info("")
        result = test_func()
        results.append((name, result))
    
    log.info("")
    log.info("=" * 60)
    log.info("Test Summary")
    log.info("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        log.info(f"{name}: {status}")
    
    log.info("")
    log.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        log.info("🎉 All tests passed! System ready for Phase 2.")
        return 0
    else:
        log.warning("⚠️  Some tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

