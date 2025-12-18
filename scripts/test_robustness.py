# scripts/test_robustness.py
"""
Comprehensive robustness testing for posture detection, activity monitoring, and patient tracking.
"""
import sys
import os
import numpy as np
import yaml
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.posture import classify_posture_state, compute_bed_angle, compute_posture_symmetry, analyze_posture
from analytics.activity import classify_activity, compute_activity_confidence
from analytics.posture_smoother import PostureStateMachine
from pipeline.pose.keypoint_validator import create_keypoint_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("robustness_test")

def test_posture_robustness():
    """Test posture detection robustness."""
    print("\n" + "="*70)
    print("POSTURE DETECTION ROBUSTNESS TEST")
    print("="*70)
    
    issues = []
    recommendations = []
    
    # Test 1: Missing keypoints
    print("\n1. Testing missing keypoints...")
    kps_missing = [
        (0.5, 0.5, 0.9),  # Nose
        (0.0, 0.0, 0.0),  # Missing
        (0.0, 0.0, 0.0),  # Missing
        (0.0, 0.0, 0.0),  # Missing
        (0.0, 0.0, 0.0),  # Missing
        (0.4, 0.4, 0.8),  # Left shoulder
        (0.6, 0.4, 0.8),  # Right shoulder
        (0.0, 0.0, 0.0),  # Missing
        (0.0, 0.0, 0.0),  # Missing
        (0.0, 0.0, 0.0),  # Missing
        (0.0, 0.0, 0.0),  # Missing
        (0.4, 0.6, 0.7),  # Left hip
        (0.6, 0.6, 0.7),  # Right hip
    ]
    
    posture = classify_posture_state(kps_missing)
    if posture == "unknown":
        print("  ✅ Correctly returns 'unknown' for insufficient keypoints")
    else:
        issues.append("Posture classification should return 'unknown' for insufficient keypoints")
        print(f"  ⚠️  Returned '{posture}' instead of 'unknown'")
    
    # Test 2: Low confidence keypoints
    print("\n2. Testing low confidence keypoints...")
    kps_low_conf = [
        (0.5, 0.5, 0.2),  # Nose (low confidence)
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.4, 0.4, 0.2),  # Left shoulder (low confidence)
        (0.6, 0.4, 0.2),  # Right shoulder (low confidence)
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.4, 0.6, 0.2),  # Left hip (low confidence)
        (0.6, 0.6, 0.2),  # Right hip (low confidence)
    ]
    
    bed_angle = compute_bed_angle(kps_low_conf)
    if bed_angle.get("orientation") == "unknown":
        print("  ✅ Correctly returns 'unknown' for low confidence keypoints")
    else:
        issues.append("Bed angle computation should handle low confidence keypoints")
        print(f"  ⚠️  Returned '{bed_angle.get('orientation')}' instead of 'unknown'")
    
    # Test 3: Edge cases - extreme angles
    print("\n3. Testing extreme angles...")
    # Supine (horizontal)
    kps_supine = [
        (0.5, 0.2, 0.9),  # Nose
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.4, 0.4, 0.9),  # Left shoulder
        (0.6, 0.4, 0.9),  # Right shoulder
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.4, 0.6, 0.9),  # Left hip
        (0.6, 0.6, 0.9),  # Right hip
    ]
    
    posture_supine = classify_posture_state(kps_supine, use_strict_thresholds=True)
    bed_angle_supine = compute_bed_angle(kps_supine)
    print(f"  Supine test: orientation={bed_angle_supine.get('orientation')}, posture={posture_supine}")
    
    # Test 4: Temporal smoothing
    print("\n4. Testing temporal smoothing...")
    smoother = PostureStateMachine(transition_threshold=5, history_size=10)
    
    # Simulate noisy transitions
    states = ["supine", "supine", "side", "supine", "side", "side", "side", "side", "side", "side"]
    results = []
    for state in states:
        smoothed = smoother.update(state)
        results.append(smoothed)
    
    if results[-1] == "side":
        print("  ✅ Temporal smoothing correctly handles noisy transitions")
    else:
        issues.append("Temporal smoothing may not be robust enough for noisy data")
        print(f"  ⚠️  Final state: {results[-1]} (expected 'side')")
    
    # Test 5: Symmetry edge cases
    print("\n5. Testing symmetry computation...")
    # Perfectly symmetric
    kps_symmetric = [
        (0.5, 0.2, 0.9),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.4, 0.4, 0.9),  # Left shoulder
        (0.6, 0.4, 0.9),  # Right shoulder (symmetric)
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.4, 0.6, 0.9),  # Left hip
        (0.6, 0.6, 0.9),  # Right hip (symmetric)
    ]
    
    symmetry = compute_posture_symmetry(kps_symmetric)
    if symmetry.get("symmetry_index", 0) > 0.9:
        print("  ✅ Correctly identifies symmetric posture")
    else:
        issues.append("Symmetry computation may not be accurate for symmetric postures")
        print(f"  ⚠️  Symmetry index: {symmetry.get('symmetry_index')} (expected >0.9)")
    
    return issues, recommendations

def test_activity_robustness():
    """Test activity monitoring robustness."""
    print("\n" + "="*70)
    print("ACTIVITY MONITORING ROBUSTNESS TEST")
    print("="*70)
    
    issues = []
    recommendations = []
    
    # Test 1: Missing keypoints
    print("\n1. Testing missing keypoints...")
    kps_minimal = [
        (0.5, 0.5, 0.9),  # Nose
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.4, 0.4, 0.8),  # Left shoulder
        (0.6, 0.4, 0.8),  # Right shoulder
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.4, 0.6, 0.7),  # Left hip
        (0.6, 0.6, 0.7),  # Right hip
    ]
    
    activity = classify_activity(kps_minimal)
    if activity.get("activity") == "unknown":
        print("  ✅ Correctly returns 'unknown' for insufficient keypoints")
    else:
        issues.append("Activity classification should return 'unknown' for insufficient keypoints")
        print(f"  ⚠️  Returned '{activity.get('activity')}' instead of 'unknown'")
    
    # Test 2: Activity confidence
    print("\n2. Testing activity confidence...")
    confidence = compute_activity_confidence(kps_minimal)
    print(f"  Confidence: {confidence:.2f}")
    if confidence < 0.5:
        print("  ✅ Low confidence for minimal keypoints")
    else:
        issues.append("Activity confidence may be too high for minimal keypoints")
    
    # Test 3: Temporal activity (walking detection)
    print("\n3. Testing temporal activity (walking)...")
    # Simulate walking motion
    kps_history = []
    for i in range(5):
        kps_walking = [
            (0.5, 0.2, 0.9),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.4, 0.4, 0.9),
            (0.6, 0.4, 0.9),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.4, 0.6, 0.9),
            (0.6, 0.6, 0.9),
            (0.4, 0.7, 0.9),  # Left knee (varying)
            (0.6, 0.7, 0.9),  # Right knee (varying)
            (0.4, 0.9, 0.9),  # Left ankle
            (0.6, 0.9, 0.9),  # Right ankle
        ]
        # Vary knee positions to simulate walking
        kps_walking[13] = (0.4, 0.7 + i * 0.05, 0.9)
        kps_walking[14] = (0.6, 0.7 - i * 0.05, 0.9)
        kps_history.append(kps_walking)
    
    activity_walking = classify_activity(kps_history[-1], kps_history=kps_history)
    print(f"  Walking detection: {activity_walking.get('activity')} (confidence: {activity_walking.get('confidence'):.2f})")
    
    if activity_walking.get("activity") == "walking":
        print("  ✅ Correctly detects walking motion")
    else:
        recommendations.append({
            "priority": "MEDIUM",
            "issue": "Walking detection may need improvement",
            "impact": "May miss patient movement"
        })
        print(f"  ⚠️  Did not detect walking (detected: {activity_walking.get('activity')})")
    
    # Test 4: Edge cases - extreme aspect ratios
    print("\n4. Testing extreme aspect ratios...")
    # Very tall (standing)
    kps_tall = [
        (0.5, 0.1, 0.9),  # Nose (high)
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.4, 0.3, 0.9),
        (0.6, 0.3, 0.9),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.4, 0.5, 0.9),
        (0.6, 0.5, 0.9),
        (0.4, 0.7, 0.9),
        (0.6, 0.7, 0.9),
        (0.4, 0.95, 0.9),  # Ankles (low)
        (0.6, 0.95, 0.9),
    ]
    
    activity_tall = classify_activity(kps_tall)
    print(f"  Tall posture: {activity_tall.get('activity')} (confidence: {activity_tall.get('confidence'):.2f})")
    
    return issues, recommendations

def test_tracking_robustness():
    """Test patient tracking robustness."""
    print("\n" + "="*70)
    print("PATIENT TRACKING ROBUSTNESS TEST")
    print("="*70)
    
    issues = []
    recommendations = []
    
    # Test 1: Track ID consistency
    print("\n1. Testing track ID consistency...")
    print("  ⚠️  Track ID consistency depends on ByteTrack/StrongSORT")
    print("  ⚠️  Need to test with actual video to verify consistency")
    recommendations.append({
        "priority": "HIGH",
        "issue": "Track ID consistency testing needed",
        "impact": "May lose patient identity during occlusions"
    })
    
    # Test 2: Multiple people handling
    print("\n2. Testing multiple people handling...")
    print("  ⚠️  Current: Uses largest bbox or face recognition")
    print("  ⚠️  Risk: May switch to visitor if face recognition fails")
    issues.append("Patient selection may switch to wrong person")
    recommendations.append({
        "priority": "HIGH",
        "issue": "Improve multi-person patient selection",
        "impact": "Use track_id persistence + face recognition + size",
        "solution": "Maintain track_id history and prefer persistent tracks"
    })
    
    # Test 3: Occlusion handling
    print("\n3. Testing occlusion handling...")
    print("  ⚠️  Current threshold: 150 frames (10 seconds at 15 FPS)")
    print("  ⚠️  May be too short for long occlusions (e.g., nurse blocking view)")
    issues.append("Occlusion threshold may be too short")
    recommendations.append({
        "priority": "MEDIUM",
        "issue": "Increase occlusion threshold or use ReID for recovery",
        "impact": "Better handling of temporary occlusions",
        "solution": "Use appearance-based ReID to recover patient after occlusion"
    })
    
    # Test 4: Track ID validation
    print("\n4. Testing track ID validation...")
    print("  ✅ Track IDs are extracted from YOLO tracker")
    print("  ⚠️  No validation of track ID consistency across frames")
    recommendations.append({
        "priority": "MEDIUM",
        "issue": "Add track ID validation and consistency checks",
        "impact": "Detect and handle track ID switches"
    })
    
    # Test 5: Patient onboarding
    print("\n5. Testing patient onboarding...")
    print("  ⚠️  Patient onboarding logic not clearly visible in code")
    print("  ⚠️  Need explicit patient onboarding with track_id assignment")
    recommendations.append({
        "priority": "HIGH",
        "issue": "Implement explicit patient onboarding",
        "impact": "Ensure patient is properly tracked from start",
        "solution": "Onboard patient when first detected + verified, assign persistent track_id"
    })
    
    return issues, recommendations

def test_integration_robustness():
    """Test integration between posture, activity, and tracking."""
    print("\n" + "="*70)
    print("INTEGRATION ROBUSTNESS TEST")
    print("="*70)
    
    issues = []
    recommendations = []
    
    # Test 1: Keypoint validation
    print("\n1. Testing keypoint validation...")
    validator = create_keypoint_validator()
    
    # Test with invalid keypoints
    kps_invalid = [
        (np.nan, 0.5, 0.9),  # NaN
        (0.5, np.inf, 0.9),  # Inf
        (1.5, 0.5, 0.9),  # Out of range
        (-0.1, 0.5, 0.9),  # Out of range
        (0.5, 0.5, 0.1),  # Low confidence
    ] + [(0.5, 0.5, 0.9)] * 12  # Fill rest
    
    validated = validator.validate(kps_invalid)
    if validated is None or len(validated) < 5:
        print("  ✅ Keypoint validator correctly filters invalid keypoints")
    else:
        issues.append("Keypoint validator may not be strict enough")
        print(f"  ⚠️  Validated {len(validated)} keypoints (expected fewer)")
    
    # Test 2: Posture + Activity consistency
    print("\n2. Testing posture + activity consistency...")
    # Lying posture should match "lying" activity
    kps_lying = [
        (0.5, 0.3, 0.9),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.3, 0.4, 0.9),  # Left shoulder
        (0.7, 0.4, 0.9),  # Right shoulder (wide)
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.3, 0.6, 0.9),  # Left hip
        (0.7, 0.6, 0.9),  # Right hip (wide)
        (0.3, 0.7, 0.9),
        (0.7, 0.7, 0.9),
        (0.3, 0.9, 0.9),
        (0.7, 0.9, 0.9),
    ]
    
    posture_lying = classify_posture_state(kps_lying)
    activity_lying = classify_activity(kps_lying)
    
    print(f"  Posture: {posture_lying}, Activity: {activity_lying.get('activity')}")
    
    # Check consistency
    if (posture_lying in ["supine", "side", "left_lateral", "right_lateral"] and 
        activity_lying.get("activity") == "lying"):
        print("  ✅ Posture and activity are consistent")
    elif posture_lying == "unknown" or activity_lying.get("activity") == "unknown":
        print("  ⚠️  One or both returned 'unknown' (may be acceptable)")
    else:
        issues.append("Posture and activity classifications may be inconsistent")
        print(f"  ⚠️  Inconsistency: posture={posture_lying}, activity={activity_lying.get('activity')}")
    
    return issues, recommendations

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE ROBUSTNESS ANALYSIS")
    print("="*70)
    
    all_issues = []
    all_recommendations = []
    
    # Run all tests
    issues, recs = test_posture_robustness()
    all_issues.extend(issues)
    all_recommendations.extend(recs)
    
    issues, recs = test_activity_robustness()
    all_issues.extend(issues)
    all_recommendations.extend(recs)
    
    issues, recs = test_tracking_robustness()
    all_issues.extend(issues)
    all_recommendations.extend(recs)
    
    issues, recs = test_integration_robustness()
    all_issues.extend(issues)
    all_recommendations.extend(recs)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total Issues Found: {len(all_issues)}")
    print(f"Total Recommendations: {len(all_recommendations)}")
    
    if all_issues:
        print("\nIssues:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
    
    if all_recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(all_recommendations, 1):
            print(f"  {i}. [{rec.get('priority', 'MEDIUM')}] {rec.get('issue')}")
            if 'impact' in rec:
                print(f"      Impact: {rec['impact']}")
            if 'solution' in rec:
                print(f"      Solution: {rec['solution']}")
    
    return 0 if len(all_issues) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

