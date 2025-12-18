# scripts/analyze_accuracy.py
"""
Analyze posture detection and patient identification (ReID) accuracy.
Provides recommendations for improvement.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import logging
import numpy as np
from analytics.posture import classify_posture_state, analyze_posture, compute_bed_angle, compute_posture_symmetry
from pipeline.pose.inference_pipeline import InferencePipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("accuracy_analysis")


def analyze_posture_accuracy():
    """Analyze posture detection accuracy and identify issues."""
    print("\n" + "="*70)
    print("POSTURE DETECTION ACCURACY ANALYSIS")
    print("="*70)
    
    issues = []
    recommendations = []
    
    # Read posture classification code
    print("\n1. POSTURE CLASSIFICATION LOGIC ANALYSIS")
    print("-" * 70)
    
    # Check thresholds
    print("\nCurrent Thresholds:")
    print("  - Bed angle < 25° + vertical_extent < 0.4 → supine")
    print("  - Bed angle > 75° + vertical_extent > 0.3 → upright/sitting")
    print("  - Bed angle 25-75° OR vertical_extent < 0.35 → side")
    print("  - Symmetry index < 0.85 → lateral (left/right)")
    print("  - MIN_CONFIDENCE = 0.3")
    
    # Identify potential issues
    issues.append({
        "issue": "Fixed thresholds may not adapt to different camera angles",
        "impact": "High",
        "description": "Camera height/angle affects vertical extent calculations"
    })
    
    issues.append({
        "issue": "No temporal consistency check",
        "impact": "Medium",
        "description": "Posture can flip between frames without smoothing"
    })
    
    issues.append({
        "issue": "Prone detection missing",
        "impact": "Medium",
        "description": "Cannot distinguish supine vs prone (both have low vertical extent)"
    })
    
    issues.append({
        "issue": "Keypoint confidence threshold too low (0.3)",
        "impact": "Medium",
        "description": "May use unreliable keypoints for classification"
    })
    
    recommendations.append({
        "area": "Posture Classification",
        "recommendation": "Add adaptive thresholds based on camera calibration",
        "priority": "High",
        "implementation": "Use camera intrinsics to normalize measurements"
    })
    
    recommendations.append({
        "area": "Posture Classification",
        "recommendation": "Implement temporal smoothing with hysteresis",
        "priority": "High",
        "implementation": "Use state machine: require N consecutive frames before transition"
    })
    
    recommendations.append({
        "area": "Posture Classification",
        "recommendation": "Add prone detection using face visibility",
        "priority": "Medium",
        "implementation": "Check if nose/face keypoints are visible (supine) vs hidden (prone)"
    })
    
    recommendations.append({
        "area": "Posture Classification",
        "recommendation": "Increase keypoint confidence threshold to 0.5",
        "priority": "Medium",
        "implementation": "Only use high-confidence keypoints for classification"
    })
    
    recommendations.append({
        "area": "Posture Classification",
        "recommendation": "Add multi-view fusion",
        "priority": "Low",
        "implementation": "If multiple cameras available, fuse posture estimates"
    })
    
    return issues, recommendations


def analyze_reid_accuracy():
    """Analyze ReID-based patient identification accuracy."""
    print("\n" + "="*70)
    print("PATIENT IDENTIFICATION (ReID) ACCURACY ANALYSIS")
    print("="*70)
    
    issues = []
    recommendations = []
    
    print("\n1. CURRENT ReID IMPLEMENTATION")
    print("-" * 70)
    print("  - Uses ByteTrack/StrongSORT for tracking")
    print("  - Patient selection: Largest bounding box area")
    print("  - Track ID persistence: 30 frames missing threshold")
    print("  - No visual appearance features (true ReID)")
    
    issues.append({
        "issue": "No visual ReID features (appearance-based)",
        "impact": "High",
        "description": "Current implementation uses only bounding box tracking, not true ReID"
    })
    
    issues.append({
        "issue": "Patient selection based only on size",
        "impact": "High",
        "description": "If visitor is larger/closer, system may switch to wrong person"
    })
    
    issues.append({
        "issue": "No face recognition for patient verification",
        "impact": "High",
        "description": "Cannot verify patient identity using facial features"
    })
    
    issues.append({
        "issue": "Track ID lost on occlusion",
        "impact": "Medium",
        "description": "30-frame threshold may be too short for long occlusions"
    })
    
    recommendations.append({
        "area": "Patient Identification",
        "recommendation": "Implement DeepFace or face recognition for patient verification",
        "priority": "High",
        "implementation": "Add face detection → face recognition → patient ID matching"
    })
    
    recommendations.append({
        "area": "Patient Identification",
        "recommendation": "Add visual ReID features (appearance embedding)",
        "priority": "High",
        "implementation": "Use ResNet/OSNet for person re-identification features"
    })
    
    recommendations.append({
        "area": "Patient Identification",
        "recommendation": "Multi-modal patient selection",
        "priority": "Medium",
        "implementation": "Combine: size + position + appearance + face recognition"
    })
    
    recommendations.append({
        "area": "Patient Identification",
        "recommendation": "Increase missing threshold for known patients",
        "priority": "Medium",
        "implementation": "Use 60-90 frames for patients with confirmed identity"
    })
    
    recommendations.append({
        "area": "Patient Identification",
        "recommendation": "Add patient enrollment system",
        "priority": "Low",
        "implementation": "Capture reference face/features when patient admitted"
    })
    
    return issues, recommendations


def print_analysis_report(posture_issues, posture_recs, reid_issues, reid_recs):
    """Print comprehensive analysis report."""
    print("\n" + "="*70)
    print("ACCURACY IMPROVEMENT RECOMMENDATIONS")
    print("="*70)
    
    # Posture issues
    print("\n📊 POSTURE DETECTION ISSUES:")
    print("-" * 70)
    for i, issue in enumerate(posture_issues, 1):
        print(f"\n{i}. [{issue['impact']}] {issue['issue']}")
        print(f"   {issue['description']}")
    
    # ReID issues
    print("\n\n🔍 PATIENT IDENTIFICATION ISSUES:")
    print("-" * 70)
    for i, issue in enumerate(reid_issues, 1):
        print(f"\n{i}. [{issue['impact']}] {issue['issue']}")
        print(f"   {issue['description']}")
    
    # Recommendations
    print("\n\n✅ RECOMMENDED IMPROVEMENTS:")
    print("-" * 70)
    
    # High priority
    print("\n🔴 HIGH PRIORITY:")
    all_recs = posture_recs + reid_recs
    high_priority = [r for r in all_recs if r['priority'] == 'High']
    for i, rec in enumerate(high_priority, 1):
        print(f"\n{i}. [{rec['area']}] {rec['recommendation']}")
        print(f"   Implementation: {rec['implementation']}")
    
    # Medium priority
    print("\n🟡 MEDIUM PRIORITY:")
    medium_priority = [r for r in all_recs if r['priority'] == 'Medium']
    for i, rec in enumerate(medium_priority, 1):
        print(f"\n{i}. [{rec['area']}] {rec['recommendation']}")
        print(f"   Implementation: {rec['implementation']}")
    
    # Low priority
    print("\n🟢 LOW PRIORITY:")
    low_priority = [r for r in all_recs if r['priority'] == 'Low']
    for i, rec in enumerate(low_priority, 1):
        print(f"\n{i}. [{rec['area']}] {rec['recommendation']}")
        print(f"   Implementation: {rec['implementation']}")


def main():
    print("\n" + "="*70)
    print("ENHANCED ACTIVITY MONITOR - ACCURACY ANALYSIS")
    print("="*70)
    
    # Analyze posture
    posture_issues, posture_recs = analyze_posture_accuracy()
    
    # Analyze ReID
    reid_issues, reid_recs = analyze_reid_accuracy()
    
    # Print report
    print_analysis_report(posture_issues, posture_recs, reid_issues, reid_recs)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review high-priority recommendations")
    print("2. Implement DeepFace for patient verification")
    print("3. Add temporal smoothing for posture")
    print("4. Test with real patient data")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

