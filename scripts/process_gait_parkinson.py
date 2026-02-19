#!/usr/bin/env python3
"""
Process PhysioNet Gait in Parkinson's Disease Dataset

This script extracts gait features from the foot force sensor data and compares
them with Video2IMU-derived features to validate the cross-modal approach.

Dataset: https://physionet.org/content/gaitpdb/1.0.0/
- 93 healthy controls (Co)
- 63 Parkinson's patients (Pt)
- Vertical ground reaction force (VGRF) from 8 sensors per foot @ 100Hz

Usage:
 python scripts/process_gait_parkinson.py [--visualize] [--output OUTPUT_DIR]
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data directory
DATA_DIR = Path("clinical_validation_data/gait-parkinson")


def load_gait_file(filepath):
 """
 Load a single gait data file.

 Returns:
 np.array of shape (N, 19) with columns:
 [time, L1-L8 (8 left foot sensors), R1-R8 (8 right foot sensors),
 total_left, total_right]
 """
 try:
 data = np.loadtxt(filepath)
 return data
 except Exception as e:
 print(f"Error loading {filepath}: {e}")
 return None


def load_demographics():
 """Load subject demographics and clinical info."""
 demo_path = DATA_DIR / "demographics.txt"
 try:
 df = pd.read_csv(demo_path, sep='\t')
 return df
 except Exception as e:
 print(f"Error loading demographics: {e}")
 return None


def parse_filename(filename):
 """
 Parse gait filename to extract metadata.

 Example: GaPt03_01.txt
 - Study: Ga (Yogev), Ju (Hausdorff), Si (Frenkel-Toledo)
 - Group: Co (Control), Pt (Patient)
 - Subject: 03
 - Walk: 01 (normal), 10 (dual-task)
 """
 name = Path(filename).stem
 study = name[:2] # Ga, Ju, Si
 group = name[2:4] # Co, Pt
 subject = int(name[4:6])
 walk = int(name[7:9]) if len(name) > 7 else 1

 return {
 'study': study,
 'group': group, # Co=Control, Pt=Patient
 'subject': subject,
 'walk': walk,
 'is_patient': group == 'Pt',
 'is_dual_task': walk == 10,
 'subject_id': f"{study}{group}{subject:02d}"
 }


def extract_gait_features(data, sampling_rate=100):
 """
 Extract gait features from foot force sensor data.

 These features are analogous to what Video2IMU computes from video.

 Args:
 data: np.array (N, 19) - time + 16 sensors + 2 totals
 sampling_rate: Hz (default 100)

 Returns:
 dict of gait features
 """
 if data is None or len(data) < 100:
 return None

 time = data[:, 0]
 left_sensors = data[:, 1:9] # 8 left foot sensors
 right_sensors = data[:, 9:17] # 8 right foot sensors
 total_left = data[:, 17]
 total_right = data[:, 18]

 dt = 1.0 / sampling_rate

 # Total force
 total_force = total_left + total_right

 # --- Kinematic-like features (analogous to Video2IMU) ---

 # 1. Force velocity (rate of change) - analogous to acceleration
 force_velocity = np.diff(total_force) / dt

 # 2. Force acceleration (second derivative) - analogous to jerk
 force_accel = np.diff(force_velocity) / dt

 # 3. Left-right force difference (asymmetry) - analogous to symmetry index
 lr_diff = total_left - total_right
 asymmetry = np.abs(lr_diff) / (total_force + 1e-6)

 # --- Statistical features (as in Video2IMU paper) ---

 # Total force statistics
 features = {
 # Force magnitude stats
 'force_avg': np.mean(total_force),
 'force_med': np.median(total_force),
 'force_var': np.var(total_force),
 'force_min': np.min(total_force),
 'force_max': np.max(total_force),
 'force_lq': np.percentile(total_force, 25),
 'force_uq': np.percentile(total_force, 75),

 # Force velocity stats (analogous to acceleration)
 'velocity_avg': np.mean(np.abs(force_velocity)),
 'velocity_var': np.var(force_velocity),
 'velocity_max': np.max(np.abs(force_velocity)),

 # Force acceleration stats (analogous to jerk)
 'accel_avg': np.mean(np.abs(force_accel)),
 'accel_var': np.var(force_accel),
 'accel_max': np.max(np.abs(force_accel)),

 # Asymmetry stats (analogous to symmetry index)
 'asymmetry_avg': np.mean(asymmetry),
 'asymmetry_var': np.var(asymmetry),

 # Left foot stats
 'left_avg': np.mean(total_left),
 'left_var': np.var(total_left),

 # Right foot stats
 'right_avg': np.mean(total_right),
 'right_var': np.var(total_right),
 }

 # --- Gait-specific features ---

 # Detect steps using force threshold
 threshold = 0.1 * np.max(total_force)

 # Left foot steps
 left_stance = total_left > threshold
 left_step_changes = np.diff(left_stance.astype(int))
 left_heel_strikes = np.where(left_step_changes == 1)[0]

 # Right foot steps
 right_stance = total_right > threshold
 right_step_changes = np.diff(right_stance.astype(int))
 right_heel_strikes = np.where(right_step_changes == 1)[0]

 # Step timing
 if len(left_heel_strikes) > 1:
 left_step_times = np.diff(left_heel_strikes) * dt
 features['left_step_time_avg'] = np.mean(left_step_times)
 features['left_step_time_var'] = np.var(left_step_times)
 features['left_step_count'] = len(left_heel_strikes)
 else:
 features['left_step_time_avg'] = 0
 features['left_step_time_var'] = 0
 features['left_step_count'] = 0

 if len(right_heel_strikes) > 1:
 right_step_times = np.diff(right_heel_strikes) * dt
 features['right_step_time_avg'] = np.mean(right_step_times)
 features['right_step_time_var'] = np.var(right_step_times)
 features['right_step_count'] = len(right_heel_strikes)
 else:
 features['right_step_time_avg'] = 0
 features['right_step_time_var'] = 0
 features['right_step_count'] = 0

 # Cadence (steps per minute)
 duration = time[-1] - time[0]
 total_steps = features['left_step_count'] + features['right_step_count']
 features['cadence'] = (total_steps / duration) * 60 if duration > 0 else 0

 # Step time variability (coefficient of variation) - key PD indicator
 all_step_times = []
 if len(left_heel_strikes) > 1:
 all_step_times.extend(np.diff(left_heel_strikes) * dt)
 if len(right_heel_strikes) > 1:
 all_step_times.extend(np.diff(right_heel_strikes) * dt)

 if len(all_step_times) > 2:
 step_cv = np.std(all_step_times) / np.mean(all_step_times)
 features['step_time_cv'] = step_cv
 else:
 features['step_time_cv'] = 0

 # Double support time (both feet on ground)
 double_support = left_stance & right_stance
 features['double_support_ratio'] = np.mean(double_support)

 # Duration
 features['duration'] = duration

 return features


def process_all_subjects(data_dir=DATA_DIR):
 """Process all gait files and extract features."""
 results = []

 # Find all .txt files (excluding metadata files)
 gait_files = [f for f in data_dir.glob("*.txt")
 if not f.name.startswith(('format', 'demographics', 'SHA'))]

 print(f"Found {len(gait_files)} gait data files")

 for filepath in sorted(gait_files):
 # Parse filename
 meta = parse_filename(filepath.name)

 # Load data
 data = load_gait_file(filepath)
 if data is None:
 continue

 # Extract features
 features = extract_gait_features(data)
 if features is None:
 continue

 # Combine metadata and features
 result = {**meta, **features}
 results.append(result)

 return pd.DataFrame(results)


def compare_control_vs_patient(df):
 """Compare features between control and patient groups."""
 print("\n" + "="*70)
 print("CONTROL vs PARKINSON'S COMPARISON")
 print("="*70)

 controls = df[df['is_patient'] == False]
 patients = df[df['is_patient'] == True]

 print(f"\nSample sizes: Controls={len(controls)}, Patients={len(patients)}")

 # Key discriminative features
 key_features = [
 'step_time_cv', # Step time variability (key PD indicator)
 'asymmetry_avg', # Gait asymmetry
 'velocity_var', # Force velocity variance
 'accel_var', # Force acceleration variance (like jerk)
 'double_support_ratio', # Time with both feet down
 'cadence', # Steps per minute
 ]

 print("\nFeature Comparison (mean ± std):")
 print("-" * 70)
 print(f"{'Feature':<25} {'Controls':>20} {'Patients':>20} {'p-value':>10}")
 print("-" * 70)

 from scipy import stats

 for feat in key_features:
 if feat not in df.columns:
 continue

 ctrl_vals = controls[feat].dropna()
 pt_vals = patients[feat].dropna()

 if len(ctrl_vals) < 2 or len(pt_vals) < 2:
 continue

 ctrl_mean = ctrl_vals.mean()
 ctrl_std = ctrl_vals.std()
 pt_mean = pt_vals.mean()
 pt_std = pt_vals.std()

 # t-test
 t_stat, p_val = stats.ttest_ind(ctrl_vals, pt_vals)

 sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

 print(f"{feat:<25} {ctrl_mean:>8.3f} ± {ctrl_std:<8.3f} "
 f"{pt_mean:>8.3f} ± {pt_std:<8.3f} {p_val:>8.4f} {sig}")

 print("-" * 70)
 print("Significance: * p<0.05, ** p<0.01, *** p<0.001")

 return controls, patients


def map_to_video2imu_features(gait_features):
 """
 Map gait sensor features to Video2IMU feature space.

 This shows the correspondence between real sensor features
 and what Video2IMU computes from video.
 """
 mapping = {
 # Gait feature -> Video2IMU equivalent
 'velocity_avg': 'linear_acceleration (ax, ay, az)',
 'velocity_var': 'acceleration_variance',
 'accel_avg': 'jerk_index',
 'accel_var': 'motion_variability',
 'asymmetry_avg': 'symmetry_index (inverted)',
 'step_time_cv': 'motion_entropy',
 'force_var': 'motion_energy',
 }

 print("\n" + "="*70)
 print("GAIT FEATURES -> VIDEO2IMU MAPPING")
 print("="*70)
 print("\nCorrespondence between real sensor data and Video2IMU:")
 print("-" * 70)
 print(f"{'Gait Sensor Feature':<25} {'Video2IMU Equivalent':<40}")
 print("-" * 70)

 for gait_feat, v2i_feat in mapping.items():
 print(f"{gait_feat:<25} {v2i_feat:<40}")

 print("-" * 70)
 print("\nThese mappings enable cross-modal validation:")
 print("- Train classifier on gait sensor features")
 print("- Apply same classifier to Video2IMU features from video")
 print("- Compare classification accuracy")


def train_simple_classifier(df):
 """Train a simple classifier to distinguish Control vs Patient."""
 from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.preprocessing import StandardScaler

 print("\n" + "="*70)
 print("CLASSIFIER TRAINING (Control vs Parkinson's)")
 print("="*70)

 # Feature columns (exclude metadata)
 feature_cols = [c for c in df.columns if c not in
 ['study', 'group', 'subject', 'walk', 'is_patient',
 'is_dual_task', 'subject_id', 'duration']]

 # Prepare data
 X = df[feature_cols].fillna(0).values
 y = df['is_patient'].astype(int).values
 groups = df['subject_id'].values # For LOSO validation

 # Scale features
 scaler = StandardScaler()
 X_scaled = scaler.fit_transform(X)

 # Train classifier with Leave-One-Subject-Out cross-validation
 clf = RandomForestClassifier(n_estimators=100, random_state=42)
 logo = LeaveOneGroupOut()

 scores = cross_val_score(clf, X_scaled, y, cv=logo, groups=groups)

 print(f"\nFeatures used: {len(feature_cols)}")
 print(f"Samples: {len(X)}")
 print(f"\nLeave-One-Subject-Out Cross-Validation:")
 print(f" Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
 print(f" Min: {scores.min():.3f}, Max: {scores.max():.3f}")

 # Train final model on all data for feature importance
 clf.fit(X_scaled, y)

 # Feature importance
 importance = clf.feature_importances_
 importance_df = pd.DataFrame({
 'feature': feature_cols,
 'importance': importance
 }).sort_values('importance', ascending=False)

 print("\nTop 10 Most Important Features:")
 print("-" * 40)
 for i, row in importance_df.head(10).iterrows():
 print(f" {row['feature']:<25} {row['importance']:.4f}")

 return clf, scaler, feature_cols


def visualize_distributions(df, output_dir=None):
 """Visualize feature distributions for controls vs patients."""
 try:
 import matplotlib.pyplot as plt
 except ImportError:
 print("Matplotlib not available, skipping visualization")
 return

 fig, axes = plt.subplots(2, 3, figsize=(15, 10))
 fig.suptitle('Gait Features: Control vs Parkinson\'s Disease', fontsize=14)

 key_features = [
 ('step_time_cv', 'Step Time Variability (CV)'),
 ('asymmetry_avg', 'Gait Asymmetry'),
 ('velocity_var', 'Force Velocity Variance'),
 ('accel_var', 'Force Acceleration Variance'),
 ('double_support_ratio', 'Double Support Ratio'),
 ('cadence', 'Cadence (steps/min)'),
 ]

 controls = df[df['is_patient'] == False]
 patients = df[df['is_patient'] == True]

 for ax, (feat, title) in zip(axes.flat, key_features):
 ctrl_vals = controls[feat].dropna()
 pt_vals = patients[feat].dropna()

 ax.hist(ctrl_vals, bins=20, alpha=0.5, label='Control', color='blue')
 ax.hist(pt_vals, bins=20, alpha=0.5, label='Parkinson\'s', color='red')
 ax.set_xlabel(title)
 ax.set_ylabel('Count')
 ax.legend()
 ax.set_title(f'{title}\n(Control: {ctrl_vals.mean():.3f}, PD: {pt_vals.mean():.3f})')

 plt.tight_layout()

 if output_dir:
 output_path = Path(output_dir) / 'gait_feature_distributions.png'
 plt.savefig(output_path, dpi=150)
 print(f"\nSaved visualization to {output_path}")

 plt.show()


def save_features(df, output_dir):
 """Save extracted features to CSV."""
 output_path = Path(output_dir) / 'gait_parkinson_features.csv'
 df.to_csv(output_path, index=False)
 print(f"\nSaved features to {output_path}")

 # Also save summary statistics
 summary_path = Path(output_dir) / 'gait_parkinson_summary.txt'
 with open(summary_path, 'w') as f:
 f.write("Gait in Parkinson's Disease - Feature Summary\n")
 f.write("=" * 50 + "\n\n")
 f.write(f"Total recordings: {len(df)}\n")
 f.write(f"Control subjects: {len(df[df['is_patient']==False])}\n")
 f.write(f"Patient subjects: {len(df[df['is_patient']==True])}\n\n")
 f.write("Feature Statistics:\n")
 f.write(df.describe().to_string())

 print(f"Saved summary to {summary_path}")


def main():
 parser = argparse.ArgumentParser(description="Process Gait-Parkinson dataset")
 parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
 parser.add_argument("--output", default="validation_results", help="Output directory")
 args = parser.parse_args()

 print("="*70)
 print("GAIT IN PARKINSON'S DISEASE - FEATURE EXTRACTION")
 print("PhysioNet Dataset Analysis for Video2IMU Validation")
 print("="*70)

 # Check data directory
 if not DATA_DIR.exists():
 print(f"\nError: Data directory not found: {DATA_DIR}")
 print("Please download the dataset from:")
 print(" https://physionet.org/content/gaitpdb/1.0.0/")
 return 1

 # Process all subjects
 print("\nProcessing gait data files...")
 df = process_all_subjects()

 if len(df) == 0:
 print("No data processed!")
 return 1

 print(f"\nProcessed {len(df)} recordings successfully")

 # Compare control vs patient
 compare_control_vs_patient(df)

 # Show feature mapping to Video2IMU
 map_to_video2imu_features(df)

 # Train classifier
 clf, scaler, feature_cols = train_simple_classifier(df)

 # Save results
 os.makedirs(args.output, exist_ok=True)
 save_features(df, args.output)

 # Visualize if requested
 if args.visualize:
 visualize_distributions(df, args.output)

 print("\n" + "="*70)
 print("NEXT STEPS FOR VIDEO2IMU VALIDATION")
 print("="*70)
 print("""
1. Collect videos of walking (control and PD patients if available)
2. Extract Video2IMU features from the videos
3. Map Video2IMU features to gait sensor feature space
4. Compare classification accuracy:
 - Real sensors: ~{:.1f}% (from this analysis)
 - Video2IMU: Should be similar if transformation is valid
5. If accuracy differs significantly, calibrate Video2IMU parameters
""".format(clf.score(scaler.transform(df[feature_cols].fillna(0).values),
 df['is_patient'].astype(int).values) * 100))

 return 0


if __name__ == "__main__":
 sys.exit(main())
