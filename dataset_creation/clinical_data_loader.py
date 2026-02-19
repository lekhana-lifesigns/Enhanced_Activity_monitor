# dataset_creation/clinical_data_loader.py
"""
Clinical Data Loader for Enhanced Activity Monitor
Loads clinical datasets from local training_data/ directory (primary) or
Hugging Face via DuckDB (secondary). Supports fall detection, posture
analysis, and activity recognition training.
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
from glob import glob

log = logging.getLogger("clinical_data_loader")

# 12 clinical activity classes (matches verified NPZ schema 1.0.0)
CLINICAL_CLASSES = [
    'lying_still', 'lying_restless', 'turning_in_bed',
    'sitting_stable', 'sitting_unstable', 'bed_exit',
    'standing', 'walking', 'agitated', 'convulsive',
    'fallen', 'occluded',
]

# Mapping from manifest labels to canonical class names
LABEL_ALIASES = {
    'hand_to_face': 'agitated',
    'hand_to_mask': 'agitated',
    'hand_to_iv': 'agitated',
    'hand_to_iv_line': 'agitated',
    'hand_to_chest_tube': 'agitated',
    'coughing': 'lying_restless',
    'laboured_breathing': 'lying_restless',
}


class ClinicalDataLoader:
    """
    Loads clinical activity datasets.
    Primary source: local training_data/ directory (NPZ + verified manifests).
    Secondary source: Hugging Face via DuckDB (requires network + httpfs).
    """

    def __init__(self, cache_dir: str = "clinical_cache",
                 local_data_dir: str = "training_data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.local_data_dir = Path(local_data_dir)

        # DuckDB connection (lazy - only created when needed for remote loading)
        self._conn = None

    def _get_conn(self):
        """Lazy-initialize DuckDB connection with httpfs (only for remote loading)."""
        if self._conn is None:
            import duckdb
            self._conn = duckdb.connect()
            self._conn.execute("INSTALL httpfs; LOAD httpfs;")
        return self._conn

    # ── Local data loading (primary) ──────────────────────────────────

    def load_local_npz(self, npz_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load pre-processed NPZ files from training_data/verified_npz/.

        Returns:
            DataFrame with columns: keypoints, activity, label_id, features,
            dataset_source
        """
        if npz_path:
            npz_files = [Path(npz_path)]
        else:
            npz_dir = self.local_data_dir / "verified_npz"
            npz_files = sorted(npz_dir.glob("*.npz"))

        if not npz_files:
            log.warning("No NPZ files found in %s", self.local_data_dir / "verified_npz")
            return pd.DataFrame()

        all_rows = []
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                keypoints = data['keypoints']          # (N, 17, 3)
                labels = data['labels']                 # (N,)
                class_names = list(data['class_names']) # (C,)
                features = data.get('features')         # (N, F) or None

                for i in range(len(labels)):
                    label_id = int(labels[i])
                    activity = class_names[label_id] if label_id < len(class_names) else 'unknown'
                    row = {
                        'keypoints': keypoints[i],
                        'activity': activity,
                        'label_id': label_id,
                        'dataset_source': str(npz_file),
                    }
                    if features is not None:
                        row['features'] = features[i]
                    all_rows.append(row)

                log.info("Loaded %d samples from %s", len(labels), npz_file.name)
            except Exception as e:
                log.warning("Failed to load NPZ %s: %s", npz_file, e)

        return pd.DataFrame(all_rows)

    def load_local_manifests(self, data_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Load verified frame manifests from training_data/verified/<activity>/manifest.json.

        Returns:
            DataFrame with columns: keypoints (None - frame-based), activity,
            label_id, bbox, quality_score, frame_file, dataset_source
        """
        verified_dir = Path(data_dir) if data_dir else self.local_data_dir / "verified"
        if not verified_dir.exists():
            log.warning("Verified data directory not found: %s", verified_dir)
            return pd.DataFrame()

        all_rows = []
        for manifest_path in sorted(verified_dir.glob("*/manifest.json")):
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)

                raw_label = manifest.get('label', 'unknown')
                activity = LABEL_ALIASES.get(raw_label, raw_label)
                frames = manifest.get('frames', [])

                for frame in frames:
                    if frame.get('status') != 'accepted':
                        continue
                    all_rows.append({
                        'activity': activity,
                        'label_id': frame.get('label_id'),
                        'bbox': frame.get('bbox'),
                        'quality_score': frame.get('quality_score', 0.0),
                        'keypoint_avg_conf': frame.get('keypoint_avg_conf', 0.0),
                        'num_keypoints_valid': frame.get('num_keypoints_valid', 0),
                        'frame_file': str(manifest_path.parent / "frames" / frame['frame_file']),
                        'dataset_source': str(manifest_path),
                    })

                log.info("Loaded %d accepted frames from %s (label=%s)",
                         len([r for r in all_rows if r['dataset_source'] == str(manifest_path)]),
                         manifest_path.parent.name, activity)
            except Exception as e:
                log.warning("Failed to load manifest %s: %s", manifest_path, e)

        return pd.DataFrame(all_rows)

    def load_local_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load all available local training data (NPZ + manifests).

        Args:
            limit: Maximum number of samples

        Returns:
            Combined DataFrame
        """
        frames = []

        npz_df = self.load_local_npz()
        if not npz_df.empty:
            frames.append(npz_df)

        manifest_df = self.load_local_manifests()
        if not manifest_df.empty:
            frames.append(manifest_df)

        if not frames:
            raise ValueError("No local training data found in %s" % self.local_data_dir)

        combined = pd.concat(frames, ignore_index=True)
        if limit and len(combined) > limit:
            combined = combined.sample(n=limit, random_state=42).reset_index(drop=True)

        log.info("Total local samples loaded: %d", len(combined))
        return combined

    # ── Activity-filtered loading ─────────────────────────────────────

    def load_fall_detection_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load fall detection data (local first, then remote)."""
        df = self.load_local_data(limit=limit)
        fall_df = df[df['activity'].isin(['fallen', 'bed_exit', 'convulsive'])]
        if not fall_df.empty:
            return fall_df
        raise ValueError("No fall detection samples found in local data")

    def load_activity_data(self, activities: Optional[List[str]] = None,
                           limit: Optional[int] = None) -> pd.DataFrame:
        """Load activity recognition data with optional activity filtering."""
        df = self.load_local_data(limit=limit)
        if activities:
            df = df[df['activity'].isin(activities)]
        return df

    def convert_to_training_format(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Convert clinical dataset to training format compatible with the activity monitor.

        Args:
            df: Clinical dataset DataFrame

        Returns:
            Dictionary with 'keypoints' (N,17,3), 'activities' (N,), 'metadata' (list)
        """
        training_data = {
            'keypoints': [],
            'activities': [],
            'metadata': []
        }

        for _, row in df.iterrows():
            # Extract keypoints
            kps = None
            if 'keypoints' in row and row['keypoints'] is not None:
                kps = self._parse_keypoints(row['keypoints'])
            else:
                # Frame-only row (from manifests) — skip if no keypoints
                continue

            training_data['keypoints'].append(kps)

            # Extract activity labels
            if 'activity' in row:
                training_data['activities'].append(row['activity'])

            # Store metadata
            metadata = {
                'dataset_source': row.get('dataset_source'),
                'label_id': row.get('label_id'),
                'quality_score': row.get('quality_score'),
            }
            training_data['metadata'].append(metadata)

        # Convert to numpy arrays
        if training_data['keypoints']:
            training_data['keypoints'] = np.array(training_data['keypoints'])
        else:
            training_data['keypoints'] = np.zeros((0, 17, 3), dtype=np.float32)
        training_data['activities'] = np.array(training_data['activities'])

        log.info("Converted %d samples to training format", len(training_data['keypoints']))

        return training_data

    def _parse_keypoints(self, keypoints_data: Union[str, list, np.ndarray]) -> np.ndarray:
        """Parse keypoints from various formats to standardized (17,3) array."""
        if isinstance(keypoints_data, str):
            keypoints_data = json.loads(keypoints_data)

        if isinstance(keypoints_data, list):
            keypoints_data = np.array(keypoints_data, dtype=np.float32)

        if isinstance(keypoints_data, np.ndarray):
            # Ensure COCO format (17 keypoints, x,y,confidence)
            if keypoints_data.ndim == 2 and keypoints_data.shape[-1] == 2:
                confidence = np.ones((keypoints_data.shape[0], 1), dtype=np.float32)
                keypoints_data = np.concatenate([keypoints_data, confidence], axis=1)
            return keypoints_data

        return np.zeros((17, 3), dtype=np.float32)

    def create_validation_split(self, data: Dict[str, np.ndarray],
                               train_ratio: float = 0.7) -> Tuple[Dict, Dict]:
        """Split data into training and validation sets."""
        n_samples = len(data['keypoints'])
        indices = np.random.permutation(n_samples)

        train_size = int(n_samples * train_ratio)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_data = {}
        val_data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                train_data[k] = v[train_indices]
                val_data[k] = v[val_indices]
            elif isinstance(v, list):
                train_data[k] = [v[i] for i in train_indices]
                val_data[k] = [v[i] for i in val_indices]
            else:
                train_data[k] = v
                val_data[k] = v

        log.info("Created train/val split: %d/%d",
                 len(train_data['keypoints']), len(val_data['keypoints']))

        return train_data, val_data

    def save_processed_data(self, data: Dict[str, np.ndarray], filename: str):
        """Save processed training data to disk."""
        output_path = self.cache_dir / f"{filename}.npz"
        np.savez(output_path, **data)
        log.info("Saved processed data to %s", output_path)

    def load_processed_data(self, filename: str) -> Dict[str, np.ndarray]:
        """Load processed training data from disk."""
        input_path = self.cache_dir / f"{filename}.npz"
        data = np.load(input_path)
        return {k: data[k] for k in data.files}


def demo_clinical_data_loading():
    """Demonstrate clinical data loading capabilities."""
    logging.basicConfig(level=logging.INFO)

    loader = ClinicalDataLoader()

    # Load all local data
    print("Loading local training data...")
    all_data = loader.load_local_data()
    print(f"Loaded {len(all_data)} total samples")
    print(f"Activities: {sorted(all_data['activity'].unique())}")

    # Load fall detection subset
    print("\nLoading fall detection data...")
    try:
        fall_data = loader.load_fall_detection_data()
        print(f"Fall detection samples: {len(fall_data)}")
    except ValueError as e:
        print(f"No fall data: {e}")

    # Convert to training format
    print("\nConverting to training format...")
    training_data = loader.convert_to_training_format(all_data)
    print(f"Keypoints shape: {training_data['keypoints'].shape}")
    print(f"Activities: {len(training_data['activities'])} samples")

    # Create train/val split
    if len(training_data['keypoints']) > 0:
        train_data, val_data = loader.create_validation_split(training_data)
        print(f"Training samples: {len(train_data['keypoints'])}")
        print(f"Validation samples: {len(val_data['keypoints'])}")

    return loader


if __name__ == "__main__":
    import logging
    demo_clinical_data_loading()