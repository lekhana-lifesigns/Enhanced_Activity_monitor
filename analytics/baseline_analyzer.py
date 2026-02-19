# analytics/baseline_analyzer.py
"""
Patient Baseline Analyzer for EAM System.

Uses DBSCAN for per-patient behavior clustering and PCA for feature
stability monitoring. Produces baseline_info consumed by ClinicalMonitor.

Design:
  - DBSCAN fitted offline (or periodically) on accumulated 9D feature vectors.
  - PCA fitted once on the same buffer; reconstruction error flags sensor drift.
  - At inference time both are cheap O(n_core) lookups — safe for Orin Nano.
"""

import numpy as np
import logging
import time
import pickle
from typing import Dict, Optional, Tuple
from collections import deque

log = logging.getLogger("baseline_analyzer")

# ---------------------------------------------------------------------------
# Lightweight DBSCAN implementation (no sklearn dependency on edge)
# ---------------------------------------------------------------------------

def _euclidean_distances(X: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Euclidean distance from every row of X to a single point."""
    return np.sqrt(np.sum((X - point) ** 2, axis=1))


def _dbscan_fit(X: np.ndarray, eps: float, min_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal DBSCAN.  Returns (labels, core_indices).
    labels: -1 = noise, >=0 = cluster id.
    """
    n = len(X)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        dists = _euclidean_distances(X, X[i])
        neighbors = np.where(dists <= eps)[0]

        if len(neighbors) < min_samples:
            continue  # noise for now

        labels[i] = cluster_id
        seed_set = list(neighbors)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_dists = _euclidean_distances(X, X[q])
                q_neighbors = np.where(q_dists <= eps)[0]
                if len(q_neighbors) >= min_samples:
                    seed_set.extend(q_neighbors.tolist())
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1
        cluster_id += 1

    core_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        dists = _euclidean_distances(X, X[i])
        if np.sum(dists <= eps) >= min_samples:
            core_mask[i] = True

    return labels, np.where(core_mask)[0]


# ---------------------------------------------------------------------------
# Lightweight PCA (NumPy only)
# ---------------------------------------------------------------------------

class _SimplePCA:
    """Minimal PCA via SVD — no sklearn needed."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        total_var = (S ** 2).sum()
        self.components_ = Vt[: self.n_components]
        self.explained_variance_ratio_ = (S[: self.n_components] ** 2) / max(total_var, 1e-12)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Per-sample reconstruction error (L2)."""
        Z = self.transform(X)
        X_hat = self.inverse_transform(Z)
        return np.sqrt(np.sum((X - X_hat) ** 2, axis=1))


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class PatientBaselineAnalyzer:
    """
    Per-patient baseline modeling using DBSCAN + PCA on 9D feature vectors.

    Lifecycle:
        1. Accumulate features via ``update(feature_vector)``.
        2. After ``min_samples_for_fit`` samples, automatically fit baseline.
        3. ``analyze(feature_vector)`` returns baseline_info dict consumed by
           ClinicalMonitor.
        4. Periodically re-fits when buffer grows by ``refit_interval`` samples.
        5. Models can be serialized / loaded for persistence across restarts.
    """

    # Baseline states
    STATE_COLLECTING = "collecting"
    STATE_READY = "ready"

    def __init__(
        self,
        patient_id: str,
        device_id: str,
        buffer_size: int = 5000,
        min_samples_for_fit: int = 300,
        refit_interval: int = 500,
        # DBSCAN params
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 10,
        # PCA params
        pca_components: int = 6,
        pca_reconstruction_threshold: float = 1.5,
        # Anomaly params
        anomaly_persistence_frames: int = 15,
    ):
        self.patient_id = patient_id
        self.device_id = device_id

        # Buffers
        self.feature_buffer = deque(maxlen=buffer_size)
        self.min_samples_for_fit = min_samples_for_fit
        self.refit_interval = refit_interval
        self._samples_since_last_fit = 0

        # DBSCAN state
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self._core_points: Optional[np.ndarray] = None
        self._core_labels: Optional[np.ndarray] = None
        self._cluster_labels: Optional[np.ndarray] = None
        self._n_clusters = 0

        # PCA state
        self.pca_components = pca_components
        self.pca_reconstruction_threshold = pca_reconstruction_threshold
        self._pca: Optional[_SimplePCA] = None

        # Feature distribution (for KL divergence)
        self._baseline_hist: Optional[np.ndarray] = None
        self._hist_bins = 20
        self._hist_ranges: Optional[list] = None

        # Anomaly persistence (must persist > N frames to count)
        self.anomaly_persistence_frames = anomaly_persistence_frames
        self._anomaly_streak = 0

        # State
        self.state = self.STATE_COLLECTING
        self.last_fit_time: float = 0.0
        self.total_samples: int = 0
        self.fit_count: int = 0

        log.info(
            "BaselineAnalyzer created for patient=%s device=%s "
            "(min_samples=%d, refit_interval=%d)",
            patient_id, device_id, min_samples_for_fit, refit_interval,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, feature_vector: np.ndarray) -> None:
        """Ingest a single 9D feature vector into the baseline buffer."""
        feat = np.asarray(feature_vector, dtype=np.float32).ravel()
        if np.isnan(feat).any() or np.isinf(feat).any():
            return  # reject bad samples

        self.feature_buffer.append(feat)
        self.total_samples += 1
        self._samples_since_last_fit += 1

        # Auto-fit when enough samples collected
        if self.state == self.STATE_COLLECTING:
            if len(self.feature_buffer) >= self.min_samples_for_fit:
                self._fit_baseline()
        elif self._samples_since_last_fit >= self.refit_interval:
            self._fit_baseline()

    def analyze(self, feature_vector: np.ndarray) -> Dict:
        """
        Score a single feature vector against the patient baseline.

        Returns:
            baseline_info dict:
                is_anomaly: bool — sustained anomaly (persisted)
                anomaly_score: float 0-1 — raw anomaly score this frame
                kl_score: float 0-1 — KL divergence from baseline distribution
                cluster_id: int — nearest DBSCAN cluster (-1 = noise)
                distance_to_core: float — distance to nearest core point
                pca_reconstruction_error: float — PCA reconstruction error
                baseline_state: str — "collecting" or "ready"
                samples_in_baseline: int
        """
        feat = np.asarray(feature_vector, dtype=np.float32).ravel()

        info: Dict = {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "kl_score": 0.0,
            "cluster_id": -1,
            "distance_to_core": 0.0,
            "pca_reconstruction_error": 0.0,
            "baseline_state": self.state,
            "samples_in_baseline": len(self.feature_buffer),
        }

        if self.state != self.STATE_READY:
            return info

        # --- DBSCAN distance ---
        dist_to_core, nearest_cluster = self._distance_to_nearest_core(feat)
        info["distance_to_core"] = float(dist_to_core)
        info["cluster_id"] = int(nearest_cluster)

        # Normalize distance to 0-1 score (sigmoid-like)
        dbscan_score = float(np.tanh(dist_to_core / max(self.dbscan_eps * 3, 1e-6)))

        # --- PCA reconstruction error ---
        pca_error = 0.0
        if self._pca is not None:
            pca_error = float(
                self._pca.reconstruction_error(feat.reshape(1, -1))[0]
            )
        info["pca_reconstruction_error"] = pca_error

        # Normalize PCA error
        pca_score = float(
            np.tanh(pca_error / max(self.pca_reconstruction_threshold * 2, 1e-6))
        )

        # --- KL divergence from baseline distribution ---
        kl = self._compute_kl_divergence(feat)
        info["kl_score"] = float(np.clip(kl, 0.0, 1.0))

        # --- Combined anomaly score ---
        anomaly_score = 0.4 * dbscan_score + 0.3 * pca_score + 0.3 * kl
        info["anomaly_score"] = float(np.clip(anomaly_score, 0.0, 1.0))

        # --- Persistence filter (reduce false positives) ---
        if anomaly_score > 0.5:
            self._anomaly_streak += 1
        else:
            self._anomaly_streak = max(0, self._anomaly_streak - 1)

        info["is_anomaly"] = self._anomaly_streak >= self.anomaly_persistence_frames

        return info

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_baseline(self) -> None:
        """Fit DBSCAN + PCA on the accumulated feature buffer."""
        X = np.array(self.feature_buffer, dtype=np.float32)
        if len(X) < self.dbscan_min_samples:
            return

        t0 = time.time()

        # --- DBSCAN ---
        labels, core_idx = _dbscan_fit(X, self.dbscan_eps, self.dbscan_min_samples)
        self._cluster_labels = labels
        if len(core_idx) > 0:
            self._core_points = X[core_idx]
            self._core_labels = labels[core_idx]
        else:
            # Fallback: use all non-noise as cores
            non_noise = labels >= 0
            if non_noise.any():
                self._core_points = X[non_noise]
                self._core_labels = labels[non_noise]
            else:
                # All noise — use centroids of data as single cluster
                self._core_points = X.mean(axis=0, keepdims=True)
                self._core_labels = np.array([0])
        self._n_clusters = max(int(labels.max()) + 1, 1)

        # --- PCA ---
        n_comp = min(self.pca_components, X.shape[1])
        self._pca = _SimplePCA(n_components=n_comp)
        self._pca.fit(X)

        # --- Baseline histogram for KL divergence ---
        self._build_baseline_histogram(X)

        self._samples_since_last_fit = 0
        self.last_fit_time = time.time()
        self.fit_count += 1
        self.state = self.STATE_READY

        elapsed = time.time() - t0
        log.info(
            "Baseline fitted for patient=%s: %d clusters, %d core points, "
            "PCA variance=%.2f, elapsed=%.1fms (fit #%d, %d samples)",
            self.patient_id,
            self._n_clusters,
            len(self._core_points),
            float(self._pca.explained_variance_ratio_.sum()),
            elapsed * 1000,
            self.fit_count,
            len(X),
        )

    # ------------------------------------------------------------------
    # Distance / scoring helpers
    # ------------------------------------------------------------------

    def _distance_to_nearest_core(self, feat: np.ndarray) -> Tuple[float, int]:
        """Return (distance, cluster_id) to nearest core point."""
        if self._core_points is None or len(self._core_points) == 0:
            return 0.0, -1
        dists = _euclidean_distances(self._core_points, feat)
        idx = int(np.argmin(dists))
        return float(dists[idx]), int(self._core_labels[idx])

    def _build_baseline_histogram(self, X: np.ndarray) -> None:
        """Build per-feature histograms for KL divergence computation."""
        self._hist_ranges = []
        hists = []
        for col in range(X.shape[1]):
            lo, hi = float(X[:, col].min()), float(X[:, col].max())
            margin = max((hi - lo) * 0.1, 1e-6)
            lo -= margin
            hi += margin
            self._hist_ranges.append((lo, hi))
            h, _ = np.histogram(X[:, col], bins=self._hist_bins, range=(lo, hi))
            h = h.astype(np.float64) + 1e-10  # Laplace smoothing
            h /= h.sum()
            hists.append(h)
        self._baseline_hist = np.array(hists)  # (n_features, bins)

    def _compute_kl_divergence(self, feat: np.ndarray) -> float:
        """
        Approximate KL divergence of a single sample from baseline.
        We place the sample in the histogram and compute per-feature surprise.
        """
        if self._baseline_hist is None or self._hist_ranges is None:
            return 0.0

        total_surprise = 0.0
        for col in range(len(self._hist_ranges)):
            lo, hi = self._hist_ranges[col]
            val = float(feat[col])
            # Which bin does val fall into?
            bin_idx = int((val - lo) / max(hi - lo, 1e-12) * self._hist_bins)
            bin_idx = np.clip(bin_idx, 0, self._hist_bins - 1)
            p = self._baseline_hist[col, bin_idx]
            total_surprise += -np.log(max(p, 1e-12))

        # Normalize by number of features and log(bins) to get 0-1 range
        max_surprise = len(self._hist_ranges) * np.log(self._hist_bins)
        return float(np.clip(total_surprise / max(max_surprise, 1e-12), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> bytes:
        """Serialize the fitted model to bytes for storage."""
        state = {
            "patient_id": self.patient_id,
            "device_id": self.device_id,
            "state": self.state,
            "fit_count": self.fit_count,
            "total_samples": self.total_samples,
            "last_fit_time": self.last_fit_time,
            "n_clusters": self._n_clusters,
            # DBSCAN
            "core_points": self._core_points,
            "core_labels": self._core_labels,
            "dbscan_eps": self.dbscan_eps,
            "dbscan_min_samples": self.dbscan_min_samples,
            # PCA
            "pca_mean": self._pca.mean_ if self._pca else None,
            "pca_components": self._pca.components_ if self._pca else None,
            "pca_variance_ratio": self._pca.explained_variance_ratio_ if self._pca else None,
            "pca_n_components": self.pca_components,
            "pca_reconstruction_threshold": self.pca_reconstruction_threshold,
            # Histogram
            "baseline_hist": self._baseline_hist,
            "hist_ranges": self._hist_ranges,
            "hist_bins": self._hist_bins,
        }
        return pickle.dumps(state)

    def load_from_bytes(self, data: bytes) -> bool:
        """Restore a fitted model from serialized bytes."""
        try:
            s = pickle.loads(data)

            self.state = s.get("state", self.STATE_COLLECTING)
            self.fit_count = s.get("fit_count", 0)
            self.total_samples = s.get("total_samples", 0)
            self.last_fit_time = s.get("last_fit_time", 0.0)
            self._n_clusters = s.get("n_clusters", 0)

            # DBSCAN
            self._core_points = s.get("core_points")
            self._core_labels = s.get("core_labels")
            self.dbscan_eps = s.get("dbscan_eps", self.dbscan_eps)
            self.dbscan_min_samples = s.get("dbscan_min_samples", self.dbscan_min_samples)

            # PCA
            pca_mean = s.get("pca_mean")
            pca_comp = s.get("pca_components")
            pca_var = s.get("pca_variance_ratio")
            if pca_mean is not None and pca_comp is not None:
                self._pca = _SimplePCA(n_components=s.get("pca_n_components", self.pca_components))
                self._pca.mean_ = pca_mean
                self._pca.components_ = pca_comp
                self._pca.explained_variance_ratio_ = pca_var
                self.pca_reconstruction_threshold = s.get(
                    "pca_reconstruction_threshold", self.pca_reconstruction_threshold
                )

            # Histogram
            self._baseline_hist = s.get("baseline_hist")
            self._hist_ranges = s.get("hist_ranges")
            self._hist_bins = s.get("hist_bins", self._hist_bins)

            log.info(
                "Baseline loaded for patient=%s: state=%s, %d clusters, fit_count=%d",
                self.patient_id, self.state, self._n_clusters, self.fit_count,
            )
            return True
        except Exception as e:
            log.warning("Failed to load baseline: %s", e)
            return False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict:
        """Return human-readable summary of baseline state."""
        summary = {
            "patient_id": self.patient_id,
            "device_id": self.device_id,
            "state": self.state,
            "total_samples": self.total_samples,
            "buffer_size": len(self.feature_buffer),
            "fit_count": self.fit_count,
            "n_clusters": self._n_clusters,
        }
        if self._pca is not None:
            summary["pca_explained_variance"] = float(
                self._pca.explained_variance_ratio_.sum()
            )
        return summary
