# analytics/vitals.py
# Vital Sign Proxies Module
# Estimates vital signs from vision-derived signals

import numpy as np
import math
import logging
from scipy import signal
from collections import deque

log = logging.getLogger("vitals")

# Keypoint indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

MIN_CONFIDENCE = 0.3


def get_keypoint(kps, idx, default=(0.0, 0.0, 0.0)):
    """Safely get keypoint with confidence check."""
    if idx < len(kps) and kps[idx][2] > MIN_CONFIDENCE:
        return kps[idx]
    return default


def get_thorax_point(kps):
    """Get thorax (chest) point as midpoint between shoulders and hips."""
    try:
        ls = get_keypoint(kps, LEFT_SHOULDER)
        rs = get_keypoint(kps, RIGHT_SHOULDER)
        lh = get_keypoint(kps, LEFT_HIP)
        rh = get_keypoint(kps, RIGHT_HIP)
        
        # Mid-shoulder
        sx = (ls[0] + rs[0]) / 2.0
        sy = (ls[1] + rs[1]) / 2.0
        
        # Mid-hip
        hx = (lh[0] + rh[0]) / 2.0
        hy = (lh[1] + rh[1]) / 2.0
        
        # Thorax = midpoint
        tx = (sx + hx) / 2.0
        ty = (sy + hy) / 2.0
        
        return (tx, ty)
    except Exception:
        return (0.5, 0.5)


def estimate_breath_rate(kps_history, fps=15.0):
    """
    Estimate breathing rate from thorax vertical oscillation.
    
    Args:
        kps_history: List of keypoint sequences over time
        fps: Frames per second
    
    Returns:
        dict with:
        - breath_rate: Estimated breaths per minute
        - confidence: 0-1 confidence score
        - waveform: Breathing waveform (optional)
    """
    if not kps_history or len(kps_history) < 10:
        return {
            "breath_rate": 0.0,
            "confidence": 0.0,
            "waveform": []
        }
    
    try:
        # Extract thorax vertical positions
        thorax_y_positions = []
        for kps in kps_history:
            if kps:
                tx, ty = get_thorax_point(kps)
                thorax_y_positions.append(ty)
        
        if len(thorax_y_positions) < 10:
            return {
                "breath_rate": 0.0,
                "confidence": 0.0,
                "waveform": []
            }
        
        # Detrend the signal
        y_signal = np.array(thorax_y_positions)
        y_detrended = signal.detrend(y_signal)
        
        # Apply bandpass filter (0.1-0.5 Hz = 6-30 breaths/min)
        try:
            sos = signal.butter(4, [0.1, 0.5], btype='band', fs=fps, output='sos')
            y_filtered = signal.sosfilt(sos, y_detrended)
        except Exception:
            y_filtered = y_detrended
        
        # FFT to find dominant frequency
        fft = np.fft.fft(y_filtered)
        freqs = np.fft.fftfreq(len(y_filtered), 1.0 / fps)
        
        # Find peak in breathing range
        valid_idx = (freqs > 0.1) & (freqs < 0.5)
        if np.any(valid_idx):
            power = np.abs(fft[valid_idx])
            peak_idx = np.argmax(power)
            peak_freq = freqs[valid_idx][peak_idx]
            breath_rate = peak_freq * 60.0  # Convert to breaths/min
            
            # Confidence based on signal quality
            peak_power = power[peak_idx]
            total_power = np.sum(power)
            confidence = min(1.0, peak_power / (total_power + 1e-6))
            
            return {
                "breath_rate": float(np.clip(breath_rate, 0.0, 40.0)),
                "confidence": float(confidence),
                "waveform": y_filtered.tolist()
            }
        else:
            return {
                "breath_rate": 0.0,
                "confidence": 0.0,
                "waveform": []
            }
            
    except Exception as e:
        log.exception("Error in breath rate estimation: %s", e)
        return {
            "breath_rate": 0.0,
            "confidence": 0.0,
            "waveform": []
        }


def estimate_hrv_proxy(kps_history, fps=15.0):
    """
    Estimate heart rate variability proxy from micro-movements.
    Uses high-frequency components of motion as HRV proxy.
    
    Args:
        kps_history: List of keypoint sequences over time
        fps: Frames per second
    
    Returns:
        dict with:
        - hrv_proxy: HRV proxy metric (normalized 0-1)
        - confidence: 0-1 confidence score
    """
    if not kps_history or len(kps_history) < 20:
        return {
            "hrv_proxy": 0.0,
            "confidence": 0.0
        }
    
    try:
        # Extract thorax micro-movements
        thorax_positions = []
        for kps in kps_history:
            if kps:
                tx, ty = get_thorax_point(kps)
                thorax_positions.append((tx, ty))
        
        if len(thorax_positions) < 20:
            return {
                "hrv_proxy": 0.0,
                "confidence": 0.0
            }
        
        # Compute micro-movements (high-frequency components)
        x_signal = np.array([p[0] for p in thorax_positions])
        y_signal = np.array([p[1] for p in thorax_positions])
        
        # Detrend
        x_detrended = signal.detrend(x_signal)
        y_detrended = signal.detrend(y_signal)
        
        # High-pass filter (remove slow movements, keep micro-movements)
        try:
            sos = signal.butter(4, 0.5, btype='high', fs=fps, output='sos')
            x_hf = signal.sosfilt(sos, x_detrended)
            y_hf = signal.sosfilt(sos, y_detrended)
        except Exception:
            x_hf = x_detrended
            y_hf = y_detrended
        
        # Compute HRV proxy as variability of high-frequency components
        hrv_x = np.std(x_hf)
        hrv_y = np.std(y_hf)
        hrv_proxy = math.sqrt(hrv_x * hrv_x + hrv_y * hrv_y)
        
        # Normalize to 0-1 (assuming max ~0.01 in normalized coordinates)
        hrv_proxy_normalized = min(1.0, hrv_proxy * 100.0)
        
        # Confidence based on signal quality
        signal_power = np.var(x_hf) + np.var(y_hf)
        confidence = min(1.0, signal_power * 10000.0)
        
        return {
            "hrv_proxy": float(hrv_proxy_normalized),
            "confidence": float(confidence)
        }
        
    except Exception as e:
        log.exception("Error in HRV proxy estimation: %s", e)
        return {
            "hrv_proxy": 0.0,
            "confidence": 0.0
        }


def estimate_vital_signs(kps_history, fps=15.0):
    """
    Comprehensive vital sign estimation.
    
    Returns:
        dict with all vital sign proxies
    """
    if not kps_history:
        return {
            "breath_rate": {},
            "hrv_proxy": {},
            "overall_confidence": 0.0
        }
    
    try:
        breath_rate = estimate_breath_rate(kps_history, fps)
        hrv_proxy = estimate_hrv_proxy(kps_history, fps)
        
        overall_confidence = (breath_rate["confidence"] + hrv_proxy["confidence"]) / 2.0
        
        return {
            "breath_rate": breath_rate,
            "hrv_proxy": hrv_proxy,
            "overall_confidence": float(overall_confidence)
        }
        
    except Exception as e:
        log.exception("Error in vital sign estimation: %s", e)
        return {
            "breath_rate": {},
            "hrv_proxy": {},
            "overall_confidence": 0.0
        }

