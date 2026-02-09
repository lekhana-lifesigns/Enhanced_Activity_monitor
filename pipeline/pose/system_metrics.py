# pipeline/system_metrics.py
"""System health metrics collection for monitoring."""
import psutil
import time


def get_health():
    """Get system health metrics (CPU, RAM, temperature)."""
    temp = None
    try:
        # Linux thermal zone (Jetson/Raspberry Pi)
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = int(f.read()) / 1000.0
    except (FileNotFoundError, IOError, ValueError):
        temp = None  # Not available on Windows/macOS

    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "temp": temp,
        "ts": time.time()
    }
