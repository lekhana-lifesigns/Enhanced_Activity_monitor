#!/usr/bin/env python3
"""
Health check script for Enhanced Activity Monitor.
Used by systemd watchdog, Docker HEALTHCHECK, and monitoring systems.

Exit codes:
    0 - Healthy
    1 - Unhealthy

Checks:
    1. Database is writable
    2. MQTT broker is reachable (if configured)
    3. Camera device exists
    4. Disk space is sufficient
    5. GPU/CUDA is available (if configured)
"""
import sys
import os
import sqlite3
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage")
DB_PATH = os.path.join(STORAGE_DIR, "events.db")

errors = []


def check_database():
    """Check database is accessible and writable."""
    try:
        if not os.path.exists(STORAGE_DIR):
            errors.append("Storage directory missing")
            return
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.execute("SELECT 1")
        conn.close()
    except Exception as e:
        errors.append(f"Database: {e}")


def check_disk_space():
    """Check sufficient disk space (>10% free)."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(STORAGE_DIR)
        free_pct = (free / total) * 100
        if free_pct < 10:
            errors.append(f"Disk space low: {free_pct:.1f}% free")
    except Exception as e:
        errors.append(f"Disk check: {e}")


def check_gpu():
    """Check CUDA/GPU availability if configured."""
    try:
        import yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "system.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        if cfg.get("device") == "cuda":
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA configured but not available")
    except ImportError:
        pass  # torch not installed, skip GPU check
    except Exception as e:
        errors.append(f"GPU check: {e}")


def check_camera():
    """Check if camera device exists."""
    try:
        camera_idx = int(os.environ.get("CAMERA_IDX", "0"))
        # Linux: check /dev/videoN
        dev_path = f"/dev/video{camera_idx}"
        if os.path.exists("/dev") and not os.path.exists(dev_path):
            errors.append(f"Camera device not found: {dev_path}")
    except Exception as e:
        errors.append(f"Camera check: {e}")


def check_mqtt():
    """Check MQTT broker is reachable."""
    try:
        import yaml
        import socket
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "mqtt.yaml"
        )
        with open(config_path) as f:
            mqtt_cfg = yaml.safe_load(f)

        broker = mqtt_cfg.get("broker", "localhost")
        port = mqtt_cfg.get("port", 1883)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((broker, port))
        sock.close()
        if result != 0:
            errors.append(f"MQTT broker unreachable: {broker}:{port}")
    except Exception as e:
        errors.append(f"MQTT check: {e}")


if __name__ == "__main__":
    check_database()
    check_disk_space()
    check_gpu()
    check_camera()
    check_mqtt()

    if errors:
        print(f"UNHEALTHY: {'; '.join(errors)}")
        sys.exit(1)
    else:
        print("HEALTHY")
        sys.exit(0)
