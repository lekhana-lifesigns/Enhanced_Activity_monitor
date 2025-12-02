# pipeline/system_metrics.py
import psutil, time
def get_health():
    try:
        temp = None
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = int(f.read())/1000.0
    except:
        temp = None
    return {"cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent, "temp": temp, "ts": time.time()}
