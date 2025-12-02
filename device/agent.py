import psutil
import time

class DeviceAgent:

    def send_heartbeat(self):
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        heartbeat = {
            "timestamp": time.time(),
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage
        }

        print("💓 Heartbeat:", heartbeat)
        # Here you would normally send the heartbeat to a server or log it