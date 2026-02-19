#!/usr/bin/env python3
"""
Control script for ICU Monitor system
Provides alternative ways to control the system when display is unresponsive
"""

import signal
import os
import sys
import time
import psutil

def find_inference_processes():
 """Find running inference_node.py processes."""
 inference_pids = []
 for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
 try:
 if proc.info['cmdline'] and 'inference_node.py' in ' '.join(proc.info['cmdline']):
 inference_pids.append(proc.info['pid'])
 except (psutil.NoSuchProcess, psutil.AccessDenied):
 continue
 return inference_pids

def stop_inference_node():
 """Stop the running inference node gracefully."""
 pids = find_inference_processes()

 if not pids:
 print("No inference_node.py processes found running.")
 return False

 print(f"Found {len(pids)} inference_node.py process(es): {pids}")

 for pid in pids:
 try:
 os.kill(pid, signal.SIGTERM)
 print(f"Sent SIGTERM to process {pid}")
 except OSError as e:
 print(f"Failed to stop process {pid}: {e}")
 return False

 # Wait a bit for graceful shutdown
 time.sleep(2)

 # Check if processes are still running
 remaining = find_inference_processes()
 if remaining:
 print(f"Processes still running: {remaining}")
 print("Force killing...")
 for pid in remaining:
 try:
 os.kill(pid, signal.SIGKILL)
 print(f"Force killed process {pid}")
 except OSError as e:
 print(f"Failed to force kill process {pid}: {e}")

 return True

def show_status():
 """Show status of inference processes."""
 pids = find_inference_processes()

 if pids:
 print(f" ICU Monitor is RUNNING ({len(pids)} process(es): {pids})")
 print(" Display window may be open but unresponsive to keyboard.")
 print(" Use this script to stop the system gracefully.")
 else:
 print(" ICU Monitor is NOT running")
 print(" Start with: .venv\\Scripts\\activate && python inference_node.py")

def main():
 if len(sys.argv) < 2:
 print("ICU Monitor Control Script")
 print("Usage: python control_monitor.py <command>")
 print("Commands:")
 print(" status - Show if ICU Monitor is running")
 print(" stop - Stop the ICU Monitor gracefully")
 return

 command = sys.argv[1].lower()

 if command == "status":
 show_status()
 elif command == "stop":
 print("Stopping ICU Monitor...")
 if stop_inference_node():
 print(" ICU Monitor stopped successfully")
 else:
 print(" Failed to stop ICU Monitor")
 else:
 print(f"Unknown command: {command}")

if __name__ == "__main__":
 main()