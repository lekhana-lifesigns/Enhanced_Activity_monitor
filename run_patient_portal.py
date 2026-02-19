#!/usr/bin/env python3
"""
Patient Portal Runner
Starts the HIPAA/GDPR compliant patient privacy portal.
"""

import sys
import os
import argparse
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from patient_portal import PatientPrivacyPortal

def main():
 """Run the patient privacy portal."""
 # Parse CLI arguments
 parser = argparse.ArgumentParser(description="Patient Privacy Portal - HIPAA/GDPR Compliant")
 parser.add_argument("--port", type=int, help="Web server port (overrides config)")
 parser.add_argument("--host", type=str, default='0.0.0.0', help="Web server host")
 parser.add_argument("--debug", action='store_true', help="Enable debug mode")
 parser.add_argument("--config", type=str, default="config/system.yaml", help="Path to system config file")
 args = parser.parse_args()

 # Load port from config or use CLI argument
 port = args.port
 if port is None:
 try:
 with open(args.config, 'r') as f:
 cfg = yaml.safe_load(f)
 port = cfg.get("patient_portal_port", 5001)
 except Exception:
 port = 5001 # Fallback to default

 print(" Starting Enhanced Activity Monitor - Patient Privacy Portal")
 print("HIPAA/GDPR Compliant Patient Data Access Portal")
 print("=" * 60)
 print(f"Server will listen on {args.host}:{port}")
 print("=" * 60)

 # Initialize and run the portal
 portal = PatientPrivacyPortal()

 try:
 portal.run(
 host=args.host,
 port=port,
 debug=args.debug
 )
 except KeyboardInterrupt:
 print("\n Patient Portal stopped by user")
 except Exception as e:
 print(f" Error starting patient portal: {e}")
 sys.exit(1)

if __name__ == "__main__":
 main()