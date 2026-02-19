#!/bin/bash
# Enhanced Activity Monitor - Jetson Orin Nano Deployment Script
# Usage: sudo ./deploy/install.sh [bed_count]
#
# Examples:
#   sudo ./deploy/install.sh 1       # Single bed (1 camera)
#   sudo ./deploy/install.sh 4       # 4 beds (4 cameras on one Jetson)

set -euo pipefail

BED_COUNT=${1:-1}
INSTALL_DIR="/opt/eam"
SERVICE_USER="eam"

echo "=============================================="
echo "Enhanced Activity Monitor - Deployment"
echo "Jetson Orin Nano Super"
echo "Beds: ${BED_COUNT}"
echo "=============================================="

# Check root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Run as root (sudo ./deploy/install.sh)"
    exit 1
fi

# Check for Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "WARNING: Not running on Jetson. Proceeding anyway..."
fi

# Create service user
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --no-create-home --shell /bin/false "$SERVICE_USER"
    echo "Created service user: ${SERVICE_USER}"
fi

# Create install directory
mkdir -p "${INSTALL_DIR}"
cp -r . "${INSTALL_DIR}/"

# Create required directories
mkdir -p "${INSTALL_DIR}/storage"
mkdir -p "${INSTALL_DIR}/logs"
mkdir -p "${INSTALL_DIR}/models/tensorrt"
mkdir -p "${INSTALL_DIR}/storage/patient_faces"

# Set up Python virtual environment
if [ ! -d "${INSTALL_DIR}/venv" ]; then
    python3 -m venv "${INSTALL_DIR}/venv"
    echo "Created virtual environment"
fi

source "${INSTALL_DIR}/venv/bin/activate"
pip install --upgrade pip
pip install -r "${INSTALL_DIR}/requirements.txt"
echo "Dependencies installed"

# Create .env from example if it doesn't exist
if [ ! -f "${INSTALL_DIR}/.env" ]; then
    cp "${INSTALL_DIR}/.env.example" "${INSTALL_DIR}/.env"
    echo ""
    echo "WARNING: .env created from .env.example"
    echo "  Edit ${INSTALL_DIR}/.env with production secrets BEFORE starting!"
    echo ""
fi

# Generate encryption key if not set
if ! grep -q "^EAM_DB_ENCRYPTION_KEY=.\+" "${INSTALL_DIR}/.env" 2>/dev/null; then
    KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    sed -i "s|^EAM_DB_ENCRYPTION_KEY=.*|EAM_DB_ENCRYPTION_KEY=${KEY}|" "${INSTALL_DIR}/.env"
    echo "Generated database encryption key"
fi

# Set ownership
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_DIR}"
chmod 600 "${INSTALL_DIR}/.env"
chmod 700 "${INSTALL_DIR}/storage"

# Install systemd service
cp "${INSTALL_DIR}/deploy/eam@.service" /etc/systemd/system/
systemctl daemon-reload

# Enable and start services for each bed
for i in $(seq -w 1 "$BED_COUNT"); do
    BED_ID="bed_$(printf '%02d' "$i")"
    systemctl enable "eam@${BED_ID}.service"
    echo "Enabled service: eam@${BED_ID}.service"
done

echo ""
echo "=============================================="
echo "Installation complete!"
echo ""
echo "BEFORE STARTING:"
echo "  1. Edit ${INSTALL_DIR}/.env with production secrets"
echo "  2. Edit ${INSTALL_DIR}/config/security.yaml - change default passwords"
echo "  3. Place YOLO models in ${INSTALL_DIR}/ (yolo11n.pt, yolo11n-pose.pt, yolo11n-seg.pt)"
echo "  4. Configure MQTT broker address in ${INSTALL_DIR}/config/mqtt.yaml"
echo ""
echo "START:"
for i in $(seq -w 1 "$BED_COUNT"); do
    BED_ID="bed_$(printf '%02d' "$i")"
    echo "  sudo systemctl start eam@${BED_ID}"
done
echo ""
echo "VIEW LOGS:"
echo "  journalctl -u 'eam@*' -f"
echo "=============================================="
