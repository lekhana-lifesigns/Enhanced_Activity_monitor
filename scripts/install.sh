#!/usr/bin/env bash
# Enhanced Activity Monitor - Installation Script for Raspberry Pi
# This script sets up the Python environment and installs dependencies

set -e

echo "============================================================"
echo "🏥 Enhanced Activity Monitor - Installation"
echo "============================================================"

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ] || ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "⚠️  Warning: This script is optimized for Raspberry Pi"
    echo "   Continuing anyway..."
fi

# Update package list
echo ""
echo "📦 Updating package list..."
sudo apt update

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-venv \
    python3-pip \
    libatlas-base-dev \
    libopenblas-dev \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5

# Create virtual environment
VENV_PATH="/opt/eac_venv"
echo ""
echo "🐍 Creating virtual environment at $VENV_PATH..."
if [ -d "$VENV_PATH" ]; then
    echo "   Virtual environment already exists, skipping..."
else
    sudo python3 -m venv $VENV_PATH
    echo "   ✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source $VENV_PATH/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo ""
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Install TensorFlow Lite runtime for Raspberry Pi (if ARM)
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "armv7l" ]; then
    echo ""
    echo "📦 Installing TensorFlow Lite runtime for ARM..."
    # Try to install tflite-runtime
    pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0-cp39-cp39-linux_aarch64.whl || \
    pip install tflite-runtime || \
    echo "   ⚠️  Could not install tflite-runtime, will use TensorFlow fallback"
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p models/yolo
mkdir -p models/pose
mkdir -p models/temporal
mkdir -p storage
mkdir -p logs
mkdir -p config

# Set permissions
echo ""
echo "🔐 Setting permissions..."
sudo chown -R $USER:$USER $VENV_PATH
chmod +x eac.py

echo ""
echo "============================================================"
echo "✅ Installation Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Place TFLite models in models/ directory:"
echo "   - models/yolo/yolo11n.tflite"
echo "   - models/pose/movenet_lightning.tflite"
echo "   - models/temporal/gru_activity.tflite"
echo ""
echo "2. Configure config/system.yaml and config/mqtt.yaml"
echo ""
echo "3. Activate virtual environment:"
echo "   source $VENV_PATH/bin/activate"
echo ""
echo "4. Run the system:"
echo "   python3 eac.py"
echo ""
echo "5. (Optional) Install as systemd service:"
echo "   sudo cp services/eac.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable eac.service"
echo "   sudo systemctl start eac.service"
echo ""
echo "============================================================"
