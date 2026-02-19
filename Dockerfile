# Enhanced Activity Monitor - Jetson Orin Nano Deployment
# Base: NVIDIA L4T with PyTorch + CUDA (JetPack 6.x)
#
# Build:  docker build -t eam:latest .
# Run:    docker run --runtime nvidia --device /dev/video0 --network host -v ./storage:/app/storage eam:latest
# Multi:  docker run --runtime nvidia --device /dev/video0 --network host -e DEVICE_ID=bed_01 eam:latest
#         docker run --runtime nvidia --device /dev/video1 --network host -e DEVICE_ID=bed_02 eam:latest

FROM nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libv4l-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (Jetson-specific — excludes torch/tf already in L4T base)
COPY requirements-jetson.txt .
RUN pip3 install --no-cache-dir -r requirements-jetson.txt

# Application code
COPY . .

# Create required directories
RUN mkdir -p storage logs models/tensorrt storage/patient_faces

# Default environment
ENV DEVICE_ID=bed_01
ENV CAMERA_IDX=0
ENV EAM_LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 deploy/health_check.py || exit 1

ENTRYPOINT ["python3", "eac.py"]
CMD ["--device-id", "bed_01", "--config", "config/system.yaml", "--mqtt-config", "config/mqtt.yaml"]
