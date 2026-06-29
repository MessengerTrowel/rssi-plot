#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$PWD}"

echo "=== Building ForestFormer3D Docker image ==="
echo "Project directory: $PROJECT_DIR"

cd "$PROJECT_DIR"

# Verify Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "ERROR: Dockerfile not found in $PROJECT_DIR"
    echo "Make sure you are in the ForestFormer3D repository root"
    exit 1
fi

# Get available system memory for shm-size
TOTAL_MEM_GB=$(free -g | awk '/Mem:/{print $2}')
SHM_SIZE="${TOTAL_MEM_GB}g"
echo "System RAM: ${TOTAL_MEM_GB} GB"
echo "Setting --shm-size=${SHM_SIZE} (matching available system RAM)"

# Build Docker image
docker build -t forestformer3d:official .

echo ""
echo "=== Docker image built successfully ==="
echo ""
echo "=== Starting container ==="

# Run container with GPU support
docker run \
  --gpus all \
  --shm-size="${SHM_SIZE}" \
  --rm \
  -it \
  -v "$PROJECT_DIR:/workspace" \
  forestformer3d:official \
  /bin/bash
