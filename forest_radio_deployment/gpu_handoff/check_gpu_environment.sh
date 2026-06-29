#!/usr/bin/env bash
set -euo pipefail

echo "=== GPU ==="
nvidia-smi

echo ""
echo "=== Docker ==="
docker --version

echo ""
echo "=== NVIDIA container runtime ==="
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

echo ""
echo "=== Disk ==="
df -h

echo ""
echo "=== Memory ==="
free -h

echo ""
echo "=== CPU ==="
lscpu | head -20

echo ""
echo "=== Environment check complete ==="
