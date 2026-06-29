# System Requirements for ForestFormer3D

## Hardware (Resource Planning)

| Component | Smoke Test | Inference | Training |
|-----------|-----------|-----------|----------|
| GPU VRAM | 16 GB+ | 24 GB+ | 24-48 GB |
| System RAM | 32 GB | 64 GB | 64-128 GB |
| Disk | 30 GB | 50 GB | 100 GB |
| CPU | 4+ cores | 8+ cores | 12+ cores |

These are resource planning suggestions based on model architecture and Dockerfile settings. They are NOT official minimum hardware requirements. Run single-sample smoke test first to determine actual resource needs.

## Software

| Component | Version | Source |
|-----------|---------|--------|
| OS | Ubuntu 22.04 LTS | - |
| NVIDIA Driver | >=515 | nvidia.com |
| Docker | >=20.10 | docker.com |
| NVIDIA Container Toolkit | Latest | nvidia.github.io |
| Python | 3.10 | Via Docker |
| PyTorch | 1.13.1 | Via Docker |
| CUDA | 11.6 | Via Docker base image |

## GPU Compatibility

The Dockerfile builds CUDA kernels for arch `8.0` (A100). For other GPUs:

| GPU | CUDA Arch | Dockerfile Change |
|-----|-----------|-------------------|
| A100 | 8.0 | Default (no change) |
| A6000 | 8.6 | Change to `"8.6"` |
| RTX 3090 | 8.6 | Change to `"8.6"` |
| RTX 4090 | 8.9 | Change to `"8.9"` |
| V100 | 7.0 | Change to `"7.0"` |
| Multi-GPU | multiple | Use `"7.0;8.0;8.6"` |

Edit the `TORCH_CUDA_ARCH_LIST` in the Dockerfile MinkowskiEngine build step.
