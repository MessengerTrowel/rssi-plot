# GPU Environment Setup for ForestFormer3D

## Quick Start

```bash
# 1. Verify GPU environment
bash check_gpu_environment.sh

# 2. Build Docker image and enter container
bash build_and_enter_container.sh /path/to/ForestFormer3D

# 3. Inside container: run smoke test
bash /workspace/gpu_handoff/run_smoke_test.sh

# 4. Inside container: preprocess data
cd /workspace/data/ForAINetV2
python batch_load_ForAINetV2_data.py
cd /workspace
python tools/create_data_forainetv2.py forainetv2

# 5. Inside container: run official inference
bash /workspace/gpu_handoff/run_official_inference.sh
```

## Prerequisites

- Ubuntu 22.04 LTS
- NVIDIA GPU with 16+ GB VRAM (resource planning suggestion)
- NVIDIA Driver compatible with CUDA 11.6
- Docker with NVIDIA Container Toolkit
- 50+ GB free disk space

## Data Layout

Before starting, ensure data is organized as:

```
forest_radio_deployment/
├── data/raw/for_instance_v2/
│   ├── train_val_data/     (65 PLY files)
│   ├── test_data/          (29 PLY files - 28 test + NIBIO_MLS)
│   └── forestformer_code/
│       └── clean_forestformer/
│           └── epoch_3000_fix.pth
└── external/ForestFormer3D/   (cloned from GitHub)
```

Symlinks will be created by the setup to avoid data duplication. See `data_paths.yaml` for full path configuration.

## Troubleshooting

See `troubleshooting.md` for common issues and solutions.
