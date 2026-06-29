# GPU Environment Migration Plan

## Target Environment

| Item | Recommendation | Notes |
|------|---------------|-------|
| OS | Ubuntu 22.04 LTS | Per official Dockerfile |
| GPU | NVIDIA (16+ GB VRAM) | A100/A6000/RTX 4090 preferred |
| GPU Driver | Compatible with CUDA 11.6 | Or rebuild Docker for newer CUDA |
| Docker | Latest stable | Required for official environment |
| NVIDIA Container Toolkit | Latest | For `--gpus all` support |
| System RAM | 64 GB+ | 128 GB recommended for large scans |
| Disk | 50+ GB free | Data + model + intermediate files |
| Python | 3.10 (via Docker) | Do not use system Python |

### GPU VRAM Resource Planning

| Task | Suggested VRAM | Notes |
|------|---------------|-------|
| Smoke test + single sample | 16 GB+ | Start here to verify |
| Official test set inference | 24 GB+ | 28 test scans |
| Fine-tuning | 24-48 GB | Depends on batch size |
| Full training (3000 epochs) | 24-48 GB | A100 used in paper |

These are resource planning suggestions, not official minimum requirements. Run single-sample test first to determine actual VRAM needs.

## Migration Steps

### Phase 1: Environment Setup

```bash
# 1. Verify GPU
nvidia-smi

# 2. Verify Docker + NVIDIA runtime
docker --version
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# 3. Transfer project data (or re-download)
# Option A: rsync from current machine
# Option B: Re-download from Zenodo (see scripts/download_for_instance_v2.ps1 for URLs)

# 4. Clone ForestFormer3D
git clone https://github.com/SmartForest-no/ForestFormer3D.git external/ForestFormer3D
cd external/ForestFormer3D
git checkout 6a75c3735e4a4108d02ee944a8b93177f2360a4f

# 5. Link data (do not copy)
ln -s /path/to/data/raw/for_instance_v2/train_val_data data/ForAINetV2/train_val_data
ln -s /path/to/data/raw/for_instance_v2/test_data data/ForAINetV2/test_data
mkdir -p work_dirs/clean_forestformer
ln -s /path/to/data/raw/for_instance_v2/forestformer_code/clean_forestformer/epoch_3000_fix.pth work_dirs/clean_forestformer/epoch_3000_fix.pth

# 6. Build Docker image
docker build -t forestformer3d:official .

# 7. Run container
docker run --gpus all --shm-size=64g --rm -it \
  -v $(pwd):/workspace \
  forestformer3d:official /bin/bash
```

Note: `--shm-size` should be adjusted based on available system RAM. Do not set higher than physical RAM.

### Phase 2: Smoke Test (Inside Container)

```bash
# 1. Verify CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 2. Verify spconv
python -c "import spconv; print('spconv OK')"

# 3. Verify torch-points-kernels
python -c "from torch_points_kernels import instance_iou; print('torch-points-kernels OK')"

# 4. Verify mmdet3d
python -c "from mmdet3d.utils import register_all_modules; print('mmdet3d OK')"

# 5. Verify checkpoint loading
python -c "
import torch
ckpt = torch.load('work_dirs/clean_forestformer/epoch_3000_fix.pth', map_location='cpu')
print('Keys:', list(ckpt.keys()))
print('State dict keys:', len(ckpt.get('state_dict', {})))
print('Checkpoint loaded OK')
"
```

### Phase 3: Data Preprocessing

```bash
# Inside container
cd /workspace

# Step 1: PLY -> NPY
cd data/ForAINetV2
pip install laspy "laspy[lazrs]"
python batch_load_ForAINetV2_data.py

# Step 2: NPY -> PKL+BIN
cd /workspace
python tools/create_data_forainetv2.py forainetv2
```

### Phase 4: Single Sample Inference

```bash
# Inside container - pick smallest test file
export PYTHONPATH=/workspace

# Run on single test sample (modify test_list.txt to contain only 1 entry)
# Or run full test set:
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/oneformer3d_qs_radius16_qp300_2many.py \
  work_dirs/clean_forestformer/epoch_3000_fix.pth
```

### Phase 5: Full Test Set

Only proceed after single sample succeeds.

### Phase 6: Training/Fine-tuning

Only proceed after all conditions in Section 六.6.5 of the task specification are met.

## Data Transfer Checklist

| File | Size | Source | Required |
|------|------|--------|----------|
| train_val_data/ | ~10 GB | Zenodo extract | Yes |
| test_data/ | ~1.3 GB | Zenodo extract | Yes |
| epoch_3000_fix.pth | ~230 MB | Zenodo extract | Yes |
| ForestFormer3D repo | ~2 MB | GitHub clone | Yes |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| CUDA arch mismatch | Rebuild Docker with correct `TORCH_CUDA_ARCH_LIST` |
| OOM on inference | Reduce `chunk` or `num_points` in config |
| OOM on training | Reduce `batch_size` or `radius` |
| Dependency conflicts | Use official Docker image as-is |
| Data corruption during transfer | Re-verify MD5 checksums after transfer |

## First Command After Migration

```bash
bash gpu_handoff/check_gpu_environment.sh
```
