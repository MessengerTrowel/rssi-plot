# Troubleshooting Guide

## Common Issues

### 1. CUDA OOM During Inference

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions** (try in order):
1. Reduce `chunk` value in config (default: 20000)
2. Reduce `num_points` in `oneformer3d/oneformer3d.py` (see README)
3. Reduce cylinder `radius` in config (default: 16)

### 2. torch-points-kernels Import Error

**Symptom**: `ModuleNotFoundError: No module named 'torch_points_kernels.points_cuda'`

**Solution**:
```bash
pip uninstall torch-points-kernels -y
pip install --no-deps --no-cache-dir torch-points-kernels==0.7.0
```

### 3. torch-cluster Import Error

**Solution**:
```bash
pip uninstall torch-cluster
pip install torch-cluster --no-cache-dir --no-deps
```

### 4. mmengine/mmdet3d Compatibility

**Symptom**: Various import errors or attribute errors

**Solution**: Replace patched files:
```bash
pip show mmengine  # find package path
cp replace_mmdetection_files/loops.py /opt/conda/lib/python3.10/site-packages/mmengine/runner/
cp replace_mmdetection_files/base_model.py /opt/conda/lib/python3.10/site-packages/mmengine/model/base_model/
cp replace_mmdetection_files/transforms_3d.py /opt/conda/lib/python3.10/site-packages/mmdet3d/datasets/transforms/
```

### 5. spconv Checkpoint Mismatch

**Symptom**: Weight shape errors during checkpoint loading

**Note**: `tools/test.py` already applies in-memory spconv weight permutation fix. If using a custom script, apply the fix from `tools/fix_spconv_checkpoint.py` first.

### 6. Docker Build Fails on CUDA Arch

**Symptom**: MinkowskiEngine build error

**Solution**: Edit `TORCH_CUDA_ARCH_LIST` in Dockerfile to match your GPU:
- A100: `"8.0"`
- A6000/RTX 3090: `"8.6"`
- RTX 4090: `"8.9"`
- V100: `"7.0"`

### 7. Shared Memory Error

**Symptom**: `RuntimeError: DataLoader worker is killed by signal: Bus error`

**Solution**: Increase `--shm-size` in docker run, but do not exceed physical RAM.

### 8. No PLY Files Found

**Symptom**: Empty data directory or file not found errors

**Solution**: Verify symlinks:
```bash
ls -la data/ForAINetV2/train_val_data/
ls -la data/ForAINetV2/test_data/
ls -la work_dirs/clean_forestformer/epoch_3000_fix.pth
```

If broken, recreate:
```bash
ln -sfn /path/to/actual/train_val_data data/ForAINetV2/train_val_data
ln -sfn /path/to/actual/test_data data/ForAINetV2/test_data
```
