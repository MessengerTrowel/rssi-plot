#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/workspace/outputs/logs/gpu_smoke_test.log"
mkdir -p "$(dirname "$LOG_FILE")"

{
echo "=== GPU Smoke Test ==="
echo "Date: $(date)"
echo ""

echo "--- 1. CUDA availability ---"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    echo 'FAIL: CUDA not available'
    exit 1
"

echo ""
echo "--- 2. spconv ---"
python -c "import spconv; print(f'spconv version: {spconv.__version__}')" || echo "FAIL: spconv import failed"

echo ""
echo "--- 3. torch-points-kernels ---"
python -c "from torch_points_kernels import instance_iou; print('torch-points-kernels: OK')" || echo "FAIL: torch-points-kernels import failed"

echo ""
echo "--- 4. torch-cluster ---"
python -c "import torch_cluster; print('torch-cluster: OK')" || echo "FAIL: torch-cluster import failed"

echo ""
echo "--- 5. mmengine ---"
python -c "import mmengine; print(f'mmengine version: {mmengine.__version__}')" || echo "FAIL: mmengine import failed"

echo ""
echo "--- 6. mmdet3d ---"
python -c "from mmdet3d.utils import register_all_modules; register_all_modules(); print('mmdet3d: OK')" || echo "FAIL: mmdet3d import failed"

echo ""
echo "--- 7. Checkpoint loading ---"
python -c "
import torch
ckpt_path = 'work_dirs/clean_forestformer/epoch_3000_fix.pth'
try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f'Checkpoint keys: {list(ckpt.keys())}')
    if 'state_dict' in ckpt:
        print(f'State dict parameters: {len(ckpt[\"state_dict\"])}')
    print('Checkpoint: OK')
except Exception as e:
    print(f'FAIL: {e}')
"

echo ""
echo "--- 8. Data files ---"
ls -la data/ForAINetV2/train_val_data/*.ply 2>/dev/null | head -5 || echo "WARNING: train_val_data not linked"
ls -la data/ForAINetV2/test_data/*.ply 2>/dev/null | head -5 || echo "WARNING: test_data not linked"
ls -la work_dirs/clean_forestformer/epoch_3000_fix.pth 2>/dev/null || echo "WARNING: pretrained weights not linked"

echo ""
echo "=== Smoke test complete ==="

} 2>&1 | tee "$LOG_FILE"
