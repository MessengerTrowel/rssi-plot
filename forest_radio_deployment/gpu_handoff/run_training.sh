#!/usr/bin/env bash
set -euo pipefail

echo "=== ForestFormer3D Training ==="
echo "Date: $(date)"
echo ""
echo "WARNING: Full training takes ~3000 epochs."
echo "Ensure all preconditions are met before proceeding."
echo ""

export PYTHONPATH=/workspace

CONFIG="configs/oneformer3d_qs_radius16_qp300_2many.py"
WORK_DIR="work_dirs/forest_structure_training"

# Precondition checks
echo "--- Precondition checks ---"

# 1. Official inference must have succeeded
if [ ! -d "work_dirs/official_inference" ]; then
    echo "ERROR: Official inference results not found."
    echo "Run official inference first: bash gpu_handoff/run_official_inference.sh"
    exit 1
fi

# 2. Data must be preprocessed
if [ ! -d "data/ForAINetV2/forainetv2_instance_data" ]; then
    echo "ERROR: Preprocessed data not found."
    exit 1
fi

# 3. GPU must be available
python -c "import torch; assert torch.cuda.is_available(), 'No GPU'" || {
    echo "ERROR: GPU not available"
    exit 1
}

echo "All preconditions met."
echo ""

# Training
echo "Starting training..."
echo "Config: $CONFIG"
echo "Work dir: $WORK_DIR"

CUDA_VISIBLE_DEVICES=0 python tools/train.py "$CONFIG" \
  --work-dir "$WORK_DIR" 2>&1 | tee outputs/logs/forest_structure_training.log

echo ""
echo "=== Training complete ==="
