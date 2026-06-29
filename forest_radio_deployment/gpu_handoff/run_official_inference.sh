#!/usr/bin/env bash
set -euo pipefail

echo "=== ForestFormer3D Official Inference ==="
echo "Date: $(date)"

export PYTHONPATH=/workspace

CONFIG="configs/oneformer3d_qs_radius16_qp300_2many.py"
CHECKPOINT="work_dirs/clean_forestformer/epoch_3000_fix.pth"

# Verify files exist
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Check if preprocessed data exists
if [ ! -d "data/ForAINetV2/forainetv2_instance_data" ]; then
    echo "ERROR: Preprocessed data not found."
    echo "Run data preprocessing first:"
    echo "  cd data/ForAINetV2 && python batch_load_ForAINetV2_data.py"
    echo "  cd /workspace && python tools/create_data_forainetv2.py forainetv2"
    exit 1
fi

echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo ""

# Run inference
CUDA_VISIBLE_DEVICES=0 python tools/test.py "$CONFIG" "$CHECKPOINT" \
  --work-dir work_dirs/official_inference 2>&1 | tee outputs/logs/official_inference.log

echo ""
echo "=== Inference complete ==="
echo "Results saved to: work_dirs/official_inference/"
echo "Log saved to: outputs/logs/official_inference.log"
