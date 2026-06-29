# Expected Outputs at Each Stage

## Smoke Test

After `run_smoke_test.sh`:
- `outputs/logs/gpu_smoke_test.log` — log of all import checks
- All 8 checks should print "OK"

## Data Preprocessing

After `batch_load_ForAINetV2_data.py`:
- `data/ForAINetV2/forainetv2_instance_data/` — NPY files per scan
- Per scan: `_vert.npy`, `_sem_label.npy`, `_ins_label.npy`, `_aligned_bbox.npy`, `_unaligned_bbox.npy`, `_axis_align_matrix.npy`, `_offsets.npy`

After `create_data_forainetv2.py`:
- `data/ForAINetV2/forainetv2_oneformer3d_infos_train.pkl`
- `data/ForAINetV2/forainetv2_oneformer3d_infos_val.pkl`
- `data/ForAINetV2/forainetv2_oneformer3d_infos_test.pkl`
- `data/ForAINetV2/points/*.bin`
- `data/ForAINetV2/instance_mask/*.bin`
- `data/ForAINetV2/semantic_mask/*.bin`

## Official Inference

After `run_official_inference.sh`:
- `work_dirs/official_inference/` — test results
- `outputs/logs/official_inference.log` — runtime log
- Per test scan: predicted PLY with instance/semantic labels
- Console output: per-class IoU, mean IoU, instance P/R/F1

## Training

After `run_training.sh`:
- `work_dirs/forest_structure_training/` — checkpoints + logs
- `work_dirs/forest_structure_training/vis_data/` — TensorBoard logs
- Checkpoints every 100 epochs (3 most recent kept)
- Best model selection requires manual analysis of validation metrics
