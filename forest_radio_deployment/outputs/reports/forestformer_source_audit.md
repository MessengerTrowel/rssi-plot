# ForestFormer3D Source Code Audit Report

## Repository Information

| Item | Value |
|------|-------|
| Repository | SmartForest-no/ForestFormer3D |
| URL | https://github.com/SmartForest-no/ForestFormer3D |
| Commit | `6a75c3735e4a4108d02ee944a8b93177f2360a4f` |
| Branch | `main` |
| License | CC BY-NC 4.0 |
| Paper | ICCV 2025 Oral |
| Clone Date | 2026-06-28 |

## 1. Directory Structure

```
ForestFormer3D/
├── configs/                    # 2 config files
│   ├── oneformer3d_qs_radius16_qp300_2many.py   (main config)
│   └── oneformer3d_radius16_qp300.py
├── data/ForAINetV2/            # Data loading and preprocessing
│   ├── batch_load_ForAINetV2_data.py
│   ├── load_forainetv2_data.py
│   ├── plyutils.py
│   ├── compare_outputs.py
│   ├── second_inference.py
│   └── meta_data/              # Train/val/test split lists
│       ├── train_list.txt      (47 scans)
│       ├── val_list.txt        (16 scans)
│       └── test_list.txt       (28 scans)
├── oneformer3d/                # Core model code
│   ├── oneformer3d.py          (226 KB - main model)
│   ├── query_decoder.py        (50 KB - decoder)
│   ├── instance_criterion.py   (53 KB - losses)
│   ├── spconv_unet.py          (SpConv backbone)
│   ├── mink_unet.py            (Minkowski backbone)
│   ├── transforms_3d.py        (augmentations)
│   └── ... (17 .py files total)
├── tools/                      # Entry points and utilities
│   ├── train.py
│   ├── test.py
│   ├── create_data_forainetv2.py
│   ├── fix_spconv_checkpoint.py
│   ├── inference_bluepoint.sh
│   ├── final_eval.py
│   └── ... (14 files total)
├── replace_mmdetection_files/  # Patches for mmdetection3d
│   ├── base_model.py
│   ├── loops.py
│   └── transforms_3d.py
├── segmentator/                # Superpoint segmentation (C++)
├── Dockerfile
├── LICENSE
└── readme.md
```

## 2. Key Dependencies (from Dockerfile)

| Package | Version | Purpose | GPU Required |
|---------|---------|---------|--------------|
| PyTorch | 1.13.1 | Deep learning framework | Yes (CUDA 11.6) |
| Python | 3.10 | Runtime | No |
| mmengine | 0.7.3 | Training framework | No |
| mmdet | 3.0.0 | Detection framework | No |
| mmdet3d | 22aaa47 (git) | 3D detection | Yes |
| mmcv | 2.0.0 (cu116) | Computer vision ops | Yes |
| MinkowskiEngine | 02fc608 (git) | Sparse conv (alternative) | Yes |
| spconv-cu116 | 2.3.6 | Sparse convolution | Yes |
| torch-scatter | 2.0.9 | Scatter operations | Yes |
| torch-points-kernels | 0.7.0 | Point cloud kernels | Yes |
| torch-cluster | latest | Clustering ops | Yes |
| Open3D | 0.17.0 | Point cloud processing | No (CPU) |
| plyfile | 1.0.2 | PLY file I/O | No |
| numpy | 1.24.1 | Numerical computing | No |
| scipy | 1.10.1 | Scientific computing | No |
| pandas | 2.0.1 | Data frames | No |
| matplotlib | 3.5.2 | Visualization | No |

## 3. Docker Environment

- Base image: `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel`
- CUDA arch: `8.0` (A100 optimized; may need modification for other GPUs)
- Shared memory: `--shm-size=128g` recommended (resource planning, not minimum)
- Port mapping: `127.0.0.1:49211:22` for SSH debugging
- PYTHONPATH: `/workspace`

## 4. Configuration Analysis

### Main Config (`oneformer3d_qs_radius16_qp300_2many.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model type | ForAINetV2OneFormer3D_XAwarequery | Custom OneFormer3D variant |
| Backbone | SpConvUNet | 5-level sparse conv |
| Voxel size | 0.2 m | Grid sampling resolution |
| Cylinder radius | 16 m | Input crop radius |
| Query points | 300 | Decoder query count |
| Num channels | 32 | Base feature channels |
| Semantic classes | 3 | ground(0), wood(1), leaf(2) |
| Instance classes | 3 | Same 3 classes |
| Decoder layers | 6 | Transformer decoder |
| Hidden dim | 256/1024 | d_model / hidden |
| Training epochs | 3000 | Full training schedule |
| Batch size | 2 | Per-GPU |
| Workers | 12 | Data loading |
| LR | 0.0001 | AdamW |
| Weight decay | 0.05 | |
| LR schedule | PolyLR (power=0.9) | Over 450K iterations |
| Max points/crop | 640,000 | After grid sampling |
| Score threshold | 0.4 | Instance filtering |
| Chunk size | 20,000 | Sequential processing |
| Stuff classes | [0] (ground) | Background |
| Thing classes | [1, 2] (wood, leaf) | Instance objects |

### Semantic Label Mapping

| Internal ID | Class Name | Original PLY `semantic_seg` |
|-------------|------------|---------------------------|
| 0 | ground | 1 (mapped: semantic_seg - 1) |
| 1 | wood/trunk | 2 |
| 2 | leaf/canopy | 3 |

## 5. Data Preprocessing Pipeline

### Step 1: PLY to NPY (`batch_load_ForAINetV2_data.py`)

- **Input**: PLY files with fields `x, y, z, semantic_seg, treeID`
- **Processing**:
  1. Read PLY via custom `plyutils.read_ply()` (binary PLY reader)
  2. Extract coordinates as float64
  3. Center: subtract mean(x), mean(y), min(z) as offsets
  4. Cast to float32
  5. Read `semantic_seg` and `treeID` as int64
  6. Compute bounding boxes per instance
  7. Apply identity alignment matrix
- **Output** (per scan): `_vert.npy`, `_sem_label.npy`, `_ins_label.npy`, `_aligned_bbox.npy`, `_unaligned_bbox.npy`, `_axis_align_matrix.npy`, `_offsets.npy`
- **Output dir**: `data/ForAINetV2/forainetv2_instance_data/`
- **GPU required**: No (pure NumPy + custom PLY reader)
- **Dependencies**: numpy, scipy (Delaunay), open3d, torch (import only), segmentator (import only)
- **Note**: `torch` and `segmentator` are imported at top but not used in core `export()` function

### Step 2: NPY to PKL+BIN (`create_data_forainetv2.py`)

- **Input**: NPY files from Step 1
- **Processing**: Creates mmdet3d info PKL files with paths to BIN files
- **Output**: `forainetv2_oneformer3d_infos_{train,val,test}.pkl`, `points/*.bin`, `instance_mask/*.bin`, `semantic_mask/*.bin`
- **GPU required**: No (but requires mmdet3d/mmengine imports)
- **Dependencies**: mmengine, mmdet3d (register_all_modules)

### CPU-Runnable Steps

| Step | Script | CPU-Only | Notes |
|------|--------|----------|-------|
| PLY reading | `plyutils.read_ply()` | Yes | Pure binary file I/O |
| Coordinate centering | `load_forainetv2_data.export()` | Partially | Imports torch/segmentator but core logic is NumPy |
| Batch NPY export | `batch_load_ForAINetV2_data.py` | Partially | Top-level imports require torch/segmentator |
| Info PKL generation | `create_data_forainetv2.py` | No | Requires mmdet3d.utils |
| Training | `tools/train.py` | No | Requires CUDA |
| Testing/inference | `tools/test.py` | No | Requires CUDA + spconv |

### Steps Requiring GPU

1. `tools/test.py` - inference with SpConv + CUDA
2. `tools/train.py` - training with CUDA
3. MinkowskiEngine operations
4. spconv operations
5. torch-points-kernels (instance IoU)
6. torch-cluster operations

## 6. Training Entry Point

```
tools/train.py CONFIG [--work-dir DIR] [--amp] [--resume]
```
- Uses mmengine Runner
- Config: `configs/oneformer3d_qs_radius16_qp300_2many.py`
- Command: `CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/oneformer3d_qs_radius16_qp300_2many.py --work-dir work_dirs/<name>`

## 7. Testing Entry Point

```
tools/test.py CONFIG CHECKPOINT [--work-dir DIR]
```
- Uses mmengine Runner
- Applies spconv checkpoint fix in-memory (weight permutation)
- Official pretrained: `work_dirs/clean_forestformer/epoch_3000_fix.pth`
- Command: `CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/oneformer3d_qs_radius16_qp300_2many.py work_dirs/clean_forestformer/epoch_3000_fix.pth`

## 8. Pretrained Weights

| File | Source | Size | Location |
|------|--------|------|----------|
| epoch_3000_fix.pth | Zenodo (clean_forestformer.zip) | ~230 MB | work_dirs/clean_forestformer/ |

The checkpoint has already been fixed with `fix_spconv_checkpoint.py`. The `test.py` script additionally applies an in-memory fix for spconv weight permutation.

## 9. Evaluation Metrics

From `unified_metric.py` and config:
- **Semantic**: Per-class IoU, Mean IoU
- **Instance**: Precision, Recall, F1
- **Instance matching**: Based on IoU overlap with configurable thresholds
- **NMS**: Matrix NMS with linear kernel
- **Score threshold**: Instance score >= 0.0 (test_cfg), NMS at sp_score_thr=0.15

## 10. Two-Pass Inference

For dense forests, the model supports iterative inference:
1. First pass: standard inference on full point cloud
2. Second pass: re-inference on "blue points" (unassigned points from first pass)
3. Results merged via `tools/merge_prediction.py`

Script: `tools/inference_bluepoint.sh`

## 11. Key Observations for Adaptation

### For Baicaowa Data Integration

1. **PLY format required**: Fields must be `x, y, z, semantic_seg, treeID`
2. **Coordinate system**: Mean-centered (x, y) and min-zeroed (z)
3. **Semantic labels**: Must be 1=ground, 2=wood, 3=leaf (converted internally to 0, 1, 2)
4. **Instance labels**: treeID > 0 for trees, 0 for ground/unannotated
5. **Binary PLY**: ASCII PLY not supported by the custom reader
6. **File naming**: `{site}_{scan}_annotated_{split}` convention

### Known Limitations

1. CUDA arch set to `8.0` (A100) in Dockerfile - needs `TORCH_CUDA_ARCH_LIST` adjustment for other GPUs
2. `--shm-size=128g` is a resource planning suggestion; actual requirement depends on batch size and data
3. Python 3.10 specifically (not 3.12)
4. PyTorch 1.13.1 specifically (not newer)
5. CC BY-NC 4.0 license - non-commercial use only

### For Forest Radio Deployment Project

1. The model outputs single-tree instances with semantic class (ground/wood/leaf)
2. From instances, we can extract: tree positions, bounding boxes, tree height, crown dimensions
3. No trunk diameter or DBH prediction - must derive from point cloud geometry
4. No color/intensity features used - only xyz coordinates
5. Cylinder-based processing (radius=16m) naturally segments forest into local patches

## 12. Files NOT to Modify

Per task specification, original repository must remain unmodified. All adaptations go to `src/forestformer_adapted/` with changes documented in `outputs/reports/forestformer_code_changes.md`.
