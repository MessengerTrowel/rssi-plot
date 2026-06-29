# Stage 0 Report: FOR-instanceV2 Data Download, Environment Check & Data Audit

**Date:** 2026-06-29
**Platform:** Windows Server 2022 (10.0.20348)
**Machine:** Intel Xeon Platinum 8259CL @ 2.50GHz, 8 cores, 32 GB RAM

---

## 1. Download Results

| File | Size (MB) | MD5 Expected | MD5 Actual | Status |
|------|-----------|-------------|------------|--------|
| clean_forestformer.zip | 188.7 | 553d67379331966509076f3fbb409e57 | 553d67379331966509076f3fbb409e57 | PASS |
| test_data.zip | 359.6 | 1c00a0f0b89f03b74064432162619136 | 1c00a0f0b89f03b74064432162619136 | PASS |
| train_val_data.zip | 2,314.2 | 5a63cc1cbe88edd9ebec28ad7e46f79b | 5a63cc1cbe88edd9ebec28ad7e46f79b | PASS |

**All 3 MD5 checksums PASSED.**

## 2. Extraction Summary

| Archive | Extracted Directory | Files | Size (GB) |
|---------|-------------------|-------|-----------|
| clean_forestformer.zip | forestformer_code/ | 1 | 0.21 |
| test_data.zip | test_data/ | 29 | 1.31 |
| train_val_data.zip | train_val_data/ | 65 | 10.13 |
| **Total** | | **95** | **11.65** |

Total FOR-instanceV2 directory (including ZIPs + extracted): ~14.45 GB

## 3. Disk Space

- C: drive total: ~127.8 GB
- C: drive free after extraction: ~98.7 GB
- Remaining for project work: sufficient (~98 GB)

## 4. GPU & Compute Resources

| Resource | Status |
|----------|--------|
| GPU | **NOT AVAILABLE** (no NVIDIA GPU detected) |
| CUDA | Not installed |
| CPU | Intel Xeon 8259CL, 8 cores @ 2.50 GHz |
| RAM | 32 GB |
| Python | 3.12.8 |
| OS | Windows Server 2022 Standard |

**Critical limitation:** No GPU available. Model inference and training for ForestFormer3D (a point cloud deep learning model) require CUDA-capable GPU. CPU-only inference is theoretically possible but extremely slow for 600M+ points.

## 5. ForestFormer3D Code Review

### What's in the download:
- `clean_forestformer.zip` contains ONLY the pretrained model weights file: `epoch_3000_fix.pth` (230.2 MB)
- **NO source code, README, configuration files, or training scripts** are included in the Zenodo download

### What's needed:
- The ForestFormer3D source code repository (likely on GitHub) must be obtained separately
- Without the source code, we cannot:
  - Load or run the pretrained model
  - Understand the model architecture
  - Perform inference on the test set
  - Fine-tune for Baicaowa domain adaptation

### Pretrained weights:
- File: `epoch_3000_fix.pth` (230.2 MB)
- Format: PyTorch checkpoint (`.pth`)
- Architecture details: Unknown without source code

## 6. FOR-instanceV2 Data Audit

### 6.1 Overview

| Metric | Value |
|--------|-------|
| Total PLY files | 94 |
| Total points | 613,942,843 (~614 million) |
| Total size (extracted) | 12.28 GB |
| Data format | PLY (binary) |
| Train files | 48 |
| Validation files | 17 |
| Test files | 29 |
| Sites/sources | 8 |
| Total tree instances | 11,134 |
| Empty files | 0 |
| Corrupt files | 0 |

### 6.2 Point Cloud Fields

All 94 files share the same schema:

| Field | Type | Description |
|-------|------|-------------|
| x | float | X coordinate |
| y | float | Y coordinate |
| z | float | Z coordinate |
| semantic_seg | int | Semantic segmentation label |
| treeID | int | Instance segmentation label (per-tree ID) |

### 6.3 Semantic Label Distribution

| Label | Likely Meaning | Points | Percentage |
|-------|---------------|--------|------------|
| 1 | Ground/Terrain | 14,016,414 | 2.3% |
| 2 | Trunk/Stem | 124,649,081 | 20.3% |
| 3 | Canopy/Foliage | 475,277,348 | 77.4% |

**Note:** No dedicated semantic labels for understory vegetation, branches, or dead wood.

### 6.4 Sites

| Site | Files | Train | Val | Test | Trees |
|------|-------|-------|-----|------|-------|
| BlueCat | 3 | 1 | 1 | 1 | 6,304 |
| CULS | 3 | 1 | 1 | 1 | 47 |
| NIBIO | 14 | 7 | 4 | 3 (ALS) + 4 (MLS) | ~500+ |
| NIBIO2 | 44 | 28 | 7 | 9 (plots) | ~2,800+ |
| NIBIO_MLS | 5 | 4 | 1 | 0 | ~334 |
| RMIT | 2 | 1 | 0 | 1 | 223 |
| SCION | 4 | 2 | 1 | 1 | ~135 |
| TUWIEN | 2 | 1 | 0 | 1 | 150 |
| Yuchen | 3 | 1 | 1 | 1 | 281 |

### 6.5 Data Quality

- **NaN coordinates:** 0 files
- **Inf coordinates:** 0 files  
- **Duplicate points:** Checked per file (exact XYZ matches)
- **Empty files:** 0
- **Corrupt files:** 0
- **Coordinate systems:** Various (local and projected), not all georeferenced

### 6.6 Minimal Sample Read Results

Successfully read `Yuchen_2023_dls_merged_230209_panoptic_test.ply`:
- 197,963 points, 5 fields
- X range: [286625, 286657] (projected coordinates)
- Z range: [19.5, 63.1] m (suggesting tree heights up to ~44 m)
- 24 tree instances, 8,331 ground points
- Semantic labels: 1 (ground 4.2%), 2 (trunk 3.5%), 3 (canopy 92.3%)

## 7. Assessment: Can We Proceed?

### 7.1 Can the code be installed?
**NO** — ForestFormer3D source code is NOT included in the Zenodo download. Only pretrained weights.

### 7.2 Can the data be read by official scripts?
**PARTIALLY** — Data is readable via `plyfile`. Official scripts are not available (no source code).

### 7.3 Does the pretrained model exist?
**YES** — `epoch_3000_fix.pth` (230 MB) exists and appears valid.

### 7.4 Is inference possible?
**NOT YET** — Missing: (1) ForestFormer3D source code, (2) GPU.

### 7.5 Is retraining feasible on this machine?
**NO** — No GPU available. Training on 614M points requires CUDA GPU with significant VRAM (16+ GB recommended).

## 8. Data Integrity Compliance

Per the project's scientific integrity rules:
- FOR-instanceV2 does **NOT** contain RSSI data → Cannot be used for LoRa propagation modeling ✓
- FOR-instanceV2 is used **ONLY** for forest point cloud structure model pre-training and testing ✓
- All data is raw, unmodified from Zenodo source ✓
- No anomalous values were deleted or modified ✓

## 9. Identified Risks

1. **No GPU:** Critical blocker for deep learning inference and training. Need GPU-equipped machine.
2. **No ForestFormer3D source code:** Only pretrained weights in Zenodo. Source code repository needed.
3. **Large dataset:** BlueCat site alone is ~8.8 GB (407M points train + 56M val + 40M test). Processing requires substantial RAM.
4. **No semantic ground truth verification:** Only 3 semantic classes (ground/trunk/canopy). No fine-grained labels for understory, branches, etc.
5. **Mixed coordinate systems:** Different sites use different projections/CRS. Need careful spatial alignment for any cross-site analysis.
6. **Windows environment:** Most point cloud deep learning frameworks are better supported on Linux.

## 10. Next Steps (Recommended)

1. **Obtain ForestFormer3D source code** — User should provide the GitHub repository URL or source code
2. **Set up GPU environment** — Either:
   - Provision a GPU machine (NVIDIA GPU with CUDA)
   - Use cloud GPU (e.g., AWS, GCP, Azure)
3. **Install PyTorch + CUDA** on GPU machine
4. **Reproduce official inference** (Stage 1, Section 7.1) using pretrained weights
5. **Await Baicaowa data** from user for subsequent stages

**Do NOT start full training until GPU environment is confirmed and official inference is verified.**
