# FOR-instanceV2 Data Audit Report

**Date:** 2026-06-29 01:34:13

## 1. Overview

- Total PLY files: 94
- Total points: 613,942,843
- Total size: 12.28 GB
- Train / Val / Test: 48 / 17 / 29
- Number of sites: 8
- Total tree instances: 11134

## 2. Data Sources / Sites

| Site | Files | Train | Val | Test | Points | Size (MB) | Trees |
|------|-------|-------|-----|------|--------|-----------|-------|
| BlueCat | 3 | 1 | 1 | 1 | 504,086,538 | 10081.8 | 6304 |
| CULS | 3 | 1 | 1 | 1 | 6,292,818 | 125.9 | 47 |
| NIBIO | 26 | 12 | 7 | 7 | 57,589,075 | 1151.8 | 932 |
| NIBIO2 | 50 | 29 | 6 | 15 | 22,948,587 | 459.1 | 3062 |
| RMIT | 2 | 1 | 0 | 1 | 1,529,042 | 30.6 | 223 |
| SCION | 5 | 2 | 1 | 2 | 10,771,797 | 215.4 | 135 |
| TUWIEN | 2 | 1 | 0 | 1 | 7,215,487 | 144.3 | 150 |
| Yuchen | 3 | 1 | 1 | 1 | 3,509,499 | 70.2 | 281 |

## 3. Point Cloud Fields

**Field set 1:** `x;y;z;semantic_seg;treeID`

## 4. Semantic Label Distribution (field: `semantic_seg`)

| Label | Likely Meaning | Total Points | Percentage |
|-------|---------------|-------------|------------|
| 1 | Ground/Terrain | 14,016,414 | 2.3% |
| 2 | Trunk/Stem | 124,649,081 | 20.3% |
| 3 | Canopy/Foliage | 475,277,348 | 77.4% |

## 5. Instance Statistics

- Trees per file: min=6, max=5032, mean=118.4
- Total tree instances: 11134

## 6. Data Quality

- Files with NaN coordinates: 0
- Files with Inf coordinates: 0
- Files with duplicate points: 93 (total duplicates: 4,889,143)
- Empty files: 0
- Corrupt files: 0

## 7. Coordinate Ranges

| File | X range | Y range | Z range |
|------|---------|---------|----------|
| BlueCat_RN_merged_trees_test.ply | [-472.5, -415.1] | [-598.3, -491.2] | [151.8, 189.6] |
| CULS_CULS_plot_2_annotated_test.ply | [-20.0, 14.4] | [-21.9, 14.4] | [334.5, 362.2] |
| NIBIO2_NIBIO2_plot10_annotated_test.ply | [0.2, 27.9] | [0.7, 28.5] | [0.0, 22.3] |
| NIBIO2_NIBIO2_plot15_annotated_test.ply | [0.3, 28.1] | [0.9, 28.7] | [0.0, 22.7] |
| NIBIO2_NIBIO2_plot1_annotated_test.ply | [0.9, 28.7] | [1.0, 28.7] | [0.0, 26.6] |
| NIBIO2_NIBIO2_plot27_annotated_test.ply | [0.5, 28.3] | [0.9, 28.6] | [0.0, 25.8] |
| NIBIO2_NIBIO2_plot32_annotated_test.ply | [0.2, 27.9] | [0.9, 28.7] | [0.0, 20.5] |
| NIBIO2_NIBIO2_plot34_annotated_test.ply | [0.7, 28.4] | [0.1, 27.9] | [0.0, 22.3] |
| NIBIO2_NIBIO2_plot35_annotated_test.ply | [0.4, 28.2] | [0.3, 28.1] | [0.0, 27.1] |
| NIBIO2_NIBIO2_plot3_annotated_test.ply | [0.5, 28.3] | [0.3, 28.2] | [0.0, 17.4] |
| ... (84 more files) | | | |

## 8. Pretrained Model

- Pretrained weights: `epoch_3000_fix.pth` (230.2 MB)
- NOTE: The `clean_forestformer.zip` contains only the pretrained model weights,
  NOT the ForestFormer3D source code. The source code must be obtained separately
  (likely from a GitHub repository).

## 9. Risks and Notes

- **No GPU available** on current machine (Windows Server 2022, Intel Xeon). GPU required for model inference and training.
- **No ForestFormer3D source code** in the Zenodo download. Only pretrained weights.
- **Large dataset** (~11.4 GB extracted). BlueCat site alone is ~8.8 GB.
- FOR-instanceV2 does **NOT contain RSSI data**. Cannot be used for LoRa propagation modeling.
