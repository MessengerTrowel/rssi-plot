# Literature Method Comparison

## Overview

This report compares 5 key reference papers relevant to the forest radio deployment project. Each paper is analyzed for its methodology, data requirements, and applicability to our specific task of combining forest structure modeling with wireless communication optimization.

---

## Paper 1: LaPS — LiDAR-assisted Placement of Wireless Sensor Networks in Forests

### Summary
LaPS uses airborne LiDAR point cloud data to model radio signal attenuation in forests by tracing signal paths through individual trees. It provides physics-based path loss predictions that account for trunk interception and canopy penetration.

| Aspect | Details |
|--------|---------|
| **Research Task** | Predict signal attenuation in forests using LiDAR-derived forest structure |
| **Input Data** | Airborne LiDAR point cloud, single-tree segmentation, Tx/Rx positions |
| **Output Data** | Path loss predictions (dB) for arbitrary Tx-Rx links |
| **Model Structure** | Physics-based ray-tracing through forest structure: free-space loss + trunk attenuation + canopy attenuation + terrain effects |
| **Mathematical Model** | `L = L0 + 10n*log10(d/d0) + alpha_trunk*L_trunk + alpha_canopy*L_canopy + terrain_term` |
| **Loss Function** | N/A (physics-based, parameters fitted via regression) |
| **Optimization Objective** | Minimize path loss prediction error (RMSE) |
| **Data Split** | Leave-one-out cross-validation on link measurements |
| **Evaluation Metrics** | RMSE, MAE, R2 of predicted vs measured path loss |
| **Reproducible Code** | Not publicly available |
| **Directly Applicable** | Forest structure → link feature extraction pipeline; path loss model framework |
| **Not Directly Applicable** | Original parameters are site-specific; cannot transfer to Baicaowa without recalibration |

### Relevance to This Project
- **High**: Core inspiration for the LaPS prior component (Stage 5)
- LaPS trunk/canopy interception method directly applicable
- Parameters must be recalibrated using Baicaowa RSSI measurements
- Serves as physics-informed prior for Forest-UNetDCN residual prediction

---

## Paper 2: Radio Map Prediction from Aerial Images and Application to Coverage Optimization

### Summary
Proposes using a U-Net architecture with deformable convolutions (UNetDCN) to predict radio signal coverage maps from aerial images. The model takes overhead imagery and transmitter location as input, and predicts spatially continuous signal strength maps.

| Aspect | Details |
|--------|---------|
| **Research Task** | Predict radio coverage maps from aerial images |
| **Input Data** | Aerial/satellite imagery, transmitter position (as Gaussian heatmap + distance map), measured RSSI at sparse points |
| **Output Data** | Pixel-wise radio coverage map (signal strength) |
| **Model Structure** | U-Net with deformable convolution layers (DCN), encoder-decoder with skip connections |
| **Mathematical Model** | `RSSI_hat = f_UNetDCN(image, tx_position)` |
| **Loss Function** | MSE on measured points; weighted loss for sparse observations |
| **Optimization Objective** | Minimize prediction error at known measurement locations |
| **Data Split** | Spatial split: different transmitter locations for train/val/test |
| **Evaluation Metrics** | RMSE, MAE, R2 on held-out measurements |
| **Reproducible Code** | Not publicly available |
| **Directly Applicable** | UNetDCN architecture; transmitter encoding (Gaussian heatmap + distance map + azimuth); sparse measurement training |
| **Not Directly Applicable** | Urban/suburban environment assumptions; RGB-only input channels; no forest-specific features |

### Relevance to This Project
- **High**: Architectural basis for Forest-UNetDCN (Stage 6)
- Tx encoding method (Gaussian heatmap, distance, azimuth) directly reusable
- Must extend input channels to include forest structure layers (CHM, canopy cover, trunk density)
- Must add LaPS residual prediction branch and uncertainty estimation
- Loss function needs modification for forest scenario (add physics prior, uncertainty, reciprocity)

---

## Paper 3: RayProNet — A Neural Point Field Framework for Radio Propagation Modeling in 3D Environments

### Summary
RayProNet uses neural point fields to model radio propagation in complex 3D environments. It represents the environment as neural point features and traces rays through them to predict signal propagation characteristics.

| Aspect | Details |
|--------|---------|
| **Research Task** | 3D radio propagation modeling using neural representations |
| **Input Data** | 3D scene geometry (point clouds/meshes), Tx/Rx positions, frequency |
| **Output Data** | Path loss, delay spread, angle of arrival |
| **Model Structure** | Neural point field + differentiable ray tracing; MLP-based feature extraction |
| **Mathematical Model** | Point-wise neural features aggregated along ray paths |
| **Loss Function** | L2 loss on propagation parameters |
| **Optimization Objective** | Accurate propagation prediction in complex 3D scenes |
| **Data Split** | Scene-based split |
| **Evaluation Metrics** | RMSE, MAE on path loss; angular errors |
| **Reproducible Code** | Partially available |
| **Directly Applicable** | Concept of aggregating 3D point features along propagation paths |
| **Not Directly Applicable** | Requires dense ray tracing (computationally expensive for forest); urban-focused; requires detailed 3D geometry |

### Relevance to This Project
- **Medium**: Conceptual reference for 3D-aware propagation modeling
- Ray-path feature aggregation concept informative for LaPS extension
- Too computationally expensive for direct application in dense forests
- Our approach (LaPS physics prior + UNetDCN residual) is more practical for forest scenarios
- Could inform future work on more sophisticated forest propagation models

---

## Paper 4: Spatial Optimization of the Multiple Coverage Mesh Network Problem for Multifunctional Smart Poles

### Summary
Addresses the multiple coverage mesh network problem (MC-MNP) for optimizing placement of smart poles that must simultaneously provide wireless coverage and other services. Uses spatial optimization with coverage constraints.

| Aspect | Details |
|--------|---------|
| **Research Task** | Optimal placement of communication nodes with multiple coverage objectives |
| **Input Data** | Demand points with coverage requirements, candidate node positions, propagation model, terrain/obstacle data |
| **Output Data** | Optimal node subset and placement coordinates |
| **Model Structure** | Mixed-integer linear programming (MILP) / heuristic optimization |
| **Mathematical Model** | `max sum(coverage) subject to: budget, connectivity, redundancy constraints` |
| **Loss Function** | N/A (combinatorial optimization) |
| **Optimization Objective** | Maximize weighted coverage while minimizing cost and ensuring connectivity |
| **Data Split** | N/A (optimization, not ML) |
| **Evaluation Metrics** | Coverage ratio, connectivity, cost, redundancy, number of nodes |
| **Reproducible Code** | Not publicly available |
| **Directly Applicable** | Multi-objective coverage formulation; candidate node scoring; connectivity constraints |
| **Not Directly Applicable** | Urban smart pole context; flat terrain assumption; fixed propagation model |

### Relevance to This Project
- **Medium**: Framework reference for candidate node evaluation (Stage 8) and graph construction (Stage 9)
- Coverage optimization formulation applicable to forest sensor network
- Connectivity and gateway reachability constraints directly relevant
- Must adapt for: 3D terrain, forest-specific deployment constraints, LoRa propagation characteristics

---

## Paper 5: Multi-objective Optimization for 3D Heterogeneous WSN Deployment

### Summary
Proposes multi-objective optimization methods for deploying wireless sensor networks in 3D environments, considering coverage, connectivity, energy efficiency, and deployment cost simultaneously.

| Aspect | Details |
|--------|---------|
| **Research Task** | 3D WSN deployment optimization with multiple objectives |
| **Input Data** | 3D terrain model, sensor specifications, coverage requirements, energy model |
| **Output Data** | Pareto-optimal sensor placements |
| **Model Structure** | Multi-objective evolutionary algorithm (NSGA-II or similar) |
| **Mathematical Model** | `minimize [f1(cost), f2(-coverage), f3(-connectivity), f4(energy)]` |
| **Loss Function** | N/A (evolutionary optimization) |
| **Optimization Objective** | Pareto front of trade-offs between coverage, connectivity, cost, energy |
| **Data Split** | N/A (optimization, not ML) |
| **Evaluation Metrics** | Hypervolume, Pareto front quality, coverage ratio, connectivity, network lifetime |
| **Reproducible Code** | Partially available |
| **Directly Applicable** | 3D deployment consideration; multi-objective framework; terrain-aware placement |
| **Not Directly Applicable** | Generic terrain (not forest-specific); simplified propagation; no ML-based radio maps |

### Relevance to This Project
- **Medium-Low**: Conceptual reference for post-candidate-node optimization (Stage 11 in full pipeline)
- Multi-objective optimization framework applicable to final deployment selection
- 3D terrain consideration directly relevant to forest hills
- For current scope (candidate node generation), our differentiable gradient-based approach is more appropriate
- May inform future work on final optimal deployment subset selection

---

## Cross-Paper Comparison Table

| Feature | LaPS | UNetDCN RadioMap | RayProNet | MC-MNP | 3D WSN Opt |
|---------|------|-----------------|-----------|--------|------------|
| **Environment** | Forest | Urban/suburban | Indoor/urban | Urban | Generic 3D |
| **Data type** | LiDAR point cloud | Aerial images | 3D geometry | Demand map | DEM/terrain |
| **Propagation model** | Physics (ray-path) | Learned (CNN) | Neural ray trace | Empirical | Empirical |
| **ML component** | No (regression) | Yes (UNetDCN) | Yes (neural fields) | No | No |
| **3D aware** | Yes | No (2D map) | Yes | Partial | Yes |
| **Forest specific** | Yes | No | No | No | No |
| **Handles uncertainty** | No | No | No | No | No |
| **Multiple objectives** | No | No | No | Yes | Yes |
| **Node placement** | Mentioned | Coverage opt | No | Core task | Core task |
| **Open source** | No | No | Partial | No | Partial |

## Integration in This Project

| Paper | Used In Stage | How Used |
|-------|--------------|----------|
| LaPS | Stage 5 | Extended path loss model with forest structure features |
| UNetDCN RadioMap | Stage 6 | Architecture basis for Forest-UNetDCN |
| RayProNet | Stage 5 (conceptual) | Ray-path feature aggregation concept |
| MC-MNP | Stage 8-9 | Coverage scoring and graph connectivity framework |
| 3D WSN Opt | Stage 8 (conceptual) | Multi-objective optimization reference |

## Key Innovations of This Project vs. Literature

1. **LaPS + deep learning hybrid**: No paper combines physics-based forest propagation prior with learned residual prediction
2. **Forest structure from point clouds**: No paper uses single-tree segmentation outputs as radio map input features
3. **Uncertainty-aware placement**: No paper uses predicted radio map uncertainty to penalize candidate node scores
4. **Differentiable node optimization**: Gradient-based candidate node refinement is novel for forest scenarios
5. **End-to-end pipeline**: No existing work connects forest point cloud segmentation → radio map → node placement
