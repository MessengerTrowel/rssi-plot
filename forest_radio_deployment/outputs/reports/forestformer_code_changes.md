# ForestFormer3D Code Changes Log

This document tracks all modifications made to the ForestFormer3D codebase for the forest radio deployment project.

## Source

- Original: `external/ForestFormer3D/` (commit `6a75c3735e4a4108d02ee944a8b93177f2360a4f`)
- Modified copy: `src/forestformer_adapted/` (created when modifications begin)

## Change Policy

1. Original repository (`external/ForestFormer3D/`) must never be modified
2. All changes are made in `src/forestformer_adapted/`
3. Each change must be documented here with: file, function, what changed, why

## Changes

No changes have been made yet. Modifications will begin after:

1. GPU environment is available
2. Official pretrained model inference is verified
3. Baicaowa data format is confirmed
4. Adaptation strategy is approved

## Planned Modifications (Pending Approval)

| File | Change | Purpose | Priority |
|------|--------|---------|----------|
| `data/ForAINetV2/load_forainetv2_data.py` | Add LAZ/LAS support | Baicaowa data may be in LAZ format | Medium |
| `data/ForAINetV2/batch_load_ForAINetV2_data.py` | Remove torch/segmentator imports for CPU preprocessing | Enable CPU-only data preparation | Low |
| `configs/` | Add baicaowa-specific config | Domain adaptation settings | High |
| `Dockerfile` | Adjust CUDA arch for target GPU | Match available hardware | High |
