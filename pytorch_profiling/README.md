# PyTorch Profiling

This directory contains PyTorch-based quantization sensitivity experiments.

## Purpose

Determine which layers and RLHF phases can tolerate MXFP4 vs MXFP8 quantization without significant quality degradation.

## Key Deliverables

1. **Layer Sensitivity Matrix**: Per-layer perplexity delta for MXFP4/MXFP8
2. **Group Size Analysis**: Compare group_size=8 vs group_size=16
3. **Phase Sensitivity**: Quantization impact on rollout/reward/gradient phases
4. **Policy Definitions**: Define policies A/B/C/D based on sensitivity results

## Scripts (to be added)

- `sensitivity_profiler.py`: Per-layer quantization testing
- `phase_analysis.py`: Phase-level quantization impact
- `policy_definitions.py`: Define MX format policies
- `utils.py`: Helper functions for MX format conversion

## Dependencies

```bash
pip install torch transformers datasets mx-pytorch
```

## Usage

```bash
# Run layer sensitivity profiling
python sensitivity_profiler.py --model Qwen/Qwen2.5-0.5B-Instruct --output results/sensitivity_matrix.csv

# Analyze phase-level impact
python phase_analysis.py --policy balanced --output results/phase_analysis.csv
```
