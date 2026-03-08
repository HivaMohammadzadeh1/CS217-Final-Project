# Testing Status

**Date**: March 8, 2026  
**Status**: Current core paths are tested; final hardware experiments are still pending.

## What has been verified

### 1. MX reference simulation

- `systemc/` reference simulation runs locally
- quantize/dequantize checks pass
- MAC and GEMM accuracy checks pass
- mode-switch safety checks pass

### 2. Precision-aware offload path

- `integration.test_mx_offload_integration` passes
- flush behavior is enforced correctly
- mode/group-size metadata stays consistent
- real-interface fallback behavior is consistent

### 3. FPGA runner / build entry points

- `fpga.test_run_fpga_flow` passes
- top-level `make fpga-doctor` works
- top-level `make fpga-ref-sim` works
- FPGA command surfaces are now consistent and documented

### 4. Profiling and policy generation

- `pytorch_profiling.test_sensitivity_profiler` passes
- `pytorch_profiling.test_policy_generation` passes
- smoke profiling run completed on a tiny cached model
- smoke policy generation completed from the resulting sensitivity matrix

## What this means

The following are now in good shape:

- MX reference model
- precision-control software path
- hardware build/deploy entry points
- profiling-to-policy workflow

## What has not been fully verified yet

- full target-model profiling on the final experiment model
- MX arithmetic deployed in the hardware compute path
- AFI generation and runtime execution on F2 for the MX design
- final baseline-vs-policy experiment table and plots

## Practical interpretation

The repo is ready for:

- local software validation
- profiling smoke runs
- Stanford-environment hardware build steps

The repo is not yet finished for:

- real MX-on-FPGA experiment results

## Commands that currently pass locally

```bash
make fpga-doctor
make fpga-ref-sim
.venv/bin/python -m unittest \
  pytorch_profiling.test_sensitivity_profiler \
  pytorch_profiling.test_policy_generation \
  integration.test_mx_offload_integration \
  fpga.test_run_fpga_flow -v
```
