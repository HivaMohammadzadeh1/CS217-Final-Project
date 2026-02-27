# PR: Milestone 3 Completion - MX Datapath + Integration Hardening

## Why this PR exists

Milestone 3 needs two things to be practically useful:

1. A correct dual-precision MX datapath model.
2. A clear integration contract so RLHF/offload code can control precision safely.

This PR delivers both.

It includes:

- MXFP8 (E4M3) and MXFP4 (E2M1) support
- Group scaling (group size 8 or 16)
- Explicit mode switching safety (flush required between precision changes)
- Integration APIs (`configure_precision`, `flush_pipeline`) in offload path
- Deterministic unit/integration tests
- Clear human-language docs for exactly what works now vs what is hardware-next

## What changed

### A) SystemC simulation implementation (from Milestone 3 build-out)

- `systemc/mx_types.h`
  - Precision modes and minifloat specs.
- `systemc/group_scaler.h`
  - Shared exponent quantize/dequantize logic.
- `systemc/mxfp_pe.h`
  - MXFP8/MXFP4 PE wrappers.
- `systemc/mx_datapath.h`
  - Dual-precision datapath with safe mode switch semantics.
- `systemc/testbench.cpp`
  - End-to-end C++ testbench for quantization/MAC/GEMM/mode switching.
- `systemc/Makefile`
  - Local build/run.
- `systemc/mx_datapath_hls.cpp`
  - HLS top function.
- `systemc/testbench_hls.cpp`
  - HLS C-sim smoke test.
- `systemc/run_simulation.tcl`
  - Vivado HLS csim/csynth script.
- `systemc/resource_estimator.py`
  - Resource summary helper.
- `systemc/README.md`
  - Rewritten in plain language with exact run flow and guarantees.

### B) Integration hardening added in this pass

- `integration/mx_precision_sim.py`
  - Deterministic Python reference model for MXFP8/MXFP4 + group scaling + flush contract.
- `integration/fpga_matmul_offload.py`
  - Added precision-aware APIs:
    - `configure_precision(mode, group_size, flush=...)`
    - `flush_pipeline()`
  - Supports `INT8`, `MXFP8`, `MXFP4`.
  - Safe switch behavior enforced in mock and fallback paths.
- `integration/lab1_fpga_interface.py`
  - Added compatible precision control state + flush semantics for integration consistency.
  - Keeps Lab1 INT8 behavior unchanged.
- `baseline_energy/config.py`
  - Added:
    - `FPGA_PRECISION_MODE`
    - `FPGA_GROUP_SIZE`
- `baseline_energy/config_lab1_fpga.py`
  - Added same precision config knobs.
- `baseline_energy/rlhf_with_fpga.py`
  - Passes precision/group config into `FPGAMatmulOffload(...)`.
- `integration/README.md`
  - Rewritten to describe concrete current integration behavior.

### C) New tests

- `integration/test_mx_precision_sim.py`
  - Unit tests for minifloat encoding, quantization quality, and mode-switch safety.
- `integration/test_mx_offload_integration.py`
  - Integration tests for offload precision API, flush requirement, and stats metadata.
- `integration/test_mx_pytorch_optional.py`
  - Optional parity check (skips cleanly if mx-pytorch/microxcaling is unavailable).

## Validation run

Executed and passing:

```bash
make -C systemc clean
make -C systemc
make -C systemc run

python3 -m unittest integration.test_mx_precision_sim integration.test_mx_offload_integration integration.test_mx_pytorch_optional -v
python3 -m py_compile integration/*.py baseline_energy/*.py
```

Result:
- SystemC testbench: `ALL CHECKS PASSED`
- Python tests: all pass, optional mx-pytorch parity test cleanly skipped when package is absent.

## Important design decisions

- Keep a single safe precision contract everywhere:
  - request/switch precision
  - flush pipeline
  - compute
- Preserve old INT8 behavior by default so baseline runs stay stable.
- Make MX behavior deterministic in software now, so hardware comparison later is straightforward.
- Keep unsupported external dependencies optional (mx-pytorch parity check is skip-safe).

## What is complete vs pending

### Complete in this PR
- Milestone 3 simulation datapath and safety semantics.
- Integration-level precision control APIs and tests.
- RLHF config plumbing for selecting precision mode.

### Still hardware-next
- True AXI-lite register write to an MX-capable FPGA bitstream.
- End-to-end adaptive controller that switches precision per layer/phase.
- Hardware-vs-software parity against deployed MX RTL.

## Suggested follow-up PRs

1. Add `AdaptiveController` and use it to call `configure_precision(...)` per layer/phase.
2. Hook AXI-lite mode register writes in real MX hardware path.
3. Add strict parity tests against deployed MX bitstream outputs.
