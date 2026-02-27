# Integration Layer (Milestone 3 Completion)

This directory now includes a clear precision-control contract between the
RLHF host code and the FPGA/offload path.

## What is implemented now

### 1) Precision-aware offload API

`fpga_matmul_offload.py` now supports:

- `INT8` mode (existing baseline behavior)
- `MXFP8` mode (software MX simulator)
- `MXFP4` mode (software MX simulator)
- explicit mode switching:
  - `configure_precision(mode, group_size, flush=...)`
  - `flush_pipeline()`

This gives a clean control path for later adaptive policy integration.

### 2) Deterministic MX simulator

`mx_precision_sim.py` provides a software reference model:

- E4M3 and E2M1 encode/decode
- shared group scaling (group size 8 or 16)
- 16x16 tile matmul
- strict "switch pending until flush" behavior

### 3) Lab1 interface compatibility

`lab1_fpga_interface.py` now exposes:

- `configure_precision(...)`
- `flush_pipeline()`

Current Lab1 hardware remains INT8. MX modes are accepted for API
compatibility and handled by software fallback until an MX bitstream exists.

## Test coverage added

### Unit tests (math + safety)

```bash
python -m unittest integration.test_mx_precision_sim -v
```

Covers:
- known minifloat code points for simple values
- MXFP8 vs MXFP4 expected accuracy ordering
- mode switch requires flush
- tile-level error bounds

### Integration tests (offload API behavior)

```bash
python -m unittest integration.test_mx_offload_integration -v
```

Covers:
- INT8 path exactness
- mode switch pending error until flush
- precision metadata in stats
- real-interface fallback behavior when MX hardware is unavailable

## How RLHF picks precision now

In `baseline_energy/config.py`:

- `FPGA_PRECISION_MODE = "INT8" | "MXFP8" | "MXFP4"`
- `FPGA_GROUP_SIZE = 8 | 16`

`rlhf_with_fpga.py` passes these into `FPGAMatmulOffload(...)`.

This gives one consistent knob for experiments while keeping code simple.

## What remains for full hardware integration

- AXI-lite mode register write to actual MX datapath in RTL
- bitstream supporting MXFP8/MXFP4 in hardware (instead of software fallback)
- adaptive per-layer/per-phase controller on top of this API
