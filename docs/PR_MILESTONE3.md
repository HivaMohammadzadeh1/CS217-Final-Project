# PR: Milestone 3 - Dual-Precision MX Datapath Simulation (MXFP4/MXFP8)

## Why this PR exists

Milestone 3 requires a working dual-precision MX datapath design that can be tested before FPGA deployment.

This PR adds a complete simulation implementation in `systemc/` with:

- MXFP8 (E4M3) and MXFP4 (E2M1) support
- Group scaling (group size 8 or 16)
- Mode switching safety (flush required between precision changes)
- A runnable testbench with clear pass/fail output
- A clear README in plain language

## What changed

### New core implementation files

- `systemc/mx_types.h`
  - Precision mode enum and format specs.
- `systemc/group_scaler.h`
  - Shared exponent computation and quantize/dequantize logic.
- `systemc/mxfp_pe.h`
  - Processing element wrappers for MXFP8 and MXFP4 MAC.
- `systemc/mx_datapath.h`
  - Dual-precision datapath with:
    - `RequestMode(...)`
    - `FlushPipeline()`
    - MAC and GEMM execution
    - safety checks that block compute if flush is pending.

### New test + build files

- `systemc/testbench.cpp`
  - End-to-end validation:
    - quantization reconstruction quality
    - MAC accuracy
    - 16x16 GEMM accuracy
    - mode switching + pipeline flush correctness
    - group-size trend check (8 vs 16).
- `systemc/Makefile`
  - `make`, `make run`, `make clean`.

### New HLS-support files

- `systemc/mx_datapath_hls.cpp`
  - fixed-size top function `mx_datapath_top` for HLS flow.
- `systemc/testbench_hls.cpp`
  - small C-sim smoke test for HLS top.
- `systemc/run_simulation.tcl`
  - Vivado HLS script for `csim_design` and `csynth_design`.
- `systemc/resource_estimator.py`
  - report parser/default estimator for DSP/LUT/BRAM summary.

### Documentation

- Rewrote `systemc/README.md` in clear human language:
  - what is implemented now
  - exact run commands
  - how group scaling works
  - how mode-switch flush works
  - optional Vivado HLS flow
  - Milestone 4 handoff notes.

## Validation run

Executed locally:

```bash
make -C systemc clean
make -C systemc
make -C systemc run
```

Result: `ALL CHECKS PASSED`

Also executed:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic systemc/mx_datapath_hls.cpp systemc/testbench_hls.cpp -o /tmp/mx_hls_tb
/tmp/mx_hls_tb
```

Result: `HLS C-sim smoke test passed.`

## Important design decisions

- Prioritized clarity and correctness over early micro-optimization.
- Enforced explicit flush on mode switch to match hardware safety behavior.
- Kept group-size validation strict (`8` or `16`) to prevent silent invalid configs.
- Added HLS wrapper separately so simulation model stays easy to read.

## What this PR does not do

- Does not integrate adaptive policy controller yet.
- Does not connect to AXI-lite registers yet.
- Does not provide real FPGA bitstream deployment (Milestone 4+).

## Suggested follow-up PRs

1. Integrate this datapath model into `integration/` precision policy flow.
2. Add `AdaptiveController` and policy-driven precision per layer/phase.
3. Add AXI-lite mode register plumbing for real hardware mode switching.
