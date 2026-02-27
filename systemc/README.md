# Milestone 3: Dual-Precision MX Datapath (Simulation)

This folder now contains a complete Milestone 3 simulation package:

- MXFP8 datapath (E4M3)
- MXFP4 datapath (E2M1)
- Group scaling (group size 8 or 16)
- Dual-precision mode switching with explicit pipeline flush
- Runnable testbench with clear pass/fail output
- Optional Vivado HLS wrapper for C simulation and synthesis setup

The code is written to be easy to read first, then optimize later.

## What this gives you right now

You can answer these Milestone 3 questions with code:

- "Can we quantize with MXFP8 and MXFP4 with shared scaling?" -> yes
- "Can we switch precision safely at runtime?" -> yes (flush required)
- "Do quantized MAC/GEMM results match float closely enough?" -> yes, tested

## File map

- `mx_types.h`
  Common types, precision enums, and MX format specs.
- `group_scaler.h`
  Shared-exponent quantization/dequantization logic.
- `mxfp_pe.h`
  Processing element implementations for MXFP8 and MXFP4.
- `mx_datapath.h`
  Dual-precision datapath with mode request + flush safety model.
- `testbench.cpp`
  End-to-end validation for quantization, MAC, GEMM, and mode switching.
- `mx_datapath_hls.cpp`
  Fixed-size top function (`mx_datapath_top`) for Vivado HLS flow.
- `testbench_hls.cpp`
  Small HLS C-sim testbench for the top function.
- `run_simulation.tcl`
  Vivado HLS script for csim + synthesis.
- `resource_estimator.py`
  Parse HLS reports (or use defaults) and print a DSP/LUT/BRAM table.
- `Makefile`
  Local compile/run for the software testbench.

## Quick start (local simulation)

From repo root:

```bash
cd systemc
make
make run
```

You should see a summary ending with:

```text
ALL CHECKS PASSED
```

## How group scaling is modeled

For each group (size 8 or 16):

1. Find `max_abs` in the group.
2. Compute one shared exponent: `floor(log2(max_abs))`.
3. Scale each value by that shared exponent.
4. Encode scaled values into minifloat (E4M3 or E2M1).
5. Decode back and rescale for MAC.

This follows the layer/group adaptive microscaling idea in a simple, explicit way.

## How mode switching is modeled

`DualPrecisionMXDatapath` uses a safe two-step mode switch:

1. `request_mode(new_mode)`
2. `flush_pipeline()`

If you request a mode change and call `mac()` or `gemm()` before flush, the datapath throws an error. This models the "flush between mode switches" requirement from Milestone 3.

## Vivado HLS flow (optional)

If Vivado HLS is available:

```bash
cd systemc
vivado_hls -f run_simulation.tcl
```

Reports are typically written under:

- `mx_datapath_hls/solution1/syn/report/`

Then summarize resources:

```bash
python3 resource_estimator.py \
  --mxfp8-report mx_datapath_hls/solution1/syn/report/mx_datapath_top_csynth.rpt \
  --mxfp4-report mx_datapath_hls/solution1/syn/report/mx_datapath_top_csynth.rpt
```

If you do not have reports yet, `resource_estimator.py` can print conservative defaults.

## Notes for integration (Milestone 4)

- Keep this simulation code as golden reference behavior.
- In HLS RTL, preserve the same mode-switch contract:
  request mode -> flush pipeline -> resume MAC.
- Connect mode to AXI-lite register later in FPGA integration.
