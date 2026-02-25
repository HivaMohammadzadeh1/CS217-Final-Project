# SystemC MX Datapath

This directory contains SystemC/HLS designs for the dual-precision MX datapath.

## Purpose

Design and simulate MXFP4/MXFP8 processing elements (PEs) before FPGA synthesis.

## Key Components

1. **MXFP8 Datapath**: E4M3 format (4 exponent, 3 mantissa bits)
2. **MXFP4 Datapath**: E2M1 format (2 exponent, 1 mantissa bit)
3. **Dual-Precision Controller**: Mode switching between FP4/FP8
4. **Group Scaling Logic**: Shared scale factor computation

## Design Files (to be added)

- `mx_datapath.h`: Main dual-precision PE design
- `mxfp8_pe.h`: MXFP8 processing element
- `mxfp4_pe.h`: MXFP4 processing element
- `group_scaler.h`: Group scaling logic
- `testbench.cpp`: SystemC testbench
- `run_simulation.tcl`: Vivado HLS simulation script

## Build Instructions

```bash
# Run SystemC simulation
vivado_hls -f run_simulation.tcl

# Check synthesis report
cat solution1/syn/report/mx_datapath_csynth.rpt
```

## Resource Estimates

Target resource usage on Xilinx VU9P:

| Component | DSPs | LUTs | BRAMs |
|-----------|------|------|-------|
| MXFP8 PE  | ~8   | ~500 | 2     |
| MXFP4 PE  | ~4   | ~300 | 2     |
| Controller| 0    | ~100 | 0     |

## Verification

Testbench validates:
1. Functional correctness vs PyTorch MX reference
2. Mode switching without state corruption
3. Pipeline throughput (1 MAC per cycle target)
