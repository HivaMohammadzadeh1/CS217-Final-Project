# Experimentation & Implementation Log

## Overview

This document tracks the end-to-end process of adding MX precision support (MXFP4/MXFP8) to the PECore accelerator — from HLS synthesis through RTL simulation to FPGA deployment. The project required 10 synthesis attempts over 6 days to produce working MX-capable RTL, followed by hardware validation across all three precision modes.

---

## Part 1: HLS Synthesis — The Road to MX-Capable RTL

### Background

The baseline PECore design supported only INT8 precision and had been previously synthesized and validated on AWS F2 FPGA hardware. Adding MX support required new minifloat decode logic (`DecodeE4M3Fixed` for MXFP8, `DecodeE2M1Fixed` for MXFP4), a new `ProductSumMX` multiply-accumulate path, and precision-mode-aware scaling in `RunScale`. Each synthesis attempt took 1-48 hours on Stanford Farmshare compute nodes using Synopsys Catapult HLS.

### Attempt 1 — Completed, but produced INT8-only RTL

- **Duration:** ~24-48 hours (completed)
- **Config:** CLK_PERIOD=4.0ns, CLOCK_OVERHEAD=20%, II=1, 16 read ports, 16 banks
- **Result:** Generated `concat_PECore.v` (70K lines), but RTL contained **only INT8 datapath**. Zero references to `ProductSumMX`, `DecodeE4M3Fixed`, or `DecodeE2M1Fixed`.
- **Root causes:**
  1. `DecodeE4M3Fixed` used conditional variable-direction barrel shifts (`if (exp >= 6) mant << (exp-6); else mant >> (6-exp)`). Catapult silently dropped the MX datapath.
  2. `DecodeE2M1Fixed` used a variable left shift (`full_mant << shift_l`) — same anti-pattern.
  3. HLS testbench only exercised `precision_mode=1` (MXFP8), never `precision_mode=2` (MXFP4). Catapult constant-propagated the MXFP4 path away.
- **FPGA results on this RTL:** INT8 passed at 0.112% error. MXFP4 failed at 91,495% error (hardware always computed INT8 regardless of precision_mode setting).

### Attempt 2 — SCHD-3 (ArbitratedScratchpadDP arbiter timing)

- **Duration:** ~8 hours before failure
- **Changes:** Fixed `DecodeE4M3Fixed` with explicit 16-case switch statement.
- **Result:** `SCHD-3: Feedback path is too long` — arbiter feedback in ArbitratedScratchpadDP exceeded effective clock period (4.0ns - 20% overhead = 3.2ns).

### Attempt 3 — SCHD-3 (weight_mem read crossbar — different arbiter)

- **Duration:** 12.5 hours before failure
- **Changes:** Added `directive set -CLOCK_OVERHEAD 0`.
- **Result:** SCHD-3 in a different arbiter (`weight_mem.read_arbxbar`). 16-port Round-robin `leading_ones<31U>` priority encoder chain couldn't close at 4ns. Peak memory: 20.3 GB.

### Attempt 4 — Directive stage restriction error

- **Duration:** ~6.5 hours (assembly completed, failed on directive)
- **Changes:** Relaxed CLK_PERIOD to 5.0ns, added REGISTER_THRESHOLD, MEM_MAP_THRESHOLD, DESIGN_GOAL, COMPGRADE directives.
- **Result:** `Error: Directive 'CLOCK_PERIOD' must be specified no later than state 'libraries'` — CLOCK_PERIOD cannot be set in `usercmd_post_assembly`.

### Attempt 5 — Invalid directive enumeration

- **Duration:** Minutes
- **Changes:** Moved directives to `usercmd_pre_compile`.
- **Result:** `Error: not a valid enumeration value` — `COMPGRADE slow` was invalid.

### Attempt 6 — TCL directive error after successful assembly

- **Duration:** 6.5 hours (assembly completed successfully)
- **Changes:** Reduced weight memory from 16 to 4 read ports, serialized MAC reads (4 batches of 4 lanes), added pipeline init interval via TCL directive.
- **Result:** Assembly completed with only 1.9 GB memory (vs 20.3 GB with 16 ports — 10x reduction). Failed on `directive set /PECore/PECoreRun -PIPELINE_INIT_INTERVAL 4` — invalid path in `usercmd_post_assembly`.
- **Key insight:** The source pragma `#pragma hls_pipeline_init_interval 4` handles II without needing a TCL directive.

### Attempt 7 — Switched to `grant-branch-interval4` (based on main)

- **Duration:** ~15 hours (reached allocate, then failed)
- **Changes:** Fresh start from `main` branch (which already had 4 read ports + serialized MAC from teammate). Added only: II 2→4, E4M3/E2M1 switch statements, MXFP4 testbench exercise.
- **Result:** Past schedule without SCHD-3 (timing solved!), but hit SCHD-4 in allocate: `insufficient resources 'Xilinx_RAMS.BLOCK_1R1W_RBW'` — 256 BRAM instances needed, only 1 available. The 16-bank write arbiter's routing arrays were being mapped to BRAM instead of registers.

### Attempt 8 — SCHD-4 with 4 banks

- **Duration:** ~1 hour
- **Changes:** Reduced `kNumBanks` from 16 to 4, increased `kEntriesPerBank` from 4096 to 16384.
- **Result:** Same SCHD-4 but needing 64 instances instead of 256. `MEM_MAP_THRESHOLD=32` still mapping arbiter internals to BRAM. Also, 16384 entries/bank may exceed Xilinx BRAM primitive sizes.

### Attempt 9 — Reverted to Lab 4 scratchpad sizes + raised MEM_MAP_THRESHOLD

- **Duration:** ~8 hours (assembly + architect completed)
- **Changes:** Reverted to 16 banks / 4096 entries (Lab 4 proven config). Raised `MEM_MAP_THRESHOLD` from 32 to 2048 so arbiter routing stays as registers.
- **Result:** Passed assembly, architect, and allocate. Failed during SCVerify (post-synthesis verification) due to compiler version mismatch (`g++ 10.3.0` not supported by VCS). **But synthesis itself completed — RTL was generated.**

### Attempt 10 — SUCCESS

- **Branch:** `grant-branch-interval4`
- **Final config:** CLK_PERIOD=4.0ns, CLOCK_OVERHEAD=20%, II=4, 4 read ports, 16 banks, 4096 entries/bank, REGISTER_THRESHOLD=256, MEM_MAP_THRESHOLD=2048
- **Changes from main (4 total):**
  1. Pipeline init interval: 2→4 (pragma)
  2. DecodeE4M3Fixed: variable barrel shift → 16-case switch statement
  3. DecodeE2M1Fixed: variable left shift → 4-case switch statement
  4. Testbench: added MXFP4 PEConfig write before MXFP8 (exercises all precision modes)
- **Result:** `concat_PECore.v` generated (6.2 MB). Grep confirmed **43,420 references** to MX datapath logic (`ProductSumMX`, `DecodeE4M3`, `DecodeE2M1`). Compare to attempt 1 which had zero.

### Key Lessons from Synthesis

1. **Variable shifts are unsynthesizable in Catapult HLS.** Any `x << variable` or `x >> variable` must be replaced with explicit switch/case statements. Catapult silently drops functions containing variable barrel shifts.
2. **Testbench coverage drives synthesis inclusion.** All precision modes must be exercised in the HLS testbench, or Catapult may constant-propagate unused branches away.
3. **16-port Round-robin arbiters create unsolvable timing.** Reducing to 4 read ports (with serialized MAC) cut memory from 20GB to 2.5GB and eliminated SCHD-3.
4. **MEM_MAP_THRESHOLD controls arbiter BRAM usage.** Setting it too low (32) maps small arbiter routing arrays to BRAM, exhausting available instances (SCHD-4). Setting to 2048 keeps these as registers.
5. **Use proven scratchpad sizes.** Lab 4's config (16 banks, 4096 entries) maps correctly to Xilinx BRAM primitives.
6. **Pipeline initiation interval matters.** II=4 gives the scheduler flexibility to pipeline feedback paths across multiple cycles.
7. **Catapult directive stages are strict.** `CLOCK_PERIOD` must be set before `libraries`. Use source pragmas instead of TCL directives for `PIPELINE_INIT_INTERVAL`.
8. **Synthesis iteration time is the bottleneck.** 1-48 hours per attempt with no intermediate feedback. The 2-day turnaround per failure made iterative debugging extremely costly.

---

## Part 2: RTL Simulation (hw_sim) — Hardware Verification

### Phase 1: MXFP8 Verification

#### First Run — Float Golden Model (2.597% average error)

After generating MX-capable RTL, we ran RTL simulation (`make hw_sim`) targeting MXFP8.

**Configuration:**
- Precision mode: MXFP8 (E4M3)
- Group size: 8
- 16 vector lanes, 16-element dot product per lane

**Results:**

| Lane | Computed | Expected (float) | Error |
|------|----------|-------------------|-------|
| 0 | -84730 | -82824 | 2.301% |
| 1 | 58989 | 58989 | 0.000% |
| 2 | 4971 | 4965 | 0.121% |
| 3 | 10377 | 10376 | 0.010% |
| 4 | -1398 | -1401 | 0.214% |
| 5 | 228 | 243 | 6.173% |
| 6 | 1196 | 1221 | 2.048% |
| 7 | -1864 | -1853 | 0.594% |
| 8 | -664 | -664 | 0.000% |
| 9 | 13331 | 13331 | 0.000% |
| 10 | -10026 | -13388 | **25.112%** |
| 11 | -3632 | -3653 | 0.575% |
| 12 | -15258 | -15261 | 0.020% |
| 13 | -121137 | -126414 | 4.174% |
| 14 | -90458 | -90465 | 0.008% |
| 15 | -6313 | -6300 | 0.206% |

**Average error: 2.597%** — exceeded the SV testbench threshold of 2.0%.

**Diagnosis:** The hardware output was correct. The error came from a **golden model mismatch** in the SystemVerilog testbench. The SV golden model used floating-point (`real`) arithmetic:

```systemverilog
// Old golden model (float-based)
wf = decode_minifloat_real(...);   // returns 'real' (double-precision float)
inf = decode_minifloat_real(...);
acc_mx += wf * inf;                // float multiply + accumulate
scaled = acc_mx;                   // float -> int
```

But the hardware computes in **fixed-point integer** arithmetic:

```cpp
// Hardware (Datapath.h)
NVINTW(16) w_val = DecodeMXByte(...);  // 16-bit fixed-point, 4 fractional bits
NVINTW(16) i_val = DecodeMXByte(...);
acc += (NVINTW(32))w_val * (NVINTW(32))i_val;  // 32-bit integer accumulate
out = (spec::AccumScalarType)(acc >> 8);         // remove 8 fractional bits
```

The float golden model preserves full precision through intermediate computations, while the hardware truncates via right-shifts (especially for small exponents where `full_mant >> 5` discards most bits). These truncation differences compound across 16 multiply-accumulate operations.

**Key insight:** The 2.597% average error represents the **quantization cost of fixed-point MX arithmetic** vs. ideal floating-point — useful data for the paper's accuracy analysis, but NOT a hardware bug.

#### Golden Model Fix

We replaced the SV testbench golden model with fixed-point integer decode functions matching the hardware exactly:

```systemverilog
// New golden model (fixed-point, matching hardware)
function automatic longint decode_e4m3_fixed(input [7:0] code);
    case (exp_f)
       0: abs_val = 0;
       1: abs_val = full_mant >> 5;
       2: abs_val = full_mant >> 4;
       // ... (all 16 cases with constant shifts)
      14: abs_val = full_mant << 8;
      default: abs_val = (mant == 7) ? 0 : full_mant << 9;
    endcase
    return sign_bit ? -abs_val : abs_val;
endfunction

// Accumulate as integers, arithmetic right-shift by 8
acc_fx += w_val * i_val;
scaled = $itor(acc_fx >>> 8);
```

This tests the correct question: "Does the synthesized RTL compute exactly what the SystemC source code specifies?" — which is the actual purpose of hw_sim (verifying Catapult's synthesis correctness).

#### Second Run — Fixed-Point Golden Model (0.000% error)

**Results:**

| Lane | Computed | Expected (fixed-point) | Error |
|------|----------|------------------------|-------|
| 0 | -84730 | -84730 | 0.000% |
| 1 | 58989 | 58989 | 0.000% |
| 2 | 4971 | 4971 | 0.000% |
| 3 | 10377 | 10377 | 0.000% |
| 4 | -1398 | -1398 | 0.000% |
| 5 | 228 | 228 | 0.000% |
| 6 | 1196 | 1196 | 0.000% |
| 7 | -1864 | -1864 | 0.000% |
| 8 | -664 | -664 | 0.000% |
| 9 | 13331 | 13331 | 0.000% |
| 10 | -10026 | -10026 | 0.000% |
| 11 | -3632 | -3632 | 0.000% |
| 12 | -15258 | -15258 | 0.000% |
| 13 | -121137 | -121137 | 0.000% |
| 14 | -90458 | -90458 | 0.000% |
| 15 | -6313 | -6313 | 0.000% |

**Average error: 0.000%. TEST PASSED.**

The synthesized RTL computes bit-identical results to the SystemC source code. Catapult's synthesis is verified correct for MXFP8.

---

### Phase 2: MXFP4 Verification

**Configuration:** Precision mode: MXFP4 (E2M1), Group size: 8

**Results:**

| Lane | Computed | Expected | Error |
|------|----------|----------|-------|
| 0 | 20 | 20 | 0.000% |
| 1 | 22 | 22 | 0.000% |
| 2 | 28 | 28 | 0.000% |
| 3 | -76 | -76 | 0.000% |
| 4 | -5 | -5 | 0.000% |
| 5 | 23 | 23 | 0.000% |
| 6 | -25 | -25 | 0.000% |
| 7 | 17 | 17 | 0.000% |
| 8 | 27 | 27 | 0.000% |
| 9 | 6 | 6 | 0.000% |
| 10 | 11 | 11 | 0.000% |
| 11 | 27 | 27 | 0.000% |
| 12 | -12 | -12 | 0.000% |
| 13 | 7 | 7 | 0.000% |
| 14 | -13 | -13 | 0.000% |
| 15 | 14 | 14 | 0.000% |

**Average error: 0.000%. TEST PASSED.**

MXFP4 values range from -76 to 28 vs MXFP8's -121,137 to 58,989 — reflecting E2M1's much lower dynamic range (max value 6.0) compared to E4M3 (max value 448.0). Compute cycles: 41 (same throughput as MXFP8).

---

### Phase 3: INT8 Regression

**Configuration:** Precision mode: INT8, Group size: 8

**Results:**

| Lane | Computed | Expected | Error |
|------|----------|----------|-------|
| 0 | 36858 | 36899 | 0.111% |
| 1 | 27072 | 27102 | 0.111% |
| 2 | 31472 | 31508 | 0.114% |
| 3 | 21472 | 21497 | 0.116% |
| 4 | 35868 | 35908 | 0.111% |
| 5 | 26598 | 26628 | 0.113% |
| 6 | 25023 | 25051 | 0.112% |
| 7 | 29020 | 29053 | 0.114% |
| 8 | 18146 | 18167 | 0.116% |
| 9 | 36853 | 36894 | 0.111% |
| 10 | 24061 | 24088 | 0.112% |
| 11 | 31926 | 31962 | 0.113% |
| 12 | 19204 | 19225 | 0.109% |
| 13 | 31344 | 31379 | 0.112% |
| 14 | 30835 | 30870 | 0.113% |
| 15 | 32230 | 32266 | 0.112% |

**Average error: 0.112%. TEST PASSED.**

INT8 values are in the 18K-37K range (unsigned 8-bit products accumulated across 16 elements, then scaled by ÷12.25). The consistent 0.11% error across all lanes comes from the integer approximation of the ÷12.25 scale factor (implemented as `×167 >> 11` in hardware). INT8 mode is verified to work identically to the original pre-MX design.

Compute cycles: 41 (same as MX modes — the serialized 4-port MAC adds cycles equally regardless of precision mode).

---

## Part 3: FPGA Build & Deployment

**Status:** Pending — all three precision modes validated in RTL simulation.

Steps:
1. `make fpga_build` — Vivado synthesis (~3-4 hours)
2. `make generate_afi` — upload DCP to AWS, register AFI (~30 min)
3. `make check_afi_available` — poll until ready
4. `make program_fpga` — load AFI onto F2 FPGA (~1 min)
5. `make run_fpga_test FPGA_TEST_ARGS="MXFP8 8"` — MXFP8 hardware test (tolerance: 8%)
6. `make run_fpga_test FPGA_TEST_ARGS="MXFP4 8"` — MXFP4 hardware test (tolerance: 25%)
7. `make run_fpga_test` — INT8 regression on hardware (tolerance: 2%)

---

## Summary: Quantization Accuracy

### Hardware fixed-point vs. ideal floating-point (from first hw_sim run)

| Metric | MXFP8 | MXFP4 | INT8 |
|--------|-------|-------|------|
| Average error vs. float | 2.597% | TBD | N/A (exact integer) |
| Max per-lane error | 25.112% | TBD | N/A |
| Compute cycles | 41 | 41 | 41 |
| Data transfer cycles | 1471 | 1471 | 1471 |

### RTL vs. SystemC source (synthesis correctness)

| Mode | Average error | Result |
|------|--------------|--------|
| MXFP8 | 0.000% | PASSED |
| MXFP4 | 0.000% | PASSED |
| INT8 | 0.112% | PASSED |

The 0.112% INT8 error is from the ÷12.25 approximation (`×167 >> 11`), not a synthesis bug. This matches the original pre-MX design behavior.

These results confirm that Catapult HLS correctly synthesized all three precision paths, and the fixed-point quantization error is within the tolerance bounds used by the adaptive precision policy engine (Milestone 3).
