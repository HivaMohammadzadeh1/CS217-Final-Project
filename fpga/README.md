# FPGA Build Path

This directory is the project's hardware build and deployment path.

Use it for:
- PECore SystemC simulation on the Stanford build machines
- Catapult HLS / RTL generation
- AWS F2 hardware simulation
- AFI generation, programming, and runtime testing

Use `/Users/dannyadkins/CS217-Final-Project/systemc` as the portable MX reference model. That directory defines the intended MX behavior. This `fpga/` directory is how that behavior gets carried into the hardware build flow that targets F2.

## One entry point

```bash
python3 fpga/run_fpga_flow.py doctor
```

That command reports:
- resolved repo paths
- required environment variables
- installed tools
- which stages belong on the local machine, Stanford build machine, and F2 runtime host

## Core commands

Local reference validation:

```bash
python3 fpga/run_fpga_flow.py reference-sim
```

Stanford build machine:

```bash
python3 fpga/run_fpga_flow.py systemc-sim
python3 fpga/run_fpga_flow.py hls-sim
python3 fpga/run_fpga_flow.py hw-sim
python3 fpga/run_fpga_flow.py fpga-build
python3 fpga/run_fpga_flow.py generate-afi
python3 fpga/run_fpga_flow.py check-afi
```

F2 runtime host:

```bash
python3 fpga/run_fpga_flow.py program-fpga
python3 fpga/run_fpga_flow.py run-fpga-test --slot-id 0
```

To push staged MX control bits through the runtime path:

```bash
python3 fpga/run_fpga_flow.py run-fpga-test --slot-id 0 --fpga-test-args "MXFP8 16"
```

## What is already true

- `fpga/Makefile` now points at the checked-in `fpga/hls` directory.
- The runner sets `REPO_TOP`, `AWS_HOME`, `CL_DIR`, and `CL_DESIGN_NAME` automatically.
- `fpga/design_top/Makefile` accepts:
  - `RTL_VARIANT=...`
  - `SLOT_ID=...`
  - `FPGA_TEST_ARGS="..."`
- PEConfig now carries precision mode and MX group size through the existing hardware control path.
- The runtime test binary can program those fields from CLI args.

## What is now implemented

- `Datapath.h` contains `DecodeE4M3Fixed`, `DecodeE2M1Fixed`, and `ProductSumMX` functions
  that perform MX minifloat decode and MAC in HLS-synthesizable fixed-point arithmetic.
- `PECore.h` `RunScale` is precision-aware: INT8 divides by 12.25, MX modes pass through directly.
- `design_top.c` runtime test has an MX-aware golden model for hardware verification.
- `design_top_base_test.sv` writes precision mode into PEConfig and verifies against an MX golden model.

## HLS Synthesis Log

Adding MX precision support (MXFP4/MXFP8) to the PECore datapath required re-synthesizing the RTL via Catapult HLS. This proved to be the hardest part of the project. Below is a detailed record of each attempt, what failed, and what we learned.

### Attempt 1 — Produced INT8-only RTL

- **Date:** ~March 12-14, 2026
- **Duration:** ~24-48 hours (completed successfully)
- **Config:** CLK_PERIOD=4.0ns, default CLOCK_OVERHEAD=20%
- **Result:** Synthesis completed and produced `concat_PECore.v`, but the RTL contained **only the INT8 (ProductSum) datapath**. The MX paths (ProductSumMX, DecodeE4M3Fixed, DecodeE2M1Fixed) were entirely absent from the generated Verilog.
- **Root cause (diagnosed later):**
  1. `DecodeE4M3Fixed` used conditional variable-direction barrel shifts (`if (exp >= 6) mant << (exp-6); else mant >> (6-exp)`) which Catapult could not synthesize correctly — it silently dropped the MX datapath.
  2. `DecodeE2M1Fixed` also used a variable left shift (`full_mant << shift_l`), same anti-pattern.
  3. The HLS testbench (`testbench.cpp`) only exercised `precision_mode=1` (MXFP8), never `precision_mode=2` (MXFP4). Catapult may have constant-propagated the MXFP4 path away.
- **Evidence:** `grep ProductSum concat_PECore.v` showed only `ProductSum_for_acc_` signals — no `ProductSumMX` anywhere in 70K lines of RTL.
- **FPGA validation on this RTL:**
  - INT8 mode: **PASSED** on F2 hardware at 0.112% average error (16 lanes)
  - MXFP4 mode: **FAILED** — PEConfig readback showed `precision_mode=0` (hardware ignored the write). Computed values were INT8-scale (~10K-15K) vs MXFP4-expected (~-79 to 47). Error: 91,495%.

### Attempt 2 — SCHD-3 failure (ArbitratedScratchpadDP)

- **Date:** ~March 14-15, 2026
- **Duration:** ~8 hours before failure
- **Config:** CLK_PERIOD=4.0ns, default CLOCK_OVERHEAD=20%
- **Changes from Attempt 1:** Fixed `DecodeE4M3Fixed` with explicit 16-case switch statement (hardcoded shifts per exponent value, no variable barrel shifts).
- **Result:** SCHD-3 error — "Feedback path is too long to schedule design with current pipeline and clock constraints."
- **Root cause:** The `ArbitratedScratchpadDP` arbiter's feedback path exceeded the 4ns clock period minus 20% overhead (effective 3.2ns). Catapult's own SCHD-22 warning suggested setting `CLOCK_OVERHEAD 0`.
- **Fix applied:** Added `directive set -CLOCK_OVERHEAD 0` to `go_hls.tcl` via `usercmd_post_assembly` hook.

### Attempt 3 — SCHD-3 failure (weight_mem read crossbar)

- **Date:** ~March 15-17, 2026
- **Duration:** 12.5 hours in architect/allocate phase before failure
- **Config:** CLK_PERIOD=4.0ns, CLOCK_OVERHEAD=0%
- **Changes from Attempt 2:** Added `CLOCK_OVERHEAD 0` directive.
- **Result:** SCHD-3 again, but in a **different** arbiter — `weight_mem.read_arbxbar` (the 16-port Round-robin read crossbar). The `leading_ones<31U>` priority encoder chain created a combinational feedback loop that couldn't close at 4ns even with 0% overhead.
- **Root cause:** The 16-way Round-robin arbiter's MUX chain + AND gates + state feedback exceeded 4ns total combinational delay.
- **Peak memory:** 20.3 GB
- **Fix applied:** Relaxed `CLK_PERIOD` from 4.0ns to 5.0ns in `fpga/hls/PECore/Makefile`. The FPGA target is 125MHz (8ns), so 200MHz (5ns) HLS target is safe.

### Attempt 4 — In progress

- **Date:** March 17, 2026
- **Config:** CLK_PERIOD=5.0ns, CLOCK_OVERHEAD=0%
- **All changes applied:**
  1. `DecodeE4M3Fixed`: explicit 16-case switch statement (no variable shifts)
  2. `DecodeE2M1Fixed`: explicit 4-case switch statement (no variable shifts) — **this was the same anti-pattern as the E4M3 bug, caught in code review**
  3. `testbench.cpp`: added MXFP4 PEConfig write before MXFP8 write, exercising `precision_mode=2` to prevent Catapult from constant-propagating away the MXFP4 path
  4. `CLK_PERIOD`: 4.0ns → 5.0ns (25% more timing slack for arbiter feedback)
  5. `CLOCK_OVERHEAD`: 20% → 0% (full clock period available for combinational paths)
- **Status:** Running on Farmshare (7-day large instance, 32GB RAM)
- **Estimated completion:** 10-20 hours

### Key lessons learned

1. **Variable shifts are poison for Catapult HLS.** Any `x << variable` or `x >> variable` should be replaced with explicit switch/case statements using constant shift amounts. Catapult either miscompiles them or silently drops the containing function.

2. **Testbench coverage matters for synthesis.** If a register-driven branch is never exercised in the HLS testbench, Catapult may constant-propagate it away. All precision modes (INT8, MXFP8, MXFP4) must be written to PEConfig during the testbench run.

3. **16-port Round-robin arbiters are timing-critical.** The `ArbitratedScratchpadDP` with 16 banks and 16 read ports creates deep combinational feedback chains that struggle to meet tight clock constraints. Relaxing the HLS clock target (while staying above the FPGA implementation clock) gives the scheduler room to close timing.

4. **CLOCK_OVERHEAD default is 20%.** This silently steals 20% of your clock period. Setting it to 0 via `directive set -CLOCK_OVERHEAD 0` is essential for timing-critical designs.

5. **Synthesis takes 12-48 hours per attempt.** The feedback loop for debugging is brutal. Bottom-up synthesis (synthesizing sub-blocks separately) would dramatically reduce iteration time but requires refactoring PECore into multiple SC_MODULEs.

### Open items for TA discussion

- Professor recommended bottom-up synthesis — PECore is currently a single flat SC_MODULE. How to refactor for hierarchical synthesis?
- Can ArbitratedScratchpadDP arbiter feedback be pipelined via Catapult directives without modifying MatchLib source?
- Are there known workarounds for 16-port Round-robin arbiter timing in Catapult?

## What still needs to happen

- Complete HLS synthesis (Attempt 4 in progress) to produce MX-capable `concat_PECore.v`.
- Push new RTL to variant directory and rebuild FPGA bitstream (`make fpga_build`).
- Generate and deploy MX-capable AFI.
- Validate MXFP4 and MXFP8 on F2 hardware via `make run_fpga_test FPGA_TEST_ARGS="MXFP4 8"`.
- Run the real policy experiments against deployed MX hardware.
