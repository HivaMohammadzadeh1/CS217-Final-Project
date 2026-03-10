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
- The RLHF runtime path does not use this MX hardware yet; on real Lab1 runs it still uses `INT8` hardware plus MX software fallback until validation is complete.

## What still needs to happen

- Validate and refine the checked-in MX datapath in Catapult/F2.
- Build and deploy an MX-capable AFI.
- Switch the RLHF runtime path to use validated MX hardware.
- Run the real policy experiments against that deployed hardware.
