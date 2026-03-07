# FPGA Flow In This Repo

This directory is the copied Stanford Lab 1 FPGA flow, not a placeholder.

Use it for:
- PECore SystemC simulation on the Stanford HLS machines
- Catapult HLS / RTL generation
- AWS F2 hardware simulation
- AFI generation, programming, and runtime test

Use the standalone `/Users/dannyadkins/CS217-Final-Project/systemc` directory as the portable MX reference model. That reference runs locally on a normal machine. The `fpga/` flow is the lab-style shell and deployment infrastructure.

## One entry point

Run everything through:

```bash
python3 fpga/run_lab_flow.py doctor
```

That command prints:
- resolved repo paths
- required environment variables
- which tools are installed on the current machine
- which stages are realistic locally vs on Stanford/AWS

## Typical commands

Local reference validation:

```bash
python3 fpga/run_lab_flow.py reference-sim
```

Stanford HLS machine:

```bash
python3 fpga/run_lab_flow.py systemc-sim
python3 fpga/run_lab_flow.py hls-sim
```

AWS F2 instance:

```bash
python3 fpga/run_lab_flow.py hw-sim
python3 fpga/run_lab_flow.py fpga-build
python3 fpga/run_lab_flow.py generate-afi
python3 fpga/run_lab_flow.py check-afi
python3 fpga/run_lab_flow.py program-fpga
python3 fpga/run_lab_flow.py run-fpga-test --slot-id 0
```

## Important details

- The top-level `fpga/Makefile` now points HLS at `/Users/dannyadkins/CS217-Final-Project/fpga/hls`, which matches the checked-in repo layout.
- The runner sets `REPO_TOP`, `AWS_HOME`, `CL_DIR`, and `CL_DESIGN_NAME` automatically so the copied lab makefiles do not depend on fragile shell state.
- `fpga/design_top/Makefile` now accepts:
  - `RTL_VARIANT=...`
  - `SLOT_ID=...`
  - `FPGA_TEST_ARGS="..."`

## Current project status

What is real today:
- the lab-derived PECore/AWS shell flow in `fpga/`
- the standalone MX reference model in `systemc/`
- the Python RLHF/controller/offload infrastructure elsewhere in the repo

What still requires FPGA-side implementation work:
- wiring the MX arithmetic path into the PECore hardware that the AWS shell instantiates
- deploying that MX-capable PECore as a new AFI

So the correct mental model is:
- `systemc/` answers "what should MX do?"
- `fpga/` answers "how do we build/program/run hardware on the Stanford/AWS stack?"
