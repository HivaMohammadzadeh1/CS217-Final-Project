# Layer-Adaptive MX Quantization for RLHF on FPGA

This repo is trying to answer one question:

Can RLHF training spend less FPGA energy if matrix multiplies use adaptive MX precision (`MXFP4` / `MXFP8`) instead of a fixed baseline precision, without hurting quality too much?

## What the repo actually contains

- [`/Users/dannyadkins/CS217-Final-Project/systemc`](/Users/dannyadkins/CS217-Final-Project/systemc)
  Portable MX reference model. This is the clean software definition of the MX behavior you want in hardware.
- [`/Users/dannyadkins/CS217-Final-Project/fpga`](/Users/dannyadkins/CS217-Final-Project/fpga)
  Hardware build/deploy path for the Stanford machines targeting AWS F2.
- [`/Users/dannyadkins/CS217-Final-Project/integration`](/Users/dannyadkins/CS217-Final-Project/integration)
  Python offload layer between RLHF code and the FPGA path.
- [`/Users/dannyadkins/CS217-Final-Project/baseline_energy`](/Users/dannyadkins/CS217-Final-Project/baseline_energy)
  RLHF runner, sweep runner, logging, and baseline experiment code.
- [`/Users/dannyadkins/CS217-Final-Project/pytorch_profiling`](/Users/dannyadkins/CS217-Final-Project/pytorch_profiling)
  Layer sensitivity profiling and policy generation.
- [`/Users/dannyadkins/CS217-Final-Project/results`](/Users/dannyadkins/CS217-Final-Project/results)
  Experiment outputs and smoke-run artifacts.

## Current architecture

The intended system is:

1. Host runs the RLHF loop.
2. Expensive matmuls are offloaded through the Python integration layer.
3. The FPGA executes those tiled matmuls.
4. A controller decides per phase / per layer which precision to use.
5. Results are compared across policies for energy vs quality.

Today, that breaks down like this:

- `systemc/` already models real MX behavior and passes local tests.
- `integration/` already supports precision-aware offload and policy control.
- `fpga/` now has a cleaned hardware build/deploy path and carries MX control bits through PEConfig.
- The checked-in PECore compute path is still baseline integer MAC arithmetic.

So the repo now has the right control path, but not yet the final MX compute datapath in deployed hardware.

## One command surface

Use the top-level `Makefile`:

```bash
make help
make fpga-doctor
make fpga-ref-sim
make py-tests
```

Or call the FPGA runner directly:

```bash
python3 fpga/run_fpga_flow.py doctor
```

## Where commands run

- Local machine:
  - `make fpga-ref-sim`
  - `make py-tests`
- Stanford build machine:
  - `make fpga-systemc-sim`
  - `make fpga-hls-sim`
  - `make fpga-hw-sim`
  - `make fpga-build`
- F2 runtime host:
  - `make fpga-program`
  - `make fpga-test`

In practice, the Stanford environment is the place you drive this from. It uses the AWS FPGA toolchain and targets F2 hardware.

## Current status

What is in good shape:

- MX reference simulation
- precision-aware Python offload layer
- policy sweep scaffolding
- hardware build/deploy entry points
- MX precision/group-size control carried into the hardware register path

What is still missing for final results:

1. Replace the baseline integer PECore arithmetic with real MX arithmetic.
2. Build and deploy an MX-capable AFI from the Stanford environment.
3. Run the real baseline vs MX policy experiments on hardware.
4. Finish layer sensitivity profiling and use those results to generate the final policies.

## Practical next commands

Local:

```bash
make fpga-doctor
make fpga-ref-sim
make py-tests
```

Stanford machine:

```bash
make fpga-systemc-sim
make fpga-hls-sim
make fpga-build
```

F2 runtime:

```bash
make fpga-program
make fpga-test
```
