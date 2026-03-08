# Layer-Adaptive MX Quantization for RLHF on FPGA

This repo asks one research question:

Can RLHF training use less energy if FPGA matmuls switch between `INT8`, `MXFP8`, and `MXFP4` adaptively, without hurting quality too much?

## Current milestone status

| Milestone | Status | What that means today |
| --- | --- | --- |
| 1. Repo setup and tooling | Complete | Core repo structure, scripts, and test entry points exist. |
| 2. Baseline RLHF + offload plumbing | Mostly complete | RLHF training, evaluation, timing, and FPGA hook-up code all exist. |
| 3. MX simulation + control path | Complete | MX reference models, precision switching, and policy control are implemented and tested. |
| 4. MX hardware integration | In progress | The hardware control path carries precision settings, but the checked-in RTL compute path is still baseline arithmetic. |
| 5. Final experiments | Partial | Smoke runs and FPGA-offload runs exist, but the final real MX-on-hardware comparison is still missing. |
| 6. Final report | In progress | Draft report material exists, but the final story depends on the missing experiments. |

## What is implemented

- `systemc/`
  Portable dual-precision MX reference model with group scaling and flush-on-mode-switch behavior.
- `integration/`
  Python matmul tiling/offload layer, Lab 1 bridge, adaptive precision controller, and MX software fallback path.
- `baseline_energy/`
  RLHF training scripts, timing/energy logging, evaluation, and policy sweep runner.
- `pytorch_profiling/`
  Layer sensitivity profiler and A/B/C/D policy generator.
- `fpga/`
  Stanford/AWS F2 build/deploy flow wrapper plus runtime control-path plumbing for precision mode and group size.
- `results/`
  Saved smoke runs and several FPGA-offload experiment directories, including a 50-step run in `results/fpga_final-full/`.

## What is not finished yet

- The checked-in hardware datapath still performs the baseline integer MAC.
  The precision bits are carried into PEConfig, but they do not yet change the actual RTL arithmetic.
- There is no checked-in proof of a deployed MX-capable FPGA bitstream doing real `MXFP8` / `MXFP4` math.
- On the Python "real FPGA" path, MX modes still fall back to software unless true MX hardware is available.
- Gradient-phase FPGA offload is not fully autograd-safe, so gradients default to native PyTorch/`FP16`.
- The final canonical policy sweep and Pareto-style energy/quality comparison table are still missing.

## Repo map

- [baseline_energy](/Users/dannyadkins/CS217-Final-Project/baseline_energy)
  Host-side RLHF experiments, configs, timing, evaluation, and sweep orchestration.
- [integration](/Users/dannyadkins/CS217-Final-Project/integration)
  FPGA matmul offload wrapper, Lab 1 interface, adaptive controller, and integration tests.
- [pytorch_profiling](/Users/dannyadkins/CS217-Final-Project/pytorch_profiling)
  Sensitivity profiling and policy generation.
- [systemc](/Users/dannyadkins/CS217-Final-Project/systemc)
  Clean MX reference implementation and local C++ testbench.
- [fpga](/Users/dannyadkins/CS217-Final-Project/fpga)
  Hardware build/deploy path for the Stanford/AWS environment.
- [results](/Users/dannyadkins/CS217-Final-Project/results)
  Saved experiment outputs and smoke artifacts.
- [docs](/Users/dannyadkins/CS217-Final-Project/docs)
  Short project-state notes and report support material.
- [report](/Users/dannyadkins/CS217-Final-Project/report)
  LaTeX report drafts.

## Command entry points

Top-level `Makefile`:

```bash
make help
make fpga-doctor
make fpga-ref-sim
make py-tests
```

Useful direct commands:

```bash
python3 fpga/run_fpga_flow.py doctor
make -C systemc run
```

Where they run:

- Local machine
  - `make fpga-ref-sim`
  - `make py-tests`
  - `make -C systemc run`
- Stanford build machine
  - `make fpga-systemc-sim`
  - `make fpga-hls-sim`
  - `make fpga-hw-sim`
  - `make fpga-build`
- F2 runtime host
  - `make fpga-program`
  - `make fpga-test`

## Verified locally

- `make py-tests`
- `make -C systemc clean all run`

Both passed in the current repo state.

## Best docs to read next

- Current state and missing work:
  [docs/CURRENT_STATE_AND_SWEEP_PLAN.md](/Users/dannyadkins/CS217-Final-Project/docs/CURRENT_STATE_AND_SWEEP_PLAN.md)
- Host-side RLHF path:
  [baseline_energy/README.md](/Users/dannyadkins/CS217-Final-Project/baseline_energy/README.md)
- Python integration layer:
  [integration/README.md](/Users/dannyadkins/CS217-Final-Project/integration/README.md)
- Profiling and policy generation:
  [pytorch_profiling/README.md](/Users/dannyadkins/CS217-Final-Project/pytorch_profiling/README.md)
- FPGA build/deploy path:
  [fpga/README.md](/Users/dannyadkins/CS217-Final-Project/fpga/README.md)

## Documentation note

Several older top-level milestone/setup notes are still kept in the repo, but they have been shortened and treated as historical snapshots. The main source of truth is this README plus the component READMEs above.
