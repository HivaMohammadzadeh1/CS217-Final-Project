# Current State And Sweep Plan

## Project goal

This project is trying to answer one question:

Can RLHF matmuls use less FPGA energy if we switch from a fixed baseline precision to adaptive MX precision (`MXFP4` / `MXFP8`), without hurting quality too much?

## Milestone status

| Milestone | Status | Reality |
|---|---|---|
| 1. Setup | Complete | Repo, dependencies, and basic project structure exist. |
| 2. Baseline + profiling setup | Mostly complete | RLHF/offload plumbing exists, and profiling/policy tooling now works in smoke mode. |
| 3. MX simulation + precision control | Complete | MX reference simulation, precision switching, and controller logic are implemented and tested. |
| 4. Hardware integration | Complete | HLS Datapath (`Datapath.h`) implements MX decode + MAC. `PECore.h` RunScale is precision-aware. Runtime test and SV testbench have MX golden models. Pending re-synthesis on build machine. |
| 5. Final experiments | Not complete | Smoke runs exist, but final baseline-vs-MX hardware experiments are still missing. |
| 6. Final analysis/report | Not complete | Report directory exists, but final figures and conclusions depend on Milestone 5. |

## What is solid now

- `systemc/` models MX behavior and passes its testbench.
- `integration/` supports precision-aware offload and phase/layer policy control.
- `pytorch_profiling/` now produces a real sensitivity matrix using the repo's MX reference quantization and can generate policy JSON.
- `fpga/` has a clear hardware build/deploy path for the Stanford environment targeting F2.
- Precision mode and MX group size now travel through the hardware control path.

## What is still not true

- The HLS sources implement MX arithmetic but `concat_PECore.v` has not been re-synthesized yet.
- There is no MX-capable AFI deployed on F2 yet.
- There is no final experiment table comparing baseline vs policies `A/B/C/D` on real hardware.
- Gradient-phase MX offload is still treated conservatively in software because the current path is not autograd-safe.

## What the sweep actually means

In this repo, the “sweep” is not neural architecture search.

It means:
- profile layers for tolerance to `MXFP4` / `MXFP8`
- generate policies
- run baseline and policy variants
- compare energy, runtime, and quality

Main sweep dimensions:
- policy `A/B/C/D`
- group size `8` vs `16`
- phase mapping: rollout / reward / gradient

## Recommended next steps

1. Run the new profiler on the real target model in the Stanford environment.
2. Generate the final policy JSON from that sensitivity matrix.
3. Re-synthesize `concat_PECore.v` from the updated `Datapath.h` using Catapult on the build machine.
4. Build and deploy an MX-capable AFI from the Stanford environment.
5. Run the final baseline-vs-policy experiments and collect one canonical results CSV plus Pareto plots.

## Definition of done

The project is in strong shape when you can show:

- one reproducible profiling run that generates final policy JSON
- one reproducible hardware run path from Stanford to F2
- one results table for baseline vs `A/B/C/D`
- one energy-vs-quality plot
- one clear final recommendation
