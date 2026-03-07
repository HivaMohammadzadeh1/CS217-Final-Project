# CS217 Project: Current State and Sweep Plan

## 1) What this project is

You are building a **research prototype** that asks:

> Can we reduce RLHF training energy on FPGA by using MX precision formats (MXFP4/MXFP8) layer-adaptively, compared to an INT8 baseline?

System split:
- Host CPU runs RLHF loop logic.
- FPGA/offload path handles GEMMs (matmuls).
- Comparison target is energy/quality tradeoff across precision policies.

## 2) What is done now

### Milestone 2 (mostly done)
- Baseline RLHF + FPGA plumbing exists.
- INT8/Lab1 integration path exists (with software fallback).
- Policy definition tooling exists in `pytorch_profiling/define_policies.py`.

### Milestone 3 (now mergeable)
- SystemC dual-precision MX simulation exists and passes its testbench.
- Integration precision-control APIs exist:
  - `configure_precision(mode, group_size, flush=...)`
  - `flush_pipeline()`
- Precision state-machine bugs were fixed:
  - flush contract enforced across INT8↔MX transitions,
  - group-size reconfigure now preserves requested mode,
  - pending-state reporting is behavior-consistent.
- Regression tests for these edge cases were added.

## 3) What is *not* done yet

- No deployed MX FPGA bitstream connected to real AXI precision register writes yet.
- No adaptive per-layer/per-phase controller wired into live RLHF matmul calls yet.
- No full end-to-end policy comparison table (A/B/C/D) with quality metrics and Pareto curve yet.
- PyTorch sensitivity profiler still uses placeholder quantization logic in `sensitivity_profiler.py`.

## 4) What teammates mean by “architecture search/sweep”

In this project, that likely means **policy/precision sweep**, not neural-architecture search.

Sweep dimensions:
- Precision policy: A/B/C/D (and variants).
- Phase mapping: rollout/reward/gradient precision choices.
- Group size: 8 vs 16.
- (Optional later) hardware knobs: tileing/batching strategy and transfer amortization.

Outputs per run:
- `energy_J`, `runtime_s`
- quality metrics (win-rate proxy / KL / reward correlation)
- aggregate tradeoff points for Pareto plot.

## 5) Immediate next steps (recommended order)

1. Replace placeholder quantization in `pytorch_profiling/sensitivity_profiler.py` with real MX library path (or clearly mark proxy mode in outputs).
2. Generate sensitivity matrix CSV from real runs.
3. Generate policy JSON via `define_policies.py`.
4. Implement/adapt `AdaptiveController` to call `configure_precision(...)` before offloaded GEMMs.
5. Run fixed benchmark across INT8 + policies A/B/C/D and write one canonical results CSV.
6. Produce Pareto curve and phase breakdown for report.

## 6) Definition of done for “next checkpoint”

Checkpoint is strong when you can show:
- One command/script that runs policy sweep reproducibly.
- A single CSV with all policy points.
- A plot showing energy vs quality tradeoff.
- A clear recommendation: conservative/balanced/aggressive/phase-adaptive winner.
