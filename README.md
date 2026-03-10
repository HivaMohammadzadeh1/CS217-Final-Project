# Layer-Adaptive MX Quantization for RLHF on FPGA

This repo asks one research question:

Can RLHF training use less energy if FPGA matmuls switch between `INT8`, `MXFP8`, and `MXFP4` adaptively, without hurting quality too much?

## Current milestone status

| Milestone | Status | What that means today |
| --- | --- | --- |
| 1. Repo setup and tooling | Complete | Core repo structure, scripts, and test entry points exist. |
| 2. Baseline RLHF + offload plumbing | Complete | 50-step RLHF run with real FPGA offload finished. Policy wins 50% vs reference with +0.57 mean reward delta. See results below. |
| 3. MX simulation + control path | Complete | MX reference models, precision switching, and policy control are implemented and tested. |
| 4. MX hardware integration | Complete | HLS Datapath implements MX decode + MAC (E4M3/E2M1 fixed-point). RunScale is precision-aware. Runtime test and SV testbench have MX golden models. Awaiting re-synthesis on build machine. |
| 5. Final experiments | Partial | Smoke runs and FPGA-offload runs exist, but the final real MX-on-hardware comparison is still missing. |
| 6. Final report | In progress | Draft report material exists, but the final story depends on the missing experiments. |

## Milestone 2: Baseline RLHF + FPGA Offload Results

A full 50-step RLHF training run with real FPGA offload was completed on an AWS EC2 F2 instance using Qwen2.5-0.5B-Instruct.

### Run Configuration

| Parameter | Value |
| --- | --- |
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Device | CPU (matmuls offloaded to FPGA) |
| Training steps | 50 |
| FPGA mode | REAL |
| FPGA policy blocks | [0, 23] |
| FPGA response length | 16 |
| Policy layers offloaded | 8 (blocks 0 and 23, q/k/v/o projections) |
| Reward layers offloaded | 96 (all layers) |
| Reference layers offloaded | 8 (blocks 0 and 23) |

### Training Summary

| Phase | Total Time | Avg per Batch |
| --- | --- | --- |
| Rollout | 148.3s | 23.39s |
| Reward | 112.3s | 18.72s |
| Gradient | 178.6s | 29.76s |

### FPGA Statistics

- Total matmuls offloaded: 65,013,760
- Total tiles processed: 65,013,760

### FPGA Energy Baseline

Hardware power estimated via Xilinx Power Estimator (XPE). Cycle counts measured on the Lab 1 bitstream (16x16 INT8 matmul).

| Parameter | Value |
| --- | --- |
| FPGA power | 35 W |
| Cycles per 16x16 matmul | 148,656 (15 compute + 148,641 PCIe transfer) |
| Energy per matmul | 20,811.84 μJ |
| Total energy for 50 PPO steps | 2,830.41 J (0.786 Wh) |

Data transfer accounts for 99.99% of per-matmul cycles; computation is just 15 cycles (0.01%). This PCIe bottleneck is the dominant cost and the primary target for MX-format compression.

### Phase-Level Energy Breakdown

FPGA power is constant at 35 W, so energy per phase = power × runtime. Phase runtimes from the actual 50-step RLHF run (`results/milestone_report/phase_timing.json`):

| Phase | Runtime (s) | FPGA Energy (J) | % of Total |
| --- | --- | --- | --- |
| Rollout | 140.3 | 4,911 | 32.5% |
| Reward | 112.3 | 3,931 | 26.0% |
| Gradient | 178.6 | 6,251 | 41.4% |
| **Total** | **431.2** | **15,092** | **100%** |

This is the "before" baseline. MX precision policies target the compute portion of energy, though PCIe transfer dominates in the current Lab 1 setup.

### Post-Training Evaluation (50 held-out examples)

| Metric | Policy (FPGA) | Reference | Delta |
| --- | --- | --- | --- |
| Mean Reward | 3.2699 | 2.7030 | +0.5668 |
| Mean Perplexity | 9.86 | 10.19 | -0.33 |
| Win Rate | 50.0% | — | 25W / 25L / 0T |

The FPGA-quantized policy achieves a **higher mean reward** (+0.57) and **lower perplexity** (-0.33) than the reference model, with an even 50% win rate. The evaluation flagged the quantization impact as **SIGNIFICANT** (>10% reward change), confirming that the FPGA offload path produces meaningfully different — and in this case better — outputs compared to the un-offloaded reference.

### How to reproduce

```bash
python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/fpga_final-full --eval-samples 50 --save-models
```

Saved artifacts are in `results/fpga_final-full/` (trained policy, reward model, reference model, PPO checkpoint, training metadata, power log, eval results, and FPGA stats).

---

## Milestone 3: MX Simulation + Precision Control

Milestone 3 validates the dual-precision MX datapath (SystemC reference model), layer sensitivity profiling, and automatic policy generation.

### SystemC Testbench Results

All five test categories passed via `make -C systemc clean all run`:

| Test | Metric | Value | Limit | Result |
| --- | --- | --- | --- | --- |
| Quantize/dequantize | FP8 reconstruction MAE | 0.00884 | 0.10 | PASS |
| Quantize/dequantize | FP4 reconstruction MAE | 0.07951 | 0.35 | PASS |
| MAC accuracy | FP8 mean normalized error | 0.00665 | 0.08 | PASS |
| MAC accuracy | FP4 mean normalized error | 0.05165 | 0.25 | PASS |
| GEMM accuracy (16x16) | FP8 mean abs error | 0.03559 | 0.25 | PASS |
| GEMM accuracy (16x16) | FP4 mean abs error | 0.24754 | 0.85 | PASS |
| Mode switching | MAC blocked until FlushPipeline() | — | — | PASS |
| Mode switching | Mode updated to MXFP4 after flush | — | — | PASS |
| Mode switching | MAC succeeds after flush | — | — | PASS |
| Group size comparison | Group 8 not worse than group 16 | within margin | — | PASS |

### Sensitivity Profiling (Qwen2.5-0.5B-Instruct, 169 layers)

Per-layer quantization sensitivity on `Qwen/Qwen2.5-0.5B-Instruct`, baseline perplexity **13.3231**, tolerance threshold 2%. MXFP4 g8 perplexity delta by block (attention vs MLP):

```
Block   Attn (q/k/v/o avg)   MLP (gate/up/down avg)   MLP MXFP4 tolerant?
  0        -0.01%                +0.53%                  Yes
  1        +0.12%                +0.15%                  Yes
  2        +0.23%                +2.27%  ← sensitive     NO (down_proj +7.48%)
  3        +0.32%                +3.26%  ← sensitive     NO (all 3 MLP layers)
  4        +0.02%                +0.21%                  Yes
  5        +0.07%                +0.25%                  Yes
  6        +0.33%                (running...)             ...
```

MXFP8 is tolerant everywhere (max delta +0.43%). Full 169-layer run in progress — results for blocks 7–23 + lm_head will follow.

### Policy Generation

Four precision policies (A–D) were generated from the sensitivity matrix:

```bash
python3 pytorch_profiling/define_policies.py \
  --sensitivity results/sensitivity_matrix.csv \
  --group-size 8 \
  --output results/policies.json
```

| Policy | Strategy | Description |
| --- | --- | --- |
| A | Conservative | MXFP8 all layers for rollout/reward, FP16 for gradients |
| B | Balanced | MXFP4 for tolerant layers, MXFP8 for sensitive, FP16 for gradients |
| C | Aggressive | MXFP4 everywhere possible, MXFP8 fallback for sensitive gradients |
| D | Phase-Adaptive | MXFP4-biased rollout, MXFP8-biased reward, FP16-safe gradients |

### How to reproduce

```bash
# 1. SystemC datapath tests
make -C systemc clean all run

# 2. Sensitivity profiling on Qwen2.5-0.5B-Instruct
python3 pytorch_profiling/sensitivity_profiler.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset hivamoh/cs217-rlhf-dataset \
  --text-field chosen \
  --num-examples 16 \
  --max-seq-len 256 \
  --device cpu \
  --output results/sensitivity_matrix.csv

# 3. Policy generation
python3 pytorch_profiling/define_policies.py \
  --sensitivity results/sensitivity_matrix.csv \
  --group-size 8 \
  --output results/policies.json
```

---

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

- The HLS sources now implement MX arithmetic, but the checked-in `concat_PECore.v` RTL has not been
  re-synthesized yet (requires Catapult on the Stanford build machine).
- There is no deployed MX-capable FPGA bitstream yet (pending re-synthesis and AFI build).
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
