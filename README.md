# Layer-Adaptive MX Quantization for RLHF on FPGA

This project asks one question:

Can RLHF matmuls use less FPGA energy if we switch from fixed `INT8` to adaptive `MXFP8` / `MXFP4`, without hurting model quality too much?

## Status

The project is in a good Milestone 4 state on the host side and a partial Milestone 4 state on the hardware side.

What is true today:

- `pytorch_profiling/` can profile layer sensitivity and generate policy JSON for policies `A/B/C/D`.
- `systemc/` has a working MX reference model with precision switching and group scaling.
- `integration/` has a phase-aware offload path, adaptive precision controller, Lab 1 bridge, and tested MX software fallback.
- `baseline_energy/rlhf_with_fpga.py` can run PPO with selective FPGA offload, save timing, and record per-phase FPGA usage.
- `baseline_energy/test_fpga_integration.py` now provides a fast smoke path and a tiny end-to-end RLHF smoke run.

What is not true yet:

- The checked-in FPGA compute path still does baseline arithmetic, not real `MXFP8` / `MXFP4` MACs.
- There is no checked-in proof of a deployed MX-capable AFI running real MX math on F2.
- Gradient-phase offload still falls back to native PyTorch because the current FPGA matmul path is not autograd-safe.
- The final canonical experiment table and Pareto plot are still missing.

## Checklist

- [x] Repo structure, configs, and test entry points are in place.
- [x] RLHF host training path exists and runs with selective FPGA offload.
- [x] MX precision control and policy selection are implemented in software.
- [x] Local and mock-FPGA smoke tests exist.
- [x] Per-phase FPGA statistics are saved for rollout, reward, and gradient.
- [ ] Replace baseline PE arithmetic with real MX arithmetic in the checked-in hardware path.
- [ ] Build and validate an MX-capable FPGA image on the Stanford/AWS flow.
- [ ] Run the final baseline vs `A/B/C/D` policy sweep on the real hardware path.
- [ ] Produce the final energy-vs-quality table, Pareto figure, and report conclusion.

## Next Steps

1. Finish the real MX datapath in [fpga/src/PECore/Datapath/Datapath.h](/Users/dannyadkins/CS217-Final-Project/fpga/src/PECore/Datapath/Datapath.h) and the connected PECore path so precision mode changes actual hardware math, not just control metadata.
2. Build and deploy an MX-capable bitstream through [fpga/README.md](/Users/dannyadkins/CS217-Final-Project/fpga/README.md), then validate outputs against the SystemC and Python MX reference paths.
3. Regenerate the final sensitivity matrix and policy JSON for the target model, then lock one canonical set of policies for experiments.
4. Run one clean experiment matrix: `INT8` baseline plus policies `A/B/C/D`, with one saved manifest, one canonical CSV, and one results directory structure.
5. Finish the report with the final figures: sensitivity heatmap, phase breakdown, energy-quality table, and Pareto curve.

## Fast Start

Local sanity checks:

```bash
make py-tests
make fpga-ref-sim
.venv/bin/python3 baseline_energy/test_fpga_integration.py
```

Tiny end-to-end smoke run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python3 baseline_energy/test_fpga_integration.py \
  --run-end-to-end \
  --steps 1 \
  --output results/milestone4_smoke_tiny \
  --model-name HuggingFaceH4/tiny-random-LlamaForCausalLM \
  --reward-model-name HuggingFaceH4/tiny-random-LlamaForCausalLM \
  --local-dataset-path baseline_energy/data/smoke_rlhf.jsonl \
  --num-samples 2 \
  --train-size 1 \
  --eval-size 1 \
  --batch-size 1 \
  --mini-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-seq-length 64 \
  --max-prompt-length 32 \
  --max-response-length 4 \
  --fpga-response-length 4 \
  --pretrain-reward-steps 1 \
  --policy-blocks 0 \
  --reward-policy-blocks 0
```

Stanford/AWS hardware flow:

```bash
make fpga-doctor
make fpga-systemc-sim
make fpga-hls-sim
make fpga-build
make fpga-program
make fpga-test
```

## Repo Map

- [baseline_energy](/Users/dannyadkins/CS217-Final-Project/baseline_energy)
  Host-side RLHF runs, smoke tests, timing, energy logging, and sweep orchestration.
- [integration](/Users/dannyadkins/CS217-Final-Project/integration)
  Adaptive controller, FPGA offload wrapper, Lab 1 bridge, and integration tests.
- [pytorch_profiling](/Users/dannyadkins/CS217-Final-Project/pytorch_profiling)
  Layer sensitivity profiling and policy generation.
- [systemc](/Users/dannyadkins/CS217-Final-Project/systemc)
  Clean MX reference implementation and local C++ testbench.
- [fpga](/Users/dannyadkins/CS217-Final-Project/fpga)
  Stanford/AWS F2 build and deploy path.
- [results](/Users/dannyadkins/CS217-Final-Project/results)
  Saved experiment outputs and report artifacts.
- [report](/Users/dannyadkins/CS217-Final-Project/report)
  LaTeX report drafts.

## Source Of Truth

Use this README as the top-level status page.

For component details:

- [baseline_energy/README.md](/Users/dannyadkins/CS217-Final-Project/baseline_energy/README.md)
- [integration/README.md](/Users/dannyadkins/CS217-Final-Project/integration/README.md)
- [pytorch_profiling/README.md](/Users/dannyadkins/CS217-Final-Project/pytorch_profiling/README.md)
- [fpga/README.md](/Users/dannyadkins/CS217-Final-Project/fpga/README.md)
- [docs/CURRENT_STATE_AND_SWEEP_PLAN.md](/Users/dannyadkins/CS217-Final-Project/docs/CURRENT_STATE_AND_SWEEP_PLAN.md)
