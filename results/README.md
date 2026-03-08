# Results

This directory stores saved experiment artifacts from the repo.

## What is actually here now

- `fpga_run1/`, `fpga_run2/`, `fpga_final/`, `fpga_final-full/`
  Saved FPGA-offload training runs with timing, stats, and some evaluation outputs.
- `policy_sweep*`
  Sweep manifests and smoke-policy runs.
- `profiling_smoke_v3/`
  Smoke sensitivity matrix and generated policy JSON.
- `milestone_report/`
  Early Lab 1 measurement and estimate files.

## What these results mean

- They show that the software stack and selective FPGA-offload path were exercised.
- They do not yet prove a final end-to-end MX hardware result.
- The saved runs are useful as smoke artifacts and report support, not as the final answer to the research question.

## Typical files

- `training_meta.json`
  Run configuration and saved artifact list.
- `training_stats.json`
  PPO metrics collected during training.
- `phase_timing.json`
  Rollout/reward/gradient timing.
- `fpga_stats.json`
  Matmul and tile counts seen by the offload layer.
- `eval_results.json`
  Held-out comparison between the trained policy and the reference model when evaluation was run.

## Source scripts

- Training:
  [/Users/dannyadkins/CS217-Final-Project/baseline_energy/rlhf_with_fpga.py](/Users/dannyadkins/CS217-Final-Project/baseline_energy/rlhf_with_fpga.py)
- Sweeps:
  [/Users/dannyadkins/CS217-Final-Project/baseline_energy/run_policy_sweep.py](/Users/dannyadkins/CS217-Final-Project/baseline_energy/run_policy_sweep.py)
- Profiling:
  [/Users/dannyadkins/CS217-Final-Project/pytorch_profiling/sensitivity_profiler.py](/Users/dannyadkins/CS217-Final-Project/pytorch_profiling/sensitivity_profiler.py)
