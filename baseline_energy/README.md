# Baseline Energy And RLHF Driver

This directory contains the host-side experiment code for the project.

## What is here

- `rlhf_baseline.py`
  Baseline PPO RLHF training path with timing and optional GPU power logging.
- `rlhf_with_fpga.py`
  RLHF path with selective FPGA matmul offload, evaluation, and saved run metadata.
- `run_policy_sweep.py`
  Runs one experiment per precision policy and records a sweep manifest.
- `test_fpga_integration.py`
  Milestone 4 smoke test for precision switching, tiled matmul, and optional end-to-end RLHF execution.
- `calculate_energy.py` and `process_energy.py`
  Convert timing/power logs into energy summaries.
- `config.py`
  Shared experiment configuration.

## Current status

- The RLHF and evaluation scripts are real and have saved outputs under `results/`.
- FPGA offload is selective rather than full-model.
- GPU power monitoring is best-effort and skips cleanly when `nvidia-smi` is unavailable.
- Gradient-phase FPGA offload stays disabled by default because the current path is not fully autograd-safe.
- `fpga_stats.json` now includes a per-phase FPGA breakdown so rollout, reward, and gradient offload activity can be audited separately.

## Main commands

```bash
python baseline_energy/rlhf_baseline.py --steps 10 --output results/gpu_baseline_smoke
python baseline_energy/rlhf_with_fpga.py --steps 2 --output results/fpga_smoke --use-mock-fpga
.venv/bin/python3 baseline_energy/test_fpga_integration.py --run-end-to-end --steps 2
python baseline_energy/run_policy_sweep.py --policy-json baseline_energy/data/smoke_policies.json --policies A,D --steps 2 --dry-run
```

## Related docs

- Project status:
  [/Users/dannyadkins/CS217-Final-Project/README.md](/Users/dannyadkins/CS217-Final-Project/README.md)
- Integration layer:
  [/Users/dannyadkins/CS217-Final-Project/integration/README.md](/Users/dannyadkins/CS217-Final-Project/integration/README.md)
- Saved outputs:
  [/Users/dannyadkins/CS217-Final-Project/results/README.md](/Users/dannyadkins/CS217-Final-Project/results/README.md)
