# GPU Baseline Energy Measurement

This directory contains scripts for measuring GPU baseline energy consumption during RLHF training.

## Purpose

Establish the reference energy consumption using standard FP16 training on GPU (NVIDIA T4), which all FPGA results will be compared against.

## Key Deliverables

1. **GPU Energy Baseline**: Total energy for 100 PPO steps on 500 HH-RLHF examples
2. **Phase Breakdown**: Energy split across rollout/reward/gradient phases
3. **Power Profile**: Time-series power consumption during training

## Measurement Protocol

See [ENERGY_MEASUREMENT_PROTOCOL.md](./ENERGY_MEASUREMENT_PROTOCOL.md) for detailed protocol.

## Scripts (to be added)

- `run_rlhf_baseline.py`: RLHF training loop with energy monitoring
- `monitor_gpu_power.sh`: nvidia-smi power logging wrapper
- `process_power_log.py`: Convert power logs to energy measurements
- `config.py`: Configuration for baseline experiments

## Quick Start

```bash
# 1. Start power monitoring
./monitor_gpu_power.sh &
MONITOR_PID=$!

# 2. Run baseline training
python run_rlhf_baseline.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset Anthropic/hh-rlhf \
    --num_steps 100 \
    --output ../results/gpu_baseline

# 3. Stop monitoring
kill $MONITOR_PID

# 4. Process results
python process_power_log.py \
    --power_log power_log.csv \
    --training_log ../results/gpu_baseline/training_log.txt \
    --output ../results/gpu_baseline/energy.csv
```

## Expected Results

Based on NVIDIA T4 specifications:

| Metric | Expected Range |
|--------|----------------|
| Average Power | 70-150 W |
| Total Runtime | 20-40 minutes |
| Total Energy | ~200-300 kJ |
| Rollout Phase | ~40% of total energy |
| Reward Phase | ~30% of total energy |
| Gradient Phase | ~30% of total energy |

## Verification

Sanity checks before using baseline:

1. **Power range**: Should be 70-250W for T4 (check if in range)
2. **Runtime**: Should be 20-40 min for 100 steps (if much longer, check batch size)
3. **Consistency**: Run 3 trials, std dev should be <5% of mean
4. **Phase timing**: Rollout should be slowest phase (largest batch processing)

## Output Format

Results saved to `../results/gpu_baseline/`:

```
gpu_baseline/
├── energy.csv              # Phase-wise energy breakdown
├── metrics.csv             # Training metrics (reward, KL)
├── training_log.txt        # Detailed training log
├── power_log.csv           # Raw nvidia-smi output
└── config.json             # Experiment configuration
```
