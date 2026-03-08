# Profiling And Policy Generation

This directory converts model sensitivity measurements into policy inputs for the FPGA experiments.

## What it does

1. Quantize one layer at a time with the repo's MX reference model.
2. Measure perplexity delta versus the floating-point baseline.
3. Save a sensitivity matrix with MXFP4/MXFP8 results for group sizes 8 and 16.
4. Generate policy files (`A/B/C/D`) from that matrix.

## Main scripts

- `sensitivity_profiler.py`
  Profiles selected linear layers and writes `sensitivity_matrix.csv` plus metadata JSON.
- `define_policies.py`
  Converts a sensitivity matrix into `A/B/C/D` policy JSON.

## Typical commands

Smoke run on a small local dataset:

```bash
.venv/bin/python pytorch_profiling/sensitivity_profiler.py \
  --model sshleifer/tiny-gpt2 \
  --dataset baseline_energy/data/smoke_rlhf.jsonl \
  --text-field chosen \
  --num-examples 4 \
  --max-seq-len 96 \
  --max-layers 4 \
  --device cpu \
  --output results/profiling_smoke/sensitivity_matrix.csv
```

Generate policies from the resulting matrix:

```bash
.venv/bin/python pytorch_profiling/define_policies.py \
  --sensitivity results/profiling_smoke/sensitivity_matrix.csv \
  --group-size 8 \
  --output results/profiling_smoke/policies_g8.json
```

## Notes

- This uses the repo's `integration/mx_precision_sim.py` quantization logic.
- It quantizes weights only; it does not emulate the full deployed hardware datapath.
- It is still useful because it gives a consistent, reproducible basis for policy selection before the final hardware experiments.
