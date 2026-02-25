# Results

This directory stores all experimental results, measurements, and analysis outputs.

## Structure

```
results/
├── gpu_baseline/              # GPU baseline measurements
│   ├── energy.csv
│   ├── metrics.csv
│   └── training_log.txt
├── fpga_fp16_baseline/        # FPGA FP16 baseline
│   └── ...
├── policy_A_conservative/     # Policy A results
│   └── ...
├── policy_B_balanced/         # Policy B results
│   └── ...
├── policy_C_aggressive/       # Policy C results
│   └── ...
├── policy_D_adaptive/         # Policy D results
│   └── ...
├── sensitivity_matrix.csv     # Layer sensitivity analysis
├── phase_analysis.csv         # Phase-level quantization impact
├── experiments_summary.csv    # All experiments summary
└── plots/                     # Generated plots
    ├── energy_comparison.png
    ├── pareto_curve.png
    ├── phase_breakdown.png
    └── sensitivity_heatmap.png
```

## File Formats

### energy.csv
```csv
phase,avg_power_W,runtime_s,energy_J
rollout,150.2,600,90120
reward_inference,145.8,400,58320
gradient_update,180.5,800,144400
total,158.8,1800,285840
```

### metrics.csv
```csv
metric,value
win_rate,0.73
kl_divergence,0.045
reward_rank_correlation,0.91
final_reward,1.52
```

### experiments_summary.csv
```csv
policy,total_energy_J,win_rate,kl_divergence,energy_savings_pct,quality_loss_pct
GPU_FP16,285840,0.78,0.000,0.0,0.0
FPGA_FP16,245000,0.78,0.002,14.3,0.0
Policy_A,210000,0.76,0.015,26.5,2.6
Policy_B,180000,0.74,0.032,37.0,5.1
Policy_C,155000,0.68,0.068,45.8,12.8
Policy_D,170000,0.75,0.028,40.5,3.8
```

## Analysis Scripts

```bash
# Generate summary CSV
python ../integration/analyze_results.py --input ./ --output experiments_summary.csv

# Plot Pareto curve
python ../integration/plot_pareto.py --input experiments_summary.csv --output plots/pareto_curve.png

# Generate all plots
python ../integration/generate_plots.py --input ./ --output plots/
```

## Interpretation

### Energy Savings
- Calculated as: `(baseline_energy - policy_energy) / baseline_energy * 100`
- Baseline: GPU FP16 (285,840 J in example above)
- Target: >30% savings

### Quality Loss
- Calculated from win rate drop: `(baseline_win_rate - policy_win_rate) / baseline_win_rate * 100`
- Baseline: GPU FP16 (0.78 win rate in example)
- Threshold: <10% loss

### Pareto Optimal
- Policies on the Pareto frontier achieve best quality for given energy budget
- Policy D is research target: balance energy savings with minimal quality loss
