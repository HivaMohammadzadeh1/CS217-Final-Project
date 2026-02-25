# Energy Measurement Protocol

This document defines the standardized protocol for measuring energy consumption across all experimental runs, ensuring apples-to-apples comparison between GPU baseline and FPGA implementations.

## Fundamental Equation

```
Energy (Joules) = Power (Watts) × Time (Seconds)
```

We measure power and runtime separately, then multiply to get total energy.

## GPU Energy Measurement

### Tool: nvidia-smi

```bash
# Start power monitoring (100ms polling interval)
nvidia-smi dmon -s u -i 0 -d 1 -c 10000 > power_log_gpu.csv &
MONITOR_PID=$!

# Run training
python run_rlhf_baseline.py

# Stop monitoring
kill $MONITOR_PID
```

### Data Processing

```python
import pandas as pd

# Load power log
df = pd.read_csv('power_log_gpu.csv', skiprows=1, sep=r'\s+')

# Extract power column (Watts)
power_watts = df['pwr'].values

# Calculate average power
avg_power = power_watts.mean()

# Get runtime from training script logs
runtime_seconds = 1800  # example: 30 minutes

# Calculate energy
energy_joules = avg_power * runtime_seconds

print(f"Average Power: {avg_power:.2f} W")
print(f"Runtime: {runtime_seconds:.2f} s")
print(f"Total Energy: {energy_joules:.2f} J ({energy_joules/3600:.2f} Wh)")
```

### Output Format

Save results to `results/gpu_baseline_energy.csv`:

```csv
phase,avg_power_W,runtime_s,energy_J
rollout,150.2,600,90120
reward_inference,145.8,400,58320
gradient_update,180.5,800,144400
total,158.8,1800,285840
```

## FPGA Energy Measurement

### Method 1: Xilinx Power Estimator (XPE)

1. After synthesis, open the Vivado project
2. Run: `Tools > Xilinx Power Estimator`
3. Load post-implementation results
4. Export power report: `Reports > Power > Export to CSV`

Example XPE output:
```
Total On-Chip Power: 12.5 W
Dynamic Power: 8.3 W
Static Power: 4.2 W
```

### Method 2: On-Board Power Sensors (if available)

For AWS F2 instances with power telemetry:

```bash
# Read FPGA power from sysfs
watch -n 0.1 cat /sys/class/hwmon/hwmon*/power1_input > fpga_power_log.txt
```

### Runtime Measurement

```python
import time

start_time = time.time()
# Run FPGA inference
run_fpga_policy()
end_time = time.time()

runtime_seconds = end_time - start_time
```

### Energy Calculation

```python
# From XPE
fpga_power_watts = 12.5  # XPE total on-chip power

# From runtime measurement
runtime_seconds = 2400  # 40 minutes

# Calculate FPGA energy
fpga_energy_joules = fpga_power_watts * runtime_seconds

# Don't forget host CPU energy!
host_power_watts = 50  # typical host CPU contribution
host_energy_joules = host_power_watts * runtime_seconds

# Total system energy
total_energy_joules = fpga_energy_joules + host_energy_joules

print(f"FPGA Power: {fpga_power_watts:.2f} W")
print(f"Host Power: {host_power_watts:.2f} W")
print(f"Runtime: {runtime_seconds:.2f} s")
print(f"FPGA Energy: {fpga_energy_joules:.2f} J")
print(f"Host Energy: {host_energy_joules:.2f} J")
print(f"Total Energy: {total_energy_joules:.2f} J")
```

### Output Format

Save results to `results/fpga_policy_X_energy.csv`:

```csv
phase,fpga_power_W,host_power_W,runtime_s,fpga_energy_J,host_energy_J,total_energy_J
rollout,10.2,45,720,7344,32400,39744
reward_inference,12.8,48,480,6144,23040,29184
gradient_update,15.5,52,1200,18600,62400,81000
total,12.5,50,2400,30000,120000,150000
```

## Fixed Measurement Run

To ensure consistent comparison, all energy measurements use the same workload:

### Parameters
- **PPO steps**: 100
- **Training examples**: 500 preference pairs from HH-RLHF
- **Batch size**: 8
- **Sequence length**: 512 tokens
- **Random seed**: 42 (for reproducibility)

### Script Template

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Fixed parameters
SEED = 42
NUM_STEPS = 100
BATCH_SIZE = 8
MAX_SEQ_LEN = 512
NUM_EXAMPLES = 500

torch.manual_seed(SEED)

# Load model and dataset
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
dataset = load_dataset('Anthropic/hh-rlhf', split='train').shuffle(seed=SEED).select(range(NUM_EXAMPLES))

# Run training with energy monitoring
# ... (actual RLHF training loop)
```

## Phase Breakdown

Each measurement run should log energy separately for each RLHF phase:

1. **Rollout Phase**: Policy model generates candidate responses
2. **Reward Inference**: Reward model scores the responses
3. **Gradient Update**: Compute and apply policy gradient updates

This breakdown helps identify which phases benefit most from quantization.

## Data Validation

Before comparing results, verify:

1. **Same workload**: All runs use identical 500 examples, same seed
2. **Same sequence length**: All padded/truncated to 512 tokens
3. **Comparable runtime**: FPGA should be within 2x of GPU (if much slower, something is wrong)
4. **Power sanity check**:
   - GPU (T4): ~70-250 W typical
   - FPGA (VU9P): ~5-25 W typical
   - If values are way outside these ranges, double-check measurement setup

## Cross-Validation

Run at least 3 measurement trials for each policy and report:
- Mean energy
- Standard deviation
- Min/max values

This accounts for system noise and validates consistency.

## Example Comparison

```python
# After collecting all measurements
import pandas as pd
import matplotlib.pyplot as plt

# Load all energy results
gpu_energy = pd.read_csv('results/gpu_baseline_energy.csv')
fpga_policy_a = pd.read_csv('results/fpga_policy_a_energy.csv')
fpga_policy_d = pd.read_csv('results/fpga_policy_d_energy.csv')

# Compare total energy
gpu_total = gpu_energy[gpu_energy['phase'] == 'total']['energy_J'].values[0]
fpga_a_total = fpga_policy_a[fpga_policy_a['phase'] == 'total']['total_energy_J'].values[0]
fpga_d_total = fpga_policy_d[fpga_policy_d['phase'] == 'total']['total_energy_J'].values[0]

# Calculate savings
savings_a = (gpu_total - fpga_a_total) / gpu_total * 100
savings_d = (gpu_total - fpga_d_total) / gpu_total * 100

print(f"GPU Baseline: {gpu_total:.2f} J")
print(f"FPGA Policy A: {fpga_a_total:.2f} J ({savings_a:.1f}% savings)")
print(f"FPGA Policy D: {fpga_d_total:.2f} J ({savings_d:.1f}% savings)")

# Plot comparison
policies = ['GPU\nBaseline', 'FPGA\nPolicy A', 'FPGA\nPolicy D']
energies = [gpu_total, fpga_a_total, fpga_d_total]

plt.bar(policies, energies)
plt.ylabel('Total Energy (J)')
plt.title('Energy Consumption Comparison')
plt.savefig('results/energy_comparison.png')
```

## Troubleshooting

### Issue: nvidia-smi not logging power
- **Solution**: Check GPU supports power monitoring: `nvidia-smi -q -d POWER`
- Some GPUs don't expose power telemetry; use TDP as approximation

### Issue: FPGA power seems too high
- **Solution**: Check if measurement includes host system. FPGA alone should be <25W

### Issue: Inconsistent measurements across runs
- **Solution**:
  - Ensure GPU/FPGA are at idle before starting
  - Check for background processes consuming resources
  - Use fixed frequency mode (disable frequency scaling)

## References

- NVIDIA SMI Documentation: https://developer.nvidia.com/nvidia-system-management-interface
- Xilinx Power Estimator User Guide: UG440
- AWS F2 Instance Power Monitoring: AWS documentation
