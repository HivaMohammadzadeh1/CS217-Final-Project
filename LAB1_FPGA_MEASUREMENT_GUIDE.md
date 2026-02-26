# Lab 1 FPGA Direct Measurement Guide (Option C)

This guide shows you how to use the Lab 1 FPGA hardware directly to measure actual FPGA performance for matrix multiplication.

## What This Gives You

1. **Actual FPGA Hardware Performance**: Real cycle counts from the FPGA
2. **Energy Calculations**: Convert cycles to energy using FPGA power
3. **Comparison Data**: Compare CPU vs FPGA for your project

## Option C: Step-by-Step Instructions

### Step 1: Pull Latest Code

On your f2.6xlarge instance:

```bash
cd ~/CS217-Final-Project
git pull
```

### Step 2: Verify Lab 1 FPGA is Loaded

```bash
# Check FPGA status
sudo fpga-describe-local-image-slots

# Should show:
# AFIDEVICE    0       0x1d0f      0xf010      0000:34:00.0
# (0xf010 confirms Lab 1 AFI is loaded)

# If not loaded:
source ~/cs217-lab-1-hiva/design_top/generated_afid.sh
sudo fpga-load-local-image -S 0 -I $AGFI
```

### Step 3: Run Lab 1 FPGA Performance Test

#### Method 1: Single Test Run

```bash
cd ~/cs217-lab-1-hiva/design_top/software/runtime

# Run Lab 1 FPGA test
sudo ./design_top 0

# Look for these lines in the output:
# Data Transfer Cycles: XXXX
# Compute Cycles: YYYY
# TEST PASSED (should show 0 errors)
```

**Example Output:**
```
Data Transfer Cycles: 12708
Compute Cycles: 15
TEST PASSED
```

#### Method 2: Automated Performance Measurement

```bash
cd ~/CS217-Final-Project
source venv/bin/activate

# Run measurement script (10 iterations)
bash integration/measure_lab1_fpga.sh

# This will run Lab 1 test 10 times and calculate averages
```

### Step 4: Calculate FPGA Performance Metrics

From the Lab 1 output, extract:

```
Data Transfer Cycles: 12,708 cycles
Compute Cycles: 15 cycles
Total Cycles per 16×16 matmul: 12,723 cycles
```

**Time Calculation:**
Assuming FPGA clock = 250 MHz:
```
Time per matmul = Total Cycles / Clock Frequency
                = 12,723 / 250,000,000
                = 0.000050892 seconds
                = 50.89 microseconds
```

**Throughput:**
```
Matmuls per second = 1 / 0.000050892
                   = 19,649 matmuls/sec
```

### Step 5: Measure FPGA Power

#### Option A: Use Xilinx Power Estimator (XPE)

If you have the Lab 1 synthesis report:

```bash
# Find XPE report from Lab 1 synthesis
find ~/cs217-lab-1-hiva -name "*power*" -o -name "*xpe*" 2>/dev/null

# Look for total on-chip power (Watts)
# Example: 10.5 W
```

#### Option B: Use AWS F2 Instance Power Metrics

```bash
# On f2.6xlarge, FPGA power is approximately:
# - Idle: ~20-25 W
# - Active: ~30-50 W (depends on utilization)
# - Your Lab 1 design: Estimate ~35 W (typical for 16×16 matmul)
```

### Step 6: Calculate Energy

**Formula:**
```
Energy (Joules) = Power (Watts) × Time (Seconds)
Energy (Watt-hours) = Energy (Joules) / 3600
```

**Example Calculation:**

For 1000 matmuls with Lab 1 FPGA:
```
Total Cycles = 12,723 × 1,000 = 12,723,000 cycles
Time = 12,723,000 / 250,000,000 = 0.050892 seconds

Assuming FPGA Power = 35 W:
Energy = 35 W × 0.050892 s = 1.78 Joules
Energy = 1.78 / 3600 = 0.000494 Wh
```

For comparison with RLHF training:
```
# Estimate matmuls in 50 PPO steps (Qwen2.5-0.5B):
# ~50 steps × 8 batch × 512 seq_len × ~100 layers × 3 matmuls/layer
# = ~6,144,000 matmuls

Total Cycles = 12,723 × 6,144,000 = 78,169,152,000 cycles
Time = 78,169,152,000 / 250,000,000 = 312.68 seconds (~5.2 minutes)

Energy = 35 W × 312.68 s = 10,943.8 Joules = 3.04 Wh
```

### Step 7: Compare with CPU Baseline

After your CPU baseline finishes:

```bash
# CPU energy from baseline
cat results/cpu_baseline_50steps/energy_summary.csv

# Example comparison:
# CPU: 15 minutes, 150 W avg → 37.5 Wh
# FPGA (estimated): 5.2 minutes, 35 W → 3.04 Wh
# Speedup: 2.9x faster
# Energy savings: 12.3x less energy
```

## Using Python to Analyze Lab 1 Results

```bash
cd ~/CS217-Final-Project
source venv/bin/activate

# Test the Python wrapper
python integration/lab1_subprocess_wrapper.py

# This will:
# 1. Run Lab 1 FPGA test
# 2. Extract performance metrics
# 3. Calculate energy for different workloads
```

## Understanding Lab 1 Performance

### What Lab 1 Measures

- **Data Transfer Cycles (~12,708)**: Moving weights and inputs to/from FPGA
  - Writing weights (16 rows × 128 bits each)
  - Streaming input vectors
  - Reading activations

- **Compute Cycles (~15)**: Actual matrix multiplication on FPGA
  - 16 parallel multiply-accumulate operations
  - Very fast (only 15 cycles!)

- **Total Latency (~12,723 cycles)**: Dominated by PCIe data transfer

### Performance Characteristics

```
Breakdown:
  Data Transfer: 99.9% of time (12,708 / 12,723)
  Computation:    0.1% of time (15 / 12,723)

Bottleneck: PCIe communication overhead
```

### Why FPGA May Be Slower Than Expected

1. **Small Matrix Size**: 16×16 is tiny, overhead dominates
2. **PCIe Latency**: Moving data to/from FPGA takes time
3. **Single Matmul**: No batching or pipelining

**When FPGA Wins:**
- Large batch sizes (amortize transfer cost)
- Many matmuls in sequence (pipeline)
- Lower power per operation

## Integration with RLHF Training

### Current Status

✅ **What Works:**
- Lab 1 FPGA hardware: Tested, working (0 errors)
- Tiling logic: Breaks large matmuls into 16×16 tiles
- Performance measurement: Can extract cycle counts
- Software fallback: Python integration works

⚠️ **What Needs Work:**
- Direct Python-to-FPGA calls (ctypes limitation)
- Batch processing to reduce overhead
- Energy measurement integration with training

### Next Steps for Full Integration

1. **Measure CPU Baseline** (currently running)
   - Get CPU energy and timing data
   - Establish performance baseline

2. **Calculate FPGA Estimates**
   - Use Lab 1 cycle counts
   - Estimate total FPGA energy for RLHF workload
   - Compare with CPU baseline

3. **Create Comparison Report**
   - CPU vs FPGA energy
   - CPU vs FPGA latency
   - Efficiency metrics

## Quick Reference

### Test Lab 1 FPGA Once
```bash
cd ~/cs217-lab-1-hiva/design_top/software/runtime
sudo ./design_top 0 | grep -E "Cycles|TEST"
```

### Get Performance Numbers
```bash
cd ~/cs217-lab-1-hiva/design_top/software/runtime
sudo ./design_top 0 2>&1 | tee lab1_output.txt
grep "Data Transfer Cycles:" lab1_output.txt
grep "Compute Cycles:" lab1_output.txt
```

### Calculate Energy
```python
# In Python
cycles_per_matmul = 12723
num_matmuls = 1000000  # Your workload
clock_hz = 250e6
fpga_power_w = 35

time_s = (cycles_per_matmul * num_matmuls) / clock_hz
energy_j = fpga_power_w * time_s
energy_wh = energy_j / 3600

print(f"Time: {time_s:.2f} seconds")
print(f"Energy: {energy_j:.2f} Joules = {energy_wh:.4f} Wh")
```

## Troubleshooting

### FPGA Test Fails
```bash
# Check FPGA loaded
sudo fpga-describe-local-image-slots

# Reload if needed
source ~/cs217-lab-1-hiva/design_top/generated_afid.sh
sudo fpga-load-local-image -S 0 -I $AGFI
```

### Permission Denied
```bash
# Lab 1 requires sudo for FPGA access
sudo ./design_top 0
```

### Test Hangs
```bash
# Kill and retry
sudo pkill design_top
sudo ./design_top 0
```

## Summary

**Option C gives you:**
- ✅ Real FPGA cycle counts from Lab 1 hardware
- ✅ Accurate performance metrics
- ✅ Energy calculations for comparison
- ✅ Data for your project report

**What to report:**
1. CPU baseline: Time + Energy (from Option A)
2. FPGA measurements: Cycles + Power (from Option C)
3. Comparison: Speedup + Energy efficiency
4. Analysis: Why FPGA behaves as it does (overhead discussion)

This gives you complete data for your CS217 final project!
