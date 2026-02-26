# Quick Setup Guide - All 3 Solutions

This guide covers all three solutions you asked for:
1. **Energy calculations** from FPGA measurements
2. **Fix Python integration** (compile Lab 1 library)
3. **Helper scripts** to automate everything

---

## Solution 1: Calculate FPGA Energy (Works Now!)

You already have FPGA performance measurements. Let's calculate energy:

```bash
cd ~/CS217-Final-Project
source venv/bin/activate

# Pull latest code
git pull

# Calculate FPGA energy for 50-step workload
python integration/calculate_fpga_energy.py \
  --cycles 148656 \
  --power 35 \
  --steps 50 \
  --output results/fpga_energy_estimate.json
```

**What this does:**
- Uses your measured 148,656 cycles per 16×16 matmul
- Estimates total matmuls in RLHF training
- Calculates time and energy for FPGA
- Saves results to JSON file

**Try different workloads:**
```bash
# For 2 steps (quick test)
python integration/calculate_fpga_energy.py --cycles 148656 --steps 2

# For 100 steps (full training)
python integration/calculate_fpga_energy.py --cycles 148656 --steps 100

# With CPU comparison (once you have CPU baseline data)
python integration/calculate_fpga_energy.py \
  --cycles 148656 \
  --steps 50 \
  --cpu-energy 0.05 \
  --cpu-time 15
```

---

## Solution 2: Fix Python Integration

To actually run Python code on the FPGA, compile Lab 1 as a shared library:

```bash
cd ~/CS217-Final-Project
source venv/bin/activate

# Pull latest code with compilation script
git pull

# Make script executable
chmod +x integration/compile_lab1_library.sh

# Run compilation script
bash integration/compile_lab1_library.sh
```

**If the script fails**, compile manually:

```bash
cd ~/cs217-lab-1-hiva/design_top/software
mkdir -p build

# Single-line command (no backslashes)
gcc -shared -fPIC -o build/libdesign_top.so src/design_top.c -I$AWS_FPGA_REPO_DIR/sdk/userspace/include -L$AWS_FPGA_REPO_DIR/sdk/userspace/lib -lfpga_mgmt -lpthread -lrt

# Verify it worked
ls -lh build/libdesign_top.so
```

**Then test Python integration:**

```bash
cd ~/CS217-Final-Project
source venv/bin/activate

# Test Lab 1 FPGA integration
python integration/test_lab1_integration.py

# If successful, run with Lab 1 config
cp baseline_energy/config_lab1_fpga.py baseline_energy/config.py
python baseline_energy/rlhf_with_fpga.py --steps 2 --output results/fpga_python_test
```

---

## Solution 3: Helper Scripts (All Created!)

I've created these helper scripts for you:

### A. Energy Calculator
```bash
python integration/calculate_fpga_energy.py --help
```
Calculates FPGA energy for any workload.

### B. Lab 1 Compiler
```bash
bash integration/compile_lab1_library.sh
```
Compiles Lab 1 as a shared library for Python.

### C. Performance Measurement
```bash
bash integration/measure_lab1_fpga.sh
```
Runs Lab 1 test 10 times and gets average cycles.

---

## Complete Workflow Example

Here's a complete workflow from scratch:

```bash
# 1. SSH to your f2.6xlarge instance
ssh -i "hiva_cs217.pem" ubuntu@ec2-52-23-236-230.compute-1.amazonaws.com

# 2. Go to project and activate environment
cd ~/CS217-Final-Project
source venv/bin/activate

# 3. Get latest code
git pull

# 4. Load FPGA bitstream
source ~/cs217-lab-1-hiva/design_top/generated_afid.sh
sudo fpga-load-local-image -S 0 -I $AGFI
sudo fpga-describe-local-image-slots  # Verify loaded

# 5. Measure FPGA performance (if not done already)
bash integration/measure_lab1_fpga.sh

# 6. Calculate energy estimates
python integration/calculate_fpga_energy.py \
  --cycles 148656 \
  --power 35 \
  --steps 50 \
  --output results/fpga_energy_50steps.json

# 7. (Optional) Try to compile Python integration
bash integration/compile_lab1_library.sh

# 8. (Optional) If compilation worked, test Python integration
python integration/test_lab1_integration.py
```

---

## Understanding Your FPGA Measurements

Your latest measurement: **148,656 cycles per 16×16 matmul**

**Breakdown:**
- Data Transfer: 148,641 cycles (99.99%)
- Computation: 15 cycles (0.01%)

**Why so slow?**
- PCIe communication overhead dominates
- Moving data to/from FPGA takes ~1000× longer than computing
- This is normal for small matrices!

**Performance at 250 MHz clock:**
```
Time per matmul = 148,656 / 250,000,000 = 0.594 ms
Throughput = 1,682 matmuls/second
```

**Energy (assuming 35W FPGA):**
```
Energy per matmul = 35 W × 0.000594 s = 0.021 J
For 1 million matmuls = 21 kJ = 5.8 Wh
```

---

## Comparing with CPU

Once your CPU baseline finishes, compare:

```bash
# Check CPU results
cat results/cpu_baseline_50steps/energy_summary.csv

# Calculate FPGA estimate with CPU comparison
python integration/calculate_fpga_energy.py \
  --cycles 148656 \
  --steps 50 \
  --cpu-energy <CPU_WH_FROM_ABOVE> \
  --cpu-time <CPU_MIN_FROM_ABOVE>
```

---

## Troubleshooting

### Script says AWS_FPGA_REPO_DIR not set
```bash
source ~/aws-fpga/sdk_setup.sh
echo $AWS_FPGA_REPO_DIR  # Should print path
```

### GCC compilation fails
Check you have the SDK:
```bash
ls $AWS_FPGA_REPO_DIR/sdk/userspace/lib/libfpga_mgmt.so
```

If not found:
```bash
cd ~/aws-fpga
source sdk_setup.sh
```

### Python integration test fails
This is OK! You can still use:
1. Energy calculator (Solution 1)
2. Direct FPGA measurements (Solution 3)
3. Report estimates based on measured cycles

### FPGA test hangs
```bash
sudo pkill design_top
sudo fpga-load-local-image -S 0 -I $AGFI  # Reload
```

---

## For Your Project Report

You can report these results:

1. **FPGA Hardware Validation**: Lab 1 FPGA tested and working (0 errors)
2. **Performance Measurement**: 148,656 cycles per 16×16 matmul
3. **Energy Calculation**: [Run calculator to get specific numbers]
4. **Analysis**: PCIe overhead dominates small matmuls

This is sufficient for CS217 - you have real FPGA measurements and can make informed estimates!

---

## Quick Commands Reference

```bash
# Measure FPGA
bash integration/measure_lab1_fpga.sh

# Calculate energy
python integration/calculate_fpga_energy.py --cycles 148656 --steps 50

# Compile library
bash integration/compile_lab1_library.sh

# Test Python
python integration/test_lab1_integration.py

# Run training (if library compiled)
cp baseline_energy/config_lab1_fpga.py baseline_energy/config.py
python baseline_energy/rlhf_with_fpga.py --steps 2
```
