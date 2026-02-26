# FPGA Quick Start - Next Steps

**IMPORTANT**: f2.6xlarge has FPGA (Xilinx VU9P), NOT GPU!

## Hardware Clarification

Your f2.6xlarge instance:
- ✅ CPU: Intel Xeon (16 vCPUs)
- ✅ FPGA: Xilinx VU9P FPGA
- ❌ NO GPU

For GPU training, you need: g4dn.xlarge, p3.2xlarge, etc.

## On the FPGA Instance (ubuntu@ip-172-31-87-76)

### Step 1: Check FPGA Status

```bash
# Verify FPGA is detected
lspci | grep -i xilinx

# Check FPGA slot status
sudo fpga-describe-local-image-slots
```

### Step 2: Clone Project to FPGA

```bash
# Clone your repository
git clone https://github.com/HivaMohammadzadeh1/CS217-Final-Project.git
cd CS217-Final-Project

# Set up environment
bash setup_environment.sh
source venv/bin/activate

# Verify installation
python baseline_energy/verify_setup.py
```

### Step 3: Choose Your Running Mode

The code now runs in **CPU-only mode** (no mock FPGA):
- ✅ Pure PyTorch on CPU
- ✅ No FPGA offload overhead
- ✅ ~2-6 hours for 50 steps

```bash
# Quick 2-step test (CPU only)
python baseline_energy/rlhf_with_fpga.py --steps 2 --output results/test_2steps

# This uses: USE_FPGA_OFFLOAD = False in config.py
```

**Alternative modes:**

**A. Mock FPGA Mode** (for testing FPGA architecture):
```bash
# Edit config.py
# Set: USE_FPGA_OFFLOAD = True, USE_MOCK_FPGA = True
python baseline_energy/rlhf_with_fpga.py --steps 2 --output results/mock_fpga_test
```

**B. Real GPU Mode** (requires g4dn.xlarge or similar):
```bash
# Copy GPU config
cp baseline_energy/config_gpu.py baseline_energy/config.py
python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/gpu_baseline
# ~5-10 minutes on GPU
```

### Step 4: Run 50-Step Baseline (As per 2-hour plan)

```bash
# This is your main experiment run
python baseline_energy/rlhf_with_fpga.py \
  --steps 50 \
  --output results/baseline_50steps

# Expected runtime: 20-40 minutes depending on hardware
```

### Step 5: Calculate Energy

```bash
# Calculate energy from power logs and timing
python baseline_energy/calculate_energy.py \
  --results results/baseline_50steps
```

### Step 6: Download Results

```bash
# On FPGA instance - create a tarball
cd results
tar -czf baseline_results.tar.gz baseline_50steps/

# On your local machine - download the results
scp -i hiva_cs217.pem ubuntu@ec2-52-23-236-230.compute-1.amazonaws.com:~/CS217-Final-Project/results/baseline_results.tar.gz .
```

## Current Status Summary

✅ **Completed:**
- Environment setup scripts created
- Matmul tiling (16x16 chunks) implemented
- FPGA offload wrapper with mock interface
- RLHF training script with FPGA hooks
- Dataset configured for 800 train / 200 eval
- Energy calculation script ready

🔧 **Ready to Run:**
- 50-step baseline experiment with FPGA offload (mock mode)
- Energy measurement and calculation
- All scripts tested and working

⏳ **Future Work (When Lab 1 Bitstream Ready):**
- Load real FPGA bitstream
- Implement RealFPGAInterface
- Re-run experiments with real hardware
- Compare mock vs real FPGA performance

## For the 2-Hour Plan

You can complete the plan using **mock FPGA mode**:

| Time | Task | Status |
|------|------|--------|
| 0:00-0:20 | Environment setup | ✅ Done |
| 0:20-0:40 | Tiling + offload script | ✅ Done |
| 0:40-1:00 | RLHF training script | ✅ Done |
| 1:00-1:20 | Run baseline (50 steps) | ⏳ Ready to run |
| 1:20-1:40 | Get energy numbers | ⏳ Ready to run |
| 1:40-2:00 | Buffer / debug | ⏳ Buffer time |

## Troubleshooting

**Out of memory?**
```python
# Edit baseline_energy/config.py
BATCH_SIZE = 4  # Reduce from 8
MAX_SEQ_LENGTH = 256  # Reduce from 512
```

**Slow training?**
- This is expected on CPU
- Models will download on first run (takes time)
- Subsequent runs will be faster

**Need to stop mid-run?**
```bash
# Press Ctrl+C to stop
# Results are saved incrementally
```

## Expected Output Files

After running the baseline, you'll have:

```
results/baseline_50steps/
├── phase_timing.json          # Time per phase (rollout/reward/gradient)
├── training_stats.json        # PPO training metrics
├── fpga_stats.json           # FPGA offload statistics
├── power_log_baseline.csv    # GPU/CPU power measurements
└── energy_summary.csv        # Final energy breakdown (after calc)
```

## Quick Commands Reference

```bash
# On FPGA Instance
ssh -i hiva_cs217.pem ubuntu@ec2-52-23-236-230.compute-1.amazonaws.com

# Clone and setup
git clone https://github.com/HivaMohammadzadeh1/CS217-Final-Project.git
cd CS217-Final-Project && bash setup_environment.sh && source venv/bin/activate

# Run experiment
python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/baseline_50

# Calculate energy
python baseline_energy/calculate_energy.py --results results/baseline_50

# Download results
tar -czf results.tar.gz results/
# Then scp from local machine
```
