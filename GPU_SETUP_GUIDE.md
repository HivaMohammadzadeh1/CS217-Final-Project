# GPU Setup Guide - How to Use Actual GPU

## Understanding Your Options

### Current Situation: f2.6xlarge (FPGA Instance)

Your f2.6xlarge instance has:
- **CPU**: Intel Xeon (16 vCPUs) ✅
- **FPGA**: Xilinx VU9P FPGA ✅
- **GPU**: NONE ❌

**You CANNOT use GPU on f2.6xlarge because there is no GPU hardware.**

### Running Modes on f2.6xlarge

#### 1. CPU-Only Mode (Current Configuration)
- No FPGA offload
- Pure PyTorch on CPU
- **Speed**: 2-6 hours for 50 steps
- **Config**: `USE_FPGA_OFFLOAD = False`

#### 2. Mock FPGA Mode
- Simulates FPGA offload
- Computation still on CPU
- **Speed**: Similar to CPU-only (2-6 hours)
- **Config**: `USE_FPGA_OFFLOAD = True, USE_MOCK_FPGA = True`

#### 3. Real FPGA Mode (Requires Bitstream)
- Actually uses FPGA hardware
- Needs: Lab 1 bitstream + driver implementation
- **Speed**: Likely slower than CPU (PCIe overhead)
- **Config**: `USE_FPGA_OFFLOAD = True, USE_MOCK_FPGA = False`

## How to Use Actual GPU

### Step 1: Launch a GPU Instance

You need to launch a **different instance type** with a GPU:

**Recommended GPU Instances:**

| Instance Type | GPU | vCPUs | GPU Memory | Cost/Hour | Best For |
|--------------|-----|-------|------------|-----------|----------|
| **g4dn.xlarge** | T4 | 4 | 16 GB | $0.526 | Budget-friendly, fast enough |
| g4dn.2xlarge | T4 | 8 | 16 GB | $0.752 | More CPU cores |
| p3.2xlarge | V100 | 8 | 16 GB | $3.06 | Fastest training |
| p3.8xlarge | 4x V100 | 32 | 64 GB | $12.24 | Multi-GPU |

**Launch via AWS Console:**

1. Go to EC2 → Launch Instance
2. **AMI**: Ubuntu 22.04 LTS
3. **Instance Type**: `g4dn.xlarge`
4. **Key pair**: Your existing `hiva_cs217.pem`
5. **Storage**: 100 GB
6. **Launch**

### Step 2: Set Up GPU Instance

```bash
# SSH to GPU instance
ssh -i hiva_cs217.pem ubuntu@<GPU-PUBLIC-IP>

# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (usually pre-installed on Deep Learning AMI)
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

**After reboot:**

```bash
# Reconnect
ssh -i hiva_cs217.pem ubuntu@<GPU-PUBLIC-IP>

# Verify GPU
nvidia-smi

# Should show something like:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx    Driver Version: 525.xx    CUDA Version: 12.x         |
# |-------------------------------+----------------------+----------------------+
# |   0  Tesla T4            Off  | 00000000:00:1E.0 Off |                    0 |
# |  70W /  70W |      0MiB / 15109MiB |      0%      Default |
# +-----------------------------------------------------------------------------+
```

### Step 3: Clone and Set Up Project

```bash
# Clone repository
git clone https://github.com/HivaMohammadzadeh1/CS217-Final-Project.git
cd CS217-Final-Project

# Run setup
bash setup_environment.sh
source venv/bin/activate

# Verify setup detects GPU
python baseline_energy/verify_setup.py
# Should now show: "✓ GPU available"
```

### Step 4: Use GPU Configuration

```bash
# Copy GPU config (disables FPGA offload, enables GPU)
cp baseline_energy/config_gpu.py baseline_energy/config.py

# Or manually edit config.py:
# USE_GPU = True
# USE_FPGA_OFFLOAD = False
# FP16 = True
```

### Step 5: Run Training on GPU

```bash
# 2-step test (~30 seconds)
python baseline_energy/rlhf_with_fpga.py --steps 2 --output results/gpu_test_2steps

# Full 50-step baseline (~5-10 minutes)
screen -S training
python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/gpu_baseline_50steps
# Press Ctrl+A then D to detach

# Calculate energy
python baseline_energy/calculate_energy.py --results results/gpu_baseline_50steps
```

### Step 6: Download Results

```bash
# On GPU instance
cd ~/CS217-Final-Project/results
tar -czf gpu_results.tar.gz gpu_baseline_50steps/

# On your local machine
scp -i hiva_cs217.pem ubuntu@<GPU-PUBLIC-IP>:~/CS217-Final-Project/results/gpu_results.tar.gz .
```

## Comparison: CPU vs GPU vs FPGA

| Hardware | Instance Type | Time (50 steps) | Cost | Energy |
|----------|--------------|-----------------|------|--------|
| **CPU** | f2.6xlarge | 2-6 hours | $1.65/hr × 3hr = $4.95 | High |
| **GPU** | g4dn.xlarge | 5-10 min | $0.526/hr × 0.15hr = $0.08 | Low |
| **FPGA (Mock)** | f2.6xlarge | 2-6 hours | $1.65/hr × 3hr = $4.95 | High |
| **FPGA (Real)** | f2.6xlarge | TBD | $1.65/hr | TBD |

## Summary

**To use actual GPU:**
1. ❌ Cannot use GPU on f2.6xlarge (no GPU hardware)
2. ✅ Launch g4dn.xlarge instance (has NVIDIA T4 GPU)
3. ✅ Use `config_gpu.py` configuration
4. ✅ Training will be ~20-40x faster

**Current f2.6xlarge options:**
- CPU-only mode (no FPGA offload) - current setting
- Mock FPGA mode (for testing architecture)
- Real FPGA mode (requires bitstream + implementation)

## Quick Commands

### On f2.6xlarge (CPU-only):
```bash
cd ~/CS217-Final-Project
source venv/bin/activate
python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/cpu_baseline_50
```

### On g4dn.xlarge (GPU):
```bash
cd ~/CS217-Final-Project
source venv/bin/activate
cp baseline_energy/config_gpu.py baseline_energy/config.py
python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/gpu_baseline_50
```

## Next Steps

If you want GPU training:
1. Launch g4dn.xlarge instance
2. Follow setup steps above
3. Run training (will be much faster)

If you want to continue with CPU on f2.6xlarge:
1. Current config is already set for CPU-only
2. Just run the training (will be slower)
3. Can compare CPU vs GPU results later
