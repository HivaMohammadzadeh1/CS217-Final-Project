# FPGA Setup Guide

## Your FPGA Instance

**Instance Details:**
- **Type**: f2.6xlarge (Xilinx FPGA)
- **Private IP**: 172.31.87.76
- **AMI**: FPGA Developer AMI (Ubuntu) - 1.16.1

## Step 1: Connect to FPGA Instance

```bash
# SSH into your FPGA instance
ssh -i your-key.pem ubuntu@172.31.87.76

# Or if using public IP (once instance is running)
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP
```

## Step 2: Verify FPGA is Available

```bash
# Check if FPGA device is present
lspci | grep -i xilinx

# Should see something like:
# 00:0f.0 Processing accelerators: Xilinx Corporation Device...

# Check FPGA management tool
sudo fpga-describe-local-image-slots

# Should show the FPGA slot status
```

## Step 3: Load Lab 1 Bitstream (If Available)

```bash
# If you have a pre-existing AFI (Amazon FPGA Image) from Lab 1:
sudo fpga-load-local-image -S 0 -I agfi-XXXXXXXXXXXXX

# Verify it loaded
sudo fpga-describe-local-image-slots
```

## Step 4: Clone Project to FPGA Instance

```bash
# On the FPGA instance:
git clone https://github.com/HivaMohammadzadeh1/CS217-Final-Project.git
cd CS217-Final-Project

# Set up environment
bash setup_environment.sh
source venv/bin/activate

# Install any additional dependencies
pip install -r requirements.txt
```

## Step 5: Test Mock FPGA Mode First

Before connecting to real hardware, test with mock mode:

```bash
# This runs entirely in software (no FPGA needed)
python baseline_energy/rlhf_with_fpga.py --steps 5 --output results/test_mock
```

## Step 6: Switch to Real FPGA

Once you've verified the bitstream and driver are working:

1. Edit `baseline_energy/config.py`:
```python
USE_FPGA_OFFLOAD = True
USE_MOCK_FPGA = False  # Switch to real FPGA
FPGA_DEVICE_ID = 0
```

2. Update `integration/fpga_matmul_offload.py` to implement `RealFPGAInterface`:
   - Use AWS FPGA SDK or PYNQ (depending on your setup)
   - Implement memory transfer to/from FPGA
   - Implement computation trigger

## Current Status

✅ **Mock FPGA mode is working** - You can run experiments with simulated FPGA

⏳ **Real FPGA needs:**
- Lab 1 bitstream loaded onto FPGA
- FPGA driver implementation in `fpga_matmul_offload.py`
- Testing matmul correctness on real hardware

## Quick Test Commands

```bash
# Test FPGA offload (mock mode)
cd /Users/hivamoh/CS217-Project/CS217-Final-Project
source venv/bin/activate
python integration/fpga_matmul_offload.py

# Run 5-step test with FPGA offload
python baseline_energy/rlhf_with_fpga.py --steps 5 --output results/quick_test

# Run full 50-step baseline (as per 2-hour plan)
python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/fpga_baseline_50
```

## Troubleshooting

**Problem**: Can't SSH to FPGA instance
- Check security group allows SSH (port 22)
- Verify instance is in "running" state (not "pending")
- Use correct private IP or public IP

**Problem**: FPGA not detected
- Run `lspci | grep -i xilinx`
- Check instance type is f2.xlarge or larger
- Verify FPGA Developer AMI is being used

**Problem**: Out of memory during training
- Reduce `BATCH_SIZE` in `config.py`
- Use FP16 mode (`FP16 = True`)
- Reduce `MAX_SEQ_LENGTH`

## For the 2-Hour Plan

**You can proceed with mock FPGA mode for now:**

1. Test with 5 steps: `python baseline_energy/rlhf_with_fpga.py --steps 5 --output results/test`
2. Run 50 steps: `python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/baseline_50`
3. Calculate energy: `python baseline_energy/calculate_energy.py --results results/baseline_50`

**Later, when FPGA bitstream is ready:**
- Set `USE_MOCK_FPGA = False`
- Implement `RealFPGAInterface` class
- Re-run experiments with real hardware
