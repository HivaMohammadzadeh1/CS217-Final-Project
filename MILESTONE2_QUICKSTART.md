# Milestone 2 Quick Start Guide

## What You Can Do Right Now (Quick Wins - 30 minutes)

Milestone 2 has two parallel tracks. Here's what you can accomplish quickly to get momentum:

### ⚡ Quick Track (Do These First - 30 min)

These tasks require minimal setup and give you quick validation:

#### 1. Environment Setup (5 min)
```bash
# Make setup script executable
chmod +x setup_environment.sh

# Run setup
./setup_environment.sh

# Activate environment
source venv/bin/activate
```

#### 2. Verify Installation (2 min)
```bash
python baseline_energy/verify_setup.py
```
**Expected output**: All packages checked, GPU status displayed

#### 3. Test Model Loading (5 min first run, 30 sec after)
```bash
python baseline_energy/test_model.py
```
**What this does**: Downloads Qwen2.5-0.5B (~1GB) and runs test inference
**Why it matters**: Confirms the model works before attempting RLHF training

#### 4. Test Dataset Loading (2 min)
```bash
python baseline_energy/test_dataset.py
```
**What this does**: Downloads HH-RLHF dataset and creates the fixed 1000-pair subset
**Why it matters**: Prepares the exact data you'll use for all experiments

#### 5. Test GPU Power Monitoring (2 min)
```bash
# If you have a GPU:
python baseline_energy/gpu_power_monitor.py
```
**What this does**: Verifies nvidia-smi can log power consumption
**Why it matters**: Required for baseline energy measurement

---

## 🎯 Main Tasks (Do These Next - Days 2-3)

After the quick wins, tackle these more substantial tasks:

### Track 1: GPU Baseline Energy Measurement

**Goal**: Measure energy consumption of standard FP16 RLHF training

**Tasks**:
1. ✅ (Done above) Set up power monitoring
2. 🔨 Implement basic PPO training loop using TRL
3. 🔨 Run 50-100 PPO steps with power logging
4. 🔨 Analyze power logs and calculate total energy

**Files to create**:
- `baseline_energy/rlhf_baseline.py` - Main RLHF training script
- `baseline_energy/config.py` - Hyperparameters
- `baseline_energy/process_power_log.py` - Energy calculation

**Expected output**:
```csv
phase,avg_power_W,runtime_s,energy_J
rollout,150.2,600,90120
reward_inference,145.8,400,58320
gradient_update,180.5,800,144400
total,158.8,1800,285840
```

### Track 2: PyTorch Layer Sensitivity Profiling

**Goal**: Figure out which layers can tolerate MXFP4 vs MXFP8

**Tasks**:
1. 🔨 Install mx-pytorch library
2. 🔨 Write per-layer quantization tester
3. 🔨 Run sensitivity sweep (layer × format × group_size)
4. 🔨 Generate sensitivity matrix

**Files to create**:
- `pytorch_profiling/sensitivity_profiler.py` - Main profiler
- `pytorch_profiling/mx_utils.py` - MX format helpers
- `pytorch_profiling/config.py` - Profiling config

**Expected output**:
```csv
layer,format,group_size,perplexity,delta_pct
transformer.h.0.attn,MXFP4,8,12.5,2.3
transformer.h.0.attn,MXFP8,8,11.8,0.5
transformer.h.0.ffn,MXFP4,8,11.5,1.2
...
```

---

## 📅 Suggested Timeline for Milestone 2

### Day 1 (Today):
- ✅ Quick wins (30 min) - **DO THIS NOW**
- 📖 Read TRL documentation for PPO
- 🎯 Start implementing basic RLHF training loop

### Day 2-3:
- Complete GPU baseline energy measurement
- Run first energy measurement
- Verify results are reasonable

### Day 4-5:
- Set up mx-pytorch
- Implement layer sensitivity profiler
- Run initial sensitivity tests

### Day 6-7:
- Complete sensitivity matrix
- Define policies A/B/C/D based on results
- Document findings

---

## 🚀 What to Run Right Now

Open a terminal and execute these commands in order:

```bash
# 1. Setup (5 min)
cd /Users/hivamoh/CS217-Project/CS217-Final-Project
chmod +x setup_environment.sh
./setup_environment.sh
source venv/bin/activate

# 2. Verify (2 min)
python baseline_energy/verify_setup.py

# 3. Test model (5 min first time)
python baseline_energy/test_model.py

# 4. Test dataset (2 min)
python baseline_energy/test_dataset.py

# 5. Test GPU monitoring (2 min) - only if you have GPU
python baseline_energy/gpu_power_monitor.py
```

**Total time**: ~15-20 minutes

After this, you'll have:
- ✅ Working Python environment
- ✅ Qwen2.5-0.5B model downloaded and tested
- ✅ HH-RLHF dataset ready (1000 pairs, seed=42)
- ✅ GPU power monitoring verified
- ✅ Confidence that everything works

---

## 🆘 Troubleshooting

### "No module named 'trl'"
The TRL library is brand new. If installation fails:
```bash
pip install git+https://github.com/huggingface/trl.git
```

### "No module named 'mx'"
mx-pytorch might not install from pip. Try:
```bash
pip install git+https://github.com/microsoft/microxcaling.git
```
If it still fails, skip it for now - you can do Track 1 (baseline) without it.

### "CUDA out of memory"
The model is small (500MB in FP16), but if you have limited GPU memory:
```python
# In the scripts, add:
model = model.to("cpu")  # Force CPU mode
```

### "nvidia-smi: command not found"
You're on a machine without NVIDIA GPU. You can:
- Use AWS g4dn instance for GPU work
- Skip power monitoring for now and estimate power later

---

## 📊 Success Criteria for Milestone 2

By end of Week 2, you should have:

### Track 1 (Baseline Energy):
- ✅ GPU baseline energy measurement (CSV file)
- ✅ Phase breakdown (rollout/reward/gradient)
- ✅ Training curves showing PPO is working

### Track 2 (Sensitivity):
- ✅ Sensitivity matrix for all layers
- ✅ Policy definitions (A/B/C/D)
- ✅ Recommendation: which layers use MXFP4, MXFP8, FP16

### Both tracks produce inputs for Week 3:
- Baseline energy = target to beat
- Policies = what to implement in FPGA

---

## 🎁 Bonus: What We've Already Created for You

You now have these ready-to-use scripts:

| Script | Purpose | Runtime |
|--------|---------|---------|
| `verify_setup.py` | Check installation | 10 sec |
| `test_model.py` | Verify model works | 2-5 min |
| `test_dataset.py` | Load and split dataset | 1-2 min |
| `gpu_power_monitor.py` | Monitor GPU power | N/A (tool) |
| `monitor_gpu_power.sh` | Shell wrapper for nvidia-smi | N/A (tool) |

**Next**: Implement the main RLHF training loop and sensitivity profiler!
