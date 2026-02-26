# Layer-Adaptive MX Format Quantization for RLHF Energy Optimization on FPGAs

**CS217: Hardware Accelerators for Machine Learning - Final Project**
Stanford University | February 2026

## Team Members
- Hiva Zaad 
- Grant Griffith 
- Daniel Adkins 

## Project Overview

This project investigates whether layer-adaptive Microscaling (MX) format quantization can reduce the energy consumption of RLHF (Reinforcement Learning from Human Feedback) training workloads on FPGA hardware, compared to a standard GPU baseline.

### Research Question

What combination of MXFP4 and MXFP8 formats — applied layer-adaptively across RLHF phases — minimizes total energy consumption on an FPGA, without degrading alignment quality below an acceptable threshold?

## Technical Approach

### Model & Dataset
- **Model**: Qwen2.5-0.5B-Instruct (494M parameters, ~942MB in FP16)
- **Dataset**: Fixed HH-RLHF subset (1,000 train samples + 200 test samples, seed=42)
  - **HuggingFace Hub**: [hivamoh/cs217-rlhf-dataset](https://huggingface.co/datasets/hivamoh/cs217-rlhf-dataset)
  - **Local Path**: `data/cs217_rlhf_dataset/` (included in repo)
- **Algorithm**: PPO (Proximal Policy Optimization) via HuggingFace TRL library

### Hardware
- **Baseline**: AWS GPU instance (g4dn.xlarge, NVIDIA T4)
- **Target**: AWS F2 FPGA (Xilinx VU9P)
- **Toolchain**: Vivado HLS for synthesis

### Key Innovation
Layer-adaptive precision selection across RLHF phases:
- **Policy rollouts**: Can tolerate MXFP4 (high tolerance for lower precision)
- **Reward model inference**: Moderate tolerance (MXFP8)
- **Policy gradient updates**: Lower tolerance (FP16 or selective MXFP8)

## Repository Structure

```
├── pytorch_profiling/     # PyTorch quantization experiments & layer sensitivity analysis
│   ├── sensitivity_profiler.py        # Main profiling script (finds sensitive layers)
│   ├── define_policies.py             # Generates quantization policies A/B/C/D
│   └── test_profiler_setup.py         # Quick verification test
├── baseline_energy/       # GPU baseline measurement scripts
│   ├── config.py                      # All hyperparameters in one place
│   ├── rlhf_baseline.py              # Main RLHF training with energy measurement
│   ├── process_energy.py             # Calculates energy from power logs
│   ├── gpu_power_monitor.py          # GPU power monitoring class
│   ├── create_fixed_dataset.py       # Creates fixed HH-RLHF subset
│   ├── verify_hub_dataset.py         # Verifies HuggingFace Hub access
│   ├── verify_setup.py               # Tests package installation
│   ├── test_model.py                 # Tests model loading
│   ├── test_dataset.py               # Tests dataset loading
│   └── test_rlhf_setup.py            # Tests RLHF components
├── systemc/              # MX datapath design files (SystemC/HLS)
├── fpga/                 # FPGA synthesis and deployment files
├── integration/          # Adaptive controller and RLHF loop integration
├── data/                 # Fixed dataset (locally cached)
│   └── cs217_rlhf_dataset/           # 1000 train + 200 test samples
├── results/              # Experimental results, CSVs, plots
├── report/               # LaTeX source for final report
├── docs/                 # Project documentation
│   └── CS217_Project_Proposal.pdf    # Original project proposal
├── requirements.txt      # Python dependencies
├── setup_environment.sh  # Automated environment setup
├── TESTING_COMPLETE.md   # Testing verification results
└── DATASET_COMPLETE.md   # Dataset creation documentation
```

## Setup Instructions

### Quick Start (Automated)

```bash
# Clone the repository
git clone https://github.com/HivaMohammadzadeh1/CS217-Final-Project.git
cd CS217-Final-Project

# Run automated setup (creates venv, installs dependencies)
bash setup_environment.sh

# Activate environment
source venv/bin/activate
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python baseline_energy/verify_setup.py
```

### Verify Everything Works

Run these quick tests to ensure all components are working:

```bash
source venv/bin/activate

# 1. Verify packages (5 sec)
python baseline_energy/verify_setup.py

# 2. Test model loading (1 min on CPU, faster on GPU)
python baseline_energy/test_model.py

# 3. Test dataset loading (10 sec)
python baseline_energy/test_dataset.py

# 4. Test RLHF components (10 sec)
python baseline_energy/test_rlhf_setup.py

# 5. Test profiler components (1 min)
python pytorch_profiling/test_profiler_setup.py

# 6. Verify HuggingFace Hub dataset access (10 sec)
python baseline_energy/verify_hub_dataset.py
```

All tests should pass with ✅ marks. See `TESTING_COMPLETE.md` for detailed test results.

### AWS GPU Setup (For Full Experiments)

#### Recommended Instance
- **Type**: g4dn.xlarge (NVIDIA T4, 16GB GPU)
- **AMI**: Deep Learning AMI (Ubuntu)
- **Storage**: 100GB EBS

#### Setup on AWS Instance

```bash
# Clone repository
git clone https://github.com/HivaMohammadzadeh1/CS217-Final-Project.git
cd CS217-Final-Project

# Run setup
bash setup_environment.sh
source venv/bin/activate

# Verify GPU available
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Run quick test
python baseline_energy/test_model.py
```

### AWS FPGA Setup (For FPGA Experiments)

- Confirm AWS F2 instance access via course credits
- Set up AWS CLI credentials
- Ensure Vivado HLS is available
- Source the Vivado environment: `source /path/to/Vivado/settings64.sh`

## Fixed Dataset for Reproducibility

To ensure reproducible experiments across all runs, we've created a fixed subset of the Anthropic HH-RLHF dataset:

### Dataset Details
- **HuggingFace Hub**: [hivamoh/cs217-rlhf-dataset](https://huggingface.co/datasets/hivamoh/cs217-rlhf-dataset)
- **Training Set**: 1,000 samples from original train split
- **Test Set**: 200 samples from original test split
- **Random Seed**: 42 (fixed for reproducibility)
- **Total Size**: 1,200 preference pairs

### Loading the Dataset

**Option 1: From HuggingFace Hub** (Recommended)
```python
from datasets import load_dataset

# Load the fixed dataset
dataset = load_dataset("hivamoh/cs217-rlhf-dataset")
train_data = dataset['train']  # 1000 samples
test_data = dataset['test']    # 200 samples
```

**Option 2: From Local Cache**
```python
from datasets import load_from_disk

# Load from local directory
dataset = load_from_disk("data/cs217_rlhf_dataset")
train_data = dataset['train']
test_data = dataset['test']
```

### Creating/Updating the Dataset

If you need to regenerate or modify the dataset:

```bash
# Create fixed dataset locally
python baseline_energy/create_fixed_dataset.py \
  --train-size 1000 \
  --test-size 200 \
  --seed 42

# Upload to HuggingFace Hub (requires authentication)
huggingface-cli login
python baseline_energy/create_fixed_dataset.py \
  --push-to-hub \
  --hub-name YOUR_USERNAME/cs217-rlhf-dataset
```

See `DATASET_COMPLETE.md` for full documentation.

## MX Format Policies

We will test four layer-adaptive precision policies:

| Policy | Rollouts | Reward Inference | Gradient Updates | Description |
|--------|----------|------------------|------------------|-------------|
| **A - Conservative** | MXFP8 all layers | MXFP8 all layers | FP16 all layers | Minimal risk, modest energy savings |
| **B - Balanced** | MXFP4 tolerant, FP8 sensitive | MXFP4 tolerant, FP8 sensitive | FP16 all layers | Based on sensitivity results |
| **C - Aggressive** | MXFP4 all layers | MXFP4 all layers | MXFP8 sensitive, FP4 rest | Maximum energy savings |
| **D - Phase-Adaptive** | MXFP4 (most layers) | MXFP8 (most layers) | FP16 (most layers) | Phase-aware, research target |

## Running Experiments

### Quick Tests (CPU or GPU)

These tests verify all components without running full training:

```bash
source venv/bin/activate

# Test all components
python baseline_energy/verify_setup.py
python baseline_energy/test_model.py
python baseline_energy/test_dataset.py
python baseline_energy/test_rlhf_setup.py
python pytorch_profiling/test_profiler_setup.py
```

### Full RLHF Baseline (Requires GPU)

Run the GPU baseline to measure energy consumption:

```bash
source venv/bin/activate

# Run RLHF baseline with energy measurement
python baseline_energy/rlhf_baseline.py \
  --steps 100 \
  --batch-size 8 \
  --output results/baseline_run1

# Process energy logs
python baseline_energy/process_energy.py \
  --results results/baseline_run1
```

**Expected Runtime**: 1-2 hours on AWS g4dn.xlarge (T4 GPU)

### Sensitivity Profiling (Requires GPU)

Profile layer-wise sensitivity to quantization:

```bash
source venv/bin/activate

# Run sensitivity profiling (tests all layers)
python pytorch_profiling/sensitivity_profiler.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset hivamoh/cs217-rlhf-dataset \
  --output results/sensitivity_matrix.csv

# Generate quantization policies based on results
python pytorch_profiling/define_policies.py \
  --sensitivity results/sensitivity_matrix.csv \
  --output results/policies.json
```

**Expected Runtime**: 2-4 hours on GPU (profiles 168 layers × 2 formats)

### Expected Outputs

After running experiments, you'll have:

```
results/
├── baseline_run1/
│   ├── power_log_baseline.csv        # nvidia-smi power samples
│   ├── phase_timing.json             # Rollout/reward/gradient times
│   ├── energy_summary.json           # Total energy by phase
│   └── training_metrics.json         # PPO training metrics
├── sensitivity_matrix.csv            # Layer sensitivity scores
└── policies.json                     # A/B/C/D quantization policies
```

## Milestones

- [x] **Week 1**: Environment Setup & Proposal
  - ✅ Repository structure created
  - ✅ Documentation written
  - ✅ Proposal submitted

- [x] **Week 2**: PyTorch Profiling + GPU Baseline Energy (In Progress)
  - ✅ Fixed HH-RLHF dataset created (1000 train, 200 test)
  - ✅ Dataset uploaded to HuggingFace Hub
  - ✅ Environment setup scripts created
  - ✅ All testing scripts verified
  - ✅ RLHF baseline script implemented
  - ✅ Energy measurement framework ready
  - ✅ Sensitivity profiler implemented
  - ✅ Policy generator ready
  - ⏳ Awaiting GPU instance for full runs

- [ ] **Week 3**: SystemC MX Datapath Design
  - [ ] Design MXFP4/MXFP8 arithmetic units
  - [ ] Implement shared exponent logic
  - [ ] Create testbenches
  - [ ] Validate against PyTorch results

- [ ] **Week 4**: FPGA Synthesis + RLHF Integration
  - [ ] Synthesize MX datapath on AWS F2
  - [ ] Integrate with RLHF loop
  - [ ] Test policy switching
  - [ ] Validate accuracy

- [ ] **Week 5**: Energy Experiments & Measurement
  - [ ] Run all policies (A/B/C/D)
  - [ ] Measure FPGA energy consumption
  - [ ] Compare against GPU baseline
  - [ ] Generate Pareto curves

- [ ] **Week 6**: Analysis & Final Report
  - [ ] Analyze energy/quality tradeoffs
  - [ ] Write final report
  - [ ] Create presentation
  - [ ] Submit deliverables

## Project Status

### ✅ Completed (Milestone 1-2)

- Repository structure and documentation
- Fixed reproducible dataset (hivamoh/cs217-rlhf-dataset)
- Complete environment setup and testing framework
- RLHF baseline implementation with energy measurement
- Sensitivity profiler for layer-wise quantization analysis
- Policy generator for A/B/C/D quantization strategies
- All components tested and verified on CPU

### 🔄 In Progress

- Awaiting AWS GPU instance for full baseline runs
- Ready to execute sensitivity profiling
- Preparing for SystemC MX datapath design

### 📋 Upcoming

- SystemC/HLS implementation of MX arithmetic
- FPGA synthesis and deployment
- Full energy experiments across all policies
- Final analysis and report

## Energy Measurement Protocol

### GPU Baseline

Our implementation uses `nvidia-smi` for power monitoring:

```bash
# Automatic monitoring (handled by rlhf_baseline.py)
python baseline_energy/rlhf_baseline.py --steps 100 --output results/baseline

# Manual monitoring (if needed)
nvidia-smi dmon -s pmu -i 0 -d 0.1 > power_log.csv &
python baseline_energy/rlhf_baseline.py
```

**Features**:
- 10 Hz power sampling (100ms intervals)
- Phase-based timing (rollout, reward inference, gradient update)
- Automatic energy calculation per phase
- CSV logs for reproducibility

**Energy Calculation**:
```python
Energy_phase (J) = Avg_Power (W) × Time_phase (s)
Total_Energy = Σ Energy_phase across all PPO steps
```

### FPGA

- Use Xilinx Power Estimator (XPE) post-synthesis for static estimates
- Measure on-board power rails during inference
- Compare against GPU baseline for same workload

### Fixed Measurement Run

All experiments use identical parameters:
- **PPO Steps**: 100 update iterations
- **Dataset**: 1000 training samples (fixed seed=42)
- **Batch Size**: 8
- **Sequence Length**: 512 tokens
- **Model**: Qwen2.5-0.5B (494M parameters)

## Success Criteria

| Level | Criteria |
|-------|----------|
| **Minimum** | FPGA runs dual-precision MX datapath; at least one policy shows measurable energy savings vs GPU FP16 baseline |
| **Target** | Policy D (phase-adaptive) achieves >30% energy reduction vs baseline with <10% win rate drop |
| **Stretch** | >50% energy reduction with <5% win rate drop; generalizes across multiple policies |

## Key Terms

- **RLHF**: Reinforcement Learning from Human Feedback
- **PPO**: Proximal Policy Optimization
- **MXFP4/MXFP8**: Microscaling Float 4-bit/8-bit formats with shared scale factors
- **Group Size**: Number of values sharing a single scale factor in MX format
- **Win Rate**: % of test cases where the trained model's output is preferred over baseline
- **Pareto Curve**: Graph showing tradeoff between energy savings and quality loss

## Available Scripts and Tools

### Setup and Verification
| Script | Purpose | Runtime |
|--------|---------|---------|
| `setup_environment.sh` | Automated environment setup | 2-5 min |
| `baseline_energy/verify_setup.py` | Test package installation | 5 sec |
| `baseline_energy/test_model.py` | Test model loading | ~1 min |
| `baseline_energy/test_dataset.py` | Test dataset loading | 10 sec |
| `baseline_energy/test_rlhf_setup.py` | Test RLHF components | 10 sec |
| `pytorch_profiling/test_profiler_setup.py` | Test profiler | ~1 min |
| `baseline_energy/verify_hub_dataset.py` | Test HF Hub access | 10 sec |

### Dataset Management
| Script | Purpose | Runtime |
|--------|---------|---------|
| `baseline_energy/create_fixed_dataset.py` | Create/upload fixed dataset | 2-5 min |
| `baseline_energy/upload_dataset_readme.py` | Update HF Hub README | 5 sec |

### Experiments (Require GPU)
| Script | Purpose | Runtime |
|--------|---------|---------|
| `baseline_energy/rlhf_baseline.py` | Run RLHF baseline with energy measurement | 1-2 hours |
| `baseline_energy/process_energy.py` | Calculate energy from power logs | 30 sec |
| `pytorch_profiling/sensitivity_profiler.py` | Profile layer quantization sensitivity | 2-4 hours |
| `pytorch_profiling/define_policies.py` | Generate quantization policies | 5 min |

### Configuration
| File | Purpose |
|------|---------|
| `baseline_energy/config.py` | All hyperparameters (learning rate, batch size, etc.) |
| `requirements.txt` | Python dependencies |

### Documentation
| File | Purpose |
|------|---------|
| `README.md` | Main project documentation (this file) |
| `TESTING_COMPLETE.md` | Detailed testing verification results |
| `DATASET_COMPLETE.md` | Dataset creation and usage documentation |
| `MILESTONE2_QUICKSTART.md` | Quick start guide for Milestone 2 |
| `MILESTONE2_COMPLETE.md` | Complete Milestone 2 checklist |
| `docs/CS217_Project_Proposal.pdf` | Original project proposal |

See individual script files for detailed usage and command-line options.

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Verify packages
python baseline_energy/verify_setup.py
```

**2. CUDA Out of Memory**
```bash
# Reduce batch size in config.py
BATCH_SIZE = 4  # Instead of 8
```

**3. Dataset Not Found**
```bash
# Re-download from HuggingFace Hub
python baseline_energy/verify_hub_dataset.py
```

**4. HuggingFace Authentication**
```bash
# Login to HuggingFace
pip install huggingface_hub
huggingface-cli login
```

### Getting Help

- Check documentation files in the repository
- Review test scripts for usage examples
- See [GitHub Issues](https://github.com/HivaMohammadzadeh1/CS217-Final-Project/issues)

## References

- [Microsoft MX Format Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [HuggingFace TRL Library](https://github.com/huggingface/trl)
- [Anthropic HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [Our Fixed Dataset](https://huggingface.co/datasets/hivamoh/cs217-rlhf-dataset)
- [Qwen2.5 Model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [AWS F2 FPGA Instances](https://aws.amazon.com/ec2/instance-types/f1/)

## License

This project is for academic purposes as part of Stanford CS217.
