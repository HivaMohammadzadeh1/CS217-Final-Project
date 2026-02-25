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
- **Model**: Qwen2.5-0.5B-Instruct (500M parameters)
- **Dataset**: Anthropic HH-RLHF (1,000 preference pairs: 800 train / 200 eval)
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
├── baseline_energy/       # GPU baseline measurement scripts
├── systemc/              # MX datapath design files (SystemC/HLS)
├── fpga/                 # FPGA synthesis and deployment files
├── integration/          # Adaptive controller and RLHF loop integration
├── results/              # Experimental results, CSVs, plots
├── report/               # LaTeX source for final report
└── requirements.txt      # Python dependencies
```

## Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd CS217-Project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. AWS Access
- Confirm AWS F2 instance access via course credits
- Set up AWS CLI credentials

### 3. Vivado HLS
- Ensure Vivado HLS is available on FarmShare or AWS
- Source the Vivado environment: `source /path/to/Vivado/settings64.sh`

## MX Format Policies

We will test four layer-adaptive precision policies:

| Policy | Rollouts | Reward Inference | Gradient Updates | Description |
|--------|----------|------------------|------------------|-------------|
| **A - Conservative** | MXFP8 all layers | MXFP8 all layers | FP16 all layers | Minimal risk, modest energy savings |
| **B - Balanced** | MXFP4 tolerant, FP8 sensitive | MXFP4 tolerant, FP8 sensitive | FP16 all layers | Based on sensitivity results |
| **C - Aggressive** | MXFP4 all layers | MXFP4 all layers | MXFP8 sensitive, FP4 rest | Maximum energy savings |
| **D - Phase-Adaptive** | MXFP4 (most layers) | MXFP8 (most layers) | FP16 (most layers) | Phase-aware, research target |

## Milestones

- [x] **Week 1**: Environment Setup & Proposal
- [ ] **Week 2**: PyTorch Profiling + GPU Baseline Energy
- [ ] **Week 3**: SystemC MX Datapath Design
- [ ] **Week 4**: FPGA Synthesis + RLHF Integration
- [ ] **Week 5**: Energy Experiments & Measurement
- [ ] **Week 6**: Analysis & Final Report

## Energy Measurement Protocol

### GPU Baseline
```bash
# Run nvidia-smi in background to log power
nvidia-smi dmon -s u -i 0 -d 1 -c 1000 > power_log.csv &
# Run training
python baseline_energy/run_rlhf_baseline.py
```

### FPGA
- Use Xilinx Power Estimator (XPE) post-synthesis for static estimates
- Measure on-board power rails during inference

### Fixed Measurement Run
- 100 PPO update steps
- 500 examples from HH-RLHF
- Batch size: 8
- Sequence length: 512 tokens

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

## References

- [Microsoft MX Format Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [HuggingFace TRL Library](https://github.com/huggingface/trl)
- [Anthropic HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [Qwen2.5 Model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## License

This project is for academic purposes as part of Stanford CS217.
