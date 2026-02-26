# Milestone Report Guide

This guide shows you how to generate all results and compile your milestone report.

---

## Quick Start

### On Your AWS Server

```bash
# 1. SSH to your server
ssh -i "hiva_cs217.pem" ubuntu@ec2-52-23-236-230.compute-1.amazonaws.com

# 2. Navigate to project
cd ~/CS217-Final-Project
source venv/bin/activate

# 3. Pull latest code
git pull

# 4. Run ALL analyses (generates everything for the report)
bash RUN_ALL_ANALYSES.sh
```

This script will:
- ✅ Verify FPGA hardware
- ✅ Measure FPGA performance
- ✅ Calculate energy for all workloads
- ✅ Generate comparison reports
- ✅ Create LaTeX snippets
- ✅ Generate comprehensive markdown summary

**Output**: `results/milestone_report/` with all results

---

## What Gets Generated

### 1. Performance Measurements
- `fpga_status.txt` - FPGA hardware status
- `fpga_performance.txt` - Raw 10-iteration measurements
- `fpga_cycles.txt` - Average cycles per matmul

### 2. Energy Calculations
- `fpga_energy_2steps.json` - Energy for 2 PPO steps
- `fpga_energy_10steps.json` - Energy for 10 PPO steps
- `fpga_energy_50steps.json` - Energy for 50 PPO steps
- `fpga_energy_100steps.json` - Energy for 100 PPO steps

### 3. Analysis Reports
- `MILESTONE_RESULTS.md` - Comprehensive markdown summary
- `results_latex_snippet.tex` - LaTeX tables ready to paste
- `cpu_vs_fpga_comparison.txt` - CPU comparison (if available)

### 4. LaTeX Report
- `report/milestone_report.tex` - Complete milestone report in LaTeX

---

## Compiling the LaTeX Report

### Option 1: On Your Local Machine

```bash
# Copy the files from server to local
scp -i "hiva_cs217.pem" -r \
  ubuntu@ec2-52-23-236-230.compute-1.amazonaws.com:~/CS217-Final-Project/report \
  ~/Downloads/

# Compile LaTeX
cd ~/Downloads/report
pdflatex milestone_report.tex
bibtex milestone_report  # If you have references
pdflatex milestone_report.tex
pdflatex milestone_report.tex
```

### Option 2: On Overleaf

1. Download the report directory from GitHub
2. Upload to Overleaf
3. Compile

### Option 3: On AWS Server

```bash
# Install LaTeX (if not already installed)
sudo apt-get update
sudo apt-get install texlive-latex-extra texlive-fonts-recommended

# Compile
cd ~/CS217-Final-Project/report
pdflatex milestone_report.tex
```

---

## Understanding Your Results

### FPGA Performance (from measurements)

Your Lab 1 FPGA achieved:
- **Total cycles**: 148,656 per 16×16 matmul
- **Data transfer**: 148,641 cycles (99.99%)
- **Computation**: 15 cycles (0.01%)

**What this means**:
- FPGA computation is very fast (15 cycles!)
- PCIe data transfer is the bottleneck
- This motivates compression (MX formats)

### Energy Estimates (from calculations)

For 50 PPO steps (136,000 matmuls):
- **Time**: ~1,686 minutes (~28 hours)
- **Energy**: ~0.98 Wh

**Why so slow?**
- Each matmul takes 594 μs due to PCIe overhead
- This is 1000× slower than the computation alone
- Batching and compression will help significantly

### Comparison with CPU

Once your CPU baseline finishes:
- CPU will likely be faster (no PCIe overhead)
- But FPGA has lower power (35W vs 150W)
- MX format optimization aims to close the speed gap

---

## Customizing the Report

### Update Tables with Your Data

The script automatically fills in these placeholders:
- `FPGA_CYCLES_VALUE` → Your measured cycles
- `DATA_TRANSFER_CYCLES` → Calculated data transfer
- `TIME_PER_MATMUL_US` → Time in microseconds
- `ENERGY_PER_MATMUL_UJ` → Energy in microjoules

These are in `results/milestone_report/results_latex_snippet.tex`

### Add Your CPU Comparison

Once CPU baseline completes, add this to the report:

```latex
\subsection{CPU vs FPGA Comparison}
\begin{table}[h]
\centering
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Metric} & \textbf{CPU} & \textbf{FPGA (INT8)} \\ \midrule
Time (50 steps) & XX.X min & 1686.1 min \\
Energy (50 steps) & X.XXX Wh & 0.982 Wh \\
Average Power & XXX W & 35 W \\
\bottomrule
\end{tabular}
\caption{CPU vs FPGA baseline comparison}
\end{table}
```

Replace `XX.X` with your CPU measurements from:
```bash
cat results/cpu_baseline_50steps/energy_summary.csv
```

### Add Figures

If you want to add plots, place them in `report/figures/` and include:

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/your_plot.png}
\caption{Your caption here}
\label{fig:your_label}
\end{figure}
```

---

## Verification Checklist

Before submitting, verify:

- [ ] `RUN_ALL_ANALYSES.sh` completed without errors
- [ ] All JSON files generated in `results/milestone_report/`
- [ ] `MILESTONE_RESULTS.md` contains your measurements
- [ ] LaTeX compiles without errors
- [ ] Tables show your actual measured values (not placeholders)
- [ ] Report includes discussion of PCIe bottleneck
- [ ] Next steps section is updated for your timeline

---

## Common Issues

### Script fails: "AWS_FPGA_REPO_DIR not set"
```bash
source ~/aws-fpga/sdk_setup.sh
```

### LaTeX compilation fails: "neurips_2019.sty not found"
Copy the style file:
```bash
cp ~/Downloads/proposal_extracted/template/neurips_2019.sty report/
```

### CPU baseline not found
This is expected if CPU run hasn't finished yet. The script will:
- Generate FPGA-only report
- Add note that CPU comparison is pending

---

## Quick Commands Reference

```bash
# Run everything
bash RUN_ALL_ANALYSES.sh

# View markdown summary
cat results/milestone_report/MILESTONE_RESULTS.md

# View LaTeX snippet
cat results/milestone_report/results_latex_snippet.tex

# Compile report
cd report && pdflatex milestone_report.tex

# Check what's generated
ls -lh results/milestone_report/
```

---

## For Your Report

### Key Points to Emphasize

1. **FPGA Hardware Works** ✅
   - 0 test errors
   - Validated on real AWS F2 hardware
   - Reproducible measurements

2. **Performance Characterized** ✅
   - 148,656 cycles per operation
   - 99.99% data transfer, 0.01% compute
   - Clear bottleneck identified

3. **Energy Framework Built** ✅
   - Automated energy calculations
   - Projections for different workloads
   - Ready for MX format comparison

4. **Insights Gained** ✅
   - PCIe overhead is the real problem
   - Compression (MXFP4) addresses this directly
   - 50% bandwidth reduction → ~25% energy savings

### What to Say About Being "Slow"

The FPGA baseline is slower than CPU, but **this is expected and okay**:
- Small matrices (16×16) have high overhead/compute ratio
- MX format optimization targets the 99.99% bottleneck
- Our research question (INT8 vs MX on FPGA) is still valid
- Commercial systems use batching to amortize overhead

---

## Timeline

- **Week 1-2 (Done)**: Infrastructure + baseline ✅
- **Week 3**: Sensitivity profiling + MX datapath design
- **Week 4**: FPGA synthesis + integration
- **Week 5**: Experiments + measurements
- **Week 6**: Analysis + final report

You're on track! 🎉

---

## Need Help?

Check these files:
- `THREE_SOLUTIONS_README.md` - Complete technical guide
- `QUICK_SETUP.md` - Quick start guide
- `LAB1_FPGA_MEASUREMENT_GUIDE.md` - Detailed FPGA guide

Run scripts with `-h` or `--help` for options:
```bash
python integration/calculate_fpga_energy.py --help
python integration/compare_cpu_vs_fpga.py --help
```

All scripts include error messages with suggestions!
