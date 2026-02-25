# Final Report

This directory contains LaTeX source files for the final project report.

## Report Structure

```
report/
├── main.tex                  # Main document
├── sections/
│   ├── 01_abstract.tex
│   ├── 02_introduction.tex
│   ├── 03_background.tex
│   ├── 04_technical_approach.tex
│   ├── 05_experimental_setup.tex
│   ├── 06_results.tex
│   ├── 07_discussion.tex
│   └── 08_conclusion.tex
├── figures/                  # Plots and diagrams
├── references.bib           # Bibliography
└── cs217_report_template.cls # Custom document class (if needed)
```

## Section Guidelines

### 1. Abstract (150 words)
- State the problem (RLHF energy consumption)
- Describe approach (layer-adaptive MX quantization)
- Report key result (X% energy savings with Y% quality loss)

### 2. Introduction (~1.5 pages)
- Motivation: Why RLHF energy matters
- Research question
- Key contributions

### 3. Background (~2 pages)
- MX formats (MXFP4, MXFP8)
- RLHF and PPO
- FPGA acceleration for ML

### 4. Technical Approach (~3 pages)
- System architecture diagram
- MX datapath design (MXFP4/FP8 PEs)
- Adaptive controller
- RLHF integration

### 5. Experimental Setup (~1.5 pages)
- Model: Qwen2.5-0.5B
- Dataset: HH-RLHF (1000 pairs)
- Hardware: AWS F2 FPGA, T4 GPU
- Energy measurement protocol
- Policy definitions (A/B/C/D)

### 6. Results (~3 pages)
- Energy measurements (all policies)
- Quality metrics (win rate, KL divergence)
- Pareto curve analysis
- Phase breakdown
- Layer sensitivity heatmap

### 7. Discussion (~2 pages)
- Which policies win and why
- Phase-level insights (rollout vs reward vs gradient)
- Layer-level insights (attention vs FFN)
- Comparison to related work

### 8. Conclusion (~0.5 pages)
- Summary of findings
- Future work: larger models, different algorithms, custom silicon

## Build Instructions

```bash
# Compile LaTeX
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Or use latexmk
latexmk -pdf main.tex

# Clean auxiliary files
latexmk -c
```

## Figures to Include

1. System architecture diagram
2. MX datapath block diagram
3. Energy comparison bar chart (all policies)
4. Pareto curve (energy savings vs quality loss)
5. Phase breakdown (rollout/reward/gradient energy)
6. Layer sensitivity heatmap
7. Training curves (reward over time)

## Key Results to Report

- Total energy savings: "Policy D achieves 40.5% energy reduction"
- Quality preservation: "with only 3.8% quality loss"
- Phase insights: "Rollout phase contributes 55% of energy savings"
- Layer insights: "FFN layers tolerate MXFP4 better than attention"
- Hardware efficiency: "MXFP4 PE uses 50% fewer DSPs than MXFP8"

## Writing Tips

- Use active voice: "We design..." not "A design is presented..."
- Be quantitative: Include specific numbers and percentages
- Explain tradeoffs: Why does Policy C have high savings but low quality?
- Compare to baseline: Always reference GPU FP16 baseline
- Discuss limitations: Model size, dataset size, FPGA vs ASIC

## Target Length

8-10 pages (including figures and references)
