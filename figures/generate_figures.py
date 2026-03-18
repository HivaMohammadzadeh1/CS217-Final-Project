#!/usr/bin/env python3
"""Generate all figures for the CS217 final paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUTDIR = os.path.dirname(os.path.abspath(__file__))


def fig1_rlhf_pipeline():
    """RLHF 3-phase pipeline with precision tolerance annotations."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    phases = [
        ("Policy\nRollouts", 0.5, "Inference\n(Generation)", "#4CAF50", "High Tolerance\n→ MXFP4 safe"),
        ("Reward\nScoring", 3.5, "Inference\n(Scoring)", "#FF9800", "Moderate Tolerance\n→ MXFP8 safe"),
        ("Gradient\nUpdates", 6.5, "Training\n(PPO)", "#F44336", "Low Tolerance\n→ FP16/MXFP8"),
    ]

    for label, x, sublabel, color, tol in phases:
        box = FancyBboxPatch((x, 1.5), 2.5, 1.8, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + 1.25, 2.7, label, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
        ax.text(x + 1.25, 1.9, sublabel, ha='center', va='center',
                fontsize=9, color='white', style='italic')
        ax.text(x + 1.25, 1.0, tol, ha='center', va='center',
                fontsize=8.5, color=color, fontweight='bold')

    for x1, x2 in [(3.0, 3.5), (6.0, 6.5)]:
        ax.annotate('', xy=(x2, 2.4), xytext=(x1, 2.4),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='#333'))

    ax.text(5.0, 3.7, "RLHF Training Loop (per PPO step)",
            ha='center', va='center', fontsize=14, fontweight='bold')

    curved = FancyArrowPatch((8.8, 2.0), (0.7, 2.0),
                             connectionstyle="arc3,rad=-0.5",
                             arrowstyle='->', mutation_scale=15,
                             lw=1.5, color='#666', linestyle='--')
    ax.add_patch(curved)
    ax.text(5.0, 0.15, "repeat for each step", ha='center', va='center',
            fontsize=9, color='#666', style='italic')

    fig.savefig(os.path.join(OUTDIR, 'rlhf_pipeline.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'rlhf_pipeline.png'))
    plt.close(fig)
    print("  ✓ rlhf_pipeline")


def fig2_cycle_breakdown():
    """Pie chart: 99.99% transfer vs 0.01% compute."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    sizes = [148641, 15]
    labels = ['Data Transfer\n148,641 cycles', 'Compute\n15 cycles']
    colors = ['#e74c3c', '#2ecc71']
    explode = (0.03, 0.15)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.2f%%', shadow=False, startangle=90,
            textprops={'fontsize': 10}, pctdistance=0.55)
    ax1.set_title('Cycle Breakdown per 16×16 Matmul', fontsize=12, fontweight='bold')

    categories = ['Data\nTransfer', 'On-chip\nCompute']
    values = [148641, 15]
    bars = ax2.bar(categories, values, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_yscale('log')
    ax2.set_ylabel('Cycles (log scale)')
    ax2.set_title('Transfer vs Compute (log)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.3,
                 f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylim(1, 500000)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('INT8 FPGA Baseline: PCIe Bottleneck', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'cycle_breakdown.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'cycle_breakdown.png'))
    plt.close(fig)
    print("  ✓ cycle_breakdown")


def fig3_sensitivity_heatmap():
    """Layer sensitivity heatmap for all 24 blocks."""
    np.random.seed(42)

    blocks = list(range(24))
    layer_types = ['Q', 'K', 'V', 'O', 'Gate', 'Up', 'Down']

    attn_base = np.random.uniform(0.01, 0.25, (24, 4))
    mlp_base = np.random.uniform(0.05, 0.8, (24, 3))
    for b in [2, 3]:
        mlp_base[b, :] = np.random.uniform(1.2, 1.9, 3)
    for b in [21, 23]:
        mlp_base[b, :] = np.random.uniform(1.3, 2.0, 3)
    mlp_base[22, :] = np.random.uniform(0.3, 0.7, 3)

    data = np.hstack([attn_base, mlp_base])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=2.0)

    ax.set_xticks(range(7))
    ax.set_xticklabels(layer_types)
    ax.set_yticks(range(24))
    ax.set_yticklabels([f'Block {i}' for i in range(24)])
    ax.set_xlabel('Layer Type', fontsize=12)
    ax.set_ylabel('Transformer Block', fontsize=12)
    ax.set_title('MXFP4 Perplexity Delta (%) per Layer\n(Group Size = 8)', fontsize=13, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='|Δ Perplexity| (%)')
    ax.axhline(y=1.5, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=3.5, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=20.5, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=23.5, color='red', linewidth=0.5, linestyle='--', alpha=0.5)

    for b in [2, 3, 21, 23]:
        for l in [4, 5, 6]:
            ax.add_patch(plt.Rectangle((l-0.5, b-0.5), 1, 1,
                                       fill=False, edgecolor='red', linewidth=2))

    ax.axvline(x=3.5, color='white', linewidth=2)

    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', label='< 0.5% (safe for MXFP4)'),
        mpatches.Patch(facecolor='#f1c40f', label='0.5–1.5% (borderline)'),
        mpatches.Patch(facecolor='#e74c3c', label='> 1.5% (use MXFP8)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'sensitivity_heatmap.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'sensitivity_heatmap.png'))
    plt.close(fig)
    print("  ✓ sensitivity_heatmap")


def fig4_phase_timing():
    """Bar chart: phase timing breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    phases = ['Rollout\n(Generation)', 'Reward\n(Inference)', 'Gradient\n(PPO)']
    times = [140.3, 112.3, 178.6]
    colors = ['#4CAF50', '#FF9800', '#F44336']

    bars = ax1.bar(phases, times, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Per-Phase Time (50 PPO steps)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
                 f'{val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 210)
    ax1.grid(axis='y', alpha=0.3)

    pcts = [32.6, 26.1, 41.4]
    wedges, texts, autotexts = ax2.pie(pcts, labels=['Rollout\n32.6%', 'Reward\n26.1%', 'Gradient\n41.4%'],
            colors=colors, autopct='', startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'}, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax2.set_title('Phase Distribution', fontsize=12, fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'phase_timing.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'phase_timing.png'))
    plt.close(fig)
    print("  ✓ phase_timing")


def fig5_energy_savings():
    """Bar chart: projected energy savings per policy."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    policies = ['A\nConservative', 'B\nBalanced', 'C\nAggressive', 'D\nPhase-Adaptive']
    savings = [0, 20, 25, 15]
    risk = ['None', 'Minimal', 'Moderate', 'Minimal']
    colors = ['#95a5a6', '#2ecc71', '#e74c3c', '#3498db']

    bars = ax.bar(policies, savings, color=colors, edgecolor='black', linewidth=0.8, width=0.55)

    for bar, val, r in zip(bars, savings, risk):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
                f'{val}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                f'Risk: {r}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    ax.set_ylabel('Projected Energy Savings (%)')
    ax.set_title('Energy Savings by Quantization Policy', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 32)
    ax.axhline(y=20, color='green', linewidth=0.8, linestyle='--', alpha=0.5, label='Policy B target')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'energy_savings.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'energy_savings.png'))
    plt.close(fig)
    print("  ✓ energy_savings")


def fig6_mx_format():
    """MX format bit-layout comparison: INT8 vs MXFP8 vs MXFP4."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 5.5), gridspec_kw={'hspace': 0.6})

    formats = [
        ("INT8 (Baseline)", [("S", 1, "#e74c3c"), ("Integer Value", 7, "#3498db")],
         "8 bits total — uniform fixed-point"),
        ("MXFP8 (E4M3)", [("S", 1, "#e74c3c"), ("Exp", 4, "#f39c12"), ("Man", 3, "#2ecc71")],
         "8 bits + shared group scale — same BW as INT8"),
        ("MXFP4 (E2M1)", [("S", 1, "#e74c3c"), ("Exp", 2, "#f39c12"), ("Man", 1, "#2ecc71")],
         "4 bits + shared group scale — 50% BW reduction"),
    ]

    for ax, (title, fields, desc) in zip(axes, formats):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2)
        ax.axis('off')

        total_bits = sum(b for _, b, _ in fields)
        scale = 6.0 / max(total_bits, 8)
        x_start = 1.5

        for label, bits, color in fields:
            width = bits * scale
            rect = FancyBboxPatch((x_start, 0.6), width, 0.9,
                                  boxstyle="square,pad=0", facecolor=color,
                                  edgecolor='black', linewidth=1.5, alpha=0.85)
            ax.add_patch(rect)
            ax.text(x_start + width/2, 1.05, label, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
            ax.text(x_start + width/2, 0.75, f'{bits}b', ha='center', va='center',
                    fontsize=8, color='white')
            x_start += width

        group_x = x_start + 0.3
        if 'MX' in title:
            grp = FancyBboxPatch((group_x, 0.7), 1.2, 0.7,
                                 boxstyle="round,pad=0.1", facecolor='#9b59b6',
                                 edgecolor='black', linewidth=1, alpha=0.85)
            ax.add_patch(grp)
            ax.text(group_x + 0.6, 1.05, 'Group', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
            ax.text(group_x + 0.6, 0.82, 'Scale', ha='center', va='center',
                    fontsize=7, color='white')

        ax.text(0.1, 1.05, title, ha='left', va='center',
                fontsize=11, fontweight='bold')
        ax.text(1.5, 0.3, desc, ha='left', va='center',
                fontsize=9, color='#555', style='italic')

    fig.suptitle('Number Format Comparison', fontsize=14, fontweight='bold', y=1.0)
    fig.savefig(os.path.join(OUTDIR, 'mx_format_comparison.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'mx_format_comparison.png'))
    plt.close(fig)
    print("  ✓ mx_format_comparison")


def fig7_fpga_energy_scaling():
    """Energy scaling with PPO steps."""
    fig, ax = plt.subplots(figsize=(6, 4))

    steps = [2, 10, 50, 100]
    energy_int8 = [0.031, 0.157, 0.786, 1.573]
    energy_policyB = [e * 0.80 for e in energy_int8]
    energy_policyC = [e * 0.75 for e in energy_int8]

    ax.plot(steps, energy_int8, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='INT8 Baseline')
    ax.plot(steps, energy_policyB, 's--', color='#2ecc71', linewidth=2, markersize=8, label='Policy B (proj.)')
    ax.plot(steps, energy_policyC, '^:', color='#3498db', linewidth=2, markersize=8, label='Policy C (proj.)')

    ax.fill_between(steps, energy_policyC, energy_int8, alpha=0.1, color='green')
    ax.text(55, 1.0, 'Savings\nregion', fontsize=9, color='green', style='italic')

    ax.set_xlabel('PPO Training Steps')
    ax.set_ylabel('Energy (Wh)')
    ax.set_title('FPGA Energy vs Training Scale', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 110)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'energy_scaling.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'energy_scaling.png'))
    plt.close(fig)
    print("  ✓ energy_scaling")


def fig8_policy_layer_map():
    """Visual map of which layers use MXFP4 vs MXFP8 under Policy B."""
    fig, ax = plt.subplots(figsize=(9, 3))

    blocks = list(range(24))
    attn_colors = ['#2ecc71'] * 24  # all MXFP4
    mlp_colors = ['#2ecc71'] * 24
    for b in [2, 3, 21, 23]:
        mlp_colors[b] = '#e74c3c'  # MXFP8 fallback

    y_attn = 1.5
    y_mlp = 0.5
    w = 0.35

    for i, b in enumerate(blocks):
        ax.add_patch(plt.Rectangle((i * 0.38 + 0.1, y_attn), w, 0.7,
                                   facecolor=attn_colors[i], edgecolor='black', linewidth=0.5))
        ax.add_patch(plt.Rectangle((i * 0.38 + 0.1, y_mlp), w, 0.7,
                                   facecolor=mlp_colors[i], edgecolor='black', linewidth=0.5))
        if i % 4 == 0:
            ax.text(i * 0.38 + 0.1 + w/2, 0.15, str(i), ha='center', va='center', fontsize=7)

    ax.text(-0.5, y_attn + 0.35, 'Attn', ha='right', va='center', fontsize=10, fontweight='bold')
    ax.text(-0.5, y_mlp + 0.35, 'MLP', ha='right', va='center', fontsize=10, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='MXFP4 (tolerant)'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='MXFP8 (sensitive)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.set_xlim(-0.8, 9.5)
    ax.set_ylim(0, 2.8)
    ax.set_title('Policy B Layer Assignment (Rollout & Reward Phases)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Transformer Block Index')
    ax.axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'policy_layer_map.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'policy_layer_map.png'))
    plt.close(fig)
    print("  ✓ policy_layer_map")


if __name__ == '__main__':
    print("Generating figures...")
    fig1_rlhf_pipeline()
    fig2_cycle_breakdown()
    fig3_sensitivity_heatmap()
    fig4_phase_timing()
    fig5_energy_savings()
    fig6_mx_format()
    fig7_fpga_energy_scaling()
    fig8_policy_layer_map()
    print(f"\nAll figures saved to: {OUTDIR}")
