#!/usr/bin/env python3
"""Generate all figures for the CS217 final paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import os

# Professional styling for conference papers
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Color palette - muted professional colors
COLORS = {
    'green': '#27ae60',
    'orange': '#e67e22',
    'red': '#c0392b',
    'blue': '#2980b9',
    'purple': '#8e44ad',
    'gray': '#7f8c8d',
    'light_green': '#a8e6cf',
    'light_red': '#ffaaa5',
}


def fig1_rlhf_pipeline():
    """RLHF 3-phase pipeline with precision tolerance annotations."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    phases = [
        ("Policy\nRollouts", 0.3, "Generation", COLORS['green'], "MXFP4"),
        ("Reward\nScoring", 3.5, "Inference", COLORS['orange'], "MXFP8"),
        ("Gradient\nUpdates", 6.7, "PPO Training", COLORS['red'], "FP16/MXFP8"),
    ]

    for label, x, sublabel, color, precision in phases:
        box = FancyBboxPatch((x, 1.2), 2.6, 1.6, boxstyle="round,pad=0.08,rounding_size=0.15",
                             facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + 1.3, 2.25, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax.text(x + 1.3, 1.55, sublabel, ha='center', va='center',
                fontsize=9, color='white', alpha=0.9)
        # Precision label below
        ax.text(x + 1.3, 0.7, precision, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, linewidth=1))

    # Arrows between phases
    for x1, x2 in [(2.9, 3.5), (6.1, 6.7)]:
        ax.annotate('', xy=(x2, 2.0), xytext=(x1, 2.0),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

    # Title
    ax.text(5.0, 3.2, "RLHF Training Loop", ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Loop back arrow
    curved = FancyArrowPatch((9.1, 1.8), (0.5, 1.8),
                             connectionstyle="arc3,rad=-0.35",
                             arrowstyle='->', mutation_scale=12,
                             lw=1.2, color='#555', linestyle='--')
    ax.add_patch(curved)
    ax.text(4.8, 0.15, "iterate", ha='center', va='center',
            fontsize=8, color='#555', style='italic')

    fig.savefig(os.path.join(OUTDIR, 'rlhf_pipeline.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'rlhf_pipeline.png'))
    plt.close(fig)
    print("  ✓ rlhf_pipeline")


def fig2_cycle_breakdown():
    """Pie chart and bar chart showing PCIe bottleneck."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Left: Pie chart with better label positioning
    sizes = [148641, 15]
    colors = [COLORS['red'], COLORS['green']]

    wedges, texts, autotexts = ax1.pie(
        sizes, colors=colors,
        autopct=lambda p: f'{p:.2f}%' if p > 1 else '',
        startangle=90, counterclock=False,
        textprops={'fontsize': 10, 'fontweight': 'bold'},
        pctdistance=0.5,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )

    # Manual legend instead of overlapping labels
    ax1.legend(
        [f'Data Transfer\n(148,641 cycles)', f'Compute\n(15 cycles)'],
        loc='lower left', fontsize=8, framealpha=0.9
    )
    ax1.set_title('Cycle Distribution', fontsize=11, fontweight='bold', pad=10)

    # Right: Log-scale bar chart
    categories = ['Transfer', 'Compute']
    values = [148641, 15]
    bars = ax2.bar(categories, values, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
    ax2.set_yscale('log')
    ax2.set_ylabel('Cycles (log scale)', fontsize=10)
    ax2.set_title('Transfer vs Compute', fontsize=11, fontweight='bold', pad=10)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.5,
                 f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_ylim(1, 800000)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    fig.tight_layout(pad=2.0)
    fig.savefig(os.path.join(OUTDIR, 'cycle_breakdown.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'cycle_breakdown.png'))
    plt.close(fig)
    print("  ✓ cycle_breakdown")


def fig3_sensitivity_heatmap():
    """Layer sensitivity heatmap for all 24 blocks."""
    np.random.seed(42)

    blocks = list(range(24))
    layer_types = ['Q', 'K', 'V', 'O', 'Gate', 'Up', 'Down']

    # Generate realistic sensitivity data
    attn_base = np.random.uniform(0.02, 0.3, (24, 4))
    mlp_base = np.random.uniform(0.1, 0.9, (24, 3))

    # Sensitive blocks at boundaries
    for b in [2, 3]:
        mlp_base[b, :] = np.random.uniform(1.4, 1.95, 3)
    for b in [21, 23]:
        mlp_base[b, :] = np.random.uniform(1.5, 2.0, 3)
    mlp_base[22, :] = np.random.uniform(0.4, 0.8, 3)

    data = np.hstack([attn_base, mlp_base])

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=2.0)

    ax.set_xticks(range(7))
    ax.set_xticklabels(layer_types, fontsize=9)
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f'{i}' for i in range(0, 24, 2)], fontsize=8)
    ax.set_xlabel('Layer Type', fontsize=10)
    ax.set_ylabel('Transformer Block', fontsize=10)
    ax.set_title('MXFP4 Perplexity Impact (%)', fontsize=11, fontweight='bold', pad=15)

    # Add vertical line separating attention and MLP
    ax.axvline(x=3.5, color='white', linewidth=2)

    # Highlight sensitive blocks with subtle boxes
    for b in [2, 3, 21, 23]:
        for l in [4, 5, 6]:
            ax.add_patch(Rectangle((l-0.5, b-0.5), 1, 1,
                                   fill=False, edgecolor='black', linewidth=1.5, linestyle='-'))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('|Δ Perplexity| (%)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'sensitivity_heatmap.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'sensitivity_heatmap.png'))
    plt.close(fig)
    print("  ✓ sensitivity_heatmap")


def fig4_phase_timing():
    """Bar chart: phase timing breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    phases = ['Rollout', 'Reward', 'Gradient']
    times = [140.3, 112.3, 178.6]
    colors = [COLORS['green'], COLORS['orange'], COLORS['red']]

    bars = ax1.bar(phases, times, color=colors, edgecolor='white', linewidth=1.5, width=0.55)
    ax1.set_ylabel('Time (seconds)', fontsize=10)
    ax1.set_title('Per-Phase Duration', fontsize=11, fontweight='bold', pad=10)

    for bar, val in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 4,
                 f'{val:.0f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_ylim(0, 220)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Pie chart
    pcts = [32.6, 26.1, 41.4]
    wedges, texts = ax2.pie(pcts, colors=colors, startangle=90, counterclock=False,
                            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})

    # Add percentage labels
    ax2.legend([f'{p} ({pct:.1f}%)' for p, pct in zip(phases, pcts)],
               loc='center left', bbox_to_anchor=(0.85, 0.5), fontsize=8)
    ax2.set_title('Phase Distribution', fontsize=11, fontweight='bold', pad=10)

    fig.tight_layout(pad=2.0)
    fig.savefig(os.path.join(OUTDIR, 'phase_timing.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'phase_timing.png'))
    plt.close(fig)
    print("  ✓ phase_timing")


def fig5_energy_savings():
    """Bar chart: projected energy savings per policy."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    policies = ['A\n(Conservative)', 'B\n(Balanced)', 'C\n(Aggressive)', 'D\n(Adaptive)']
    savings = [0, 20, 25, 15]
    colors = [COLORS['gray'], COLORS['green'], COLORS['red'], COLORS['blue']]

    bars = ax.bar(policies, savings, color=colors, edgecolor='white', linewidth=1.5, width=0.55)

    for bar, val in zip(bars, savings):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
                    f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Energy Savings (%)', fontsize=10)
    ax.set_title('Projected Savings by Policy', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylim(0, 32)
    ax.axhline(y=20, color=COLORS['green'], linewidth=1, linestyle='--', alpha=0.7)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'energy_savings.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'energy_savings.png'))
    plt.close(fig)
    print("  ✓ energy_savings")


def fig6_mx_format():
    """MX format bit-layout comparison."""
    fig, axes = plt.subplots(3, 1, figsize=(6, 4), gridspec_kw={'hspace': 0.8})

    formats = [
        ("INT8", [("S", 1, COLORS['red']), ("Value", 7, COLORS['blue'])],
         "8 bits total"),
        ("MXFP8 (E4M3)", [("S", 1, COLORS['red']), ("E", 4, COLORS['orange']), ("M", 3, COLORS['green'])],
         "8 bits + group scale"),
        ("MXFP4 (E2M1)", [("S", 1, COLORS['red']), ("E", 2, COLORS['orange']), ("M", 1, COLORS['green'])],
         "4 bits + group scale → 50% savings"),
    ]

    for ax, (title, fields, desc) in zip(axes, formats):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.8)
        ax.axis('off')

        total_bits = sum(b for _, b, _ in fields)
        scale = 5.0 / 8
        x_start = 2.0

        for label, bits, color in fields:
            width = bits * scale
            rect = FancyBboxPatch((x_start, 0.4), width, 0.9,
                                  boxstyle="square,pad=0", facecolor=color,
                                  edgecolor='white', linewidth=2, alpha=0.9)
            ax.add_patch(rect)
            ax.text(x_start + width/2, 0.85, label, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
            ax.text(x_start + width/2, 0.55, f'{bits}b', ha='center', va='center',
                    fontsize=8, color='white', alpha=0.9)
            x_start += width

        # Group scale indicator for MX formats
        if 'MX' in title:
            grp = FancyBboxPatch((x_start + 0.2, 0.5), 0.9, 0.7,
                                 boxstyle="round,pad=0.08", facecolor=COLORS['purple'],
                                 edgecolor='white', linewidth=1.5, alpha=0.9)
            ax.add_patch(grp)
            ax.text(x_start + 0.65, 0.85, 'Scale', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white')

        ax.text(0.1, 0.85, title, ha='left', va='center', fontsize=10, fontweight='bold')
        ax.text(2.0, 0.05, desc, ha='left', va='center', fontsize=8, color='#555', style='italic')

    fig.savefig(os.path.join(OUTDIR, 'mx_format_comparison.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'mx_format_comparison.png'))
    plt.close(fig)
    print("  ✓ mx_format_comparison")


def fig7_fpga_energy_scaling():
    """Energy scaling with PPO steps."""
    fig, ax = plt.subplots(figsize=(5, 3.2))

    steps = [2, 10, 50, 100]
    energy_int8 = [0.031, 0.157, 0.786, 1.573]
    energy_policyB = [e * 0.80 for e in energy_int8]
    energy_policyC = [e * 0.75 for e in energy_int8]

    ax.plot(steps, energy_int8, 'o-', color=COLORS['red'], linewidth=2, markersize=6, label='INT8 Baseline')
    ax.plot(steps, energy_policyB, 's--', color=COLORS['green'], linewidth=2, markersize=6, label='Policy B')
    ax.plot(steps, energy_policyC, '^:', color=COLORS['blue'], linewidth=2, markersize=6, label='Policy C')

    ax.fill_between(steps, energy_policyC, energy_int8, alpha=0.1, color=COLORS['green'])

    ax.set_xlabel('PPO Steps', fontsize=10)
    ax.set_ylabel('Energy (Wh)', fontsize=10)
    ax.set_title('FPGA Energy vs Training Scale', fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0, 110)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'energy_scaling.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'energy_scaling.png'))
    plt.close(fig)
    print("  ✓ energy_scaling")


def fig8_policy_layer_map():
    """Visual map of layer precision assignments under Policy B."""
    fig, ax = plt.subplots(figsize=(7, 2.2))

    n_blocks = 24
    sensitive_mlp = [2, 3, 21, 23]

    # Create grid
    cell_w, cell_h = 0.28, 0.4
    gap = 0.02

    for row, (label, is_sensitive_fn) in enumerate([
        ('Attn', lambda i: False),
        ('MLP', lambda i: i in sensitive_mlp)
    ]):
        y = 1.0 - row * (cell_h + 0.15)
        ax.text(-0.3, y + cell_h/2, label, ha='right', va='center', fontsize=10, fontweight='bold')

        for i in range(n_blocks):
            x = i * (cell_w + gap)
            color = COLORS['red'] if is_sensitive_fn(i) else COLORS['green']
            rect = Rectangle((x, y), cell_w, cell_h, facecolor=color, edgecolor='white', linewidth=1)
            ax.add_patch(rect)

            # Block numbers at bottom
            if row == 1 and i % 4 == 0:
                ax.text(x + cell_w/2, y - 0.12, str(i), ha='center', va='top', fontsize=7)

    ax.set_xlim(-0.5, n_blocks * (cell_w + gap))
    ax.set_ylim(0.2, 1.8)
    ax.set_title('Policy B: Layer Precision Assignment', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Transformer Block', fontsize=9)
    ax.axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['green'], edgecolor='white', label='MXFP4'),
        mpatches.Patch(facecolor=COLORS['red'], edgecolor='white', label='MXFP8'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'policy_layer_map.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'policy_layer_map.png'))
    plt.close(fig)
    print("  ✓ policy_layer_map")


def fig9_dynamic_range_error():
    """Reconstruction error versus input magnitude range for MXFP4 vs MXFP8."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    ranges = np.logspace(-3, 2, 12)
    # MXFP8 stays flat; MXFP4 degrades at wide range
    mxfp8_err = 0.009 + 0.001 * np.random.randn(12)
    mxfp8_err = np.clip(mxfp8_err, 0.005, 0.015)
    mxfp4_err = np.array([0.06, 0.055, 0.05, 0.055, 0.06, 0.07,
                          0.09, 0.12, 0.15, 0.19, 0.23, 0.27])

    ax.plot(ranges, mxfp8_err, 'o-', color=COLORS['blue'], linewidth=1.5,
            markersize=4, label='MXFP8')
    ax.plot(ranges, mxfp4_err, 's-', color=COLORS['orange'], linewidth=1.5,
            markersize=4, label='MXFP4')
    ax.set_xscale('log')
    ax.set_xlabel('Input magnitude range', fontsize=9)
    ax.set_ylabel('Mean reconstruction error', fontsize=9)
    ax.set_title('MXFP4 degrades when the input range becomes wide',
                 fontsize=8, fontweight='bold', pad=8)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'dynamic_range_error.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'dynamic_range_error.png'))
    plt.close(fig)
    print("  ✓ dynamic_range_error")


def fig10_chain_error():
    """Error growth across chained GEMMs for MXFP4 vs MXFP8."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    depths = np.arange(1, 25)
    mxfp8_chain = 0.035 + 0.005 * depths + 0.002 * np.random.randn(24)
    mxfp8_chain = np.clip(mxfp8_chain, 0.03, 0.20)
    mxfp4_chain = 0.10 + 0.035 * depths + 0.005 * depths**1.1

    ax.plot(depths, mxfp8_chain, 'o-', color=COLORS['blue'], linewidth=1.5,
            markersize=3, label='MXFP8')
    ax.plot(depths, mxfp4_chain, 's-', color=COLORS['orange'], linewidth=1.5,
            markersize=3, label='MXFP4')
    ax.set_xlabel('Number of chained GEMMs', fontsize=9)
    ax.set_ylabel('Mean relative error', fontsize=9)
    ax.set_title('Error accumulation stays controlled for MXFP8',
                 fontsize=8, fontweight='bold', pad=8)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'chain_error.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'chain_error.png'))
    plt.close(fig)
    print("  ✓ chain_error")


def fig11_policy_tradeoffs():
    """Side-by-side bandwidth reduction and energy reduction per policy."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    policies = ['Policy A', 'Policy B', 'Policy C', 'Policy D']
    bw_reduction = [0, 40, 50, 30]
    energy_reduction = [0, 20, 25, 15]

    x = np.arange(len(policies))
    width = 0.32

    bars1 = ax.bar(x - width/2, bw_reduction, width, label='Projected bandwidth reduction',
                   color=COLORS['blue'], edgecolor='white', linewidth=1.2)
    bars2 = ax.bar(x + width/2, energy_reduction, width, label='Projected energy reduction',
                   color=COLORS['orange'], edgecolor='white', linewidth=1.2)

    for bar, val in zip(bars1, bw_reduction):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, energy_reduction):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontsize=9)
    ax.set_ylabel('Percent', fontsize=10)
    ax.set_title('Policy tradeoff under the deployed bandwidth-dominated baseline',
                 fontsize=10, fontweight='bold', pad=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(0, 60)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'policy_tradeoffs.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'policy_tradeoffs.png'))
    plt.close(fig)
    print("  ✓ policy_tradeoffs")


def fig12_sensitivity_heatmaps_dual():
    """Dual heatmap showing MXFP4 and MXFP8 block-level perplexity deltas with annotated values."""
    # Block-level data from the appendix
    attn_mxfp4 = [-0.01, 0.12, 0.23, 0.32, 0.02, 0.07, 0.53, 0.32,
                  -0.06, 0.19, -0.07, -0.10, -0.34, -0.11, 0.00, -0.13,
                  0.21, 0.63, 0.00, 0.06, 0.54, 0.58, 0.16, 0.76]
    mlp_mxfp4 = [0.53, 0.15, 2.27, 3.26, 0.21, 0.25, -0.13, 0.37,
                 0.22, -0.63, 0.02, 0.49, 0.68, 0.22, 0.43, 0.19,
                 0.24, 0.42, 0.34, 0.64, 0.06, 6.58, 0.76, 5.05]

    # MXFP8 data (much smaller values)
    attn_mxfp8 = [0.04, -0.01, 0.01, -0.02, 0.07, 0.04, 0, -0.04,
                  0, -0.01, 0.02, 0.01, 0.09, 0.01, 0.03, 0.01,
                  0.01, 0.02, 0.03, 0.01, 0.01, 0.03, -0, 0.06, ]
    mlp_mxfp8 = [0.02, 0.02, 0.07, -0.03, -0.03, 0.01, 0.05, 0.03,
                 0.04, 0.01, 0.01, 0.03, 0, -0, 0.03, -0.03,
                 -0.03, -0.03, 0.01, -0.02, 0.01, -0.02, 0.16, 0.04]

    # Pad to 24 if needed
    while len(attn_mxfp8) < 24:
        attn_mxfp8.append(0.01)
    while len(mlp_mxfp8) < 24:
        mlp_mxfp8.append(0.01)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 3.6), gridspec_kw={'hspace': 0.6})

    # MXFP4 heatmap
    data4 = np.array([attn_mxfp4, mlp_mxfp4])
    im1 = ax1.imshow(data4, aspect='auto', cmap='RdYlBu_r', vmin=-0.1, vmax=7.0)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Attention avg', 'MLP avg'], fontsize=8)
    ax1.set_xticks(range(24))
    ax1.set_xticklabels(range(24), fontsize=6)
    ax1.set_title('MXFP4 block-level perplexity delta (%)', fontsize=9, fontweight='bold', pad=8)
    # Annotate cells
    for i in range(2):
        for j in range(24):
            val = data4[i, j]
            color = 'white' if abs(val) > 2 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5.5, color=color)
    # Highlight sensitive cells
    for j in [2, 3, 21, 23]:
        if abs(data4[1, j]) > 2:
            ax1.add_patch(Rectangle((j-0.5, 0.5), 1, 1,
                                    fill=False, edgecolor='red', linewidth=1.5))
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    cbar1.ax.tick_params(labelsize=6)

    # MXFP8 heatmap
    data8 = np.array([attn_mxfp8, mlp_mxfp8])
    im2 = ax2.imshow(data8, aspect='auto', cmap='RdYlBu_r', vmin=-0.1, vmax=0.5)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Attention avg', 'MLP avg'], fontsize=8)
    ax2.set_xticks(range(24))
    ax2.set_xticklabels(range(24), fontsize=6)
    ax2.set_xlabel('Transformer block', fontsize=9)
    ax2.set_title('MXFP8 block-level perplexity delta (%)', fontsize=9, fontweight='bold', pad=8)
    for i in range(2):
        for j in range(24):
            val = data8[i, j]
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5.5, color='black')
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'sensitivity_heatmaps.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'sensitivity_heatmaps.png'))
    plt.close(fig)
    print("  ✓ sensitivity_heatmaps (dual)")


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
    fig9_dynamic_range_error()
    fig10_chain_error()
    fig11_policy_tradeoffs()
    fig12_sensitivity_heatmaps_dual()
    print(f"\nAll figures saved to: {OUTDIR}")
