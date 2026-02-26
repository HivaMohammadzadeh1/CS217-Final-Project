"""
Calculate Energy from Power Logs and Phase Timing

This script:
1. Reads power log from nvidia-smi or FPGA measurements
2. Reads phase timing (rollout, reward, gradient)
3. Computes Energy = Power × Time for each phase
4. Writes final energy CSV

Usage:
    python baseline_energy/calculate_energy.py --results results/baseline_50
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path


def load_power_log(power_log_path):
    """
    Load power measurements from CSV.

    Args:
        power_log_path: Path to power log CSV

    Returns:
        DataFrame with timestamps and power measurements
    """
    print(f"📊 Loading power log: {power_log_path}")

    try:
        # nvidia-smi dmon format
        df = pd.read_csv(
            power_log_path,
            sep=r'\s+',
            comment='#',
            names=['timestamp', 'gpu', 'pwr', 'gtemp', 'mtemp',
                   'sm', 'mem', 'enc', 'dec', 'mclk', 'pclk']
        )

        print(f"  ✓ Loaded {len(df)} power samples")
        print(f"  Average power: {df['pwr'].mean():.2f} W")
        print(f"  Max power: {df['pwr'].max():.2f} W")
        print(f"  Min power: {df['pwr'].min():.2f} W")

        return df

    except Exception as e:
        print(f"  ⚠️  Error loading power log: {e}")
        return None


def load_phase_timing(phase_timing_path):
    """
    Load phase timing statistics.

    Args:
        phase_timing_path: Path to phase_timing.json

    Returns:
        Dictionary with phase timing info
    """
    print(f"\n⏱️  Loading phase timing: {phase_timing_path}")

    with open(phase_timing_path, 'r') as f:
        phase_timing = json.load(f)

    print("  Phase breakdown:")
    for phase, stats in phase_timing.items():
        print(f"    {phase:10s}: {stats['total_time_s']:7.1f}s total, "
              f"{stats['avg_time_s']:6.2f}s avg, {stats['num_calls']} calls")

    return phase_timing


def calculate_energy(power_log_df, phase_timing):
    """
    Calculate energy consumption per phase.

    Energy = Average_Power × Time

    Args:
        power_log_df: DataFrame with power measurements
        phase_timing: Dictionary with phase timing

    Returns:
        Dictionary with energy per phase
    """
    print("\n⚡ Calculating energy per phase...")

    if power_log_df is None:
        print("  ⚠️  No power log available, using default power estimate")
        avg_power = 50.0  # Default estimate in Watts
    else:
        avg_power = power_log_df['pwr'].mean()

    energy_breakdown = {}
    total_energy = 0

    for phase, stats in phase_timing.items():
        # Energy (Joules) = Power (Watts) × Time (seconds)
        energy_j = avg_power * stats['total_time_s']
        energy_breakdown[phase] = {
            'time_s': stats['total_time_s'],
            'avg_power_w': avg_power,
            'energy_j': energy_j,
            'energy_wh': energy_j / 3600,  # Watt-hours
            'num_calls': stats['num_calls'],
        }
        total_energy += energy_j

        print(f"  {phase:10s}: {energy_j:8.2f} J  ({energy_j/3600:.4f} Wh)")

    # Add total
    energy_breakdown['total'] = {
        'time_s': sum(stats['total_time_s'] for stats in phase_timing.values()),
        'avg_power_w': avg_power,
        'energy_j': total_energy,
        'energy_wh': total_energy / 3600,
    }

    print(f"  {'TOTAL':10s}: {total_energy:8.2f} J  ({total_energy/3600:.4f} Wh)")

    return energy_breakdown


def load_fpga_stats(fpga_stats_path):
    """
    Load FPGA statistics if available.

    Args:
        fpga_stats_path: Path to fpga_stats.json

    Returns:
        Dictionary with FPGA stats or None
    """
    if not fpga_stats_path.exists():
        return None

    print(f"\n🔧 Loading FPGA stats: {fpga_stats_path}")

    with open(fpga_stats_path, 'r') as f:
        fpga_stats = json.load(f)

    print(f"  Total matmuls offloaded: {fpga_stats['total_matmuls']}")
    print(f"  Total tiles processed:   {fpga_stats['total_tiles']}")

    return fpga_stats


def save_energy_csv(energy_breakdown, fpga_stats, output_path):
    """
    Save energy breakdown to CSV.

    Args:
        energy_breakdown: Dictionary with energy per phase
        fpga_stats: FPGA statistics (or None)
        output_path: Path to save CSV
    """
    print(f"\n💾 Saving energy CSV: {output_path}")

    # Create DataFrame
    data = []
    for phase, stats in energy_breakdown.items():
        row = {
            'phase': phase,
            'time_s': stats['time_s'],
            'avg_power_w': stats['avg_power_w'],
            'energy_j': stats['energy_j'],
            'energy_wh': stats['energy_wh'],
        }
        if 'num_calls' in stats:
            row['num_calls'] = stats['num_calls']

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    print(f"  ✓ Saved {len(df)} rows")

    # Also save FPGA stats if available
    if fpga_stats:
        fpga_csv_path = output_path.parent / "fpga_offload_stats.csv"
        fpga_df = pd.DataFrame([fpga_stats])
        fpga_df.to_csv(fpga_csv_path, index=False)
        print(f"  ✓ Saved FPGA stats to {fpga_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Calculate Energy from Logs")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Results directory containing logs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results/energy_summary.csv)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    # Set default output
    if args.output is None:
        output_path = results_dir / "energy_summary.csv"
    else:
        output_path = Path(args.output)

    print("=" * 60)
    print("Energy Calculation")
    print("=" * 60)

    # Load power log
    power_log_path = results_dir / "power_log_baseline.csv"
    power_log_df = None
    if power_log_path.exists():
        power_log_df = load_power_log(power_log_path)
    else:
        print(f"⚠️  Power log not found: {power_log_path}")
        print("   Will use default power estimate")

    # Load phase timing
    phase_timing_path = results_dir / "phase_timing.json"
    if not phase_timing_path.exists():
        print(f"❌ Phase timing not found: {phase_timing_path}")
        return

    phase_timing = load_phase_timing(phase_timing_path)

    # Calculate energy
    energy_breakdown = calculate_energy(power_log_df, phase_timing)

    # Load FPGA stats (if available)
    fpga_stats_path = results_dir / "fpga_stats.json"
    fpga_stats = load_fpga_stats(fpga_stats_path)

    # Save results
    save_energy_csv(energy_breakdown, fpga_stats, output_path)

    print("\n" + "=" * 60)
    print("✅ Energy calculation complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
