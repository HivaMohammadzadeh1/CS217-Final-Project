"""
Process power logs and calculate energy consumption.

Reads nvidia-smi power log and training timing data,
then calculates total energy and phase breakdown.

Usage:
    python baseline_energy/process_energy.py --results results/gpu_baseline
"""

import pandas as pd
import json
import argparse
from pathlib import Path


def parse_power_log(power_log_path):
    """
    Parse nvidia-smi dmon output.

    Returns:
        DataFrame with columns: timestamp, gpu, pwr (watts), ...
    """
    try:
        # Read nvidia-smi dmon output
        # Format: # timestamp gpu pwr gtemp mtemp sm mem enc dec mclk pclk
        df = pd.read_csv(
            power_log_path,
            sep=r'\s+',
            comment='#',
            names=['timestamp', 'gpu', 'pwr', 'gtemp', 'mtemp',
                   'sm', 'mem', 'enc', 'dec', 'mclk', 'pclk']
        )
        return df
    except Exception as e:
        print(f"Error parsing power log: {e}")
        return None


def calculate_energy(results_dir):
    """
    Calculate energy consumption from power log and timing data.

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary with energy breakdown
    """
    results_dir = Path(results_dir)

    # Load power log
    power_log_path = results_dir / "power_log_baseline.csv"
    if not power_log_path.exists():
        print(f"Error: Power log not found at {power_log_path}")
        return None

    power_df = parse_power_log(power_log_path)
    if power_df is None:
        return None

    # Load phase timing
    timing_path = results_dir / "phase_timing.json"
    if not timing_path.exists():
        print(f"Error: Phase timing not found at {timing_path}")
        return None

    with open(timing_path, 'r') as f:
        phase_timing = json.load(f)

    # Calculate average power
    avg_power_W = power_df['pwr'].mean()
    max_power_W = power_df['pwr'].max()
    min_power_W = power_df['pwr'].min()

    print(f"\n{'='*60}")
    print("GPU Power Statistics")
    print(f"{'='*60}")
    print(f"Average Power: {avg_power_W:.2f} W")
    print(f"Max Power:     {max_power_W:.2f} W")
    print(f"Min Power:     {min_power_W:.2f} W")
    print(f"{'='*60}\n")

    # Calculate energy per phase
    # Energy (J) = Power (W) × Time (s)
    energy_breakdown = {}

    total_time_s = 0
    total_energy_J = 0

    for phase, stats in phase_timing.items():
        time_s = stats['total_time_s']
        energy_J = avg_power_W * time_s

        energy_breakdown[phase] = {
            'avg_power_W': avg_power_W,
            'runtime_s': time_s,
            'energy_J': energy_J,
        }

        total_time_s += time_s
        total_energy_J += energy_J

    # Add total
    energy_breakdown['total'] = {
        'avg_power_W': avg_power_W,
        'runtime_s': total_time_s,
        'energy_J': total_energy_J,
    }

    return energy_breakdown


def save_energy_csv(energy_breakdown, output_path):
    """Save energy breakdown to CSV."""
    rows = []
    for phase, data in energy_breakdown.items():
        rows.append({
            'phase': phase,
            'avg_power_W': data['avg_power_W'],
            'runtime_s': data['runtime_s'],
            'energy_J': data['energy_J'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✓ Energy breakdown saved to {output_path}")


def print_energy_breakdown(energy_breakdown):
    """Print energy breakdown table."""
    print(f"\n{'='*60}")
    print("Energy Breakdown by Phase")
    print(f"{'='*60}")
    print(f"{'Phase':<20} {'Power (W)':<12} {'Time (s)':<12} {'Energy (J)':<12}")
    print(f"{'-'*60}")

    for phase in ['rollout', 'reward', 'gradient', 'total']:
        if phase in energy_breakdown:
            data = energy_breakdown[phase]
            phase_name = phase.replace('_', ' ').title()
            print(f"{phase_name:<20} {data['avg_power_W']:>10.2f}  "
                  f"{data['runtime_s']:>10.2f}  {data['energy_J']:>10.0f}")

    print(f"{'='*60}\n")

    # Energy in more familiar units
    total_energy_J = energy_breakdown['total']['energy_J']
    total_energy_Wh = total_energy_J / 3600
    print(f"Total Energy: {total_energy_J:,.0f} J ({total_energy_Wh:.2f} Wh)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Process energy measurements")
    parser.add_argument(
        "--results",
        type=str,
        default="results/gpu_baseline",
        help="Results directory containing power log and timing data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results_dir/energy.csv)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Calculate energy
    energy_breakdown = calculate_energy(results_dir)
    if energy_breakdown is None:
        print("Failed to calculate energy")
        return

    # Print breakdown
    print_energy_breakdown(energy_breakdown)

    # Save to CSV
    output_path = args.output or (results_dir / "energy.csv")
    save_energy_csv(energy_breakdown, output_path)

    print("✅ Energy analysis complete!")


if __name__ == "__main__":
    main()
