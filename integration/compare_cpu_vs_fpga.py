#!/usr/bin/env python3
"""
Compare CPU vs FPGA for RLHF Training

This script compares energy and performance between CPU baseline and FPGA estimates.

Usage:
    # Automatic comparison (reads from result files)
    python integration/compare_cpu_vs_fpga.py

    # Manual comparison
    python integration/compare_cpu_vs_fpga.py \
        --cpu-energy 0.05 \
        --cpu-time 15.5 \
        --fpga-cycles 148656 \
        --fpga-power 35 \
        --steps 50
"""

import argparse
import json
from pathlib import Path
import csv


def load_cpu_baseline(steps=50):
    """
    Try to load CPU baseline results.

    Args:
        steps: Number of PPO steps

    Returns:
        dict with cpu_energy_wh and cpu_time_min, or None if not found
    """
    # Try to find CPU baseline results
    result_dirs = [
        f"results/cpu_baseline_{steps}steps",
        f"results/baseline_{steps}steps",
        "results/cpu_baseline",
    ]

    for result_dir in result_dirs:
        result_path = Path(result_dir)
        if not result_path.exists():
            continue

        # Look for energy summary
        summary_file = result_path / "energy_summary.csv"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                    if data:
                        return {
                            'cpu_energy_wh': float(data[0].get('total_energy_wh', 0)),
                            'cpu_time_min': float(data[0].get('total_time_min', 0)),
                            'avg_power_w': float(data[0].get('avg_power_w', 0)),
                            'source': str(summary_file)
                        }
            except Exception as e:
                print(f"⚠️  Could not parse {summary_file}: {e}")

        # Look for JSON summary
        json_file = result_path / "run_summary.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'energy' in data:
                        return {
                            'cpu_energy_wh': data['energy'].get('total_energy_wh', 0),
                            'cpu_time_min': data['timing'].get('total_time_min', 0),
                            'avg_power_w': data['energy'].get('avg_power_w', 0),
                            'source': str(json_file)
                        }
            except Exception as e:
                print(f"⚠️  Could not parse {json_file}: {e}")

    return None


def load_fpga_estimate(steps=50):
    """
    Try to load FPGA energy estimate.

    Args:
        steps: Number of PPO steps

    Returns:
        dict with FPGA estimate data, or None if not found
    """
    result_file = Path(f"results/fpga_energy_{steps}steps.json")

    if result_file.exists():
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                return {
                    **data,
                    'source': str(result_file)
                }
        except Exception as e:
            print(f"⚠️  Could not parse {result_file}: {e}")

    return None


def calculate_comparison(cpu_data, fpga_data):
    """
    Calculate comparison metrics.

    Args:
        cpu_data: dict with cpu_energy_wh, cpu_time_min, avg_power_w
        fpga_data: dict with total_energy_wh, total_time_min, fpga_power_w

    Returns:
        dict with comparison metrics
    """
    speedup = cpu_data['cpu_time_min'] / fpga_data['total_time_min']
    energy_ratio = cpu_data['cpu_energy_wh'] / fpga_data['total_energy_wh']
    power_ratio = cpu_data['avg_power_w'] / fpga_data['fpga_power_w']

    # Cost calculations (assuming AWS pricing)
    cpu_cost = cpu_data['cpu_time_min'] / 60 * 0.15  # ~$0.15/hr for CPU instance
    fpga_cost = fpga_data['total_time_min'] / 60 * 1.65  # ~$1.65/hr for f2.6xlarge
    cost_ratio = cpu_cost / fpga_cost if fpga_cost > 0 else 0

    return {
        'speedup': speedup,
        'energy_ratio': energy_ratio,
        'power_ratio': power_ratio,
        'cost_ratio': cost_ratio,
        'cpu_cost_usd': cpu_cost,
        'fpga_cost_usd': fpga_cost
    }


def print_comparison(cpu_data, fpga_data, comparison):
    """Pretty print comparison results."""
    print("\n" + "="*80)
    print("CPU vs FPGA Comparison for RLHF Training")
    print("="*80)

    # Data sources
    print(f"\n{'Data Sources:':<30}")
    print(f"  CPU: {cpu_data.get('source', 'manual input')}")
    print(f"  FPGA: {fpga_data.get('source', 'manual input')}")

    # Time comparison
    print(f"\n{'Training Time:':<30} {'CPU':<20} {'FPGA (Est.)':<20} {'Ratio':<15}")
    print("-" * 85)
    print(f"{'Total Time':<30} {cpu_data['cpu_time_min']:.2f} min{'':<13} "
          f"{fpga_data['total_time_min']:.2f} min{'':<13} "
          f"{comparison['speedup']:.2f}x faster")

    # Energy comparison
    print(f"\n{'Energy Consumption:':<30} {'CPU':<20} {'FPGA (Est.)':<20} {'Ratio':<15}")
    print("-" * 85)
    print(f"{'Total Energy':<30} {cpu_data['cpu_energy_wh']:.4f} Wh{'':<11} "
          f"{fpga_data['total_energy_wh']:.4f} Wh{'':<11} "
          f"{comparison['energy_ratio']:.2f}x less")
    print(f"{'Total Energy (Joules)':<30} {cpu_data['cpu_energy_wh'] * 3600:.1f} J{'':<14} "
          f"{fpga_data['total_energy_wh'] * 3600:.1f} J{'':<14} "
          f"")

    # Power comparison
    print(f"\n{'Power Consumption:':<30} {'CPU':<20} {'FPGA (Est.)':<20} {'Ratio':<15}")
    print("-" * 85)
    print(f"{'Average Power':<30} {cpu_data['avg_power_w']:.1f} W{'':<15} "
          f"{fpga_data['fpga_power_w']:.1f} W{'':<15} "
          f"{comparison['power_ratio']:.2f}x lower")

    # Cost comparison (if applicable)
    if comparison['cost_ratio'] > 0:
        print(f"\n{'Cost (AWS Pricing):':<30} {'CPU':<20} {'FPGA (Est.)':<20} {'Ratio':<15}")
        print("-" * 85)
        print(f"{'Estimated Cost':<30} ${comparison['cpu_cost_usd']:.4f}{'':<14} "
              f"${comparison['fpga_cost_usd']:.4f}{'':<14} "
              f"{comparison['cost_ratio']:.2f}x cheaper")

    # Efficiency metrics
    print(f"\n{'Efficiency Metrics:':<30}")
    print(f"  Energy Efficiency (ops/Wh): FPGA is {comparison['energy_ratio']:.1f}x more efficient")
    print(f"  Performance/Watt: FPGA is {comparison['speedup'] * comparison['power_ratio']:.1f}x better")

    # Key insights
    print(f"\n{'Key Insights:':<30}")
    if comparison['speedup'] > 1:
        print(f"  ✓ FPGA is {comparison['speedup']:.1f}x faster than CPU")
    else:
        print(f"  ⚠  FPGA is {1/comparison['speedup']:.1f}x slower than CPU (PCIe overhead)")

    if comparison['energy_ratio'] > 1:
        print(f"  ✓ FPGA uses {comparison['energy_ratio']:.1f}x less energy than CPU")
    else:
        print(f"  ⚠  FPGA uses {1/comparison['energy_ratio']:.1f}x more energy than CPU")

    if comparison['power_ratio'] > 1:
        print(f"  ✓ FPGA has {comparison['power_ratio']:.1f}x lower power consumption")

    # Breakdown analysis
    print(f"\n{'FPGA Performance Breakdown:':<30}")
    data_transfer_pct = ((fpga_data['cycles_per_matmul'] - 15) / fpga_data['cycles_per_matmul']) * 100
    compute_pct = (15 / fpga_data['cycles_per_matmul']) * 100
    print(f"  Data Transfer: {data_transfer_pct:.2f}% of time")
    print(f"  Computation: {compute_pct:.2f}% of time")
    print(f"  → Bottleneck: PCIe communication overhead")

    print("\n" + "="*80)


def generate_report(cpu_data, fpga_data, comparison, output_file):
    """Generate JSON report with comparison."""
    report = {
        'comparison_date': str(Path().resolve()),
        'cpu_baseline': cpu_data,
        'fpga_estimate': fpga_data,
        'comparison_metrics': comparison,
        'summary': {
            'speedup': f"{comparison['speedup']:.2f}x",
            'energy_savings': f"{comparison['energy_ratio']:.2f}x less",
            'power_reduction': f"{comparison['power_ratio']:.2f}x lower",
            'faster': comparison['speedup'] > 1,
            'more_efficient': comparison['energy_ratio'] > 1,
            'recommendation': (
                "FPGA is both faster and more energy efficient" if
                (comparison['speedup'] > 1 and comparison['energy_ratio'] > 1) else
                "CPU is more practical for this workload size"
            )
        }
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description='Compare CPU vs FPGA for RLHF training')
    parser.add_argument('--cpu-energy', type=float, default=None,
                        help='CPU energy in Wh (will try to auto-detect if not provided)')
    parser.add_argument('--cpu-time', type=float, default=None,
                        help='CPU time in minutes (will try to auto-detect if not provided)')
    parser.add_argument('--cpu-power', type=float, default=None,
                        help='CPU average power in W (will try to auto-detect if not provided)')
    parser.add_argument('--fpga-cycles', type=int, default=148656,
                        help='FPGA cycles per 16x16 matmul (default: 148656 from measurements)')
    parser.add_argument('--fpga-power', type=float, default=35,
                        help='FPGA power in Watts (default: 35W)')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of PPO training steps (default: 50)')
    parser.add_argument('--output', type=str, default='results/cpu_vs_fpga_comparison.json',
                        help='Output JSON file for comparison results')

    args = parser.parse_args()

    print("Loading data...")

    # Try to load CPU baseline
    if args.cpu_energy is None or args.cpu_time is None:
        print("Attempting to load CPU baseline results...")
        cpu_baseline = load_cpu_baseline(args.steps)
        if cpu_baseline:
            print(f"✓ Loaded CPU baseline from: {cpu_baseline['source']}")
            cpu_data = cpu_baseline
        else:
            print("⚠️  No CPU baseline found. Please provide manual values.")
            print("   Usage: --cpu-energy <WH> --cpu-time <MIN> --cpu-power <W>")
            return
    else:
        cpu_data = {
            'cpu_energy_wh': args.cpu_energy,
            'cpu_time_min': args.cpu_time,
            'avg_power_w': args.cpu_power or (args.cpu_energy * 60 / args.cpu_time),
            'source': 'manual input'
        }
        print("✓ Using manually provided CPU data")

    # Try to load FPGA estimate
    fpga_estimate = load_fpga_estimate(args.steps)
    if fpga_estimate:
        print(f"✓ Loaded FPGA estimate from: {fpga_estimate['source']}")
        fpga_data = fpga_estimate
    else:
        print("⚠️  No FPGA estimate found. Calculating now...")
        from calculate_fpga_energy import FPGAEnergyCalculator
        calc = FPGAEnergyCalculator(
            cycles_per_matmul=args.fpga_cycles,
            fpga_power_w=args.fpga_power
        )
        fpga_data = calc.calculate_rlhf_energy(args.steps)
        fpga_data['source'] = 'calculated on demand'
        print("✓ FPGA estimate calculated")

    # Calculate comparison
    comparison = calculate_comparison(cpu_data, fpga_data)

    # Print comparison
    print_comparison(cpu_data, fpga_data, comparison)

    # Generate report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(cpu_data, fpga_data, comparison, output_path)

    print(f"\nComparison report saved to: {output_path}")


if __name__ == '__main__':
    main()
