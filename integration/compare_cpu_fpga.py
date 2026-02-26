"""
CPU vs FPGA Performance Comparison Tool

This script compares CPU baseline results with FPGA estimates to calculate:
1. Speedup (time improvement)
2. Energy efficiency (energy reduction)
3. Performance per watt
4. Cost analysis

Usage:
    # After CPU baseline completes:
    python integration/compare_cpu_fpga.py \
        --cpu-results results/cpu_baseline_50steps/energy_summary.csv \
        --fpga-results results/lab1_fpga_analysis.json \
        --output results/cpu_vs_fpga_comparison.json
"""

import argparse
import json
import csv
from pathlib import Path


class CPUFPGAComparator:
    """Compares CPU and FPGA performance metrics."""

    def __init__(self, cpu_data, fpga_data):
        self.cpu_data = cpu_data
        self.fpga_data = fpga_data

    def calculate_speedup(self):
        """Calculate time speedup (CPU time / FPGA time)."""
        cpu_time = self.cpu_data['total_time_seconds']
        fpga_time = self.fpga_data['rlhf_workload_estimate']['total_time_seconds']

        speedup = cpu_time / fpga_time
        time_saved = cpu_time - fpga_time

        return {
            'cpu_time_seconds': cpu_time,
            'fpga_time_seconds': fpga_time,
            'speedup_factor': speedup,
            'time_saved_seconds': time_saved,
            'time_saved_minutes': time_saved / 60,
            'faster_by_pct': ((speedup - 1) * 100),
        }

    def calculate_energy_efficiency(self):
        """Calculate energy efficiency improvement."""
        cpu_energy = self.cpu_data['total_energy_wh']
        fpga_energy = self.fpga_data['rlhf_workload_estimate']['total_energy_wh']

        efficiency_factor = cpu_energy / fpga_energy
        energy_saved = cpu_energy - fpga_energy

        return {
            'cpu_energy_wh': cpu_energy,
            'fpga_energy_wh': fpga_energy,
            'efficiency_factor': efficiency_factor,
            'energy_saved_wh': energy_saved,
            'energy_reduction_pct': ((efficiency_factor - 1) * 100),
        }

    def calculate_performance_per_watt(self):
        """Calculate performance per watt for both."""
        cpu_time = self.cpu_data['total_time_seconds']
        cpu_energy = self.cpu_data['total_energy_wh'] * 3600  # Convert to Joules
        cpu_power = cpu_energy / cpu_time if cpu_time > 0 else 0
        cpu_perf_per_watt = 1.0 / cpu_power if cpu_power > 0 else 0

        fpga_time = self.fpga_data['rlhf_workload_estimate']['total_time_seconds']
        fpga_energy = self.fpga_data['rlhf_workload_estimate']['total_energy_joules']
        fpga_power = fpga_energy / fpga_time if fpga_time > 0 else 0
        fpga_perf_per_watt = 1.0 / fpga_power if fpga_power > 0 else 0

        return {
            'cpu_avg_power_w': cpu_power,
            'fpga_avg_power_w': fpga_power,
            'cpu_perf_per_watt': cpu_perf_per_watt,
            'fpga_perf_per_watt': fpga_perf_per_watt,
            'power_efficiency_improvement': fpga_perf_per_watt / cpu_perf_per_watt if cpu_perf_per_watt > 0 else 0,
        }

    def generate_report(self):
        """Generate comprehensive comparison report."""
        speedup = self.calculate_speedup()
        energy_eff = self.calculate_energy_efficiency()
        perf_per_watt = self.calculate_performance_per_watt()

        report = {
            'cpu_baseline': {
                'time_seconds': self.cpu_data['total_time_seconds'],
                'time_minutes': self.cpu_data['total_time_seconds'] / 60,
                'energy_wh': self.cpu_data['total_energy_wh'],
                'avg_power_w': perf_per_watt['cpu_avg_power_w'],
            },
            'fpga_estimate': {
                'time_seconds': self.fpga_data['rlhf_workload_estimate']['total_time_seconds'],
                'time_minutes': self.fpga_data['rlhf_workload_estimate']['total_time_minutes'],
                'energy_wh': self.fpga_data['rlhf_workload_estimate']['total_energy_wh'],
                'avg_power_w': perf_per_watt['fpga_avg_power_w'],
            },
            'speedup': speedup,
            'energy_efficiency': energy_eff,
            'performance_per_watt': perf_per_watt,
            'summary': {
                'fpga_is_faster': speedup['speedup_factor'] > 1,
                'fpga_uses_less_energy': energy_eff['efficiency_factor'] > 1,
                'recommendation': self._get_recommendation(speedup, energy_eff),
            }
        }

        return report

    def _get_recommendation(self, speedup, energy_eff):
        """Generate recommendation based on results."""
        if speedup['speedup_factor'] > 1 and energy_eff['efficiency_factor'] > 1:
            return f"FPGA is {speedup['speedup_factor']:.1f}x faster and {energy_eff['efficiency_factor']:.1f}x more energy efficient. FPGA acceleration is beneficial."
        elif speedup['speedup_factor'] > 1:
            return f"FPGA is {speedup['speedup_factor']:.1f}x faster but uses more energy. Consider for latency-critical applications."
        elif energy_eff['efficiency_factor'] > 1:
            return f"FPGA uses {energy_eff['efficiency_factor']:.1f}x less energy but is slower. Consider for energy-constrained applications."
        else:
            return "CPU baseline is better for this workload. FPGA overhead (PCIe transfer) dominates performance."

    def print_report(self):
        """Print human-readable comparison report."""
        report = self.generate_report()

        print("=" * 80)
        print("CPU vs FPGA Performance Comparison")
        print("=" * 80)

        print("\n### CPU Baseline ###")
        print(f"  Time:         {report['cpu_baseline']['time_seconds']:.1f} seconds ({report['cpu_baseline']['time_minutes']:.1f} min)")
        print(f"  Energy:       {report['cpu_baseline']['energy_wh']:.3f} Wh")
        print(f"  Avg Power:    {report['cpu_baseline']['avg_power_w']:.1f} W")

        print("\n### FPGA Estimate ###")
        print(f"  Time:         {report['fpga_estimate']['time_seconds']:.1f} seconds ({report['fpga_estimate']['time_minutes']:.1f} min)")
        print(f"  Energy:       {report['fpga_estimate']['energy_wh']:.3f} Wh")
        print(f"  Avg Power:    {report['fpga_estimate']['avg_power_w']:.1f} W")

        print("\n" + "=" * 80)
        print("Performance Comparison")
        print("=" * 80)

        speedup = report['speedup']
        if speedup['speedup_factor'] > 1:
            print(f"\n⚡ SPEEDUP: {speedup['speedup_factor']:.2f}x faster")
            print(f"   Time saved: {speedup['time_saved_minutes']:.1f} minutes ({speedup['faster_by_pct']:.1f}% faster)")
        else:
            slowdown = 1.0 / speedup['speedup_factor']
            print(f"\n⚠️  SLOWDOWN: {slowdown:.2f}x slower")
            print(f"   Time added: {-speedup['time_saved_minutes']:.1f} minutes")

        energy_eff = report['energy_efficiency']
        if energy_eff['efficiency_factor'] > 1:
            print(f"\n🔋 ENERGY EFFICIENCY: {energy_eff['efficiency_factor']:.2f}x less energy")
            print(f"   Energy saved: {energy_eff['energy_saved_wh']:.3f} Wh ({energy_eff['energy_reduction_pct']:.1f}% reduction)")
        else:
            inefficiency = 1.0 / energy_eff['efficiency_factor']
            print(f"\n⚠️  ENERGY INEFFICIENCY: {inefficiency:.2f}x more energy")
            print(f"   Extra energy: {-energy_eff['energy_saved_wh']:.3f} Wh")

        print("\n" + "=" * 80)
        print("Recommendation")
        print("=" * 80)
        print(f"\n{report['summary']['recommendation']}")

        print("\n" + "=" * 80)
        print("Notes")
        print("=" * 80)
        print("1. FPGA estimate based on Lab 1 hardware measurements")
        print("2. Assumes 35W FPGA power (verify with XPE report)")
        print("3. Lab 1 bottleneck: PCIe data transfer (99.9% of time)")
        print("4. FPGA efficiency improves with:")
        print("   - Larger batch sizes (amortize transfer cost)")
        print("   - Pipelined operations")
        print("   - On-chip data reuse")
        print("=" * 80)


def load_cpu_results(csv_path):
    """Load CPU baseline results from CSV."""
    data = {
        'total_time_seconds': 0,
        'total_energy_wh': 0,
    }

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Sum up time and energy across phases
            if 'time_seconds' in row:
                data['total_time_seconds'] += float(row['time_seconds'])
            if 'energy_wh' in row:
                data['total_energy_wh'] += float(row['energy_wh'])

    return data


def load_fpga_results(json_path):
    """Load FPGA results from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compare CPU baseline with FPGA performance estimates"
    )
    parser.add_argument(
        "--cpu-results",
        type=str,
        required=True,
        help="Path to CPU baseline energy_summary.csv"
    )
    parser.add_argument(
        "--fpga-results",
        type=str,
        required=True,
        help="Path to FPGA analysis JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save comparison to JSON file"
    )

    args = parser.parse_args()

    # Load results
    cpu_path = Path(args.cpu_results)
    fpga_path = Path(args.fpga_results)

    if not cpu_path.exists():
        print(f"❌ CPU results not found: {cpu_path}")
        print("   Run CPU baseline first:")
        print("   python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/cpu_baseline_50steps")
        return 1

    if not fpga_path.exists():
        print(f"❌ FPGA results not found: {fpga_path}")
        print("   Run FPGA analysis first:")
        print("   python integration/analyze_lab1_results.py --output-json results/lab1_fpga_analysis.json")
        return 1

    print("Loading results...")
    cpu_data = load_cpu_results(cpu_path)
    fpga_data = load_fpga_results(fpga_path)

    # Create comparator
    comparator = CPUFPGAComparator(cpu_data, fpga_data)

    # Print report
    comparator.print_report()

    # Save to JSON if requested
    if args.output:
        report = comparator.generate_report()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Comparison saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
