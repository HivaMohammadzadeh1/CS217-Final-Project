"""
Analyze Lab 1 FPGA Test Results

This script takes the cycle counts from Lab 1 FPGA tests and calculates:
1. Time per matmul at different clock frequencies
2. Energy consumption estimates
3. Projected performance for RLHF workload
4. Comparison metrics for CPU vs FPGA

Usage:
    python integration/analyze_lab1_results.py --data-cycles 15357 --compute-cycles 15
"""

import argparse
import json
from pathlib import Path


class Lab1PerformanceAnalyzer:
    """Analyzes Lab 1 FPGA performance and calculates energy metrics."""

    # Default assumptions
    DEFAULT_CLOCK_MHZ = 250  # Lab 1 FPGA clock frequency
    DEFAULT_FPGA_POWER_W = 35  # Typical power for Lab 1 design (conservative estimate)

    # RLHF workload estimates (for 50 PPO steps with Qwen2.5-0.5B)
    # These are rough estimates based on:
    # - 50 PPO steps
    # - Batch size 8
    # - Sequence length 512
    # - ~100 transformer layers
    # - ~3 matmuls per layer (Q, K, V projections)
    MATMULS_PER_PPO_STEP = 122880  # Approximate

    def __init__(self, data_cycles, compute_cycles, clock_mhz=None, fpga_power_w=None):
        self.data_cycles = data_cycles
        self.compute_cycles = compute_cycles
        self.total_cycles = data_cycles + compute_cycles

        self.clock_mhz = clock_mhz or self.DEFAULT_CLOCK_MHZ
        self.clock_hz = self.clock_mhz * 1e6

        self.fpga_power_w = fpga_power_w or self.DEFAULT_FPGA_POWER_W

    def time_per_matmul(self):
        """Calculate time per 16×16 matmul in seconds."""
        return self.total_cycles / self.clock_hz

    def energy_per_matmul(self):
        """Calculate energy per 16×16 matmul in Joules."""
        return self.fpga_power_w * self.time_per_matmul()

    def throughput_matmuls_per_sec(self):
        """Calculate throughput in matmuls per second."""
        return 1.0 / self.time_per_matmul()

    def estimate_rlhf_workload(self, num_ppo_steps=50):
        """Estimate total time and energy for RLHF workload."""
        total_matmuls = self.MATMULS_PER_PPO_STEP * num_ppo_steps
        total_cycles = self.total_cycles * total_matmuls
        total_time_s = total_cycles / self.clock_hz
        total_energy_j = self.fpga_power_w * total_time_s
        total_energy_wh = total_energy_j / 3600

        return {
            'num_ppo_steps': num_ppo_steps,
            'total_matmuls': total_matmuls,
            'total_cycles': total_cycles,
            'total_time_seconds': total_time_s,
            'total_time_minutes': total_time_s / 60,
            'total_energy_joules': total_energy_j,
            'total_energy_wh': total_energy_wh,
        }

    def breakdown_analysis(self):
        """Analyze the breakdown of data transfer vs compute."""
        data_pct = (self.data_cycles / self.total_cycles) * 100
        compute_pct = (self.compute_cycles / self.total_cycles) * 100

        return {
            'data_transfer_cycles': self.data_cycles,
            'data_transfer_pct': data_pct,
            'compute_cycles': self.compute_cycles,
            'compute_pct': compute_pct,
            'bottleneck': 'Data Transfer (PCIe)' if data_pct > 50 else 'Computation',
        }

    def generate_report(self, num_ppo_steps=50):
        """Generate comprehensive performance report."""
        time_us = self.time_per_matmul() * 1e6
        energy_mj = self.energy_per_matmul() * 1e3
        throughput = self.throughput_matmuls_per_sec()

        breakdown = self.breakdown_analysis()
        rlhf_estimate = self.estimate_rlhf_workload(num_ppo_steps)

        report = {
            'lab1_fpga_performance': {
                'cycles_per_matmul': {
                    'data_transfer': self.data_cycles,
                    'compute': self.compute_cycles,
                    'total': self.total_cycles,
                },
                'timing': {
                    'clock_frequency_mhz': self.clock_mhz,
                    'time_per_matmul_us': time_us,
                    'throughput_matmuls_per_sec': throughput,
                },
                'energy': {
                    'fpga_power_w': self.fpga_power_w,
                    'energy_per_matmul_mj': energy_mj,
                },
                'breakdown': breakdown,
            },
            'rlhf_workload_estimate': rlhf_estimate,
        }

        return report

    def print_report(self, num_ppo_steps=50):
        """Print human-readable performance report."""
        print("=" * 70)
        print("Lab 1 FPGA Performance Analysis")
        print("=" * 70)

        print("\n### Per 16×16 Matrix Multiplication ###")
        print(f"  Data Transfer Cycles:  {self.data_cycles:,} cycles")
        print(f"  Compute Cycles:        {self.compute_cycles:,} cycles")
        print(f"  Total Cycles:          {self.total_cycles:,} cycles")

        breakdown = self.breakdown_analysis()
        print(f"\n  Breakdown:")
        print(f"    Data Transfer: {breakdown['data_transfer_pct']:.1f}%")
        print(f"    Computation:   {breakdown['compute_pct']:.1f}%")
        print(f"    Bottleneck:    {breakdown['bottleneck']}")

        time_us = self.time_per_matmul() * 1e6
        energy_mj = self.energy_per_matmul() * 1e3
        throughput = self.throughput_matmuls_per_sec()

        print(f"\n### Timing (@ {self.clock_mhz} MHz) ###")
        print(f"  Time per matmul:       {time_us:.2f} μs")
        print(f"  Throughput:            {throughput:,.0f} matmuls/sec")

        print(f"\n### Energy (assuming {self.fpga_power_w} W FPGA power) ###")
        print(f"  Energy per matmul:     {energy_mj:.3f} mJ")

        print("\n" + "=" * 70)
        print(f"RLHF Workload Estimate ({num_ppo_steps} PPO steps)")
        print("=" * 70)

        rlhf = self.estimate_rlhf_workload(num_ppo_steps)

        print(f"\n  Total 16×16 matmuls:   {rlhf['total_matmuls']:,}")
        print(f"  Total FPGA cycles:     {rlhf['total_cycles']:,}")

        print(f"\n  Estimated Time:")
        print(f"    {rlhf['total_time_seconds']:.1f} seconds")
        print(f"    {rlhf['total_time_minutes']:.1f} minutes")

        print(f"\n  Estimated Energy:")
        print(f"    {rlhf['total_energy_joules']:,.1f} Joules")
        print(f"    {rlhf['total_energy_wh']:.3f} Wh")

        print("\n" + "=" * 70)
        print("Notes:")
        print("=" * 70)
        print("1. FPGA power estimate: Based on typical F2 instance power (~35W)")
        print("   For accurate power, check Lab 1 synthesis XPE report")
        print("\n2. RLHF matmul count: Rough estimate for Qwen2.5-0.5B")
        print("   Actual count may vary based on model architecture")
        print("\n3. Bottleneck: Data transfer dominates (PCIe overhead)")
        print("   FPGA efficiency improves with batching/pipelining")
        print("\n4. Compare with CPU baseline when Option A completes")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Lab 1 FPGA test results and calculate energy metrics"
    )
    parser.add_argument(
        "--data-cycles",
        type=int,
        default=15357,
        help="Data transfer cycles from Lab 1 test (default: 15357)"
    )
    parser.add_argument(
        "--compute-cycles",
        type=int,
        default=15,
        help="Compute cycles from Lab 1 test (default: 15)"
    )
    parser.add_argument(
        "--clock-mhz",
        type=float,
        help=f"FPGA clock frequency in MHz (default: {Lab1PerformanceAnalyzer.DEFAULT_CLOCK_MHZ})"
    )
    parser.add_argument(
        "--fpga-power",
        type=float,
        help=f"FPGA power in Watts (default: {Lab1PerformanceAnalyzer.DEFAULT_FPGA_POWER_W})"
    )
    parser.add_argument(
        "--ppo-steps",
        type=int,
        default=50,
        help="Number of PPO steps for RLHF workload estimate (default: 50)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = Lab1PerformanceAnalyzer(
        data_cycles=args.data_cycles,
        compute_cycles=args.compute_cycles,
        clock_mhz=args.clock_mhz,
        fpga_power_w=args.fpga_power
    )

    # Print report
    analyzer.print_report(num_ppo_steps=args.ppo_steps)

    # Save to JSON if requested
    if args.output_json:
        report = analyzer.generate_report(num_ppo_steps=args.ppo_steps)
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
