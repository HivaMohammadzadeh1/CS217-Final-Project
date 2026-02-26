#!/usr/bin/env python3
"""
FPGA Energy Calculator for CS217 Project

Calculates energy consumption for FPGA-based RLHF training using:
1. Measured FPGA cycle counts from Lab 1 hardware
2. Estimated matmul operations from RLHF workload
3. FPGA power consumption estimates

Usage:
    python integration/calculate_fpga_energy.py
    python integration/calculate_fpga_energy.py --cycles 85344 --power 35 --steps 50
"""

import argparse
import json
from pathlib import Path


class FPGAEnergyCalculator:
    """Calculate FPGA energy consumption for RLHF workload."""

    def __init__(self, cycles_per_matmul, fpga_power_w, fpga_clock_hz=250e6):
        """
        Initialize calculator.

        Args:
            cycles_per_matmul: Total FPGA cycles per 16x16 matmul (from Lab 1 measurement)
            fpga_power_w: FPGA power consumption in Watts
            fpga_clock_hz: FPGA clock frequency in Hz (default 250 MHz)
        """
        self.cycles_per_matmul = cycles_per_matmul
        self.fpga_power_w = fpga_power_w
        self.fpga_clock_hz = fpga_clock_hz

        # Time per matmul
        self.time_per_matmul_s = cycles_per_matmul / fpga_clock_hz

        # Energy per matmul
        self.energy_per_matmul_j = fpga_power_w * self.time_per_matmul_s
        self.energy_per_matmul_wh = self.energy_per_matmul_j / 3600

    def estimate_rlhf_matmuls(self, num_steps, batch_size=8, seq_len=512,
                               num_layers=24, matmuls_per_layer=7):
        """
        Estimate total matmul operations in RLHF training.

        For Qwen2.5-0.5B:
        - 24 transformer layers
        - Each layer has 7 matmuls: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        - Plus lm_head and value head

        Args:
            num_steps: Number of PPO steps
            batch_size: Batch size per step
            seq_len: Sequence length
            num_layers: Number of transformer layers (24 for Qwen2.5-0.5B)
            matmuls_per_layer: Matmuls per layer (7 for attention + FFN)

        Returns:
            dict with matmul estimates
        """
        # Forward pass matmuls per sample
        # Each layer: 7 matmuls (attention + FFN)
        # Output layer: 2 matmuls (lm_head + value_head)
        forward_matmuls_per_sample = (num_layers * matmuls_per_layer) + 2

        # Backward pass (gradients) - approximately same as forward
        backward_matmuls_per_sample = forward_matmuls_per_sample

        # Total per sample (forward + backward)
        total_matmuls_per_sample = forward_matmuls_per_sample + backward_matmuls_per_sample

        # Total for all steps
        total_samples = num_steps * batch_size
        total_matmuls = total_matmuls_per_sample * total_samples

        return {
            'num_steps': num_steps,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'total_samples': total_samples,
            'forward_matmuls_per_sample': forward_matmuls_per_sample,
            'backward_matmuls_per_sample': backward_matmuls_per_sample,
            'total_matmuls_per_sample': total_matmuls_per_sample,
            'total_matmuls': total_matmuls
        }

    def calculate_energy(self, num_matmuls):
        """
        Calculate energy for given number of matmuls.

        Args:
            num_matmuls: Number of 16x16 matmul operations

        Returns:
            dict with energy calculations
        """
        total_cycles = self.cycles_per_matmul * num_matmuls
        total_time_s = total_cycles / self.fpga_clock_hz
        total_time_min = total_time_s / 60

        total_energy_j = self.fpga_power_w * total_time_s
        total_energy_wh = total_energy_j / 3600

        return {
            'num_matmuls': num_matmuls,
            'total_cycles': total_cycles,
            'total_time_s': total_time_s,
            'total_time_min': total_time_min,
            'total_energy_j': total_energy_j,
            'total_energy_wh': total_energy_wh,
            'fpga_power_w': self.fpga_power_w,
            'cycles_per_matmul': self.cycles_per_matmul
        }

    def calculate_rlhf_energy(self, num_steps, batch_size=8, seq_len=512):
        """
        Calculate complete RLHF training energy.

        Args:
            num_steps: Number of PPO steps
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            dict with complete analysis
        """
        # Estimate matmuls
        matmul_est = self.estimate_rlhf_matmuls(num_steps, batch_size, seq_len)

        # Calculate energy
        energy = self.calculate_energy(matmul_est['total_matmuls'])

        # Combine results
        return {
            **matmul_est,
            **energy,
            'fpga_clock_mhz': self.fpga_clock_hz / 1e6,
            'time_per_matmul_us': self.time_per_matmul_s * 1e6,
            'energy_per_matmul_uj': self.energy_per_matmul_j * 1e6
        }


def print_analysis(results):
    """Pretty print analysis results."""
    print("\n" + "="*70)
    print("FPGA Energy Analysis for RLHF Training")
    print("="*70)

    print(f"\n{'FPGA Configuration:':<30}")
    print(f"  Clock Frequency: {results['fpga_clock_mhz']:.1f} MHz")
    print(f"  Power Consumption: {results['fpga_power_w']:.1f} W")
    print(f"  Cycles per 16×16 matmul: {results['cycles_per_matmul']:,} cycles")
    print(f"  Time per matmul: {results['time_per_matmul_us']:.2f} μs")
    print(f"  Energy per matmul: {results['energy_per_matmul_uj']:.2f} μJ")

    print(f"\n{'Workload Configuration:':<30}")
    print(f"  PPO Steps: {results['num_steps']}")
    print(f"  Batch Size: {results['batch_size']}")
    print(f"  Sequence Length: {results['seq_len']}")
    print(f"  Total Samples: {results['total_samples']:,}")

    print(f"\n{'Matmul Operations:':<30}")
    print(f"  Forward matmuls per sample: {results['forward_matmuls_per_sample']:,}")
    print(f"  Backward matmuls per sample: {results['backward_matmuls_per_sample']:,}")
    print(f"  Total matmuls per sample: {results['total_matmuls_per_sample']:,}")
    print(f"  Total matmuls: {results['total_matmuls']:,}")

    print(f"\n{'FPGA Performance:':<30}")
    print(f"  Total Cycles: {results['total_cycles']:,.0f} cycles")
    print(f"  Total Time: {results['total_time_min']:.2f} minutes ({results['total_time_s']:.1f} seconds)")
    print(f"  Total Energy: {results['total_energy_wh']:.4f} Wh ({results['total_energy_j']:.1f} Joules)")

    print(f"\n{'Breakdown:':<30}")
    data_transfer_pct = 99.98  # From measurements
    compute_pct = 0.02
    print(f"  Data Transfer: {data_transfer_pct:.2f}% of cycles")
    print(f"  Computation: {compute_pct:.2f}% of cycles")

    print("\n" + "="*70)


def compare_with_cpu(fpga_results, cpu_energy_wh=None, cpu_time_min=None):
    """
    Compare FPGA with CPU baseline.

    Args:
        fpga_results: FPGA calculation results
        cpu_energy_wh: CPU energy in Watt-hours (if known)
        cpu_time_min: CPU time in minutes (if known)
    """
    if cpu_energy_wh is None or cpu_time_min is None:
        print("\n" + "="*70)
        print("CPU Comparison")
        print("="*70)
        print("\nCPU baseline data not available yet.")
        print("Run CPU baseline first:")
        print("  python baseline_energy/rlhf_baseline.py --steps 50")
        print("Then check: results/cpu_baseline_50steps/energy_summary.csv")
        return

    print("\n" + "="*70)
    print("FPGA vs CPU Comparison")
    print("="*70)

    fpga_energy = fpga_results['total_energy_wh']
    fpga_time = fpga_results['total_time_min']

    speedup = cpu_time_min / fpga_time
    energy_savings = cpu_energy_wh / fpga_energy

    print(f"\n{'Metric':<25} {'CPU':<20} {'FPGA (Est.)':<20} {'Improvement':<20}")
    print("-" * 85)
    print(f"{'Time':<25} {cpu_time_min:.2f} min{'':<13} {fpga_time:.2f} min{'':<13} {speedup:.2f}x faster")
    print(f"{'Energy':<25} {cpu_energy_wh:.4f} Wh{'':<11} {fpga_energy:.4f} Wh{'':<11} {energy_savings:.2f}x less energy")
    print(f"{'Power':<25} {(cpu_energy_wh * 60 / cpu_time_min):.1f} W (avg){'':<9} {fpga_results['fpga_power_w']:.1f} W{'':<15} {((cpu_energy_wh * 60 / cpu_time_min) / fpga_results['fpga_power_w']):.2f}x lower")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Calculate FPGA energy for RLHF training')
    parser.add_argument('--cycles', type=int, default=85344,
                        help='FPGA cycles per 16x16 matmul (default: 85344 from measurements)')
    parser.add_argument('--power', type=float, default=35,
                        help='FPGA power in Watts (default: 35W)')
    parser.add_argument('--clock', type=float, default=250,
                        help='FPGA clock in MHz (default: 250 MHz)')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of PPO training steps (default: 50)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Sequence length (default: 512)')
    parser.add_argument('--cpu-energy', type=float, default=None,
                        help='CPU energy in Wh for comparison (optional)')
    parser.add_argument('--cpu-time', type=float, default=None,
                        help='CPU time in minutes for comparison (optional)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results (optional)')

    args = parser.parse_args()

    # Create calculator
    calc = FPGAEnergyCalculator(
        cycles_per_matmul=args.cycles,
        fpga_power_w=args.power,
        fpga_clock_hz=args.clock * 1e6
    )

    # Calculate RLHF energy
    results = calc.calculate_rlhf_energy(
        num_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len
    )

    # Print analysis
    print_analysis(results)

    # Compare with CPU if data available
    compare_with_cpu(results, args.cpu_energy, args.cpu_time)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Example calculations for different workloads
    print("\n" + "="*70)
    print("Quick Reference - Different Workloads")
    print("="*70)

    for steps in [2, 10, 50, 100]:
        result = calc.calculate_rlhf_energy(steps, args.batch_size, args.seq_len)
        print(f"\n{steps} steps: {result['total_time_min']:.2f} min, "
              f"{result['total_energy_wh']:.4f} Wh, "
              f"{result['total_matmuls']:,} matmuls")


if __name__ == '__main__':
    main()
