"""
GPU Power Monitoring Wrapper
Provides Python interface to nvidia-smi power monitoring.
"""

import subprocess
import time
import os
import signal
import pandas as pd
from pathlib import Path

class GPUPowerMonitor:
    """Context manager for GPU power monitoring."""

    def __init__(self, output_file="power_log.csv", interval_ms=100):
        """
        Initialize GPU power monitor.

        Args:
            output_file: Path to save power measurements
            interval_ms: Sampling interval in milliseconds
        """
        self.output_file = output_file
        self.interval_ms = interval_ms
        self.process = None
        self.start_time = None

    def __enter__(self):
        """Start monitoring."""
        print(f"🔋 Starting GPU power monitoring...")
        print(f"   Output: {self.output_file}")
        print(f"   Interval: {self.interval_ms}ms")

        # Start nvidia-smi in background
        self.start_time = time.time()
        self.process = subprocess.Popen(
            ["bash", "baseline_energy/monitor_gpu_power.sh",
             self.output_file, str(self.interval_ms)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Give it a moment to start
        time.sleep(0.5)
        print("✓ Monitoring started")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring."""
        if self.process:
            print("\n🛑 Stopping GPU power monitoring...")
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=5)

            runtime = time.time() - self.start_time
            print(f"✓ Monitoring stopped (runtime: {runtime:.1f}s)")
            print(f"   Data saved to: {self.output_file}")

    def get_average_power(self):
        """
        Parse logged power data and return average power.

        Returns:
            float: Average power in watts
        """
        if not os.path.exists(self.output_file):
            raise FileNotFoundError(f"Power log not found: {self.output_file}")

        # Read nvidia-smi dmon output
        # Format: # timestamp gpu pwr gtemp mtemp sm mem enc dec mclk pclk
        try:
            df = pd.read_csv(
                self.output_file,
                sep=r'\s+',
                comment='#',
                names=['timestamp', 'gpu', 'pwr', 'gtemp', 'mtemp',
                       'sm', 'mem', 'enc', 'dec', 'mclk', 'pclk']
            )

            avg_power = df['pwr'].mean()
            max_power = df['pwr'].max()
            min_power = df['pwr'].min()

            print(f"\n📊 Power Statistics:")
            print(f"   Average: {avg_power:.2f} W")
            print(f"   Max:     {max_power:.2f} W")
            print(f"   Min:     {min_power:.2f} W")

            return avg_power

        except Exception as e:
            print(f"⚠️  Error parsing power log: {e}")
            return None


def test_monitor():
    """Test the GPU power monitor."""
    print("Testing GPU Power Monitor")
    print("=" * 60)

    with GPUPowerMonitor("test_power_log.csv", interval_ms=100) as monitor:
        print("\n⏱  Running test workload for 10 seconds...")

        # Simulate some GPU work
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Create some tensors and do operations
            for i in range(100):
                x = torch.randn(1000, 1000, device=device)
                y = torch.randn(1000, 1000, device=device)
                z = torch.matmul(x, y)
                time.sleep(0.1)
        else:
            print("   No GPU available - just waiting...")
            time.sleep(10)

    # Get average power
    avg_power = monitor.get_average_power()

    print("\n" + "=" * 60)
    print(f"✅ Test complete!")
    if avg_power:
        print(f"   Average power during test: {avg_power:.2f} W")
    print("=" * 60)


if __name__ == "__main__":
    test_monitor()
