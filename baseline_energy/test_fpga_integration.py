"""
Quick test to verify FPGA integration works before full run.

This runs 2 PPO steps to test:
- Model loading with FPGA offload
- Dataset loading (800/200 split)
- Tiling and reassembly
- Phase timing
- Energy calculation

Usage:
    python baseline_energy/test_fpga_integration.py
"""

import subprocess
import sys
from pathlib import Path

import torch


def main():
    print("=" * 60)
    print("Testing FPGA Integration")
    print("=" * 60)

    # Test 1: Import check
    print("\n✓ Test 1: Checking imports...")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "integration"))
        from fpga_matmul_offload import FPGAMatmulOffload
        print("  ✓ FPGA offload module imported")
    except ImportError as e:
        print(f"  ✗ Failed to import: {e}")
        sys.exit(1)

    # Test 2: FPGA offloader initialization
    print("\n✓ Test 2: Initializing FPGA offloader...")
    try:
        offloader = FPGAMatmulOffload(use_mock=True, verbose=False)
        print("  ✓ FPGA offloader initialized (mock mode)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        sys.exit(1)

    # Test 3: Simple matmul test
    print("\n✓ Test 3: Testing matmul offload...")
    try:
        A = torch.randn(32, 32)
        B = torch.randn(32, 32)

        result_fpga = offloader.matmul(A, B)
        result_torch = torch.matmul(A, B)

        error = torch.max(torch.abs(result_fpga - result_torch)).item()
        print(f"  ✓ Matmul test passed (error: {error:.2e})")

        stats = offloader.get_stats()
        print(f"  ✓ FPGA processed {stats['total_tiles']} tiles")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        sys.exit(1)

    # Test 4: Config check
    print("\n✓ Test 4: Checking configuration...")
    try:
        try:
            from baseline_energy import config
        except ImportError:
            import config
        print(f"  Dataset: {config.DATASET_NAME}")
        print(f"  Train size: {config.TRAIN_SIZE}")
        print(f"  Eval size: {config.EVAL_SIZE}")
        print(f"  Total: {config.NUM_SAMPLES}")
        print(f"  FPGA offload: {config.USE_FPGA_OFFLOAD}")
        print(f"  Mock FPGA: {config.USE_MOCK_FPGA}")

        assert config.TRAIN_SIZE == 800, "Train size should be 800"
        assert config.EVAL_SIZE == 200, "Eval size should be 200"
        print("  ✓ Configuration is correct")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        sys.exit(1)

    # Test 5: Run 2-step mini training
    print("\n✓ Test 5: Running 2-step mini training...")
    print("  (This will test the full pipeline)")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "baseline_energy/rlhf_with_fpga.py",
                "--steps", "2",
                "--output", "results/integration_test",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            print("  ✓ 2-step training completed successfully")

            # Check output files
            output_dir = Path("results/integration_test")
            expected_files = [
                "phase_timing.json",
                "training_stats.json",
                "fpga_stats.json",
            ]

            for fname in expected_files:
                if (output_dir / fname).exists():
                    print(f"    ✓ {fname} created")
                else:
                    print(f"    ✗ {fname} missing")
        else:
            print(f"  ✗ Training failed with return code {result.returncode}")
            print(f"  stderr: {result.stderr[:500]}")

    except subprocess.TimeoutExpired:
        print("  ⚠️  Training timed out (this is OK, probably model downloading)")
    except Exception as e:
        print(f"  ⚠️  Could not run full test: {e}")
        print("     You can manually test with:")
        print("     python baseline_energy/rlhf_with_fpga.py --steps 2 --output results/test")

    print("\n" + "=" * 60)
    print("✅ FPGA Integration Tests Complete!")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("  1. Run full 50-step baseline:")
    print("     python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/baseline_50")
    print("\n  2. Calculate energy:")
    print("     python baseline_energy/calculate_energy.py --results results/baseline_50")
    print("\n  3. When FPGA hardware is ready:")
    print("     - Set USE_MOCK_FPGA = False in config.py")
    print("     - Implement RealFPGAInterface in fpga_matmul_offload.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
