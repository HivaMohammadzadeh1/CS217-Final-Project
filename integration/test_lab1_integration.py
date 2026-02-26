"""
Test Lab 1 FPGA Integration

This script tests the Lab 1 FPGA interface at multiple levels:
1. Basic 16×16 matmul on Lab 1 FPGA
2. Tiled matmul (32×32, 64×64) using FPGAMatmulOffload
3. Verifies correctness against numpy reference

Usage:
    # Test with mock FPGA (software fallback)
    python integration/test_lab1_integration.py --mock

    # Test with real Lab 1 FPGA hardware
    sudo python integration/test_lab1_integration.py
"""

import numpy as np
import argparse
import sys
from pathlib import Path

# Add integration directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lab1_fpga_interface import Lab1FPGAInterface
from fpga_matmul_offload import FPGAMatmulOffload


def test_lab1_basic():
    """Test basic 16×16 matmul on Lab 1 FPGA."""
    print("\n" + "="*60)
    print("TEST 1: Basic 16×16 Matrix Multiplication")
    print("="*60)

    # Create Lab 1 FPGA interface
    fpga = Lab1FPGAInterface(device_id=0, verbose=True)

    # Create random 16×16 matrices
    np.random.seed(42)
    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)

    print("\nTest matrices:")
    print(f"  A: {A.shape}, range [{A.min():.2f}, {A.max():.2f}]")
    print(f"  B: {B.shape}, range [{B.min():.2f}, {B.max():.2f}]")

    # Compute on FPGA (or software fallback)
    print("\nComputing C = A @ B on Lab 1 FPGA...")
    C_fpga = fpga.matmul_16x16(A, B)

    # Compute ground truth
    C_true = np.matmul(A, B)

    # Check error
    max_error = np.max(np.abs(C_fpga - C_true))
    rel_error = max_error / (np.abs(C_true).max() + 1e-8)

    print(f"\nResults:")
    print(f"  C_fpga: {C_fpga.shape}, range [{C_fpga.min():.2f}, {C_fpga.max():.2f}]")
    print(f"  C_true: {C_true.shape}, range [{C_true.min():.2f}, {C_true.max():.2f}]")
    print(f"  Max absolute error: {max_error:.2e}")
    print(f"  Max relative error: {rel_error:.2e}")

    # Get FPGA stats
    stats = fpga.get_stats()
    print(f"\nFPGA Statistics:")
    print(f"  Num calls: {stats['num_calls']}")
    print(f"  Total tiles: {stats['total_tiles']}")
    print(f"  Using hardware: {stats['using_hardware']}")

    # Check if test passed
    tolerance = 1e-3  # Allow some error for FPGA quantization
    if max_error < tolerance:
        print(f"\n✓ TEST PASSED (error < {tolerance})")
        return True
    else:
        print(f"\n✗ TEST FAILED (error > {tolerance})")
        return False


def test_fpga_offload_tiled():
    """Test tiled matmul using FPGAMatmulOffload."""
    print("\n" + "="*60)
    print("TEST 2: Tiled Matrix Multiplication (32×32 and 64×64)")
    print("="*60)

    # Create FPGA offloader
    offloader = FPGAMatmulOffload(
        use_mock=False,  # Try to use real hardware
        use_lab1=True,
        verbose=True
    )

    # Test 32×32
    print("\n--- Test 2a: 32×32 matrix multiplication ---")
    np.random.seed(43)
    A32 = np.random.randn(32, 32).astype(np.float32)
    B32 = np.random.randn(32, 32).astype(np.float32)

    print(f"Computing 32×32 matmul (should tile to 2×2 = 4 tiles)...")
    C32_fpga = offloader.matmul(A32, B32)
    C32_true = np.matmul(A32, B32)

    error_32 = np.max(np.abs(C32_fpga - C32_true))
    print(f"  Max error: {error_32:.2e}")

    # Test 64×64
    print("\n--- Test 2b: 64×64 matrix multiplication ---")
    A64 = np.random.randn(64, 64).astype(np.float32)
    B64 = np.random.randn(64, 64).astype(np.float32)

    print(f"Computing 64×64 matmul (should tile to 4×4 = 16 tiles)...")
    C64_fpga = offloader.matmul(A64, B64)
    C64_true = np.matmul(A64, B64)

    error_64 = np.max(np.abs(C64_fpga - C64_true))
    print(f"  Max error: {error_64:.2e}")

    # Get stats
    stats = offloader.get_stats()
    print(f"\nFPGA Statistics:")
    total_matmuls = stats.get('total_matmuls', stats.get('num_calls', 0))
    total_tiles = stats.get('total_tiles', 0)
    print(f"  Total matmuls: {total_matmuls}")
    print(f"  Total tiles: {total_tiles}")
    if total_matmuls > 0:
        print(f"  Avg tiles per matmul: {total_tiles / total_matmuls:.1f}")

    # Check if tests passed
    tolerance = 1e-3
    passed_32 = error_32 < tolerance
    passed_64 = error_64 < tolerance

    if passed_32 and passed_64:
        print(f"\n✓ TEST PASSED (both errors < {tolerance})")
        return True
    else:
        print(f"\n✗ TEST FAILED")
        if not passed_32:
            print(f"  32×32 error: {error_32:.2e} > {tolerance}")
        if not passed_64:
            print(f"  64×64 error: {error_64:.2e} > {tolerance}")
        return False


def test_mock_mode():
    """Test with mock FPGA (software fallback)."""
    print("\n" + "="*60)
    print("TEST 3: Mock FPGA Mode (Software Fallback)")
    print("="*60)

    offloader = FPGAMatmulOffload(
        use_mock=True,
        verbose=True
    )

    np.random.seed(44)
    A = np.random.randn(32, 32).astype(np.float32)
    B = np.random.randn(32, 32).astype(np.float32)

    print("\nComputing 32×32 matmul with mock FPGA...")
    C_mock = offloader.matmul(A, B)
    C_true = np.matmul(A, B)

    error = np.max(np.abs(C_mock - C_true))
    print(f"  Max error: {error:.2e}")

    stats = offloader.get_stats()
    print(f"\nMock FPGA Statistics:")
    print(f"  Total matmuls: {stats.get('total_matmuls', 'N/A')}")
    print(f"  Total tiles: {stats.get('total_tiles', stats.get('num_calls', 0))}")

    if error < 1e-6:  # Mock should be exact
        print(f"\n✓ TEST PASSED (error < 1e-6)")
        return True
    else:
        print(f"\n✗ TEST FAILED (error > 1e-6)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Lab 1 FPGA Integration")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock FPGA mode (software fallback)"
    )
    args = parser.parse_args()

    print("="*60)
    print("Lab 1 FPGA Integration Test Suite")
    print("="*60)
    print(f"Mode: {'MOCK (software fallback)' if args.mock else 'REAL Hardware (if available)'}")

    results = []

    # Test 1: Basic 16×16 matmul
    try:
        result = test_lab1_basic()
        results.append(("Basic 16×16 matmul", result))
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED with exception: {e}")
        results.append(("Basic 16×16 matmul", False))

    # Test 2: Tiled matmul (only if not mock mode)
    if not args.mock:
        try:
            result = test_fpga_offload_tiled()
            results.append(("Tiled matmul", result))
        except Exception as e:
            print(f"\n✗ TEST 2 FAILED with exception: {e}")
            results.append(("Tiled matmul", False))

    # Test 3: Mock mode
    try:
        result = test_mock_mode()
        results.append(("Mock mode", result))
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED with exception: {e}")
        results.append(("Mock mode", False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:30s}: {status}")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
