#!/usr/bin/env python3
"""
Test FPGA Connection
Quick test to verify Lab 1 FPGA is accessible from Python
"""

import sys
import numpy as np
from pathlib import Path

# Add integration directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from lab1_fpga_interface import Lab1FPGAInterface
except ImportError as e:
    print(f"❌ Failed to import Lab1FPGAInterface: {e}")
    sys.exit(1)

def main():
    print("=" * 60)
    print("Lab 1 FPGA Connection Test")
    print("=" * 60)

    # Test 1: Initialize FPGA
    print("\n📝 Test 1: Initialize FPGA interface")
    print("-" * 60)

    try:
        fpga = Lab1FPGAInterface(device_id=0, verbose=True)
        print(f"\n✓ FPGA interface created")
        print(f"  Using hardware: {fpga.use_hardware}")
    except Exception as e:
        print(f"\n❌ Failed to create FPGA interface: {e}")
        sys.exit(1)

    # Test 2: Simple 16x16 matmul
    print("\n📝 Test 2: 16×16 matrix multiplication")
    print("-" * 60)

    # Create small test matrices
    A = np.random.randn(16, 16).astype(np.float32) * 0.1  # Small values for quantization
    B = np.random.randn(16, 16).astype(np.float32) * 0.1

    try:
        # Compute on FPGA (or software fallback)
        C_fpga = fpga.matmul_16x16(A, B)

        # Compute ground truth
        C_true = np.matmul(A, B)

        # Check error
        error = np.max(np.abs(C_fpga - C_true))
        rel_error = error / (np.max(np.abs(C_true)) + 1e-10)

        print(f"\n✓ Matmul completed")
        print(f"  Max absolute error: {error:.6f}")
        print(f"  Max relative error: {rel_error:.2%}")

        if fpga.use_hardware:
            if error < 1.0:  # Allow some quantization error for INT8
                print(f"  ✓ Error within acceptable range for hardware FPGA")
            else:
                print(f"  ⚠️  Error seems high - check quantization/scaling")
        else:
            if error < 1e-5:  # Software should be exact
                print(f"  ✓ Software fallback working correctly")
            else:
                print(f"  ❌ Software fallback has errors!")

    except Exception as e:
        print(f"\n❌ Matmul test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 3: Get statistics
    print("\n📝 Test 3: FPGA statistics")
    print("-" * 60)

    stats = fpga.get_stats()
    print(f"  Number of calls: {stats['num_calls']}")
    print(f"  Total tiles: {stats['total_tiles']}")
    print(f"  Using hardware: {stats['using_hardware']}")

    # Summary
    print("\n" + "=" * 60)
    if fpga.use_hardware:
        print("✅ SUCCESS: Lab 1 FPGA hardware is working!")
        print("   Your RLHF pipeline will use the real FPGA")
    else:
        print("⚠️  NOTICE: Using software fallback")
        print("   Your RLHF pipeline will use CPU (not FPGA)")
        print("")
        print("   To enable hardware:")
        print("   1. Compile wrapper: bash compile_lab1_wrapper.sh")
        print("   2. Load AFI: sudo fpga-load-local-image -S 0 -I <afi-id>")
        print("   3. Run this test again")
    print("=" * 60)

if __name__ == "__main__":
    main()
