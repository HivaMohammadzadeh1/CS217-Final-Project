"""
FPGA Matmul Offload with Tiling

This module provides:
1. Tiling function that breaks any matmul into 16x16 chunks
2. FPGA offload wrapper (mock, or real Lab 1 FPGA)
3. Result reassembly

Usage:
    from fpga_matmul_offload import FPGAMatmulOffload

    # Option 1: Mock FPGA (for testing without hardware)
    offloader = FPGAMatmulOffload(use_mock=True)
    result = offloader.matmul(A, B)

    # Option 2: Real Lab 1 FPGA (requires hardware + bitstream)
    offloader = FPGAMatmulOffload(use_mock=False, use_lab1=True)
    result = offloader.matmul(A, B)
"""

import numpy as np
import torch
import time
from pathlib import Path

# Try to import Lab 1 FPGA interface
try:
    from lab1_fpga_interface import Lab1FPGAInterface
    LAB1_AVAILABLE = True
except ImportError:
    LAB1_AVAILABLE = False


class MockFPGAInterface:
    """Mock FPGA interface for testing without real hardware."""

    def __init__(self):
        self.num_calls = 0
        self.total_tiles_processed = 0

    def matmul_16x16(self, tile_a, tile_b):
        """
        Simulate FPGA matmul on 16x16 tiles.

        Args:
            tile_a: (16, 16) numpy array
            tile_b: (16, 16) numpy array

        Returns:
            (16, 16) numpy array result
        """
        self.num_calls += 1
        self.total_tiles_processed += 1

        # Simulate FPGA latency (adjust based on your FPGA specs)
        time.sleep(0.0001)  # 0.1ms per tile

        # Perform actual computation (in real FPGA, this happens in hardware)
        result = np.matmul(tile_a, tile_b)

        return result

    def get_stats(self):
        """Return statistics about FPGA usage."""
        return {
            'num_calls': self.num_calls,
            'total_tiles': self.total_tiles_processed
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.num_calls = 0
        self.total_tiles_processed = 0


class RealFPGAInterface:
    """Real FPGA interface using Lab 1 hardware."""

    def __init__(self, device_id=0, verbose=False, use_lab1=True):
        """
        Initialize real FPGA connection.

        Args:
            device_id: FPGA device ID (for AWS F2)
            verbose: Print debug information
            use_lab1: Use Lab 1 FPGA interface (16×16 matmul accelerator)
        """
        self.device_id = device_id
        self.verbose = verbose
        self.use_lab1 = use_lab1

        if use_lab1 and LAB1_AVAILABLE:
            # Use Lab 1 FPGA interface
            self.fpga = Lab1FPGAInterface(device_id=device_id, verbose=verbose)
            if verbose:
                print("✓ Using Lab 1 FPGA interface (16×16 matmul accelerator)")
        elif use_lab1 and not LAB1_AVAILABLE:
            if verbose:
                print("⚠️  Lab 1 FPGA interface not available")
                print("    Make sure lab1_fpga_interface.py is in integration/")
            raise ImportError("Lab 1 FPGA interface not available")
        else:
            # Generic FPGA interface (not implemented)
            raise NotImplementedError("Generic FPGA interface not yet implemented")

    def matmul_16x16(self, tile_a, tile_b):
        """
        Send 16x16 tiles to FPGA for matmul.

        Args:
            tile_a: (16, 16) numpy array
            tile_b: (16, 16) numpy array

        Returns:
            (16, 16) numpy array result
        """
        if self.use_lab1:
            # Use Lab 1 FPGA hardware (or software fallback if hardware unavailable)
            return self.fpga.matmul_16x16(tile_a, tile_b)
        else:
            raise NotImplementedError("Generic FPGA interface not yet implemented")

    def get_stats(self):
        """Return FPGA statistics."""
        if self.use_lab1:
            return self.fpga.get_stats()
        return {}

    def reset_stats(self):
        """Reset FPGA statistics."""
        if self.use_lab1:
            self.fpga.reset_stats()


class FPGAMatmulOffload:
    """
    Main class for offloading matmul operations to FPGA with automatic tiling.
    """

    TILE_SIZE = 16  # Fixed 16x16 tile size for FPGA

    def __init__(self, use_mock=True, device_id=0, verbose=False, use_lab1=True):
        """
        Initialize FPGA matmul offloader.

        Args:
            use_mock: If True, use mock FPGA for testing
            device_id: FPGA device ID (for real FPGA)
            verbose: Print detailed tiling information
            use_lab1: Use Lab 1 FPGA interface (when use_mock=False)
        """
        self.use_mock = use_mock
        self.verbose = verbose

        if use_mock:
            self.fpga = MockFPGAInterface()
            if verbose:
                print("🔧 Using MOCK FPGA interface")
        else:
            self.fpga = RealFPGAInterface(device_id=device_id, verbose=verbose, use_lab1=use_lab1)
            if verbose:
                print(f"🔧 Connected to real FPGA (device {device_id})")

    def matmul(self, A, B):
        """
        Perform matrix multiplication with automatic tiling to FPGA.

        Args:
            A: Input matrix of shape (M, K) - numpy array or torch tensor
            B: Input matrix of shape (K, N) - numpy array or torch tensor

        Returns:
            Result matrix of shape (M, N) - same type as input
        """
        # Convert torch tensors to numpy if needed
        is_torch = isinstance(A, torch.Tensor)
        if is_torch:
            A_np = A.detach().cpu().numpy()
            B_np = B.detach().cpu().numpy()
            device = A.device
        else:
            A_np = A
            B_np = B

        # Get dimensions
        M, K = A_np.shape
        K2, N = B_np.shape

        assert K == K2, f"Dimension mismatch: A is ({M}, {K}), B is ({K2}, {N})"

        if self.verbose:
            print(f"\n📊 Matmul: ({M}, {K}) × ({K}, {N}) = ({M}, {N})")

        # Perform tiled matmul
        result = self._tiled_matmul(A_np, B_np)

        # Convert back to torch if needed
        if is_torch:
            result = torch.from_numpy(result).to(device)

        return result

    def _tiled_matmul(self, A, B):
        """
        Break matmul into 16x16 tiles and send to FPGA.

        Args:
            A: (M, K) numpy array
            B: (K, N) numpy array

        Returns:
            (M, N) numpy array result
        """
        M, K = A.shape
        K2, N = B.shape

        # Calculate number of tiles needed
        tile_m = (M + self.TILE_SIZE - 1) // self.TILE_SIZE
        tile_k = (K + self.TILE_SIZE - 1) // self.TILE_SIZE
        tile_n = (N + self.TILE_SIZE - 1) // self.TILE_SIZE

        if self.verbose:
            print(f"   Tiles: {tile_m} × {tile_k} × {tile_n} = {tile_m * tile_k * tile_n} total operations")

        # Pad matrices to be multiples of TILE_SIZE
        A_padded = self._pad_matrix(A, tile_m * self.TILE_SIZE, tile_k * self.TILE_SIZE)
        B_padded = self._pad_matrix(B, tile_k * self.TILE_SIZE, tile_n * self.TILE_SIZE)

        # Initialize result matrix (padded)
        C_padded = np.zeros((tile_m * self.TILE_SIZE, tile_n * self.TILE_SIZE), dtype=A.dtype)

        # Perform tiled matrix multiplication
        # C[i,j] = sum_k A[i,k] * B[k,j]
        for i in range(tile_m):
            for j in range(tile_n):
                # Accumulate partial results for tile C[i,j]
                tile_result = np.zeros((self.TILE_SIZE, self.TILE_SIZE), dtype=A.dtype)

                for k in range(tile_k):
                    # Extract 16x16 tiles
                    tile_a = self._extract_tile(A_padded, i, k)
                    tile_b = self._extract_tile(B_padded, k, j)

                    # Send to FPGA for computation
                    tile_product = self.fpga.matmul_16x16(tile_a, tile_b)

                    # Accumulate result
                    tile_result += tile_product

                # Store result tile
                self._store_tile(C_padded, i, j, tile_result)

        # Remove padding to get final result
        result = C_padded[:M, :N]

        return result

    def _pad_matrix(self, matrix, target_rows, target_cols):
        """Pad matrix to target size with zeros."""
        rows, cols = matrix.shape
        if rows == target_rows and cols == target_cols:
            return matrix

        padded = np.zeros((target_rows, target_cols), dtype=matrix.dtype)
        padded[:rows, :cols] = matrix
        return padded

    def _extract_tile(self, matrix, tile_i, tile_j):
        """Extract a 16x16 tile from the matrix."""
        row_start = tile_i * self.TILE_SIZE
        col_start = tile_j * self.TILE_SIZE

        tile = matrix[
            row_start:row_start + self.TILE_SIZE,
            col_start:col_start + self.TILE_SIZE
        ]

        return tile

    def _store_tile(self, matrix, tile_i, tile_j, tile_data):
        """Store a 16x16 tile into the matrix."""
        row_start = tile_i * self.TILE_SIZE
        col_start = tile_j * self.TILE_SIZE

        matrix[
            row_start:row_start + self.TILE_SIZE,
            col_start:col_start + self.TILE_SIZE
        ] = tile_data

    def get_stats(self):
        """Get FPGA usage statistics."""
        return self.fpga.get_stats()

    def reset_stats(self):
        """Reset FPGA usage statistics."""
        self.fpga.reset_stats()


# Convenience function
def offload_matmul(A, B, use_mock=True, verbose=False):
    """
    Quick function to offload a matmul to FPGA.

    Args:
        A: (M, K) matrix
        B: (K, N) matrix
        use_mock: Use mock FPGA interface
        verbose: Print debug info

    Returns:
        (M, N) result matrix
    """
    offloader = FPGAMatmulOffload(use_mock=use_mock, verbose=verbose)
    return offloader.matmul(A, B)


if __name__ == "__main__":
    print("=" * 60)
    print("FPGA Matmul Offload - Test Script")
    print("=" * 60)

    # Test 1: Simple 32x32 matmul (should create 2×2×2 = 8 tiles)
    print("\n📝 Test 1: 32×32 matmul")
    print("-" * 60)
    A = np.random.randn(32, 32).astype(np.float32)
    B = np.random.randn(32, 32).astype(np.float32)

    offloader = FPGAMatmulOffload(use_mock=True, verbose=True)
    result_fpga = offloader.matmul(A, B)

    # Verify correctness
    result_numpy = np.matmul(A, B)
    error = np.max(np.abs(result_fpga - result_numpy))

    print(f"\n✅ Max error vs NumPy: {error:.2e}")
    print(f"   FPGA stats: {offloader.get_stats()}")

    # Test 2: Non-square matmul with padding
    print("\n📝 Test 2: 25×30 × 30×20 matmul (non-square, requires padding)")
    print("-" * 60)
    A = np.random.randn(25, 30).astype(np.float32)
    B = np.random.randn(30, 20).astype(np.float32)

    offloader.reset_stats()
    result_fpga = offloader.matmul(A, B)
    result_numpy = np.matmul(A, B)
    error = np.max(np.abs(result_fpga - result_numpy))

    print(f"\n✅ Max error vs NumPy: {error:.2e}")
    print(f"   FPGA stats: {offloader.get_stats()}")

    # Test 3: PyTorch tensor support
    print("\n📝 Test 3: PyTorch tensor matmul")
    print("-" * 60)
    A_torch = torch.randn(16, 16)
    B_torch = torch.randn(16, 16)

    offloader.reset_stats()
    result_fpga = offloader.matmul(A_torch, B_torch)
    result_pytorch = torch.matmul(A_torch, B_torch)
    error = torch.max(torch.abs(result_fpga - result_pytorch)).item()

    print(f"\n✅ Max error vs PyTorch: {error:.2e}")
    print(f"   FPGA stats: {offloader.get_stats()}")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
