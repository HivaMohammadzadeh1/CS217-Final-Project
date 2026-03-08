"""
Lab 1 FPGA Interface for RLHF Training
Integrates your CS217 Lab 1 16x16 matmul FPGA with the RLHF training pipeline.

Uses liblab1_wrapper.so which properly initializes the FPGA via AWS SDK
(fpga_mgmt_init + fpga_pci_attach) and exposes fpga_matmul_16x16().

Build the wrapper first:
    export SDK_DIR=/home/ubuntu/src/project_data/aws-fpga/sdk
    bash integration/compile_lab1_wrapper.sh
"""

import numpy as np
import ctypes
import os
import time
from pathlib import Path


class Lab1FPGAInterface:
    """
    Interface to Lab 1 FPGA hardware for 16x16 matrix multiplication.

    Hardware specs from Lab 1:
    - kVectorSize = 16, kNumVectorLanes = 16
    - INT8 precision (kIntWordWidth = 8)
    - Output: INT32 activations (kActWordWidth = 32)

    Uses liblab1_wrapper.so for proper FPGA init and register access.
    """

    def __init__(self, device_id=0, verbose=False):
        self.device_id = device_id
        self.verbose = verbose

        self.TILE_SIZE = 16

        # Milestone 3 precision control state.
        # Current Lab 1 bitstream supports INT8 compute. MX modes are modeled
        # in software above this layer until MX hardware is deployed.
        self.precision_mode = "INT8"
        self.pending_precision_mode = "INT8"
        self.precision_switch_pending = False
        self.group_size = 8
        self.pipeline_flush_count = 0

        # Performance tracking
        self.total_calls = 0
        self.total_data_transfer_cycles = 0
        self.total_compute_cycles = 0
        self.use_hardware = False

        self._init_fpga()

    def _init_fpga(self):
        """Load liblab1_wrapper.so and initialize the FPGA device."""
        search_paths = [
            Path(__file__).parent / "liblab1_wrapper.so",
            Path.home() / "CS217-Final-Project" / "integration" / "liblab1_wrapper.so",
            Path.home() / "cs217-lab-1-hiva" / "design_top" / "software" / "build" / "liblab1_wrapper.so",
        ]

        lib_path = None
        for p in search_paths:
            if p.exists():
                lib_path = p
                break

        if lib_path is None:
            if self.verbose:
                print("⚠️  liblab1_wrapper.so not found. Searched:")
                for p in search_paths:
                    print(f"    {p}")
                print("    Build it with: bash integration/compile_lab1_wrapper.sh")
                print("    Using software fallback")
            self.use_hardware = False
            return

        try:
            self.lib = ctypes.CDLL(str(lib_path))

            # int fpga_init(int slot_id)
            self.lib.fpga_init.argtypes = [ctypes.c_int]
            self.lib.fpga_init.restype = ctypes.c_int

            # int fpga_matmul_16x16(float *A, float *B, float *C)
            self.lib.fpga_matmul_16x16.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            self.lib.fpga_matmul_16x16.restype = ctypes.c_int

            # void fpga_cleanup()
            self.lib.fpga_cleanup.argtypes = []
            self.lib.fpga_cleanup.restype = None

            # int fpga_is_initialized()
            self.lib.fpga_is_initialized.argtypes = []
            self.lib.fpga_is_initialized.restype = ctypes.c_int

            if self.verbose:
                print(f"✓ Loaded {lib_path}")

            rc = self.lib.fpga_init(self.device_id)
            if rc != 0:
                if self.verbose:
                    print(f"⚠️  fpga_init(slot={self.device_id}) failed (rc={rc})")
                    print("    Make sure FPGA bitstream is loaded and you're running as root/sudo")
                    print("    Using software fallback")
                self.use_hardware = False
                return

            self.use_hardware = True
            if self.verbose:
                print(f"✓ FPGA initialized on slot {self.device_id} — using REAL hardware")

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to load/init FPGA wrapper: {e}")
                print("    Using software fallback")
            self.use_hardware = False

    def configure_precision(self, precision_mode, group_size=None, flush=True):
        """
        Configure compute precision mode.

        Supported:
          - INT8  (native Lab 1 support)
          - MXFP8 (accepted for API compatibility; hardware fallback path)
          - MXFP4 (accepted for API compatibility; hardware fallback path)

        If/when a dedicated MX bitstream is deployed, this method is where the
        PEConfig precision/group-size fields should be programmed before start.
        """
        mode = str(precision_mode).upper()
        if mode not in ("INT8", "MXFP8", "MXFP4"):
            raise ValueError(f"Unsupported precision mode: {precision_mode}")
        if group_size is not None:
            if group_size not in (8, 16):
                raise ValueError("group_size must be 8 or 16.")
            self.group_size = group_size

        self.pending_precision_mode = mode
        self.precision_switch_pending = (self.pending_precision_mode != self.precision_mode)

        if flush:
            self.flush_pipeline()

    def flush_pipeline(self):
        """
        Apply pending precision change.

        In current Lab 1 INT8 hardware this is a software-level state update.
        For future MX hardware this should also trigger hardware pipeline drain
        semantics around mode transitions.
        """
        if not self.precision_switch_pending:
            return
        self.precision_mode = self.pending_precision_mode
        self.precision_switch_pending = False
        self.pipeline_flush_count += 1

    def matmul_16x16(self, A, B):
        """
        Perform 16x16 matrix multiplication.
        Uses FPGA hardware if available, otherwise numpy fallback.
        """
        assert A.shape == (16, 16), f"A must be 16x16, got {A.shape}"
        assert B.shape == (16, 16), f"B must be 16x16, got {B.shape}"

        if self.precision_switch_pending:
            raise RuntimeError(
                "Precision switch pending. Call flush_pipeline() before matmul_16x16()."
            )

        if self.use_hardware:
            return self._matmul_hardware(A, B)
        else:
            return self._matmul_software(A, B)

    def _matmul_hardware(self, A, B):
        """Execute 16x16 matmul on Lab 1 FPGA via liblab1_wrapper.so."""
        A_f32 = np.ascontiguousarray(A, dtype=np.float32)
        B_f32 = np.ascontiguousarray(B, dtype=np.float32)
        C_f32 = np.zeros((16, 16), dtype=np.float32)

        A_ptr = A_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        C_ptr = C_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        rc = self.lib.fpga_matmul_16x16(A_ptr, B_ptr, C_ptr)

        if rc != 0:
            if self.verbose:
                print(f"⚠️  FPGA matmul failed (rc={rc}), falling back to software")
            return self._matmul_software(A, B)

        self.total_calls += 1
        self.total_compute_cycles += 1
        self.total_data_transfer_cycles += 16 + 16  # weight rows + input columns
        return C_f32

    def _matmul_software(self, A, B):
        """Software fallback when FPGA is not available."""
        self.total_calls += 1
        return np.matmul(A, B).astype(np.float32)

    def get_stats(self):
        return {
            'num_calls': self.total_calls,
            'total_tiles': self.total_calls,
            'data_transfer_cycles': self.total_data_transfer_cycles,
            'compute_cycles': self.total_compute_cycles,
            'using_hardware': self.use_hardware,
            'precision_mode': self.precision_mode,
            'group_size': self.group_size,
            'switch_pending': self.precision_switch_pending,
            'pipeline_flush_count': self.pipeline_flush_count,
        }

    def reset_stats(self):
        self.total_calls = 0
        self.total_data_transfer_cycles = 0
        self.total_compute_cycles = 0
        self.pipeline_flush_count = 0

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'use_hardware') and self.use_hardware:
            try:
                self.lib.fpga_cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    print("Testing Lab 1 FPGA Interface")
    print("=" * 50)

    fpga = Lab1FPGAInterface(verbose=True)

    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)

    C_fpga = fpga.matmul_16x16(A, B)
    C_true = np.matmul(A, B)

    error = np.max(np.abs(C_fpga - C_true))
    print(f"\nMax error vs numpy: {error:.2e}")
    print(f"Using hardware: {fpga.use_hardware}")
    print(f"Stats: {fpga.get_stats()}")
