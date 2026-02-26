# Lab 1 FPGA Integration

This directory contains the Lab 1 FPGA integration code for offloading matrix multiplications to your CS217 Lab 1 16×16 matmul accelerator.

## Files

- **`fpga_matmul_offload.py`**: Main FPGA offload wrapper with automatic tiling
  - Breaks large matmuls into 16×16 tiles for FPGA processing
  - Supports mock FPGA mode for testing
  - Supports real Lab 1 FPGA hardware

- **`lab1_fpga_interface.py`**: Lab 1 FPGA hardware interface
  - Python wrapper for Lab 1 16×16 matmul accelerator
  - Uses ctypes to call Lab 1 C library functions
  - Implements Lab 1 communication protocol (RVA, AXI-lite)
  - Software fallback when hardware unavailable

- **`test_lab1_integration.py`**: Test suite for Lab 1 integration
  - Tests basic 16×16 matmul
  - Tests tiled matmul (32×32, 64×64)
  - Verifies correctness against numpy

## Quick Start

### Option 1: Mock FPGA (Software Fallback)

For testing without FPGA hardware:

```python
from fpga_matmul_offload import FPGAMatmulOffload

# Create offloader with mock FPGA
offloader = FPGAMatmulOffload(use_mock=True, verbose=True)

# Use for matmul
result = offloader.matmul(A, B)  # Automatically tiled to 16×16
```

### Option 2: Lab 1 FPGA Hardware

For using actual Lab 1 FPGA on f2.6xlarge:

```python
from fpga_matmul_offload import FPGAMatmulOffload

# Create offloader with Lab 1 FPGA
offloader = FPGAMatmulOffload(
    use_mock=False,
    use_lab1=True,
    device_id=0,
    verbose=True
)

# Use for matmul (will use hardware if available, software fallback otherwise)
result = offloader.matmul(A, B)
```

## Lab 1 FPGA Setup

See `../LAB1_FPGA_INTEGRATION.md` for detailed setup instructions.

Quick steps:

1. **Load Lab 1 bitstream**: `sudo fpga-load-local-image -S 0 -I <afi_id>`
2. **Compile Lab 1 library**: See LAB1_FPGA_INTEGRATION.md
3. **Test integration**: `python integration/test_lab1_integration.py --mock`
4. **Run training**: `sudo -E python baseline_energy/rlhf_with_fpga.py --steps 2 --output results/test`

## Usage in RLHF Training

Use the pre-configured Lab 1 config:

```bash
cp baseline_energy/config_lab1_fpga.py baseline_energy/config.py
sudo -E python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/lab1_fpga_baseline
```

## Architecture

```
nn.Linear layers in PyTorch model
    ↓
FPGALinearLayer wrapper (replaces nn.Linear)
    ↓
FPGAMatmulOffload (automatic 16×16 tiling)
    ↓
Lab1FPGAInterface (hardware communication)
    ↓
libdesign_top.so (Lab 1 C library)
    ↓
AWS FPGA SDK (PCIe communication)
    ↓
Lab 1 FPGA Hardware (16×16 matmul accelerator)
```

## References

- Full Integration Guide: `../LAB1_FPGA_INTEGRATION.md`
- Training Script: `../baseline_energy/rlhf_with_fpga.py`
- Lab 1 Code: `/Users/hivamoh/CS217-Project/cs217-lab-1-hiva/`
