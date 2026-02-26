# Lab 1 FPGA Integration Guide

## Overview

This document explains how to integrate your CS217 Lab 1 FPGA hardware with the RLHF training pipeline.

## Lab 1 FPGA Specifications

From `cs217-lab-1-hiva/design_top/`:

- **Matrix Size**: 16×16
- **Precision**: INT8 inputs/weights → INT32 outputs
- **Scaling**: Hardware applies 1/12.25 scaling factor
- **Interface**: AXI-lite for configuration, RVA for data transfer
- **Performance Counters**: Data transfer cycles, compute cycles

## Integration Architecture

```
RLHF Training (Python)
    ↓
fpga_matmul_offload.py (tiles large matmuls into 16×16)
    ↓
lab1_fpga_interface.py (Python wrapper)
    ↓
libdesign_top.so (C library with ctypes)
    ↓
AWS FPGA SDK (fpga_mgmt, fpga_pci)
    ↓
Lab 1 FPGA Hardware (Xilinx VU9P on f2.6xlarge)
```

## Integration Options

### Option 1: Direct C Integration (Recommended for Real Hardware)

**Pros**: Full AWS SDK support, direct hardware access
**Cons**: Requires C compilation, less flexible

**Steps**:

1. Load your Lab 1 FPGA bitstream:
   ```bash
   # On f2.6xlarge instance
   fpga-describe-local-image-slots  # Check FPGA status

   # Load your Lab 1 AFI (Amazon FPGA Image)
   sudo fpga-load-local-image -S 0 -I <your_afi_id>

   # Verify
   fpga-describe-local-image-slots
   # Should show "loaded" status
   ```

2. Compile Lab 1 software:
   ```bash
   cd ~/cs217-lab-1-hiva/design_top/software

   # Build testbench (original Lab 1 test)
   make

   # Run test to verify FPGA works
   sudo ./design_top_test 0
   ```

3. Create a C wrapper for Python integration:
   ```bash
   # Create shared library
   cd ~/cs217-lab-1-hiva/design_top/software
   mkdir -p build

   gcc -shared -fPIC \
       -o build/libdesign_top.so \
       src/design_top.c \
       -I$SDK_DIR/userspace/include \
       -L$SDK_DIR/userspace/lib \
       -lfpga_mgmt -lpthread
   ```

4. Update Python interface:
   ```python
   # In integration/lab1_fpga_interface.py
   # Update path to shared library
   lab1_lib_path = Path("~/cs217-lab-1-hiva/design_top/software/build/libdesign_top.so").expanduser()
   ```

### Option 2: Python + ctypes (Current Implementation)

**Pros**: Pure Python, easier to integrate with training
**Cons**: Requires ctypes bindings, AWS SDK initialization tricky

**Status**: `lab1_fpga_interface.py` provides the Python interface, but requires:
1. Compiled shared library (see Option 1)
2. Proper FPGA initialization (bar_handle)
3. Loaded bitstream on FPGA

### Option 3: Software Fallback (Current Default)

**Pros**: Works without FPGA hardware, good for testing
**Cons**: No actual hardware acceleration

**Usage**: Automatically enabled when FPGA library not found.

## Hardware Communication Flow

Based on `design_top.c`, the Lab 1 FPGA communication flow is:

### 1. Initialization
```c
fpga_mgmt_init();
fpga_pci_attach(slot_id, FPGA_APP_PF, APP_PF_BAR0, 0, &bar_handle);
```

### 2. Configure PE (Processing Element)
```c
// Address: 0x400010, Data: 0x0000010100000001
rva_format(true, 0x400010, config_data, rva_msg);
ocl_rva_wr32(bar_handle, rva_msg);
```

### 3. Load Weight Matrix (16 rows)
```c
for (int lane = 0; lane < 16; lane++) {
    uint64_t weight_data[2];  // 128-bit = 16 INT8 weights
    // Pack weights...
    uint32_t addr = 0x500000 + (lane << 4);
    rva_format(true, addr, weight_data, rva_msg);
    ocl_rva_wr32(bar_handle, rva_msg);
}
```

### 4. Configure Manager
```c
// Address: 0x400020, Data: 0x0000000000000100
rva_format(true, 0x400020, manager_data, rva_msg);
ocl_rva_wr32(bar_handle, rva_msg);
```

### 5. Process Each Input Vector (Matrix Column)
```c
for (int col = 0; col < 16; col++) {
    // Write input vector (address 0x600000)
    uint64_t input_data[2];  // 128-bit = 16 INT8 inputs
    rva_format(true, 0x600000, input_data, rva_msg);
    ocl_rva_wr32(bar_handle, rva_msg);

    // Start computation
    ocl_wr32(bar_handle, 0x0404, 0x1);  // START
    usleep(50);
    ocl_wr32(bar_handle, 0x0404, 0x0);  // STOP

    // Read outputs (16 INT32 activations)
    int32_t outputs[16];
    for (int i = 0; i < 16; i++) {
        ocl_rd32(bar_handle, 0x0440 + i*4, &outputs[i]);
    }
}
```

## Performance Counters

Lab 1 FPGA provides cycle counters:

```c
// Start counter
start_data_transfer_counter(bar_handle);

// ... perform operations ...

// Stop counter
stop_data_transfer_counter(bar_handle);

// Read cycle counts
uint32_t transfer_cycles, compute_cycles;
get_data_transfer_cycles(bar_handle, &transfer_cycles);
get_compute_cycles(bar_handle, &compute_cycles);
```

These can be used to measure actual FPGA energy consumption.

## Integration Status

### ✅ Completed
- [x] `lab1_fpga_interface.py` - Python interface with ctypes bindings
- [x] Hardware communication flow implementation
- [x] Performance counter tracking
- [x] Software fallback for testing

### ⏳ To Complete
- [ ] Compile Lab 1 as shared library
- [ ] Load Lab 1 bitstream on f2.6xlarge FPGA
- [ ] Test 16×16 matmul on actual hardware
- [ ] Integrate with RLHF training pipeline
- [ ] Measure energy with real FPGA

## Testing Lab 1 FPGA

### Test 1: Verify FPGA Hardware
```bash
# On f2.6xlarge instance
cd ~/cs217-lab-1-hiva/design_top/software
make
sudo ./design_top_test 0

# Should output:
# TEST PASSED
# Data Transfer Cycles: XXXX
# Compute Cycles: YYYY
```

### Test 2: Test Python Interface
```bash
cd ~/CS217-Final-Project
source venv/bin/activate

python integration/lab1_fpga_interface.py

# Should output:
# Testing Lab 1 FPGA Interface
# ⚠️  Lab 1 library not found (or)
# ✓ Lab 1 FPGA initialized
# Max error: X.XXe-XX
```

### Test 3: Full RLHF Integration
```bash
# Update config to enable FPGA offload
# In baseline_energy/config.py:
# USE_FPGA_OFFLOAD = True
# USE_MOCK_FPGA = False  # Use real FPGA

# Run 2-step test
python baseline_energy/rlhf_with_fpga.py --steps 2 --output results/fpga_test_2steps

# Check logs for FPGA usage
grep "FPGA" results/fpga_test_2steps/training.log
```

## Energy Measurement

With real Lab 1 FPGA:

1. **Hardware Cycles** (from FPGA counters):
   ```python
   stats = fpga.get_stats()
   total_cycles = stats['data_transfer_cycles'] + stats['compute_cycles']
   ```

2. **FPGA Power** (from XPE or measurements):
   ```
   # Option A: Use Xilinx Power Estimator (XPE) output
   fpga_power = XPE_total_power_watts  # From synthesis report

   # Option B: Measure actual power
   # Read from f2.6xlarge power monitoring (if available)
   ```

3. **Energy Calculation**:
   ```python
   # Assuming FPGA clock frequency (e.g., 250 MHz)
   clock_freq_hz = 250e6
   time_seconds = total_cycles / clock_freq_hz
   energy_joules = fpga_power * time_seconds
   ```

## Next Steps

To use actual Lab 1 FPGA hardware:

1. **Load bitstream** on f2.6xlarge:
   ```bash
   sudo fpga-load-local-image -S 0 -I <your_afi_id>
   ```

2. **Verify hardware** works:
   ```bash
   cd ~/cs217-lab-1-hiva/design_top/software
   make && sudo ./design_top_test 0
   ```

3. **Compile shared library**:
   ```bash
   gcc -shared -fPIC -o build/libdesign_top.so src/design_top.c \
       -I$SDK_DIR/userspace/include \
       -L$SDK_DIR/userspace/lib \
       -lfpga_mgmt
   ```

4. **Update config** to enable FPGA:
   ```python
   # baseline_energy/config.py
   USE_FPGA_OFFLOAD = True
   USE_MOCK_FPGA = False
   ```

5. **Run experiments**:
   ```bash
   python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/lab1_fpga_50steps
   ```

## Troubleshooting

### FPGA Not Detected
```bash
# Check FPGA slots
fpga-describe-local-image-slots

# Should show slot 0 with your AFI loaded
# If not, load bitstream first
```

### Permission Denied
```bash
# FPGA access requires sudo
sudo -E python baseline_energy/rlhf_with_fpga.py --steps 2 --output results/test
```

### Library Not Found
```bash
# Verify shared library exists
ls ~/cs217-lab-1-hiva/design_top/software/build/libdesign_top.so

# If not, compile it (see Option 1 above)
```

### Software Fallback Used
This is normal if:
- FPGA library not compiled
- FPGA bitstream not loaded
- Running on instance without FPGA

Check logs for specific reason.

## References

- Lab 1 Code: `/Users/hivamoh/CS217-Project/cs217-lab-1-hiva/`
- RLHF Integration: `integration/lab1_fpga_interface.py`
- Training Script: `baseline_energy/rlhf_with_fpga.py`
- AWS FPGA SDK: https://github.com/aws/aws-fpga
