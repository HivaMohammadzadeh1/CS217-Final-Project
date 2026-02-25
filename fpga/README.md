# FPGA Synthesis and Deployment

This directory contains FPGA synthesis scripts and bitstream deployment files for AWS F2.

## Purpose

Synthesize the MX datapath design and deploy to AWS F2 FPGA (Xilinx VU9P).

## Target Platform

- **Device**: Xilinx VU9P (AWS F2 instance)
- **Toolchain**: Vivado HLS 2021.1+
- **Interface**: AXI4-Lite for control, AXI4 for data

## Files (to be added)

- `synthesis_script.tcl`: Vivado synthesis script
- `constraints.xdc`: Timing and pin constraints
- `deploy_to_f2.sh`: AWS F2 deployment script
- `fpga_driver.py`: Host-side Python driver for FPGA
- `power_measurement.py`: XPE-based power estimation

## Build Flow

```bash
# 1. Synthesize design
vivado -mode batch -source synthesis_script.tcl

# 2. Check timing
grep "WNS" vivado.log  # Should be positive (no violations)

# 3. Generate bitstream
# (Included in synthesis_script.tcl)

# 4. Deploy to AWS F2
./deploy_to_f2.sh
```

## Power Measurement

```bash
# Run Xilinx Power Estimator
vivado -mode batch -source run_xpe.tcl

# Extract power estimate
python power_measurement.py --report xpe_report.xml --output ../results/fpga_power.csv
```

## Host-FPGA Interface

```python
from fpga_driver import FPGAController

# Initialize FPGA
fpga = FPGAController(device_id=0)

# Set precision mode
fpga.set_mode('MXFP4')  # or 'MXFP8'

# Load quantized weights
fpga.load_weights(weights_fp4)

# Run inference
outputs = fpga.matmul(inputs)
```

## Expected Performance

| Metric | Target |
|--------|--------|
| Frequency | 200-250 MHz |
| Throughput | ~50 GOPS (MXFP8) / ~80 GOPS (MXFP4) |
| Power | <20W total on-chip |
| Latency | <10ms per layer (batch=1) |
