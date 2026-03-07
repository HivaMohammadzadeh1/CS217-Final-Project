#!/bin/bash
# Compile Lab 1 FPGA Python Wrapper
# Run this on your FPGA instance (f2.xlarge or larger)

set -e  # Exit on error

echo "=============================================="
echo "Compiling Lab 1 FPGA Python Wrapper"
echo "=============================================="

# Check if we're on an FPGA instance
if ! command -v fpga-describe-local-image-slots &> /dev/null; then
    echo "⚠️  Warning: fpga-describe-local-image-slots not found"
    echo "    This should be run on an AWS F2 instance with FPGA Developer AMI"
    echo "    Continuing anyway..."
fi

# Set SDK directory
if [ -z "$SDK_DIR" ]; then
    SDK_DIR="/home/ubuntu/src/project_data/aws-fpga/sdk"
    echo "Using default SDK_DIR: $SDK_DIR"
fi

if [ ! -d "$SDK_DIR" ]; then
    echo "❌ Error: SDK directory not found at $SDK_DIR"
    echo "   Please set SDK_DIR environment variable or install AWS FPGA SDK"
    exit 1
fi

echo "✓ SDK directory found: $SDK_DIR"

# Navigate to integration directory
cd "$(dirname "$0")"
echo "✓ Working directory: $(pwd)"

# Compile the wrapper
echo ""
echo "Compiling lab1_wrapper.c..."
gcc -shared -fPIC -o liblab1_wrapper.so lab1_wrapper.c \
    -I$SDK_DIR/userspace/include \
    -L$SDK_DIR/userspace/lib \
    -lfpga_mgmt \
    -lrt \
    -Wall -Wextra

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo ""
    echo "Library created: liblab1_wrapper.so"
    ls -lh liblab1_wrapper.so
    echo ""
    echo "=============================================="
    echo "Next steps:"
    echo "1. Make sure your AFI is loaded:"
    echo "   sudo fpga-describe-local-image-slots"
    echo ""
    echo "2. If AFI not loaded, load it:"
    echo "   sudo fpga-load-local-image -S 0 -I agfi-XXXXX"
    echo ""
    echo "3. Run your RLHF pipeline:"
    echo "   python baseline_energy/rlhf_with_fpga.py"
    echo "=============================================="
else
    echo "❌ Compilation failed!"
    exit 1
fi
