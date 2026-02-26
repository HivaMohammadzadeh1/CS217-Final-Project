#!/bin/bash
################################################################################
# Compile Lab 1 FPGA as Shared Library for Python Integration
#
# This script compiles your Lab 1 design_top.c as a shared library (.so)
# so that Python can call the FPGA functions via ctypes.
#
# Prerequisites:
# 1. Lab 1 FPGA bitstream loaded: sudo fpga-load-local-image -S 0 -I $AGFI
# 2. Running on f2.6xlarge with AWS FPGA SDK environment
# 3. Lab 1 code at ~/cs217-lab-1-hiva/design_top/
#
# Usage:
#   bash integration/compile_lab1_library.sh
################################################################################

set -e  # Exit on error

echo "========================================"
echo "Compiling Lab 1 as Shared Library"
echo "========================================"
echo ""

# Check environment
if [ -z "$AWS_FPGA_REPO_DIR" ]; then
    echo "❌ AWS_FPGA_REPO_DIR not set"
    echo "   Run: source ~/aws-fpga/sdk_setup.sh"
    exit 1
fi

LAB1_DIR="$HOME/cs217-lab-1-hiva/design_top/software"
if [ ! -d "$LAB1_DIR" ]; then
    echo "❌ Lab 1 directory not found: $LAB1_DIR"
    exit 1
fi

echo "✓ Environment OK"
echo "  AWS_FPGA_REPO_DIR: $AWS_FPGA_REPO_DIR"
echo "  Lab 1 directory: $LAB1_DIR"
echo ""

# Navigate to Lab 1 software directory
cd "$LAB1_DIR"
echo "Working directory: $(pwd)"
echo ""

# Create build directory
mkdir -p build
echo "✓ Build directory created"
echo ""

# Check if design_top.c exists
if [ ! -f "src/design_top.c" ]; then
    echo "❌ design_top.c not found in src/"
    exit 1
fi

echo "Compiling shared library..."
echo ""

# Compile as shared library
# -shared: Create shared library
# -fPIC: Position Independent Code (required for shared libraries)
# -o: Output file
# -I: Include directories
# -L: Library directories
# -l: Link libraries

echo "Running: gcc -shared -fPIC -o build/libdesign_top.so src/design_top.c -I\$AWS_FPGA_REPO_DIR/sdk/userspace/include -L\$AWS_FPGA_REPO_DIR/sdk/userspace/lib -lfpga_mgmt -lpthread -lrt"
echo ""

gcc -shared -fPIC -o build/libdesign_top.so src/design_top.c -I"$AWS_FPGA_REPO_DIR/sdk/userspace/include" -L"$AWS_FPGA_REPO_DIR/sdk/userspace/lib" -lfpga_mgmt -lpthread -lrt 2>&1 | tee build/compile.log

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Compilation Successful!"
    echo "========================================"
    echo ""
    echo "Shared library created:"
    ls -lh build/libdesign_top.so
    echo ""

    # Verify it's a valid shared library
    echo "Library information:"
    file build/libdesign_top.so
    echo ""

    # Show exported symbols
    echo "Exported functions (first 10):"
    nm -D build/libdesign_top.so | grep " T " | head -10
    echo ""

    echo "Next steps:"
    echo "1. Test Python integration:"
    echo "   cd ~/CS217-Final-Project"
    echo "   source venv/bin/activate"
    echo "   python integration/test_lab1_integration.py"
    echo ""
    echo "2. Run RLHF with Lab 1 FPGA:"
    echo "   cp baseline_energy/config_lab1_fpga.py baseline_energy/config.py"
    echo "   python baseline_energy/rlhf_with_fpga.py --steps 2 --output results/fpga_test"
    echo ""
else
    echo ""
    echo "========================================"
    echo "❌ Compilation Failed"
    echo "========================================"
    echo ""
    echo "Check build/compile.log for details"
    exit 1
fi
