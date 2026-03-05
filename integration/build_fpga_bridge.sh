#!/bin/bash
# Build the FPGA bridge shared library on the AWS F2 instance.
#
# Prerequisites:
#   1. AWS FPGA SDK sourced:  source $AWS_FPGA_REPO_DIR/sdk_setup.sh
#   2. AFI loaded:            fpga-load-local-image -S 0 -I <your-afi-id>
#
# Usage:
#   cd CS217-Final-Project/integration
#   bash build_fpga_bridge.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/fpga_bridge.c"
OUT="$SCRIPT_DIR/libfpga_bridge.so"

# Locate AWS FPGA SDK
if [ -z "${SDK_DIR:-}" ]; then
    SDK_DIR="${AWS_FPGA_REPO_DIR:-$HOME/src/project_data/aws-fpga}/sdk"
fi

SDK_INCLUDE="$SDK_DIR/userspace/include"
SDK_LIBDIR="$SDK_DIR/userspace/lib/so"

if [ ! -d "$SDK_INCLUDE" ]; then
    echo "ERROR: AWS FPGA SDK headers not found at $SDK_INCLUDE"
    echo "       Set SDK_DIR or source sdk_setup.sh first."
    exit 1
fi

echo "Compiling fpga_bridge.c -> libfpga_bridge.so"
echo "  SDK include : $SDK_INCLUDE"
echo "  SDK lib dir : $SDK_LIBDIR"

gcc -shared -fPIC -O2 \
    -o "$OUT" \
    "$SRC" \
    -I"$SDK_INCLUDE" \
    -L"$SDK_LIBDIR" \
    -lfpga_mgmt \
    -Wl,-rpath,"$SDK_LIBDIR"

echo "Built: $OUT"
echo ""
echo "Verify FPGA is ready:"
echo "  fpga-describe-local-image -S 0 -H"
echo ""
echo "Then run the RLHF training:"
echo "  python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/fpga_run"
