#!/bin/bash
################################################################################
# Compile the repo-local FPGA bridge and keep a legacy libdesign_top.so copy.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "========================================"
echo "Compiling FPGA bridge as Shared Library"
echo "========================================"
echo ""

bash "$SCRIPT_DIR/build_fpga_bridge.sh"

mkdir -p "$BUILD_DIR"
cp "$SCRIPT_DIR/libfpga_bridge.so" "$BUILD_DIR/libdesign_top.so"

echo ""
echo "Compatibility artifact created:"
echo "  $BUILD_DIR/libdesign_top.so"
echo ""
file "$BUILD_DIR/libdesign_top.so"
