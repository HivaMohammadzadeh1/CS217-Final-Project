#!/bin/bash
# Backward-compatible wrapper around the repo-local FPGA bridge build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo "Compiling Lab 1 FPGA Python Wrapper"
echo "=============================================="
echo ""

bash "$SCRIPT_DIR/build_fpga_bridge.sh"

echo ""
echo "Compatibility note:"
echo "  The project now builds the repo-local bridge implementation"
echo "  against fpga/design_top/software/src/design_top.h."
