#!/bin/bash
# Build the repo-local FPGA bridge shared library on the AWS F2 instance.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$SCRIPT_DIR/lab1_wrapper.c"
OUT="$SCRIPT_DIR/libfpga_bridge.so"
COMPAT_OUT="$SCRIPT_DIR/liblab1_wrapper.so"
DESIGN_TOP_INCLUDE="$REPO_ROOT/fpga/design_top/software/src"

if [ -z "${AWS_FPGA_REPO_DIR:-}" ] || [ ! -f "${AWS_FPGA_REPO_DIR}/sdk_setup.sh" ]; then
    if [ -d "$HOME/aws-fpga" ]; then
        AWS_FPGA_REPO_DIR="$HOME/aws-fpga"
    else
        AWS_FPGA_REPO_DIR="$HOME/src/project_data/aws-fpga"
    fi
fi

if [ -z "${SDK_DIR:-}" ] || [ ! -d "${SDK_DIR}/userspace/include" ]; then
    SDK_DIR="$AWS_FPGA_REPO_DIR/sdk"
fi

SDK_INCLUDE="$SDK_DIR/userspace/include"
SDK_LIBDIR="$SDK_DIR/userspace/lib/so"

if [ ! -f "$SRC" ]; then
    echo "ERROR: bridge source not found at $SRC"
    exit 1
fi

if [ ! -d "$SDK_INCLUDE" ]; then
    echo "ERROR: AWS FPGA SDK headers not found at $SDK_INCLUDE"
    echo "       Source ~/aws-fpga/sdk_setup.sh or set SDK_DIR first."
    exit 1
fi

if [ ! -f "$DESIGN_TOP_INCLUDE/design_top.h" ]; then
    echo "ERROR: design_top.h not found at $DESIGN_TOP_INCLUDE"
    exit 1
fi

echo "Compiling repo-local FPGA bridge"
echo "  source      : $SRC"
echo "  sdk include : $SDK_INCLUDE"
echo "  sdk lib dir : $SDK_LIBDIR"
echo "  design_top  : $DESIGN_TOP_INCLUDE"

gcc -shared -fPIC -O2 \
    -o "$OUT" \
    "$SRC" \
    -I"$SDK_INCLUDE" \
    -I"$DESIGN_TOP_INCLUDE" \
    -L"$SDK_LIBDIR" \
    -lfpga_mgmt \
    -lm \
    -Wl,-rpath,"$SDK_LIBDIR" \
    -Wall -Wextra

cp "$OUT" "$COMPAT_OUT"

echo "Built:"
echo "  $OUT"
echo "  $COMPAT_OUT"
echo ""
echo "Verify the loaded image before using hardware mode:"
echo "  sudo fpga-describe-local-image -S 0 -H"
echo ""
echo "Then test the bridge:"
echo "  python integration/test_fpga_connection.py"
