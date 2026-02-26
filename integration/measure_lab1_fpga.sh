#!/bin/bash
# Measure Lab 1 FPGA Performance
# This script runs the Lab 1 FPGA test multiple times and extracts performance metrics

set -e

echo "========================================"
echo "Lab 1 FPGA Performance Measurement"
echo "========================================"
echo ""

# Check FPGA is loaded
echo "Checking FPGA status..."
sudo fpga-describe-local-image-slots

echo ""
echo "Running Lab 1 FPGA test 10 times to get average performance..."
echo ""

OUTPUT_FILE="lab1_fpga_performance.txt"
cd ~/cs217-lab-1-hiva/design_top/software/runtime

# Run test 10 times
for i in {1..10}; do
    echo "Run $i/10..."
    sudo ./design_top 0 >> $OUTPUT_FILE 2>&1
done

echo ""
echo "Extracting performance metrics..."
echo ""

# Extract cycle counts
echo "=== Data Transfer Cycles ==="
grep "Data Transfer Cycles:" $OUTPUT_FILE

echo ""
echo "=== Compute Cycles ==="
grep "Compute Cycles:" $OUTPUT_FILE

echo ""
echo "=== Average Performance ==="
DATA_CYCLES=$(grep "Data Transfer Cycles:" $OUTPUT_FILE | awk '{sum+=$4; count++} END {print sum/count}')
COMPUTE_CYCLES=$(grep "Compute Cycles:" $OUTPUT_FILE | awk '{sum+=$3; count++} END {print sum/count}')

echo "Average Data Transfer Cycles: $DATA_CYCLES"
echo "Average Compute Cycles: $COMPUTE_CYCLES"
echo "Total Cycles per 16x16 matmul: $(echo "$DATA_CYCLES + $COMPUTE_CYCLES" | bc)"

echo ""
echo "Full results saved to: $OUTPUT_FILE"
echo ""
echo "To calculate energy:"
echo "  Assuming 250 MHz clock: $(echo "scale=6; ($DATA_CYCLES + $COMPUTE_CYCLES) / 250000000" | bc) seconds per matmul"
echo "  With FPGA power (from XPE): Energy = Power × Time"
