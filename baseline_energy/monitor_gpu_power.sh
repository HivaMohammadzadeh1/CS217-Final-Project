#!/bin/bash

# GPU Power Monitoring Script
# Logs GPU power consumption using nvidia-smi
# Usage: ./monitor_gpu_power.sh [output_file] [interval_ms]

OUTPUT_FILE=${1:-power_log.csv}
INTERVAL_MS=${2:-100}  # 100ms default (10 samples/second)

echo "🔋 Starting GPU power monitoring..."
echo "   Output file: $OUTPUT_FILE"
echo "   Sampling interval: ${INTERVAL_MS}ms"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found"
    echo "   This script requires an NVIDIA GPU"
    exit 1
fi

# Check if GPU is available
if ! nvidia-smi &> /dev/null; then
    echo "❌ Error: No NVIDIA GPU detected"
    exit 1
fi

# Get GPU name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "✓ GPU detected: $GPU_NAME"
echo ""

# Start monitoring
# Format: timestamp, power (W), utilization (%), temperature (C)
nvidia-smi dmon -s u -d $((INTERVAL_MS / 1000)) -c 100000 | while IFS= read -r line; do
    echo "$line"
    echo "$line" >> "$OUTPUT_FILE"
done

echo ""
echo "✓ Monitoring stopped. Data saved to $OUTPUT_FILE"
