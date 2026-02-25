#!/bin/bash

# CS217 Final Project - Environment Setup Script
# Run this script to set up your Python environment

set -e  # Exit on error

echo "🚀 Setting up CS217 Final Project Environment"
echo "=============================================="

# Check if Python 3.8+ is available
python3 --version || { echo "Error: Python 3 not found"; exit 1; }

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "    source venv/bin/activate"
echo ""
echo "Quick start commands:"
echo "    python baseline_energy/verify_setup.py    # Test installation"
echo "    python baseline_energy/test_model.py      # Test model loading"
