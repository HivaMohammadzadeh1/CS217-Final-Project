"""
Verify that all required packages are installed and working.
Quick test to ensure environment is set up correctly.
"""

import sys

def check_import(package_name, import_name=None):
    """Try to import a package and report success/failure."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name:20s} installed")
        return True
    except ImportError as e:
        print(f"✗ {package_name:20s} NOT installed - {e}")
        return False

def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n✓ GPU Available: {gpu_name}")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("\n⚠ GPU NOT available - will run on CPU (slower)")
            return False
    except:
        return False

def main():
    print("=" * 60)
    print("CS217 Final Project - Environment Verification")
    print("=" * 60)
    print("\nChecking required packages...\n")

    all_ok = True

    # Core ML packages
    all_ok &= check_import("torch")
    all_ok &= check_import("transformers")
    all_ok &= check_import("datasets")
    all_ok &= check_import("accelerate")
    all_ok &= check_import("trl")

    # MX format library
    # Note: mx-pytorch might not install easily, we'll check it separately
    mx_ok = check_import("mx (PyTorch MX)", "mx")
    if not mx_ok:
        print("  ℹ️  mx-pytorch is optional for Week 2 - needed later for profiling")

    # Utilities
    all_ok &= check_import("tqdm")
    all_ok &= check_import("pandas")
    all_ok &= check_import("matplotlib")
    all_ok &= check_import("numpy")

    # Check GPU
    check_gpu()

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All core packages installed successfully!")
        print("   You're ready to start Milestone 2!")
    else:
        print("⚠️  Some packages are missing. Run:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
