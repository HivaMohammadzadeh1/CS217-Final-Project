#!/usr/bin/env python3
"""
Single entry point for the project's FPGA build and deployment path.

This script does two things:
1. Sets the environment variables expected by the Stanford/AWS FPGA tooling.
2. Gives one stable command surface for local reference simulation, HLS/RTL
   build steps, and F2 deployment/runtime steps.

Current repo split:
- `systemc/` is the portable MX reference model.
- `fpga/` is the hardware build/deploy path that targets AWS F2 from the
  Stanford development environment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
FPGA_ROOT = Path(__file__).resolve().parent
DESIGN_TOP = FPGA_ROOT / "design_top"


def build_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("REPO_TOP", str(FPGA_ROOT))
    env.setdefault("AWS_HOME", str(DESIGN_TOP))
    env.setdefault("CL_DIR", str(DESIGN_TOP))
    env.setdefault("CL_DESIGN_NAME", "design_top")
    return env


def tool_status() -> List[Tuple[str, str]]:
    checks = [
        ("python3", shutil.which("python3")),
        ("make", shutil.which("make")),
        ("aws", shutil.which("aws")),
        ("vivado_hls", shutil.which("vivado_hls")),
        ("vitis_hls", shutil.which("vitis_hls")),
        ("catapult", shutil.which("catapult")),
        ("fpga-load-local-image", shutil.which("fpga-load-local-image")),
        ("fpga-clear-local-image", shutil.which("fpga-clear-local-image")),
    ]
    return [(name, path or "missing") for name, path in checks]


def render_doctor(env: Dict[str, str]) -> int:
    print("FPGA build environment")
    print(f"  repo root:   {ROOT}")
    print(f"  fpga root:   {FPGA_ROOT}")
    print(f"  systemc ref: {ROOT / 'systemc'}")
    print(f"  design_top:  {DESIGN_TOP}")
    print("")
    print("Resolved environment")
    for key in ("REPO_TOP", "AWS_HOME", "CL_DIR", "CL_DESIGN_NAME", "AWS_FPGA_REPO_DIR"):
        print(f"  {key:<16} {env.get(key, '<unset>')}")
    print("")
    print("Tool availability")
    for name, status in tool_status():
        print(f"  {name:<22} {status}")
    print("")
    print("Execution path")
    print("  local machine:             python fpga/run_fpga_flow.py reference-sim")
    print("  Stanford build machine:    python fpga/run_fpga_flow.py systemc-sim / hls-sim / hw-sim / fpga-build / generate-afi")
    print("  F2 runtime host:           python fpga/run_fpga_flow.py program-fpga / run-fpga-test")
    return 0


def make_command(directory: Path, target: str, variables: Dict[str, str] | None = None) -> List[str]:
    cmd = ["make", "-C", str(directory), target]
    for key, value in sorted((variables or {}).items()):
        cmd.append(f"{key}={value}")
    return cmd


def dispatch_command(action: str, args: argparse.Namespace) -> Tuple[List[str], Path]:
    if action == "reference-sim":
        return ["make", "-C", str(ROOT / "systemc"), "clean", "all", "run"], ROOT
    if action == "systemc-sim":
        return make_command(FPGA_ROOT, "systemc_sim"), ROOT
    if action == "hls-sim":
        return make_command(FPGA_ROOT, "hls_sim"), ROOT
    if action == "hw-sim":
        return make_command(DESIGN_TOP, "hw_sim"), ROOT
    if action == "fpga-build":
        variables = {"RTL_VARIANT": args.rtl_variant} if args.rtl_variant else {}
        return make_command(DESIGN_TOP, "fpga_build", variables), ROOT
    if action == "generate-afi":
        return make_command(DESIGN_TOP, "generate_afi"), ROOT
    if action == "check-afi":
        return make_command(DESIGN_TOP, "check_afi_available"), ROOT
    if action == "program-fpga":
        return make_command(DESIGN_TOP, "program_fpga"), ROOT
    if action == "run-fpga-test":
        variables = {"SLOT_ID": str(args.slot_id)}
        if args.fpga_test_args:
            variables["FPGA_TEST_ARGS"] = args.fpga_test_args
        return make_command(DESIGN_TOP, "run_fpga_test", variables), ROOT
    raise ValueError(f"Unsupported action: {action}")


def run_command(command: Sequence[str], cwd: Path, env: Dict[str, str], dry_run: bool) -> int:
    pretty = " ".join(command)
    print(pretty, flush=True)
    if dry_run:
        return 0
    result = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the project's FPGA build/deploy path.")
    parser.add_argument(
        "action",
        choices=[
            "doctor",
            "reference-sim",
            "systemc-sim",
            "hls-sim",
            "hw-sim",
            "fpga-build",
            "generate-afi",
            "check-afi",
            "program-fpga",
            "run-fpga-test",
        ],
        help="Flow stage to run.",
    )
    parser.add_argument(
        "--rtl-variant",
        default="kIntWordWidth_8_kVectorSize_16_kNumVectorLanes_16",
        help="RTL variant folder under fpga/design_top/design/concat_PECore/.",
    )
    parser.add_argument(
        "--slot-id",
        type=int,
        default=0,
        help="AWS FPGA slot for run-fpga-test.",
    )
    parser.add_argument(
        "--fpga-test-args",
        default="",
        help="Extra argv passed through to the runtime test binary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = build_env()

    if args.action == "doctor":
        return render_doctor(env)

    command, cwd = dispatch_command(args.action, args)
    return run_command(command, cwd, env, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
