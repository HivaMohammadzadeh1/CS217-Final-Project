#!/usr/bin/env python3
"""
Batch runner for SystemC MX datapath experiments.

Builds the local testbench once, then sweeps seeds and workload profiles.
Outputs a CSV with pass/fail status and parsed quality metrics.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
BINARY = ROOT / "mx_datapath_tb"
RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run many SystemC MX experiments.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="First RNG seed in the sweep.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=8,
        help="How many consecutive seeds to run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use lighter trial counts for faster turnaround.",
    )
    return parser.parse_args()


def build_binary() -> None:
    result = subprocess.run(["make", "clean", "all"], cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError("Build failed; aborting experiment sweep.")
    if not BINARY.exists():
        raise RuntimeError("Expected binary mx_datapath_tb was not produced.")


def profiles(quick: bool) -> List[Tuple[str, int, int, int]]:
    if quick:
        return [
            ("smoke", 120, 8, 80),
            ("quality", 240, 16, 160),
        ]
    return [
        ("baseline", 300, 20, 200),
        ("stress", 1200, 80, 600),
    ]


def parse_metrics(stdout: str) -> Dict[str, str]:
    patterns = {
        "fp8_mac_mean": r"FP8 normalized error mean/max:\s+([0-9.eE+-]+)\s*/",
        "fp4_mac_mean": r"FP4 normalized error mean/max:\s+([0-9.eE+-]+)\s*/",
        "fp8_gemm_mean": r"FP8 GEMM abs error mean/max:\s+([0-9.eE+-]+)\s*/",
        "fp4_gemm_mean": r"FP4 GEMM abs error mean/max:\s+([0-9.eE+-]+)\s*/",
        "group8_mean": r"FP8 mean normalized error:\s+group8=([0-9.eE+-]+)",
        "group16_mean": r"group16=([0-9.eE+-]+)",
    }
    out: Dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        out[key] = match.group(1) if match else ""
    return out


def main() -> int:
    args = parse_args()
    seeds = [args.seed_start + i for i in range(args.seed_count)]
    run_profiles = profiles(args.quick)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"systemc_experiments_{timestamp}.csv"

    print(f"[1/3] Building testbench in {ROOT}")
    build_binary()

    rows: List[Dict[str, str]] = []
    total_runs = len(seeds) * len(run_profiles)
    run_idx = 0

    print(f"[2/3] Running {total_runs} simulations")
    for seed in seeds:
        for profile_name, mac_trials, gemm_trials, group_trials in run_profiles:
            run_idx += 1
            cmd = [
                str(BINARY),
                "--seed",
                str(seed),
                "--mac-trials",
                str(mac_trials),
                "--gemm-trials",
                str(gemm_trials),
                "--group-trials",
                str(group_trials),
            ]
            print(f"  [{run_idx}/{total_runs}] seed={seed} profile={profile_name}")
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            metrics = parse_metrics(result.stdout)
            passed = "ALL CHECKS PASSED" in result.stdout and result.returncode == 0
            rows.append(
                {
                    "seed": str(seed),
                    "profile": profile_name,
                    "mac_trials": str(mac_trials),
                    "gemm_trials": str(gemm_trials),
                    "group_trials": str(group_trials),
                    "exit_code": str(result.returncode),
                    "pass": "1" if passed else "0",
                    **metrics,
                }
            )

    print(f"[3/3] Writing results to {csv_path}")
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    num_pass = sum(1 for row in rows if row["pass"] == "1")
    print(f"Completed {len(rows)} runs: {num_pass} passed, {len(rows) - num_pass} failed.")
    print(f"CSV: {csv_path}")
    return 0 if num_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
