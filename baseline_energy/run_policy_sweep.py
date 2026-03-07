"""
Run a reproducible precision-policy sweep for FPGA RLHF experiments.

This script shells out to ``baseline_energy/rlhf_with_fpga.py`` once per
policy so each run gets its own output directory and metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def build_command(args, policy_name: str, output_dir: Path, passthrough_args=None):
    cmd = [
        sys.executable,
        "baseline_energy/rlhf_with_fpga.py",
        "--steps", str(args.steps),
        "--output", str(output_dir),
        "--policy-json", args.policy_json,
        "--policy-name", policy_name,
    ]

    if args.eval_samples is not None:
        cmd.extend(["--eval-samples", str(args.eval_samples)])
    if args.skip_eval:
        cmd.append("--skip-eval")
    if args.save_models:
        cmd.append("--save-models")
    if args.precision_mode is not None:
        cmd.extend(["--precision-mode", args.precision_mode])
    if args.group_size is not None:
        cmd.extend(["--group-size", str(args.group_size)])
    if args.allow_gradient_offload:
        cmd.append("--allow-gradient-offload")
    if passthrough_args:
        cmd.extend(passthrough_args)

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run FPGA precision policy sweep")
    parser.add_argument(
        "--policy-json",
        type=str,
        required=True,
        help="Path to JSON created by pytorch_profiling/define_policies.py",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="A,B,C,D",
        help="Comma-separated policy names to run",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="PPO steps per policy run",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=20,
        help="Held-out evaluation samples per run",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/policy_sweep",
        help="Root directory for sweep outputs",
    )
    parser.add_argument(
        "--precision-mode",
        type=str,
        default=None,
        help="Fallback precision if a policy does not specify a layer",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Default MX group size override",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        help="Skip post-training evaluation for each run",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        default=False,
        help="Save full model checkpoints for each run",
    )
    parser.add_argument(
        "--allow-gradient-offload",
        action="store_true",
        default=False,
        help="Allow gradient-phase offload in the controller.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        default=False,
        help="Continue running remaining policies even if one fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print commands without executing them",
    )
    args, passthrough_args = parser.parse_known_args()

    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "policy_json": args.policy_json,
        "policies": policies,
        "steps": args.steps,
        "eval_samples": args.eval_samples,
        "runs": [],
    }

    failed = False
    for policy_name in policies:
        output_dir = output_root / f"policy_{policy_name}"
        cmd = build_command(args, policy_name, output_dir, passthrough_args=passthrough_args)
        manifest["runs"].append({
            "policy": policy_name,
            "output_dir": str(output_dir),
            "command": cmd,
        })

        print(f"\n=== Policy {policy_name} ===")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failed = True
            print(f"Policy {policy_name} failed with exit code {result.returncode}")
            if not args.keep_going:
                break

    if not args.dry_run:
        manifest_path = output_root / "sweep_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\nWrote sweep manifest to {manifest_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
