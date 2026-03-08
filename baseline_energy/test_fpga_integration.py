"""
Milestone 4 smoke test for the FPGA integration path.

This script has two layers:
1. A quick local control-path check for tiled matmul + precision switching.
2. An optional end-to-end subprocess run of ``rlhf_with_fpga.py``.

Examples:
    .venv/bin/python3 baseline_energy/test_fpga_integration.py
    .venv/bin/python3 baseline_energy/test_fpga_integration.py --run-end-to-end --steps 2
    .venv/bin/python3 baseline_energy/test_fpga_integration.py --run-end-to-end --use-real-fpga --policy-json results/policies.json --policy-name D
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_OUTPUTS = (
    "training_stats.json",
    "phase_timing.json",
    "fpga_stats.json",
    "training_meta.json",
)


def append_optional_flag(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def build_end_to_end_command(args):
    cmd = [
        args.python,
        "baseline_energy/rlhf_with_fpga.py",
        "--steps", str(args.steps),
        "--output", args.output,
    ]

    if not args.use_real_fpga:
        cmd.append("--use-mock-fpga")
    if not args.run_eval:
        cmd.append("--skip-eval")
    if args.allow_gradient_offload:
        cmd.append("--allow-gradient-offload")

    append_optional_flag(cmd, "--eval-samples", args.eval_samples if args.run_eval else None)
    append_optional_flag(cmd, "--policy-json", args.policy_json)
    append_optional_flag(cmd, "--policy-name", args.policy_name)
    append_optional_flag(cmd, "--precision-mode", args.precision_mode)
    append_optional_flag(cmd, "--group-size", args.group_size)
    append_optional_flag(cmd, "--model-name", args.model_name)
    append_optional_flag(cmd, "--reward-model-name", args.reward_model_name)
    append_optional_flag(cmd, "--dataset-name", args.dataset_name)
    append_optional_flag(cmd, "--local-dataset-path", args.local_dataset_path)
    append_optional_flag(cmd, "--num-samples", args.num_samples)
    append_optional_flag(cmd, "--train-size", args.train_size)
    append_optional_flag(cmd, "--eval-size", args.eval_size)
    append_optional_flag(cmd, "--batch-size", args.batch_size)
    append_optional_flag(cmd, "--mini-batch-size", args.mini_batch_size)
    append_optional_flag(cmd, "--gradient-accumulation-steps", args.gradient_accumulation_steps)
    append_optional_flag(cmd, "--max-seq-length", args.max_seq_length)
    append_optional_flag(cmd, "--max-prompt-length", args.max_prompt_length)
    append_optional_flag(cmd, "--max-response-length", args.max_response_length)
    append_optional_flag(cmd, "--policy-blocks", args.policy_blocks)
    append_optional_flag(cmd, "--reward-policy-blocks", args.reward_policy_blocks)
    append_optional_flag(cmd, "--fpga-response-length", args.fpga_response_length)
    append_optional_flag(cmd, "--pretrain-reward-steps", args.pretrain_reward_steps)

    return cmd


def quick_offload_check(args):
    sys.path.insert(0, str(REPO_ROOT / "integration"))
    from fpga_matmul_offload import FPGAMatmulOffload

    print("=" * 60, flush=True)
    print("Milestone 4 Quick Offload Check", flush=True)
    print("=" * 60, flush=True)

    offloader = FPGAMatmulOffload(
        use_mock=not args.use_real_fpga,
        use_lab1=True,
        verbose=False,
        precision_mode="INT8",
        group_size=args.group_size or 8,
    )

    a = torch.randn(32, 32, dtype=torch.float32)
    b = torch.randn(32, 32, dtype=torch.float32)

    int8_result = offloader.matmul(a, b)
    int8_ref = torch.matmul(a, b)
    int8_error = torch.max(torch.abs(int8_result - int8_ref)).item()

    offloader.configure_precision("MXFP8", group_size=args.group_size or 8, flush=True)
    mxfp8_result = offloader.matmul(a, b)
    stats = offloader.get_stats()

    print(f"INT8 max error: {int8_error:.2e}", flush=True)
    print(f"Active precision after switch: {stats.get('precision_mode')}", flush=True)
    print(f"Tiles processed: {stats.get('total_tiles')}", flush=True)
    if "using_hardware" in stats:
        print(f"Using hardware: {stats.get('using_hardware')}", flush=True)

    if not torch.isfinite(mxfp8_result).all():
        raise RuntimeError("MXFP8 smoke result contains non-finite values.")
    if int8_error > 1e-4:
        raise RuntimeError(f"INT8 smoke error too large: {int8_error:.2e}")

    print("Quick offload check passed.\n", flush=True)


def verify_end_to_end_outputs(output_dir: Path):
    missing = [name for name in EXPECTED_OUTPUTS if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"End-to-end run completed but missing outputs: {missing}")

    phase_timing = json.loads((output_dir / "phase_timing.json").read_text())
    fpga_stats = json.loads((output_dir / "fpga_stats.json").read_text())

    print("Generated outputs:", flush=True)
    for name in EXPECTED_OUTPUTS:
        print(f"  - {output_dir / name}", flush=True)

    print("\nPhase summary:", flush=True)
    for phase in ("rollout", "reward", "gradient"):
        phase_info = phase_timing[phase]
        phase_fpga = phase_info.get("fpga", {})
        print(
            f"  {phase:8s} time={phase_info['total_time_s']:.2f}s "
            f"tiles={phase_fpga.get('tile_ops', 0)} "
            f"matmuls={phase_fpga.get('matmul_calls', 0)}",
            flush=True,
        )

    print("\nFPGA totals:", flush=True)
    print(f"  matmuls={fpga_stats.get('total_matmuls', 0)}", flush=True)
    print(f"  tiles={fpga_stats.get('total_tiles', 0)}", flush=True)


def run_end_to_end_smoke(args):
    output_dir = REPO_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_end_to_end_command(args)
    print("=" * 60, flush=True)
    print("Milestone 4 End-to-End Smoke Test", flush=True)
    print("=" * 60, flush=True)
    print("Command:", flush=True)
    print(" ".join(cmd), flush=True)
    print("", flush=True)

    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        timeout=args.timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"End-to-end smoke test failed with exit code {result.returncode}.")

    verify_end_to_end_outputs(output_dir)
    print("\nEnd-to-end smoke test passed.\n", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Milestone 4 FPGA integration smoke test")
    parser.add_argument("--python", default=sys.executable, help="Python executable for subprocess runs.")
    parser.add_argument("--run-end-to-end", action="store_true", default=False, help="Run the RLHF smoke subprocess after the quick offload check.")
    parser.add_argument("--run-eval", action="store_true", default=False, help="Keep evaluation enabled during the end-to-end run.")
    parser.add_argument("--use-real-fpga", action="store_true", default=False, help="Use the Lab 1 runtime path instead of the mock FPGA path.")
    parser.add_argument("--steps", type=int, default=2, help="PPO steps for the end-to-end smoke run.")
    parser.add_argument("--eval-samples", type=int, default=4, help="Held-out evaluation samples when --run-eval is enabled.")
    parser.add_argument("--output", default="results/milestone4_smoke", help="Output directory for the end-to-end smoke run.")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout in seconds for the end-to-end subprocess.")
    parser.add_argument("--policy-json", default=None, help="Optional policy JSON path.")
    parser.add_argument("--policy-name", default=None, help="Optional policy name (A/B/C/D).")
    parser.add_argument("--precision-mode", default=None, help="Optional fallback precision override.")
    parser.add_argument("--group-size", type=int, default=None, help="Optional MX group-size override.")
    parser.add_argument("--allow-gradient-offload", action="store_true", default=False, help="Allow gradient-phase offload during the smoke run.")
    parser.add_argument("--model-name", default=None, help="Optional model override.")
    parser.add_argument("--reward-model-name", default=None, help="Optional reward-model override.")
    parser.add_argument("--dataset-name", default=None, help="Optional dataset override.")
    parser.add_argument("--local-dataset-path", default=None, help="Optional local dataset path.")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional dataset sample-count override.")
    parser.add_argument("--train-size", type=int, default=None, help="Optional train-size override.")
    parser.add_argument("--eval-size", type=int, default=None, help="Optional eval-size override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional PPO batch-size override.")
    parser.add_argument("--mini-batch-size", type=int, default=None, help="Optional PPO mini-batch-size override.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, help="Optional PPO gradient-accumulation override.")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Optional max sequence-length override.")
    parser.add_argument("--max-prompt-length", type=int, default=None, help="Optional max prompt-length override.")
    parser.add_argument("--max-response-length", type=int, default=None, help="Optional max response-length override.")
    parser.add_argument("--policy-blocks", default=None, help="Optional comma-separated policy/reference block list.")
    parser.add_argument("--reward-policy-blocks", default=None, help="Optional comma-separated reward-model block list.")
    parser.add_argument("--fpga-response-length", type=int, default=None, help="Optional FPGA response-length override.")
    parser.add_argument("--pretrain-reward-steps", type=int, default=None, help="Optional reward-head pretraining-step override.")
    return parser.parse_args()


def main():
    args = parse_args()
    quick_offload_check(args)
    if args.run_end_to_end:
        run_end_to_end_smoke(args)


if __name__ == "__main__":
    main()
