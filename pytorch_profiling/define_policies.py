from __future__ import annotations

"""Generate A/B/C/D precision policies from a sensitivity matrix."""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


VALID_GROUP_SIZES = (8, 16)


def load_sensitivity_matrix(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded sensitivity data for {len(df)} layers")
    return df


def tolerant_column(precision: str, group_size: int) -> str:
    return f"{precision.lower()}_g{group_size}_tolerant"


def require_columns(df: pd.DataFrame, group_size: int) -> None:
    required = [tolerant_column("MXFP4", group_size), tolerant_column("MXFP8", group_size), "layer"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Sensitivity matrix is missing required columns for group_size={group_size}: {missing}"
        )


def layer_policy(rollout: str, reward: str, gradient: str) -> Dict[str, str]:
    return {
        "rollout": rollout,
        "reward": reward,
        "gradient": gradient,
    }


def define_policy_a_conservative(df: pd.DataFrame, group_size: int) -> Dict[str, object]:
    policy = {
        "name": "A - Conservative",
        "description": "MXFP8 all layers for rollout/reward, FP16 for gradients",
        "group_size": group_size,
        "layers": {},
    }
    for _, row in df.iterrows():
        policy["layers"][row["layer"]] = layer_policy("MXFP8", "MXFP8", "FP16")
    return policy


def define_policy_b_balanced(df: pd.DataFrame, group_size: int) -> Dict[str, object]:
    fp4_col = tolerant_column("MXFP4", group_size)
    policy = {
        "name": "B - Balanced",
        "description": "MXFP4 for tolerant layers, MXFP8 otherwise, FP16 for gradients",
        "group_size": group_size,
        "layers": {},
    }
    for _, row in df.iterrows():
        preferred = "MXFP4" if bool(row[fp4_col]) else "MXFP8"
        policy["layers"][row["layer"]] = layer_policy(preferred, preferred, "FP16")
    return policy


def define_policy_c_aggressive(df: pd.DataFrame, group_size: int) -> Dict[str, object]:
    fp4_col = tolerant_column("MXFP4", group_size)
    policy = {
        "name": "C - Aggressive",
        "description": "MXFP4 everywhere possible, MXFP8 fallback for sensitive gradients",
        "group_size": group_size,
        "layers": {},
    }
    for _, row in df.iterrows():
        gradient = "MXFP4" if bool(row[fp4_col]) else "MXFP8"
        policy["layers"][row["layer"]] = layer_policy("MXFP4", "MXFP4", gradient)
    return policy


def define_policy_d_phase_adaptive(df: pd.DataFrame, group_size: int) -> Dict[str, object]:
    fp4_col = tolerant_column("MXFP4", group_size)
    fp8_col = tolerant_column("MXFP8", group_size)
    policy = {
        "name": "D - Phase-Adaptive",
        "description": "MXFP4-biased rollout, MXFP8-biased reward, FP16-safe gradients",
        "group_size": group_size,
        "layers": {},
    }
    for _, row in df.iterrows():
        rollout = "MXFP4" if bool(row[fp4_col]) else "MXFP8"
        reward = "MXFP8"
        gradient = "MXFP8" if bool(row[fp8_col]) and bool(row[fp4_col]) else "FP16"
        policy["layers"][row["layer"]] = layer_policy(rollout, reward, gradient)
    return policy


def analyze_policy(policy: Dict[str, object]) -> Dict[str, Dict[str, int]]:
    counts = {
        "rollout": {"MXFP4": 0, "MXFP8": 0, "FP16": 0},
        "reward": {"MXFP4": 0, "MXFP8": 0, "FP16": 0},
        "gradient": {"MXFP4": 0, "MXFP8": 0, "FP16": 0},
    }
    for _, phases in policy["layers"].items():
        for phase, precision in phases.items():
            counts[phase][precision] += 1
    return counts


def print_policy_analysis(policy_id: str, policy: Dict[str, object]) -> None:
    counts = analyze_policy(policy)
    print(f"\n{policy_id}: {policy['name']} (group_size={policy['group_size']})")
    print("-" * 60)
    for phase in ("rollout", "reward", "gradient"):
        total = sum(counts[phase].values())
        summary = "  ".join(
            f"{fmt}:{counts[phase][fmt]:2d} ({(counts[phase][fmt] / total * 100) if total else 0:4.1f}%)"
            for fmt in ("MXFP4", "MXFP8", "FP16")
        )
        print(f"{phase:>8}: {summary}")


def build_policies(df: pd.DataFrame, group_size: int) -> Dict[str, Dict[str, object]]:
    require_columns(df, group_size)
    return {
        "A": define_policy_a_conservative(df, group_size),
        "B": define_policy_b_balanced(df, group_size),
        "C": define_policy_c_aggressive(df, group_size),
        "D": define_policy_d_phase_adaptive(df, group_size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate A/B/C/D policies from sensitivity results.")
    parser.add_argument("--sensitivity", default="results/sensitivity_matrix.csv")
    parser.add_argument("--output", default="results/policies.json")
    parser.add_argument("--group-size", type=int, default=8, choices=VALID_GROUP_SIZES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_sensitivity_matrix(args.sensitivity)
    policies = build_policies(df, args.group_size)

    for policy_id, policy in policies.items():
        print_policy_analysis(policy_id, policy)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(policies, indent=2))
    print(f"\nSaved policies to {output_path}")


if __name__ == "__main__":
    main()
