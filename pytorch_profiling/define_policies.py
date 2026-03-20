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


def define_policy_a_mxfp4_all(df: pd.DataFrame, group_size: int) -> Dict[str, object]:
    fp4_col = tolerant_column("MXFP4", group_size)
    policy = {
        "name": "A - MXFP4 All Phases",
        "description": "MXFP4 for all phases on tolerant layers, MXFP8 fallback for sensitive layers (>2% perplexity delta)",
        "group_size": group_size,
        "layers": {},
    }
    for _, row in df.iterrows():
        precision = "MXFP4" if bool(row[fp4_col]) else "MXFP8"
        policy["layers"][row["layer"]] = layer_policy(precision, precision, precision)
    return policy


def define_policy_b_mxfp8_all(df: pd.DataFrame, group_size: int) -> Dict[str, object]:
    policy = {
        "name": "B - MXFP8 All Phases",
        "description": "MXFP8 for all layers across all phases",
        "group_size": group_size,
        "layers": {},
    }
    for _, row in df.iterrows():
        policy["layers"][row["layer"]] = layer_policy("MXFP8", "MXFP8", "MXFP8")
    return policy


def define_policy_c_mxfp8_int8_gradient(df: pd.DataFrame, group_size: int) -> Dict[str, object]:
    policy = {
        "name": "C - MXFP8 Inference / INT8 Gradient",
        "description": "Rollout and reward use MXFP8, gradient uses INT8",
        "group_size": group_size,
        "layers": {},
    }
    for _, row in df.iterrows():
        policy["layers"][row["layer"]] = layer_policy("MXFP8", "MXFP8", "INT8")
    return policy


def define_policy_d_mxfp4_int8_gradient(df: pd.DataFrame, group_size: int) -> Dict[str, object]:
    fp4_col = tolerant_column("MXFP4", group_size)
    policy = {
        "name": "D - MXFP4 Inference / INT8 Gradient",
        "description": "Rollout and reward use MXFP4 (MXFP8 for sensitive layers), gradient uses INT8",
        "group_size": group_size,
        "layers": {},
    }
    for _, row in df.iterrows():
        inference = "MXFP4" if bool(row[fp4_col]) else "MXFP8"
        policy["layers"][row["layer"]] = layer_policy(inference, inference, "INT8")
    return policy


def define_policy_e_mxfp4_mxfp8_gradient(df: pd.DataFrame, group_size: int) -> Dict[str, object]:
    fp4_col = tolerant_column("MXFP4", group_size)
    policy = {
        "name": "E - MXFP4 Inference / MXFP8 Gradient",
        "description": "Rollout and reward use MXFP4 (MXFP8 for sensitive layers), gradient uses MXFP8",
        "group_size": group_size,
        "layers": {},
    }
    for _, row in df.iterrows():
        inference = "MXFP4" if bool(row[fp4_col]) else "MXFP8"
        policy["layers"][row["layer"]] = layer_policy(inference, inference, "MXFP8")
    return policy


def analyze_policy(policy: Dict[str, object]) -> Dict[str, Dict[str, int]]:
    counts = {
        "rollout": {"INT8": 0, "MXFP4": 0, "MXFP8": 0, "FP16": 0},
        "reward": {"INT8": 0, "MXFP4": 0, "MXFP8": 0, "FP16": 0},
        "gradient": {"INT8": 0, "MXFP4": 0, "MXFP8": 0, "FP16": 0},
    }
    for _, phases in policy["layers"].items():
        for phase, precision in phases.items():
            counts[phase][precision] = counts[phase].get(precision, 0) + 1
    return counts


def print_policy_analysis(policy_id: str, policy: Dict[str, object]) -> None:
    counts = analyze_policy(policy)
    print(f"\n{policy_id}: {policy['name']} (group_size={policy['group_size']})")
    print("-" * 60)
    for phase in ("rollout", "reward", "gradient"):
        total = sum(counts[phase].values())
        summary = "  ".join(
            f"{fmt}:{counts[phase][fmt]:2d} ({(counts[phase][fmt] / total * 100) if total else 0:4.1f}%)"
            for fmt in ("INT8", "MXFP4", "MXFP8", "FP16")
        )
        print(f"{phase:>8}: {summary}")


def build_policies(df: pd.DataFrame, group_size: int) -> Dict[str, Dict[str, object]]:
    require_columns(df, group_size)
    return {
        "A": define_policy_a_mxfp4_all(df, group_size),
        "B": define_policy_b_mxfp8_all(df, group_size),
        "C": define_policy_c_mxfp8_int8_gradient(df, group_size),
        "D": define_policy_d_mxfp4_int8_gradient(df, group_size),
        "E": define_policy_e_mxfp4_mxfp8_gradient(df, group_size),
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
