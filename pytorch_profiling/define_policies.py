"""
Define MX Format Policies based on sensitivity profiling results.

Reads sensitivity matrix and defines 4 policies (A/B/C/D) that specify
which format to use for each layer in each RLHF phase.

Usage:
    python pytorch_profiling/define_policies.py --sensitivity results/sensitivity_matrix.csv
"""

import pandas as pd
import json
import argparse
from pathlib import Path


def load_sensitivity_matrix(csv_path):
    """Load sensitivity profiling results."""
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded sensitivity data for {len(df)} layers\n")
    return df


def define_policy_a_conservative(df):
    """
    Policy A: Conservative
    - MXFP8 for rollout and reward
    - FP16 for gradient updates
    - Minimal risk, modest energy savings
    """
    policy = {
        "name": "A - Conservative",
        "description": "MXFP8 all layers for rollout/reward, FP16 for gradients",
        "layers": {}
    }

    for _, row in df.iterrows():
        layer_name = row['layer']
        policy["layers"][layer_name] = {
            "rollout": "MXFP8",
            "reward": "MXFP8",
            "gradient": "FP16"
        }

    return policy


def define_policy_b_balanced(df):
    """
    Policy B: Balanced
    - MXFP4 for tolerant layers, MXFP8 for sensitive
    - Applies to rollout and reward phases
    - FP16 for gradient updates
    """
    policy = {
        "name": "B - Balanced",
        "description": "Layer-adaptive based on sensitivity, FP16 for gradients",
        "layers": {}
    }

    for _, row in df.iterrows():
        layer_name = row['layer']
        # Use MXFP4 if tolerant, else MXFP8
        rollout_format = "MXFP4" if row['mxfp4_tolerant'] else "MXFP8"
        reward_format = "MXFP4" if row['mxfp4_tolerant'] else "MXFP8"

        policy["layers"][layer_name] = {
            "rollout": rollout_format,
            "reward": reward_format,
            "gradient": "FP16"
        }

    return policy


def define_policy_c_aggressive(df):
    """
    Policy C: Aggressive
    - MXFP4 for all layers in rollout and reward
    - MXFP8 for sensitive layers in gradients, MXFP4 for tolerant
    - Maximum energy savings, possible quality degradation
    """
    policy = {
        "name": "C - Aggressive",
        "description": "MXFP4 everywhere, MXFP8 for sensitive gradient layers",
        "layers": {}
    }

    for _, row in df.iterrows():
        layer_name = row['layer']
        # MXFP4 for rollout/reward everywhere
        # Gradients: MXFP8 if sensitive to MXFP4, else MXFP4
        gradient_format = "MXFP8" if not row['mxfp4_tolerant'] else "MXFP4"

        policy["layers"][layer_name] = {
            "rollout": "MXFP4",
            "reward": "MXFP4",
            "gradient": gradient_format
        }

    return policy


def define_policy_d_phase_adaptive(df):
    """
    Policy D: Phase-Adaptive (Research Target)
    - MXFP4 for most layers in rollout (high tolerance)
    - MXFP8 for most layers in reward (moderate tolerance)
    - FP16 for most layers in gradient (low tolerance)
    - Adaptive within each phase based on sensitivity
    """
    policy = {
        "name": "D - Phase-Adaptive",
        "description": "Phase-aware precision selection based on sensitivity",
        "layers": {}
    }

    for _, row in df.iterrows():
        layer_name = row['layer']

        # Rollout: prefer MXFP4 (stochastic, high tolerance)
        rollout_format = "MXFP4" if row['mxfp4_tolerant'] else "MXFP8"

        # Reward: prefer MXFP8 (scoring, moderate tolerance)
        reward_format = "MXFP8"

        # Gradient: prefer FP16 (precision matters for convergence)
        # Only use MXFP8 for very tolerant layers
        gradient_format = "MXFP8" if row['mxfp8_tolerant'] and row['mxfp4_tolerant'] else "FP16"

        policy["layers"][layer_name] = {
            "rollout": rollout_format,
            "reward": reward_format,
            "gradient": gradient_format
        }

    return policy


def analyze_policy(policy, df):
    """Analyze a policy's format distribution."""
    analysis = {
        "name": policy["name"],
        "rollout": {"MXFP4": 0, "MXFP8": 0, "FP16": 0},
        "reward": {"MXFP4": 0, "MXFP8": 0, "FP16": 0},
        "gradient": {"MXFP4": 0, "MXFP8": 0, "FP16": 0},
    }

    for layer_name, formats in policy["layers"].items():
        for phase in ["rollout", "reward", "gradient"]:
            fmt = formats[phase]
            analysis[phase][fmt] += 1

    return analysis


def print_policy_analysis(analysis):
    """Print policy analysis."""
    print(f"\n{analysis['name']}:")
    print("-" * 60)

    for phase in ["rollout", "reward", "gradient"]:
        total = sum(analysis[phase].values())
        print(f"  {phase.capitalize():15s}: ", end="")

        for fmt in ["MXFP4", "MXFP8", "FP16"]:
            count = analysis[phase][fmt]
            pct = (count / total * 100) if total > 0 else 0
            print(f"{fmt}: {count:2d} ({pct:4.1f}%)  ", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Define MX format policies")
    parser.add_argument(
        "--sensitivity",
        type=str,
        default="results/sensitivity_matrix.csv",
        help="Path to sensitivity matrix CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/policies.json",
        help="Output JSON path for policies",
    )
    args = parser.parse_args()

    # Load sensitivity data
    print(f"📊 Loading sensitivity data from {args.sensitivity}...")
    df = load_sensitivity_matrix(args.sensitivity)

    # Define all policies
    print("🎯 Defining policies...\n")
    policies = {
        "A": define_policy_a_conservative(df),
        "B": define_policy_b_balanced(df),
        "C": define_policy_c_aggressive(df),
        "D": define_policy_d_phase_adaptive(df),
    }

    # Analyze policies
    print(f"{'='*60}")
    print("Policy Analysis")
    print(f"{'='*60}")

    for policy_id, policy in policies.items():
        analysis = analyze_policy(policy, df)
        print_policy_analysis(analysis)

    print(f"{'='*60}\n")

    # Save policies
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(policies, f, indent=2)

    print(f"✓ Policies saved to {output_path}")

    # Print recommendations
    print(f"\n{'='*60}")
    print("Recommendations")
    print(f"{'='*60}")
    print("Policy A: Use if quality is critical (minimal risk)")
    print("Policy B: Balanced approach (recommended starting point)")
    print("Policy C: Use if energy is critical (accepts some quality loss)")
    print("Policy D: Research target (best energy/quality tradeoff)")
    print(f"{'='*60}\n")

    print("✅ Policy definitions complete!")


if __name__ == "__main__":
    main()
