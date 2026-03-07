"""
Unit tests for policy sweep command construction.
"""

import unittest
from types import SimpleNamespace
from pathlib import Path

from baseline_energy.run_policy_sweep import build_command


class TestPolicySweepRunner(unittest.TestCase):
    def test_build_command_includes_policy_and_optional_flags(self):
        args = SimpleNamespace(
            steps=12,
            policy_json="results/policies.json",
            eval_samples=5,
            skip_eval=True,
            save_models=False,
            precision_mode="INT8",
            group_size=16,
            allow_gradient_offload=True,
        )

        cmd = build_command(args, "D", Path("results/policy_sweep/policy_D"))

        self.assertIn("baseline_energy/rlhf_with_fpga.py", cmd)
        self.assertIn("--policy-json", cmd)
        self.assertIn("results/policies.json", cmd)
        self.assertIn("--policy-name", cmd)
        self.assertIn("D", cmd)
        self.assertIn("--skip-eval", cmd)
        self.assertIn("--allow-gradient-offload", cmd)
        self.assertIn("16", cmd)


if __name__ == "__main__":
    unittest.main()
