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
            allow_mx_software_fallback=True,
        )

        cmd = build_command(args, "D", Path("results/policy_sweep/policy_D"))

        self.assertIn("baseline_energy/rlhf_with_fpga.py", cmd)
        self.assertIn("--policy-json", cmd)
        self.assertIn("results/policies.json", cmd)
        self.assertIn("--policy-name", cmd)
        self.assertIn("D", cmd)
        self.assertIn("--skip-eval", cmd)
        self.assertIn("--allow-gradient-offload", cmd)
        self.assertIn("--allow-mx-software-fallback", cmd)
        self.assertIn("16", cmd)

    def test_build_command_appends_passthrough_args(self):
        args = SimpleNamespace(
            steps=2,
            policy_json="baseline_energy/data/smoke_policies.json",
            eval_samples=None,
            skip_eval=False,
            save_models=False,
            precision_mode=None,
            group_size=None,
            allow_gradient_offload=False,
            allow_mx_software_fallback=False,
        )

        cmd = build_command(
            args,
            "A",
            Path("results/policy_sweep/policy_A"),
            passthrough_args=["--use-mock-fpga", "--model-name", "tiny-model"],
        )

        self.assertEqual(cmd[-3:], ["--use-mock-fpga", "--model-name", "tiny-model"])


if __name__ == "__main__":
    unittest.main()
