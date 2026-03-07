"""
Unit tests for adaptive precision policy control.
"""

import json
from pathlib import Path
import tempfile
import unittest

from integration.adaptive_controller import AdaptivePrecisionController


class TestAdaptivePrecisionController(unittest.TestCase):
    def write_policy_file(self, payload):
        tempdir = tempfile.TemporaryDirectory()
        path = Path(tempdir.name) / "policies.json"
        path.write_text(json.dumps(payload))
        self.addCleanup(tempdir.cleanup)
        return path

    def test_named_policy_collection_lookup(self):
        payload = {
            "D": {
                "name": "D - Phase-Adaptive",
                "layers": {
                    "model.layers.23.self_attn.q_proj": {
                        "rollout": "MXFP4",
                        "reward": "MXFP8",
                        "gradient": "FP16",
                    }
                }
            }
        }
        policy_path = self.write_policy_file(payload)
        controller = AdaptivePrecisionController(
            default_precision="INT8",
            policy_name="D",
            policy_path=str(policy_path),
        )

        rollout = controller.get_decision(
            "pretrained_model.model.layers.23.self_attn.q_proj",
            phase="rollout",
        )
        reward = controller.get_decision(
            "pretrained_model.model.layers.23.self_attn.q_proj",
            phase="reward",
        )

        self.assertEqual(rollout.precision, "MXFP4")
        self.assertEqual(reward.precision, "MXFP8")

    def test_unknown_layer_falls_back_to_default_precision(self):
        controller = AdaptivePrecisionController(default_precision="MXFP8", default_group_size=16)
        decision = controller.get_decision("some.unknown.layer", phase="rollout")
        self.assertEqual(decision.precision, "MXFP8")
        self.assertEqual(decision.group_size, 16)
        self.assertEqual(decision.source, "default")

    def test_gradient_phase_defaults_to_fp16_safety_fallback(self):
        payload = {
            "C": {
                "name": "C - Aggressive",
                "layers": {
                    "model.layers.1.self_attn.q_proj": {
                        "rollout": "MXFP4",
                        "reward": "MXFP4",
                        "gradient": "MXFP8",
                    }
                }
            }
        }
        policy_path = self.write_policy_file(payload)
        controller = AdaptivePrecisionController(
            default_precision="INT8",
            policy_name="C",
            policy_path=str(policy_path),
            allow_gradient_offload=False,
        )

        decision = controller.get_decision("model.layers.1.self_attn.q_proj", phase="gradient")
        self.assertEqual(decision.precision, "FP16")
        self.assertEqual(decision.source, "safety:gradient-fallback")
        self.assertFalse(decision.should_offload)

    def test_phase_scope_restores_previous_phase(self):
        controller = AdaptivePrecisionController(default_precision="INT8")
        controller.set_phase("rollout")
        with controller.phase_scope("reward"):
            self.assertEqual(controller.current_phase, "reward")
        self.assertEqual(controller.current_phase, "rollout")

    def test_stats_count_decisions_by_phase_and_precision(self):
        controller = AdaptivePrecisionController(default_precision="INT8")
        controller.get_decision("layer.q_proj", phase="rollout")
        controller.get_decision("layer.q_proj", phase="reward")
        stats = controller.get_stats()
        self.assertEqual(stats["decision_counts"]["rollout"]["INT8"], 1)
        self.assertEqual(stats["decision_counts"]["reward"]["INT8"], 1)


if __name__ == "__main__":
    unittest.main()
