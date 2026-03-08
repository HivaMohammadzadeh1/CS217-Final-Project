"""
Unit tests for lightweight helper behavior in rlhf_with_fpga.py.
"""

import unittest

import torch
import torch.nn as nn

from baseline_energy.rlhf_with_fpga import (
    FPGALinearLayer,
    RLHFWithFPGATrainer,
    replace_linear_with_fpga_selective,
)


class FakeOffloader:
    def __init__(self):
        self.configure_calls = []

    def configure_precision(self, precision, group_size=None, flush=True):
        self.configure_calls.append((precision, group_size, flush))

    def matmul(self, a, b):
        return a @ b


class TinySelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)
        self.extra = nn.Linear(4, 4)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinySelfAttention()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock() for _ in range(3)])


class TestRLHFWithFPGAHelpers(unittest.TestCase):
    def test_fpga_linear_layer_supports_higher_rank_inputs(self):
        torch.manual_seed(7)
        linear = nn.Linear(4, 3, bias=True)
        offloader = FakeOffloader()
        layer = FPGALinearLayer(
            linear,
            offloader,
            layer_name="model.layers.0.self_attn.q_proj",
        )
        x = torch.randn(2, 3, 5, 4)

        got = layer(x)
        expected = linear(x)

        self.assertEqual(tuple(got.shape), (2, 3, 5, 3))
        self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_replace_linear_with_fpga_selective_only_targets_requested_blocks(self):
        model = TinyModel()
        offloader = FakeOffloader()

        replaced = replace_linear_with_fpga_selective(
            model,
            offloader,
            target_blocks=[1],
            model_role="policy",
            verbose=False,
        )

        self.assertEqual(replaced, 4)
        self.assertIsInstance(model.model.layers[1].self_attn.q_proj, FPGALinearLayer)
        self.assertIsInstance(model.model.layers[1].self_attn.o_proj, FPGALinearLayer)
        self.assertIsInstance(model.model.layers[0].self_attn.q_proj, nn.Linear)
        self.assertIsInstance(model.model.layers[2].self_attn.v_proj, nn.Linear)
        self.assertIsInstance(model.model.layers[1].self_attn.extra, nn.Linear)

    def test_record_phase_fpga_stats_accumulates_numeric_deltas(self):
        trainer = object.__new__(RLHFWithFPGATrainer)
        trainer.phase_fpga_stats = {
            "rollout": {
                "phase_invocations": 0,
                "matmul_calls": 0,
                "tile_ops": 0,
                "mode_switches": 0,
                "flush_count": 0,
            }
        }
        trainer.fpga_stats = {
            "total_matmuls": 0,
            "total_tiles": 0,
        }

        delta = RLHFWithFPGATrainer._record_phase_fpga_stats(
            trainer,
            "rollout",
            {"num_calls": 2, "total_tiles": 4, "mode_switches": 1, "flush_count": 1},
            {"num_calls": 7, "total_tiles": 10, "mode_switches": 3, "flush_count": 2},
        )

        self.assertEqual(delta["num_calls"], 5)
        self.assertEqual(delta["total_tiles"], 6)
        self.assertEqual(delta["mode_switches"], 2)
        self.assertEqual(delta["flush_count"], 1)
        self.assertEqual(trainer.phase_fpga_stats["rollout"]["phase_invocations"], 1)
        self.assertEqual(trainer.phase_fpga_stats["rollout"]["matmul_calls"], 5)
        self.assertEqual(trainer.phase_fpga_stats["rollout"]["tile_ops"], 6)
        self.assertEqual(trainer.fpga_stats["total_matmuls"], 5)
        self.assertEqual(trainer.fpga_stats["total_tiles"], 6)


if __name__ == "__main__":
    unittest.main()
