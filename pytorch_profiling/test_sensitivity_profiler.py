import unittest

import torch

from pytorch_profiling.sensitivity_profiler import (
    FORMAT_SPECS,
    build_combinations,
    layer_type,
    quantize_weight_tensor,
)


class SensitivityProfilerUnitTests(unittest.TestCase):
    def test_build_combinations_default(self):
        combos = build_combinations(None)
        self.assertEqual(combos, (("MXFP4", 8), ("MXFP4", 16), ("MXFP8", 8), ("MXFP8", 16)))

    def test_layer_type_classification(self):
        self.assertEqual(layer_type("model.layers.0.self_attn.q_proj"), "attention")
        self.assertEqual(layer_type("model.layers.0.mlp.down_proj"), "mlp")
        self.assertEqual(layer_type("lm_head"), "lm_head")
        self.assertEqual(layer_type("model.layers.0.input_layernorm"), "other")

    def test_quantize_weight_tensor_preserves_shape_and_dtype(self):
        weight = torch.tensor([[0.1, -0.3, 1.2, -2.4], [0.5, 0.7, -0.8, 0.9]], dtype=torch.float32)
        quantized = quantize_weight_tensor(weight, FORMAT_SPECS["MXFP8"], 8)
        self.assertEqual(tuple(quantized.shape), tuple(weight.shape))
        self.assertEqual(quantized.dtype, weight.dtype)
        self.assertTrue(torch.isfinite(quantized).all())

    def test_mxfp4_is_more_aggressive_than_mxfp8(self):
        weight = torch.linspace(-3.0, 3.0, steps=32, dtype=torch.float32).reshape(2, 16)
        q4 = quantize_weight_tensor(weight, FORMAT_SPECS["MXFP4"], 8)
        q8 = quantize_weight_tensor(weight, FORMAT_SPECS["MXFP8"], 8)
        err4 = torch.mean(torch.abs(q4 - weight)).item()
        err8 = torch.mean(torch.abs(q8 - weight)).item()
        self.assertGreater(err4, err8)


if __name__ == "__main__":
    unittest.main()
