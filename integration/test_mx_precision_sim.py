"""
Unit tests for MX precision simulation behavior.

Run:
  python -m unittest integration.test_mx_precision_sim -v
"""

import unittest

import numpy as np

from integration.mx_precision_sim import (
    DualPrecisionMXSimulator,
    MXFP4_SPEC,
    MXFP8_SPEC,
    PrecisionMode,
    decode_minifloat,
    encode_minifloat,
    quantize_dequantize_vector,
)


class TestMXPrecisionSim(unittest.TestCase):
    def test_encode_decode_zero(self):
        self.assertEqual(encode_minifloat(0.0, MXFP8_SPEC), 0)
        self.assertEqual(encode_minifloat(0.0, MXFP4_SPEC), 0)
        self.assertEqual(decode_minifloat(0, MXFP8_SPEC), 0.0)
        self.assertEqual(decode_minifloat(0, MXFP4_SPEC), 0.0)

    def test_known_codes_for_simple_values(self):
        # MXFP8 (E4M3, bias 7)
        self.assertEqual(encode_minifloat(1.0, MXFP8_SPEC), 0x38)
        self.assertEqual(encode_minifloat(2.0, MXFP8_SPEC), 0x40)
        self.assertEqual(encode_minifloat(0.5, MXFP8_SPEC), 0x30)
        self.assertEqual(encode_minifloat(-1.0, MXFP8_SPEC), 0xB8)

        # MXFP4 (E2M1, bias 1)
        self.assertEqual(encode_minifloat(1.0, MXFP4_SPEC), 0x2)
        self.assertEqual(encode_minifloat(2.0, MXFP4_SPEC), 0x4)
        self.assertEqual(encode_minifloat(-1.0, MXFP4_SPEC), 0xA)

    def test_mxfp8_is_more_accurate_than_mxfp4_on_average(self):
        rng = np.random.default_rng(42)
        values = rng.normal(0.0, 1.0, size=256).astype(np.float32)

        q8 = quantize_dequantize_vector(values, MXFP8_SPEC, group_size=8)
        q4 = quantize_dequantize_vector(values, MXFP4_SPEC, group_size=8)

        mae8 = float(np.mean(np.abs(values - q8)))
        mae4 = float(np.mean(np.abs(values - q4)))
        self.assertLess(mae8, mae4)

    def test_mode_switch_requires_flush(self):
        sim = DualPrecisionMXSimulator(group_size=8, initial_mode=PrecisionMode.MXFP8)
        a = np.ones(16, dtype=np.float32)
        b = np.ones(16, dtype=np.float32)

        sim.request_mode(PrecisionMode.MXFP4)
        with self.assertRaises(RuntimeError):
            _ = sim.dot(a, b)

        sim.flush_pipeline()
        out = sim.dot(a, b)
        self.assertTrue(np.isfinite(out))
        self.assertEqual(sim.active_mode, PrecisionMode.MXFP4)

    def test_tile_matmul_error_reasonable(self):
        rng = np.random.default_rng(123)
        a = rng.normal(0.0, 1.0, size=(16, 16)).astype(np.float32)
        b = rng.normal(0.0, 1.0, size=(16, 16)).astype(np.float32)

        sim = DualPrecisionMXSimulator(group_size=8, initial_mode=PrecisionMode.MXFP8)
        got = sim.matmul_16x16(a, b)
        ref = a @ b

        mae = float(np.mean(np.abs(got - ref)))
        self.assertLess(mae, 0.2)

    def test_invalid_group_size_rejected(self):
        with self.assertRaises(ValueError):
            DualPrecisionMXSimulator(group_size=4)


if __name__ == "__main__":
    unittest.main()
