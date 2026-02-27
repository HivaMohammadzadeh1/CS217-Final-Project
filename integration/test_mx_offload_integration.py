"""
Integration tests for precision control in FPGAMatmulOffload.

Run:
  python -m unittest integration.test_mx_offload_integration -v
"""

import unittest

import numpy as np

from integration.fpga_matmul_offload import FPGAMatmulOffload


class TestMXOffloadIntegration(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(99)
        self.a = self.rng.normal(0.0, 1.0, size=(16, 16)).astype(np.float32)
        self.b = self.rng.normal(0.0, 1.0, size=(16, 16)).astype(np.float32)
        self.ref = self.a @ self.b

    def test_int8_mock_matches_numpy(self):
        offloader = FPGAMatmulOffload(use_mock=True, precision_mode="INT8")
        got = offloader.matmul(self.a, self.b)
        max_err = float(np.max(np.abs(got - self.ref)))
        self.assertLess(max_err, 1e-5)

    def test_mode_switch_requires_flush(self):
        offloader = FPGAMatmulOffload(use_mock=True, precision_mode="INT8")
        offloader.configure_precision("MXFP4", flush=False)

        with self.assertRaises(RuntimeError):
            _ = offloader.matmul(self.a, self.b)

        offloader.flush_pipeline()
        got = offloader.matmul(self.a, self.b)
        self.assertTrue(np.isfinite(got).all())

    def test_mxfp8_more_accurate_than_mxfp4(self):
        offloader = FPGAMatmulOffload(use_mock=True, precision_mode="MXFP8", group_size=8)
        got8 = offloader.matmul(self.a, self.b)
        mae8 = float(np.mean(np.abs(got8 - self.ref)))

        offloader.configure_precision("MXFP4", flush=True)
        got4 = offloader.matmul(self.a, self.b)
        mae4 = float(np.mean(np.abs(got4 - self.ref)))

        self.assertLess(mae8, mae4)

    def test_stats_include_precision_metadata(self):
        offloader = FPGAMatmulOffload(use_mock=True, precision_mode="MXFP8", group_size=16)
        _ = offloader.matmul(self.a, self.b)
        stats = offloader.get_stats()

        self.assertIn("precision_mode", stats)
        self.assertIn("group_size", stats)
        self.assertIn("mode_switches", stats)
        self.assertEqual(stats["precision_mode"], "MXFP8")
        self.assertEqual(stats["group_size"], 16)

    def test_real_interface_mxfp_fallback_path(self):
        # In development environments this usually uses software fallback because
        # Lab 1 shared library / hardware may not be available. The behavior
        # should still be deterministic and valid.
        offloader = FPGAMatmulOffload(
            use_mock=False,
            use_lab1=True,
            precision_mode="MXFP8",
            group_size=8,
            verbose=False,
        )
        got = offloader.matmul(self.a, self.b)
        self.assertTrue(np.isfinite(got).all())
        stats = offloader.get_stats()
        self.assertEqual(stats.get("precision_mode"), "MXFP8")


if __name__ == "__main__":
    unittest.main()

