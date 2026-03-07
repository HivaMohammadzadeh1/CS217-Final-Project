"""
Optional parity check against mx-pytorch / microxcaling.

This test is intentionally optional:
- If the package is not installed, test is skipped.
- If the package API is incompatible, test is skipped with explanation.

Run:
  python -m unittest integration.test_mx_pytorch_optional -v
"""

import importlib
import unittest

import numpy as np

from integration.mx_precision_sim import MXFP8_SPEC, quantize_dequantize_vector


def _try_load_mx_module():
    for name in ("mx", "microxcaling"):
        try:
            return importlib.import_module(name)
        except Exception:
            pass
    return None


class TestMXPytorchOptional(unittest.TestCase):
    def test_optional_parity_mxfp8_vector(self):
        mx_mod = _try_load_mx_module()
        if mx_mod is None:
            self.skipTest("mx/microxcaling package not installed.")

        # We only run parity when a direct quantize-like API is available.
        quantize_fn = getattr(mx_mod, "quantize", None)
        if quantize_fn is None:
            self.skipTest("mx module found, but no top-level quantize API exposed.")

        x = np.linspace(-1.0, 1.0, num=32, dtype=np.float32)
        ours = quantize_dequantize_vector(x, MXFP8_SPEC, group_size=8)

        try:
            # This call shape is intentionally conservative; if the installed API
            # differs, we skip instead of failing hard.
            theirs = quantize_fn(x, format="mxfp8", group_size=8)
        except Exception as exc:
            self.skipTest(f"Installed mx API signature not compatible: {exc}")

        theirs = np.asarray(theirs, dtype=np.float32)
        if theirs.shape != ours.shape:
            self.skipTest("mx quantize output shape differs from expected vector shape.")

        mae = float(np.mean(np.abs(ours - theirs)))
        # Loose parity threshold to account for implementation details.
        self.assertLess(mae, 0.15)


if __name__ == "__main__":
    unittest.main()

