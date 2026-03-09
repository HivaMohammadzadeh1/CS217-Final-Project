import unittest

import numpy as np

from integration.lab1_fpga_interface import Lab1FPGAInterface


class _FakeLib:
    def __init__(self, configure_rc=0):
        self.configure_rc = configure_rc
        self.configure_calls = []

    def fpga_configure_mode(self, precision_mode, group_size):
        self.configure_calls.append((precision_mode, group_size))
        return self.configure_rc

    def fpga_cleanup(self):
        return None


class _StubLab1FPGAInterface(Lab1FPGAInterface):
    def __init__(self, use_hardware=False, configure_rc=0):
        self._stub_use_hardware = use_hardware
        self._stub_configure_rc = configure_rc
        super().__init__(device_id=0, verbose=False)

    def _init_fpga(self):
        self.use_hardware = self._stub_use_hardware
        if self.use_hardware:
            self.lib = _FakeLib(configure_rc=self._stub_configure_rc)


class TestLab1PrecisionConfig(unittest.TestCase):
    def setUp(self):
        self.a = np.ones((16, 16), dtype=np.float32)
        self.b = np.eye(16, dtype=np.float32)

    def test_group_size_change_requires_flush(self):
        fpga = _StubLab1FPGAInterface(use_hardware=False)

        fpga.configure_precision("INT8", group_size=16, flush=False)

        stats = fpga.get_stats()
        self.assertTrue(stats["switch_pending"])
        self.assertEqual(stats["group_size"], 8)
        self.assertEqual(stats["pending_group_size"], 16)
        with self.assertRaises(RuntimeError):
            _ = fpga.matmul_16x16(self.a, self.b)

        fpga.flush_pipeline()
        stats = fpga.get_stats()
        self.assertFalse(stats["switch_pending"])
        self.assertEqual(stats["group_size"], 16)

    def test_flush_programs_hardware_mode(self):
        fpga = _StubLab1FPGAInterface(use_hardware=True)

        fpga.configure_precision("MXFP4", group_size=16, flush=False)
        fpga.flush_pipeline()

        self.assertEqual(fpga.lib.configure_calls, [(2, 16)])
        stats = fpga.get_stats()
        self.assertEqual(stats["precision_mode"], "MXFP4")
        self.assertEqual(stats["group_size"], 16)
        self.assertFalse(stats["switch_pending"])

    def test_failed_hardware_mode_programming_keeps_request_pending(self):
        fpga = _StubLab1FPGAInterface(use_hardware=True, configure_rc=-1)

        fpga.configure_precision("MXFP8", group_size=8, flush=False)
        with self.assertRaises(RuntimeError):
            fpga.flush_pipeline()

        stats = fpga.get_stats()
        self.assertEqual(stats["precision_mode"], "INT8")
        self.assertEqual(stats["group_size"], 8)
        self.assertTrue(stats["switch_pending"])
        self.assertEqual(fpga.lib.configure_calls, [(1, 8)])


if __name__ == "__main__":
    unittest.main()
