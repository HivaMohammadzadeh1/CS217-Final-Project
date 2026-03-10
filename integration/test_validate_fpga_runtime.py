import unittest

from integration.validate_fpga_runtime import classify_backend, run_validation


class TestValidateFPGARuntime(unittest.TestCase):
    def test_classify_backend_prefers_real_mx_hardware_when_reported(self):
        backend = classify_backend(
            "MXFP8",
            {"supports_real_mx_hardware": True, "last_tile_backend": "lab1_fpga_mx"},
            {"hardware_tile_calls": 4, "mx_software_fallback_tile_calls": 0},
        )
        self.assertEqual(backend, "real_mx_hardware")

    def test_mock_validation_reports_expected_backends(self):
        summary = run_validation(
            use_real_fpga=False,
            device_id=0,
            group_size=8,
            samples=2,
            seed=7,
            value_scale=1.0,
        )
        self.assertEqual(summary["results"]["INT8"]["backend"], "software_int8_fallback")
        self.assertEqual(summary["results"]["MXFP8"]["backend"], "software_mx_fallback")
        self.assertEqual(summary["results"]["MXFP4"]["backend"], "software_mx_fallback")
        self.assertGreaterEqual(summary["results"]["MXFP8"]["counter_delta"]["mx_software_fallback_tile_calls"], 1)


if __name__ == "__main__":
    unittest.main()
