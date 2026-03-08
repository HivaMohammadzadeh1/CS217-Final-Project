import unittest

from fpga.run_fpga_flow import DESIGN_TOP, FPGA_ROOT, build_env, dispatch_command


class RunFPGAFlowTests(unittest.TestCase):
    def test_build_env_sets_repo_defaults(self):
        env = build_env()
        self.assertEqual(env["REPO_TOP"], str(FPGA_ROOT))
        self.assertEqual(env["AWS_HOME"], str(DESIGN_TOP))
        self.assertEqual(env["CL_DIR"], str(DESIGN_TOP))
        self.assertEqual(env["CL_DESIGN_NAME"], "design_top")

    def test_dispatch_fpga_build_includes_variant(self):
        class Args:
            action = "fpga-build"
            rtl_variant = "kIntWordWidth_8_kVectorSize_16_kNumVectorLanes_8"
            slot_id = 0
            fpga_test_args = ""

        cmd, cwd = dispatch_command("fpga-build", Args())
        self.assertEqual(cwd, FPGA_ROOT.parent)
        self.assertIn(f"RTL_VARIANT={Args.rtl_variant}", cmd)
        self.assertEqual(cmd[:4], ["make", "-C", str(DESIGN_TOP), "fpga_build"])

    def test_dispatch_run_fpga_test_passes_slot_and_extra_args(self):
        class Args:
            action = "run-fpga-test"
            rtl_variant = ""
            slot_id = 3
            fpga_test_args = "MXFP8 16"

        cmd, _ = dispatch_command("run-fpga-test", Args())
        self.assertEqual(cmd[:4], ["make", "-C", str(DESIGN_TOP), "run_fpga_test"])
        self.assertIn("SLOT_ID=3", cmd)
        self.assertIn("FPGA_TEST_ARGS=MXFP8 16", cmd)


if __name__ == "__main__":
    unittest.main()
