import unittest

import pandas as pd

from pytorch_profiling.define_policies import build_policies


class PolicyGenerationTests(unittest.TestCase):
    def test_build_policies_uses_requested_group_size_columns(self):
        df = pd.DataFrame([
            {
                "layer": "model.layers.0.self_attn.q_proj",
                "mxfp4_g8_tolerant": False,
                "mxfp8_g8_tolerant": True,
                "mxfp4_g16_tolerant": True,
                "mxfp8_g16_tolerant": True,
            }
        ])
        policies = build_policies(df, group_size=16)
        self.assertEqual(policies["B"]["layers"]["model.layers.0.self_attn.q_proj"]["rollout"], "MXFP4")
        self.assertEqual(policies["D"]["layers"]["model.layers.0.self_attn.q_proj"]["gradient"], "MXFP8")
        self.assertEqual(policies["A"]["group_size"], 16)


if __name__ == "__main__":
    unittest.main()
