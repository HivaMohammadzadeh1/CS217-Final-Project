"""
Validate the current FPGA runtime contract against the repo reference math.

This script is meant to answer one practical question before a larger run:
for INT8 / MXFP8 / MXFP4, what backend actually executed, and how far was it
from the corresponding reference implementation on 16x16 tiles?

Examples:
  .venv/bin/python integration/validate_fpga_runtime.py
  .venv/bin/python integration/validate_fpga_runtime.py --use-real-fpga
  .venv/bin/python integration/validate_fpga_runtime.py --use-real-fpga --require-real-mx-hardware
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np

try:
    from .fpga_matmul_offload import FPGAMatmulOffload
    from .mx_precision_sim import DualPrecisionMXSimulator, PrecisionMode
except ImportError:
    from fpga_matmul_offload import FPGAMatmulOffload
    from mx_precision_sim import DualPrecisionMXSimulator, PrecisionMode


REFERENCE_MODES = {
    "MXFP8": PrecisionMode.MXFP8,
    "MXFP4": PrecisionMode.MXFP4,
}
COUNTER_KEYS = (
    "num_calls",
    "total_tiles",
    "mode_switches",
    "flush_count",
    "hardware_tile_calls",
    "mx_software_fallback_tile_calls",
)


def snapshot_counters(stats: Dict[str, object]) -> Dict[str, int]:
    return {key: int(stats.get(key, 0) or 0) for key in COUNTER_KEYS}


def diff_counters(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {key: after.get(key, 0) - before.get(key, 0) for key in COUNTER_KEYS}


def classify_backend(mode: str, final_stats: Dict[str, object], delta: Dict[str, int]) -> str:
    if mode == "INT8":
        if bool(final_stats.get("using_hardware")) and delta.get("hardware_tile_calls", 0) > 0:
            return "lab1_fpga_int8"
        if delta.get("num_calls", 0) > 0:
            return "software_int8_fallback"
        return "no_activity"

    if delta.get("mx_software_fallback_tile_calls", 0) > 0:
        return "software_mx_fallback"
    if bool(final_stats.get("supports_real_mx_hardware")) and delta.get("hardware_tile_calls", 0) > 0:
        return "real_mx_hardware"
    return str(final_stats.get("last_tile_backend", "unknown"))


def reference_matmul(a: np.ndarray, b: np.ndarray, mode: str, group_size: int) -> np.ndarray:
    if mode == "INT8":
        return (a @ b).astype(np.float32)
    sim = DualPrecisionMXSimulator(
        group_size=group_size,
        initial_mode=REFERENCE_MODES[mode],
    )
    return sim.matmul_16x16(a, b).astype(np.float32)


def validate_mode(offloader: FPGAMatmulOffload,
                  mode: str,
                  group_size: int,
                  samples: int,
                  rng: np.random.Generator,
                  value_scale: float) -> Dict[str, object]:
    offloader.configure_precision(mode, group_size=group_size, flush=True)
    before = snapshot_counters(offloader.get_stats())

    worst_max_abs_error = 0.0
    mean_abs_errors = []
    for _ in range(samples):
        a = rng.uniform(-value_scale, value_scale, size=(16, 16)).astype(np.float32)
        b = rng.uniform(-value_scale, value_scale, size=(16, 16)).astype(np.float32)
        got = offloader.matmul(a, b).astype(np.float32)
        ref = reference_matmul(a, b, mode, group_size)
        abs_err = np.abs(got - ref)
        worst_max_abs_error = max(worst_max_abs_error, float(np.max(abs_err)))
        mean_abs_errors.append(float(np.mean(abs_err)))

    final_stats = offloader.get_stats()
    delta = diff_counters(before, snapshot_counters(final_stats))
    backend = classify_backend(mode, final_stats, delta)

    return {
        "mode": mode,
        "group_size": group_size,
        "backend": backend,
        "using_hardware": bool(final_stats.get("using_hardware", False)),
        "supports_real_mx_hardware": bool(final_stats.get("supports_real_mx_hardware", False)),
        "worst_max_abs_error": worst_max_abs_error,
        "avg_mean_abs_error": float(np.mean(mean_abs_errors)) if mean_abs_errors else 0.0,
        "counter_delta": delta,
        "final_stats": final_stats,
    }


def run_validation(use_real_fpga: bool,
                   device_id: int,
                   group_size: int,
                   samples: int,
                   seed: int,
                   value_scale: float) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    offloader = FPGAMatmulOffload(
        use_mock=not use_real_fpga,
        device_id=device_id,
        verbose=False,
        use_lab1=True,
        precision_mode="INT8",
        group_size=group_size,
    )

    results = {
        mode: validate_mode(
            offloader=offloader,
            mode=mode,
            group_size=group_size,
            samples=samples,
            rng=rng,
            value_scale=value_scale,
        )
        for mode in ("INT8", "MXFP8", "MXFP4")
    }
    return {
        "use_real_fpga": use_real_fpga,
        "device_id": device_id,
        "group_size": group_size,
        "samples": samples,
        "seed": seed,
        "value_scale": value_scale,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate current FPGA runtime backends against repo reference math.")
    parser.add_argument("--use-real-fpga", action="store_true", default=False, help="Use the Lab1 runtime path instead of the mock FPGA path.")
    parser.add_argument("--device-id", type=int, default=0, help="FPGA device/slot id for the real runtime path.")
    parser.add_argument("--group-size", type=int, default=8, help="MX group size (8 or 16).")
    parser.add_argument("--samples", type=int, default=4, help="Number of random 16x16 tile samples per mode.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--value-scale", type=float, default=1.0, help="Uniform random input range is [-value-scale, value-scale].")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output path for a JSON summary.")
    parser.add_argument("--require-real-mx-hardware", action="store_true", default=False, help="Fail if MXFP8/MXFP4 do not execute on real MX hardware.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_validation(
        use_real_fpga=args.use_real_fpga,
        device_id=args.device_id,
        group_size=args.group_size,
        samples=args.samples,
        seed=args.seed,
        value_scale=args.value_scale,
    )

    print("=" * 60)
    print("FPGA Runtime Validation")
    print("=" * 60)
    print(f"Real FPGA path: {summary['use_real_fpga']}")
    print(f"Device id:      {summary['device_id']}")
    print(f"Group size:     {summary['group_size']}")
    print(f"Samples/mode:   {summary['samples']}")
    print("")

    failed = False
    for mode in ("INT8", "MXFP8", "MXFP4"):
        result = summary["results"][mode]
        delta = result["counter_delta"]
        print(f"{mode}:")
        print(f"  backend:             {result['backend']}")
        print(f"  using_hardware:      {result['using_hardware']}")
        print(f"  supports_real_mx_hw: {result['supports_real_mx_hardware']}")
        print(f"  worst_max_abs_error: {result['worst_max_abs_error']:.6f}")
        print(f"  avg_mean_abs_error:  {result['avg_mean_abs_error']:.6f}")
        print(
            "  counter_delta:       "
            f"calls={delta['num_calls']} tiles={delta['total_tiles']} "
            f"hw_tiles={delta['hardware_tile_calls']} mx_sw_tiles={delta['mx_software_fallback_tile_calls']}"
        )
        print("")

        if args.require_real_mx_hardware and mode in ("MXFP8", "MXFP4"):
            if result["backend"] != "real_mx_hardware":
                failed = True

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary to {output_path}")

    if failed:
        print("FAIL: MX modes did not execute on real MX hardware.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
