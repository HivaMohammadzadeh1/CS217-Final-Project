"""
MX Precision Simulation Utilities (Milestone 3)

This module provides a deterministic software model for:
- MXFP8 (E4M3)
- MXFP4 (E2M1)
- Shared group scaling (group size 8 or 16)
- Safe mode switching with explicit pipeline flush

It is intentionally straightforward and readable so it can act as a
reference model for integration tests and future hardware validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, List

import numpy as np


class PrecisionMode(str, Enum):
    MXFP8 = "MXFP8"
    MXFP4 = "MXFP4"


@dataclass(frozen=True)
class MiniFloatSpec:
    name: str
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int


MXFP8_SPEC = MiniFloatSpec(name="E4M3", exponent_bits=4, mantissa_bits=3, exponent_bias=7)
MXFP4_SPEC = MiniFloatSpec(name="E2M1", exponent_bits=2, mantissa_bits=1, exponent_bias=1)


def validate_group_size(group_size: int) -> None:
    if group_size not in (8, 16):
        raise ValueError("group_size must be 8 or 16.")


def _floor_log2_positive(x: float) -> int:
    if x <= 0.0:
        raise ValueError("_floor_log2_positive requires x > 0.")
    return int(math.floor(math.log2(x)))


def encode_minifloat(value: float, spec: MiniFloatSpec) -> int:
    exp_bits = spec.exponent_bits
    mant_bits = spec.mantissa_bits
    sign_shift = exp_bits + mant_bits
    exp_mask = (1 << exp_bits) - 1
    mant_mask = (1 << mant_bits) - 1

    if (not math.isfinite(value)) or value == 0.0:
        return 0

    neg = math.copysign(1.0, value) < 0.0
    ax = abs(value)

    exponent = _floor_log2_positive(ax)
    normalized = math.ldexp(ax, -exponent)
    exp_field = exponent + spec.exponent_bias

    if exp_field <= 0:
        # Subnormal region.
        scaled = math.ldexp(ax, -(1 - spec.exponent_bias))
        mantissa = int(round(scaled * (1 << mant_bits)))
        mantissa = max(0, min(mantissa, mant_mask))
        exp_field = 0
    else:
        mantissa_f = (normalized - 1.0) * float(1 << mant_bits)
        mantissa = int(round(mantissa_f))

        # Carry when rounding 1.111.. to 10.000..
        if mantissa == (1 << mant_bits):
            mantissa = 0
            exp_field += 1

        if exp_field >= exp_mask:
            # Saturate to max finite value.
            exp_field = exp_mask
            mantissa = mant_mask

    code = ((exp_field & exp_mask) << mant_bits) | (mantissa & mant_mask)
    if neg:
        code |= (1 << sign_shift)
    return code


def decode_minifloat(code: int, spec: MiniFloatSpec) -> float:
    exp_bits = spec.exponent_bits
    mant_bits = spec.mantissa_bits
    sign_shift = exp_bits + mant_bits
    exp_mask = (1 << exp_bits) - 1
    mant_mask = (1 << mant_bits) - 1

    neg = ((code >> sign_shift) & 0x1) != 0
    exp_field = (code >> mant_bits) & exp_mask
    mantissa = code & mant_mask

    if exp_field == 0 and mantissa == 0:
        return 0.0

    if exp_field == 0:
        frac = mantissa / float(1 << mant_bits)
        value = math.ldexp(frac, 1 - spec.exponent_bias)
    else:
        frac = 1.0 + (mantissa / float(1 << mant_bits))
        value = math.ldexp(frac, exp_field - spec.exponent_bias)

    return -value if neg else value


def quantize_dequantize_vector(values: Iterable[float],
                               spec: MiniFloatSpec,
                               group_size: int) -> np.ndarray:
    validate_group_size(group_size)
    arr = np.asarray(list(values), dtype=np.float32)
    out = np.zeros_like(arr)

    for start in range(0, arr.size, group_size):
        end = min(start + group_size, arr.size)
        group = arr[start:end]

        max_abs = float(np.max(np.abs(group))) if group.size else 0.0
        if max_abs == 0.0:
            continue

        shared_exp = _floor_log2_positive(max_abs)

        for i in range(start, end):
            scaled = math.ldexp(float(arr[i]), -shared_exp)
            code = encode_minifloat(scaled, spec)
            decoded = decode_minifloat(code, spec)
            out[i] = np.float32(math.ldexp(decoded, shared_exp))

    return out


def dot_quantized(a: Iterable[float], b: Iterable[float],
                  spec: MiniFloatSpec, group_size: int) -> float:
    a_arr = np.asarray(list(a), dtype=np.float32)
    b_arr = np.asarray(list(b), dtype=np.float32)
    if a_arr.shape != b_arr.shape:
        raise ValueError("dot_quantized requires equal-length vectors.")
    a_q = quantize_dequantize_vector(a_arr, spec, group_size)
    b_q = quantize_dequantize_vector(b_arr, spec, group_size)
    return float(np.dot(a_q, b_q))


class DualPrecisionMXSimulator:
    """
    Reference simulator for dual-precision MX tile matmul.

    Safe switching contract:
    - request_mode(...) marks switch pending.
    - flush_pipeline() must be called before compute.
    """

    def __init__(self, group_size: int = 8, initial_mode: PrecisionMode = PrecisionMode.MXFP8):
        validate_group_size(group_size)
        self.group_size = group_size
        self.active_mode = initial_mode
        self.pending_mode = initial_mode
        self.switch_pending = False
        self.flush_count = 0

    @staticmethod
    def spec_for_mode(mode: PrecisionMode) -> MiniFloatSpec:
        if mode == PrecisionMode.MXFP8:
            return MXFP8_SPEC
        if mode == PrecisionMode.MXFP4:
            return MXFP4_SPEC
        raise ValueError(f"Unsupported mode: {mode}")

    def request_mode(self, mode: PrecisionMode) -> None:
        self.pending_mode = mode
        self.switch_pending = self.pending_mode != self.active_mode

    def flush_pipeline(self) -> None:
        if not self.switch_pending:
            return
        self.active_mode = self.pending_mode
        self.switch_pending = False
        self.flush_count += 1

    def _assert_ready(self) -> None:
        if self.switch_pending:
            raise RuntimeError(
                "Mode switch pending. Call flush_pipeline() before compute.")

    def dot(self, a: np.ndarray, b: np.ndarray) -> float:
        self._assert_ready()
        spec = self.spec_for_mode(self.active_mode)
        return dot_quantized(a, b, spec, self.group_size)

    def matmul_16x16(self, tile_a: np.ndarray, tile_b: np.ndarray) -> np.ndarray:
        self._assert_ready()

        tile_a = np.asarray(tile_a, dtype=np.float32)
        tile_b = np.asarray(tile_b, dtype=np.float32)
        if tile_a.shape != (16, 16) or tile_b.shape != (16, 16):
            raise ValueError("matmul_16x16 expects two 16x16 tiles.")

        spec = self.spec_for_mode(self.active_mode)

        # Quantize-dequantize each row/column independently, then dot.
        out = np.zeros((16, 16), dtype=np.float32)
        b_t = tile_b.T
        for i in range(16):
            row = quantize_dequantize_vector(tile_a[i], spec, self.group_size)
            for j in range(16):
                col = quantize_dequantize_vector(b_t[j], spec, self.group_size)
                out[i, j] = float(np.dot(row, col))
        return out

