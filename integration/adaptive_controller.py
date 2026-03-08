"""
Adaptive precision controller for FPGA offload experiments.

This module turns policy definitions into concrete per-layer precision choices
for rollout, reward, and gradient phases.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


VALID_PHASES = ("rollout", "reward", "gradient")
VALID_PRECISIONS = ("INT8", "MXFP8", "MXFP4", "FP16")


def validate_group_size(group_size: int, field_name: str = "group_size") -> int:
    if group_size not in (8, 16):
        raise ValueError(f"{field_name} must be 8 or 16.")
    return group_size


def normalize_phase(phase: str) -> str:
    phase_key = str(phase).strip().lower()
    if phase_key not in VALID_PHASES:
        raise ValueError(f"Unsupported phase: {phase}")
    return phase_key


def normalize_precision(precision: str) -> str:
    precision_key = str(precision).strip().upper()
    if precision_key not in VALID_PRECISIONS:
        raise ValueError(f"Unsupported precision: {precision}")
    return precision_key


def layer_key_candidates(layer_name: str) -> Iterable[str]:
    """Yield progressively normalized keys for policy lookup."""
    name = str(layer_name).strip()
    seen = set()

    def emit(candidate: str) -> Iterator[str]:
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate

    for candidate in emit(name):
        yield candidate

    prefixes = (
        "pretrained_model.",
        "model.",
        "base_model.",
    )

    changed = True
    current = name
    while changed:
        changed = False
        for prefix in prefixes:
            if current.startswith(prefix):
                current = current[len(prefix):]
                changed = True
                for candidate in emit(current):
                    yield candidate


def load_policy_definition(policy_name: Optional[str], policy_path: Optional[str]):
    if not policy_path:
        return None

    payload = json.loads(Path(policy_path).read_text())
    if "layers" in payload:
        return payload

    if policy_name is None:
        raise ValueError(
            "policy_path points to a policy collection. Set policy_name to select one."
        )

    if policy_name not in payload:
        raise ValueError(
            f"Policy '{policy_name}' not found in {policy_path}. "
            f"Available: {sorted(payload.keys())}"
        )

    return payload[policy_name]


def resolve_policy_group_size(default_group_size: int,
                              policy_name: Optional[str] = None,
                              policy_path: Optional[str] = None) -> int:
    policy = load_policy_definition(policy_name, policy_path)
    if policy and policy.get("group_size") is not None:
        return validate_group_size(
            int(policy["group_size"]),
            field_name="policy group_size",
        )
    return validate_group_size(default_group_size, field_name="default_group_size")


@dataclass(frozen=True)
class PrecisionDecision:
    precision: str
    group_size: int
    phase: str
    layer_name: str
    source: str

    @property
    def should_offload(self) -> bool:
        return self.precision != "FP16"


class AdaptivePrecisionController:
    """Selects precision from a named policy or a global default."""

    def __init__(self,
                 default_precision: str = "INT8",
                 default_group_size: int = 8,
                 allow_gradient_offload: bool = False,
                 policy_name: Optional[str] = None,
                 policy_path: Optional[str] = None):
        self.default_precision = normalize_precision(default_precision)
        self.default_group_size = validate_group_size(
            default_group_size,
            field_name="default_group_size",
        )
        self.allow_gradient_offload = allow_gradient_offload
        self.current_phase = "rollout"
        self.policy_name = policy_name
        self.policy_path = str(policy_path) if policy_path else None
        self.policy = self._load_policy(policy_name, policy_path)
        self.decision_counts = {
            phase: {precision: 0 for precision in VALID_PRECISIONS}
            for phase in VALID_PHASES
        }

    @staticmethod
    def _load_policy(policy_name: Optional[str], policy_path: Optional[str]):
        return load_policy_definition(policy_name, policy_path)

    def set_phase(self, phase: str) -> None:
        self.current_phase = normalize_phase(phase)

    @contextmanager
    def phase_scope(self, phase: str):
        previous = self.current_phase
        self.current_phase = normalize_phase(phase)
        try:
            yield self
        finally:
            self.current_phase = previous

    def _policy_precision(self, layer_name: str, phase: str) -> Optional[str]:
        if not self.policy:
            return None

        layers = self.policy.get("layers", {})
        for candidate in layer_key_candidates(layer_name):
            formats = layers.get(candidate)
            if formats and phase in formats:
                return normalize_precision(formats[phase])
        return None

    def get_decision(self,
                     layer_name: str,
                     phase: Optional[str] = None,
                     model_role: Optional[str] = None) -> PrecisionDecision:
        del model_role  # Reserved for later policy specialization.
        phase_key = normalize_phase(phase or self.current_phase)
        policy_precision = self._policy_precision(layer_name, phase_key)
        precision = policy_precision or self.default_precision
        source = f"policy:{self.policy_name}" if policy_precision else "default"

        # The current FPGA matmul path is inference-only. Keep gradient phase
        # on the native PyTorch linear path unless explicitly overridden.
        if phase_key == "gradient" and precision != "FP16" and not self.allow_gradient_offload:
            precision = "FP16"
            source = "safety:gradient-fallback"

        decision = PrecisionDecision(
            precision=precision,
            group_size=self.default_group_size,
            phase=phase_key,
            layer_name=str(layer_name),
            source=source,
        )
        self.decision_counts[phase_key][decision.precision] += 1
        return decision

    def get_stats(self) -> Dict[str, object]:
        return {
            "default_precision": self.default_precision,
            "default_group_size": self.default_group_size,
            "allow_gradient_offload": self.allow_gradient_offload,
            "policy_name": self.policy_name,
            "policy_path": self.policy_path,
            "current_phase": self.current_phase,
            "decision_counts": self.decision_counts,
        }
