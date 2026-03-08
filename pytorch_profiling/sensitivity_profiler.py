from __future__ import annotations

"""
Layer sensitivity profiling using the repo's MX reference quantization.

This script profiles selected linear layers by quantizing one layer at a time,
measuring perplexity delta, and writing a policy-ready sensitivity matrix.

It is explicit about what it does:
- Uses the repo's MX quantize/dequantize reference model, not custom kernels.
- Quantizes weights only, one layer at a time.
- Supports smoke-mode runs on tiny models and local JSONL datasets.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integration.mx_precision_sim import (  # noqa: E402
    MXFP4_SPEC,
    MXFP8_SPEC,
    MiniFloatSpec,
    quantize_dequantize_vector,
)


FORMAT_SPECS: Dict[str, MiniFloatSpec] = {
    "MXFP4": MXFP4_SPEC,
    "MXFP8": MXFP8_SPEC,
}
VALID_GROUP_SIZES = (8, 16)
DEFAULT_COMBINATIONS: Tuple[Tuple[str, int], ...] = (
    ("MXFP4", 8),
    ("MXFP4", 16),
    ("MXFP8", 8),
    ("MXFP8", 16),
)


@dataclass(frozen=True)
class ProfileConfig:
    model_name: str
    dataset: str
    dataset_split: str
    dataset_config: Optional[str]
    text_field: str
    num_examples: int
    max_seq_len: int
    max_layers: Optional[int]
    tolerance_pct: float
    device: str


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_dtype_for_device(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


def parse_combo(combo: str) -> Tuple[str, int]:
    parts = combo.strip().upper().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid combo '{combo}'. Expected PRECISION:GROUP_SIZE.")
    precision, group_size_str = parts
    if precision not in FORMAT_SPECS:
        raise ValueError(f"Unsupported precision '{precision}'.")
    group_size = int(group_size_str)
    if group_size not in VALID_GROUP_SIZES:
        raise ValueError("group_size must be 8 or 16.")
    return precision, group_size


def build_combinations(raw: Optional[Sequence[str]]) -> Tuple[Tuple[str, int], ...]:
    if not raw:
        return DEFAULT_COMBINATIONS
    return tuple(parse_combo(item) for item in raw)


def layer_type(layer_name: str) -> str:
    name = layer_name.lower()
    if "attn" in name or "attention" in name:
        return "attention"
    if "mlp" in name or "ffn" in name or "feed_forward" in name:
        return "mlp"
    if "lm_head" in name or name.endswith("output"):
        return "lm_head"
    return "other"


def quantize_weight_tensor(weight: torch.Tensor, spec: MiniFloatSpec, group_size: int) -> torch.Tensor:
    arr = weight.detach().to(torch.float32).cpu().numpy()
    reshaped = arr.reshape(arr.shape[0], -1)
    out = np.empty_like(reshaped, dtype=np.float32)

    for row_idx, row in enumerate(reshaped):
        out[row_idx] = quantize_dequantize_vector(row, spec, group_size)

    out = out.reshape(arr.shape)
    return torch.from_numpy(out).to(device=weight.device, dtype=weight.dtype)


@contextmanager
def patched_linear_weight(layer: torch.nn.Module,
                          precision: str,
                          group_size: int) -> Iterator[None]:
    if not hasattr(layer, "weight"):
        yield
        return

    original = layer.weight.data.detach().clone()
    quantized = quantize_weight_tensor(original, FORMAT_SPECS[precision], group_size)
    layer.weight.data.copy_(quantized)
    try:
        yield
    finally:
        layer.weight.data.copy_(original)


class SensitivityProfiler:
    def __init__(self, config: ProfileConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = None
        self.model = None
        self.eval_texts: List[str] = []

    def load(self) -> None:
        print("=" * 60)
        print("MX Sensitivity Profiler")
        print("=" * 60)
        print(f"Model:  {self.config.model_name}")
        print(f"Device: {self.device}")
        print(f"Dataset: {self.config.dataset} [{self.config.dataset_split}]")
        print("")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype_for_device(self.config.device),
        ).to(self.device)
        self.model.eval()

        self.eval_texts = self._load_eval_texts()
        print(f"Loaded {len(self.eval_texts)} evaluation examples")
        print("")

    def _load_eval_texts(self) -> List[str]:
        dataset_spec = self.config.dataset
        path = Path(dataset_spec)
        if path.exists():
            dataset = load_dataset("json", data_files=str(path), split="train")
        else:
            kwargs = {}
            if self.config.dataset_config:
                kwargs["name"] = self.config.dataset_config
            dataset = load_dataset(dataset_spec, split=self.config.dataset_split, **kwargs)

        dataset = dataset.shuffle(seed=42)
        if self.config.num_examples:
            dataset = dataset.select(range(min(self.config.num_examples, len(dataset))))

        texts: List[str] = []
        for example in dataset:
            value = example.get(self.config.text_field)
            if value is None:
                available = ", ".join(sorted(example.keys()))
                raise KeyError(
                    f"Field '{self.config.text_field}' not found in dataset row. Available: {available}"
                )
            texts.append(str(value))
        return texts

    def get_layer_names(self) -> List[str]:
        assert self.model is not None
        names: List[str] = []
        for name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            kind = layer_type(name)
            if kind in ("attention", "mlp", "lm_head"):
                names.append(name)

        if self.config.max_layers is not None:
            return names[: self.config.max_layers]
        return names

    def compute_perplexity(self) -> float:
        assert self.model is not None and self.tokenizer is not None
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in self.eval_texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.config.max_seq_len,
                    truncation=True,
                    padding=False,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                token_count = int(inputs["input_ids"].numel())
                total_loss += float(outputs.loss.item()) * token_count
                total_tokens += token_count

        avg_loss = total_loss / max(total_tokens, 1)
        return float(torch.exp(torch.tensor(avg_loss)).item())

    def profile_layer(self,
                      layer_name: str,
                      baseline_perplexity: float,
                      combinations: Sequence[Tuple[str, int]]) -> Dict[str, object]:
        assert self.model is not None
        layer = dict(self.model.named_modules())[layer_name]
        row: Dict[str, object] = {
            "layer": layer_name,
            "layer_type": layer_type(layer_name),
        }
        best_combo: Optional[Tuple[str, int, float]] = None

        for precision, group_size in combinations:
            with patched_linear_weight(layer, precision, group_size):
                ppl = self.compute_perplexity()
            delta_pct = ((ppl - baseline_perplexity) / baseline_perplexity) * 100.0
            tolerant = delta_pct < self.config.tolerance_pct
            prefix = f"{precision.lower()}_g{group_size}"
            row[f"{prefix}_ppl"] = ppl
            row[f"{prefix}_delta_pct"] = delta_pct
            row[f"{prefix}_tolerant"] = tolerant

            if best_combo is None or delta_pct < best_combo[2]:
                best_combo = (precision, group_size, delta_pct)

        # Backward-compatible columns expected by the policy tooling.
        row["mxfp4_tolerant"] = bool(row.get("mxfp4_g8_tolerant", False))
        row["mxfp8_tolerant"] = bool(row.get("mxfp8_g8_tolerant", False))

        if best_combo is not None:
            row["best_precision"] = best_combo[0]
            row["best_group_size"] = best_combo[1]
            row["best_delta_pct"] = best_combo[2]

        return row

    def run(self,
            output_csv: Path,
            combinations: Sequence[Tuple[str, int]]) -> pd.DataFrame:
        self.load()
        print("Computing FP baseline perplexity...")
        baseline_perplexity = self.compute_perplexity()
        print(f"Baseline perplexity: {baseline_perplexity:.4f}\n")

        layer_names = self.get_layer_names()
        print(f"Profiling {len(layer_names)} layers")
        rows: List[Dict[str, object]] = []
        for idx, name in enumerate(layer_names, start=1):
            print(f"[{idx}/{len(layer_names)}] {name}")
            row = self.profile_layer(name, baseline_perplexity, combinations)
            rows.append(row)
            for precision, group_size in combinations:
                prefix = f"{precision.lower()}_g{group_size}"
                print(
                    f"  {precision} g{group_size}: "
                    f"delta={row[f'{prefix}_delta_pct']:+.2f}% "
                    f"tolerant={'yes' if row[f'{prefix}_tolerant'] else 'no'}"
                )
            print("")

        df = pd.DataFrame(rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

        metadata = {
            "baseline_perplexity": baseline_perplexity,
            "combinations": [{"precision": p, "group_size": g} for p, g in combinations],
            "config": self.config.__dict__,
            "num_layers_profiled": len(layer_names),
        }
        output_csv.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))
        self.print_summary(df, baseline_perplexity, combinations)
        print(f"Saved sensitivity matrix to {output_csv}")
        return df

    def print_summary(self,
                      df: pd.DataFrame,
                      baseline_perplexity: float,
                      combinations: Sequence[Tuple[str, int]]) -> None:
        print("=" * 60)
        print("Profiling Summary")
        print("=" * 60)
        print(f"Baseline perplexity: {baseline_perplexity:.4f}")
        print(f"Layers profiled:     {len(df)}")
        for precision, group_size in combinations:
            prefix = f"{precision.lower()}_g{group_size}"
            tolerant = int(df[f"{prefix}_tolerant"].sum())
            avg_delta = float(df[f"{prefix}_delta_pct"].mean())
            print(
                f"{precision} g{group_size}: tolerant={tolerant}/{len(df)} "
                f"avg_delta={avg_delta:+.2f}%"
            )
        print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile layer-wise MX sensitivity.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", default="Anthropic/hh-rlhf")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--text-field", default="chosen")
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--tolerance-pct", type=float, default=2.0)
    parser.add_argument("--device", default=auto_device())
    parser.add_argument(
        "--combo",
        action="append",
        default=None,
        help="Quantization combo in PRECISION:GROUP form. Repeatable. Default profiles MXFP4/8 at g8/g16.",
    )
    parser.add_argument("--output", default="results/sensitivity_matrix.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProfileConfig(
        model_name=args.model,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        dataset_config=args.dataset_config,
        text_field=args.text_field,
        num_examples=args.num_examples,
        max_seq_len=args.max_seq_len,
        max_layers=args.max_layers,
        tolerance_pct=args.tolerance_pct,
        device=args.device,
    )
    profiler = SensitivityProfiler(config)
    profiler.run(Path(args.output), build_combinations(args.combo))


if __name__ == "__main__":
    main()
