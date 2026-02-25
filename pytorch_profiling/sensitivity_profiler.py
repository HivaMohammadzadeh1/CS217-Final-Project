"""
Layer Sensitivity Profiler for MX Format Quantization

Tests each layer with MXFP4 and MXFP8 to measure quality degradation.
Outputs a sensitivity matrix showing which layers tolerate which formats.

Usage:
    python pytorch_profiling/sensitivity_profiler.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import torch
import argparse
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np


class SensitivityProfiler:
    """Profile layer-wise sensitivity to MX quantization."""

    def __init__(self, model_name, device="cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"\n{'='*60}")
        print("MX Format Sensitivity Profiler")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        # Load model and tokenizer
        print("📥 Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.model.eval()
        print("✓ Model loaded\n")

        # Load evaluation dataset
        print("📊 Loading evaluation dataset...")
        self.eval_dataset = self.load_eval_dataset()
        print(f"✓ Loaded {len(self.eval_dataset)} examples\n")

    def load_eval_dataset(self):
        """Load a small evaluation set for perplexity measurement."""
        # Use HH-RLHF for consistency
        dataset = load_dataset('Anthropic/hh-rlhf', split='train')
        # Take 100 examples for quick eval
        dataset = dataset.shuffle(seed=42).select(range(100))
        return dataset

    def compute_perplexity(self):
        """
        Compute perplexity on evaluation set.

        Returns:
            float: Perplexity score
        """
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for example in self.eval_dataset:
                # Use 'chosen' response
                text = example['chosen'][:512]  # Limit length

                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                ).to(self.device)

                # Forward pass
                outputs = self.model(**inputs, labels=inputs.input_ids)

                # Accumulate loss
                total_loss += outputs.loss.item() * inputs.input_ids.size(1)
                total_tokens += inputs.input_ids.size(1)

        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity

    def get_layer_names(self):
        """Extract names of quantizable layers."""
        layer_names = []

        for name, module in self.model.named_modules():
            # Focus on transformer layers
            if 'mlp' in name.lower() or 'attn' in name.lower():
                # Only include Linear layers
                if isinstance(module, torch.nn.Linear):
                    layer_names.append(name)

        return layer_names

    def quantize_layer_mxfp4(self, layer_name, group_size=8):
        """
        Quantize a specific layer to MXFP4 format.

        Note: This is a placeholder. Real implementation requires mx-pytorch library.
        For now, we simulate quantization with simple rounding.

        Args:
            layer_name: Name of layer to quantize
            group_size: MX group size (8 or 16)
        """
        # Placeholder: In real implementation, use mx.quantize()
        # For now, we'll just reduce precision by rounding
        print(f"  [Placeholder] Quantizing {layer_name} to MXFP4 (group_size={group_size})")

        # Get the layer
        layer = dict(self.model.named_modules())[layer_name]

        if hasattr(layer, 'weight'):
            # Simulate aggressive quantization by reducing dynamic range
            original_weight = layer.weight.data.clone()
            # Simple quantization: reduce to 4-bit equivalent dynamic range
            max_val = original_weight.abs().max()
            # MXFP4 has ~2 bits mantissa, so ~4 levels per exponent
            quantized = torch.round(original_weight / max_val * 8) / 8 * max_val
            layer.weight.data = quantized

            return original_weight

        return None

    def quantize_layer_mxfp8(self, layer_name, group_size=8):
        """
        Quantize a specific layer to MXFP8 format.

        Placeholder implementation.
        """
        print(f"  [Placeholder] Quantizing {layer_name} to MXFP8 (group_size={group_size})")

        layer = dict(self.model.named_modules())[layer_name]

        if hasattr(layer, 'weight'):
            original_weight = layer.weight.data.clone()
            # MXFP8 has ~3 bits mantissa, so ~8 levels per exponent
            max_val = original_weight.abs().max()
            quantized = torch.round(original_weight / max_val * 16) / 16 * max_val
            layer.weight.data = quantized

            return original_weight

        return None

    def restore_layer(self, layer_name, original_weight):
        """Restore original weights to a layer."""
        if original_weight is None:
            return

        layer = dict(self.model.named_modules())[layer_name]
        if hasattr(layer, 'weight'):
            layer.weight.data = original_weight

    def profile_layer(self, layer_name, baseline_perplexity):
        """
        Profile sensitivity of a single layer to quantization.

        Args:
            layer_name: Name of layer to test
            baseline_perplexity: FP16 baseline perplexity

        Returns:
            Dictionary with results for this layer
        """
        results = {
            'layer': layer_name,
        }

        # Test MXFP4 with group_size=8
        original_weight = self.quantize_layer_mxfp4(layer_name, group_size=8)
        ppl_mxfp4_g8 = self.compute_perplexity()
        delta_mxfp4_g8 = ((ppl_mxfp4_g8 - baseline_perplexity) / baseline_perplexity) * 100
        results['mxfp4_g8_ppl'] = ppl_mxfp4_g8
        results['mxfp4_g8_delta_pct'] = delta_mxfp4_g8
        self.restore_layer(layer_name, original_weight)

        # Test MXFP8 with group_size=8
        original_weight = self.quantize_layer_mxfp8(layer_name, group_size=8)
        ppl_mxfp8_g8 = self.compute_perplexity()
        delta_mxfp8_g8 = ((ppl_mxfp8_g8 - baseline_perplexity) / baseline_perplexity) * 100
        results['mxfp8_g8_ppl'] = ppl_mxfp8_g8
        results['mxfp8_g8_delta_pct'] = delta_mxfp8_g8
        self.restore_layer(layer_name, original_weight)

        # Determine if layer is "tolerant"
        # Threshold: < 2% perplexity increase
        results['mxfp4_tolerant'] = delta_mxfp4_g8 < 2.0
        results['mxfp8_tolerant'] = delta_mxfp8_g8 < 2.0

        return results

    def run_profiling(self, output_path="results/sensitivity_matrix.csv"):
        """
        Run full sensitivity profiling on all layers.

        Args:
            output_path: Where to save results CSV
        """
        print("🔍 Starting sensitivity profiling...")

        # Compute baseline perplexity (FP16)
        print("\n1️⃣  Computing FP16 baseline perplexity...")
        baseline_perplexity = self.compute_perplexity()
        print(f"   Baseline perplexity (FP16): {baseline_perplexity:.2f}\n")

        # Get all quantizable layers
        layer_names = self.get_layer_names()
        print(f"2️⃣  Found {len(layer_names)} quantizable layers\n")

        # Profile each layer
        print("3️⃣  Profiling layers...\n")
        all_results = []

        for i, layer_name in enumerate(layer_names[:10]):  # Limit to first 10 for demo
            print(f"  [{i+1}/{min(10, len(layer_names))}] {layer_name}")
            results = self.profile_layer(layer_name, baseline_perplexity)
            all_results.append(results)

            print(f"    MXFP4: Δ{results['mxfp4_g8_delta_pct']:+.2f}% "
                  f"({'✓ tolerant' if results['mxfp4_tolerant'] else '✗ sensitive'})")
            print(f"    MXFP8: Δ{results['mxfp8_g8_delta_pct']:+.2f}% "
                  f"({'✓ tolerant' if results['mxfp8_tolerant'] else '✗ sensitive'})")
            print()

        # Save results
        print(f"💾 Saving results to {output_path}...")
        df = pd.DataFrame(all_results)

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"✓ Sensitivity matrix saved!\n")

        # Print summary
        self.print_summary(df, baseline_perplexity)

    def print_summary(self, df, baseline_perplexity):
        """Print summary of profiling results."""
        print(f"{'='*60}")
        print("Sensitivity Profiling Summary")
        print(f"{'='*60}")
        print(f"Baseline Perplexity (FP16): {baseline_perplexity:.2f}")
        print(f"Layers profiled: {len(df)}")
        print()

        mxfp4_tolerant = df['mxfp4_tolerant'].sum()
        mxfp8_tolerant = df['mxfp8_tolerant'].sum()

        print(f"MXFP4 tolerant layers: {mxfp4_tolerant}/{len(df)} "
              f"({mxfp4_tolerant/len(df)*100:.1f}%)")
        print(f"MXFP8 tolerant layers: {mxfp8_tolerant}/{len(df)} "
              f"({mxfp8_tolerant/len(df)*100:.1f}%)")
        print()

        print("Average perplexity increase:")
        print(f"  MXFP4: {df['mxfp4_g8_delta_pct'].mean():+.2f}%")
        print(f"  MXFP8: {df['mxfp8_g8_delta_pct'].mean():+.2f}%")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="MX Format Sensitivity Profiler")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model to profile",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/sensitivity_matrix.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Run profiling
    profiler = SensitivityProfiler(args.model)
    profiler.run_profiling(args.output)

    print("✅ Sensitivity profiling complete!")
    print("\n⚠️  Note: This uses placeholder quantization.")
    print("   Install mx-pytorch for real MX format support:")
    print("   pip install git+https://github.com/microsoft/microxcaling.git")


if __name__ == "__main__":
    main()
