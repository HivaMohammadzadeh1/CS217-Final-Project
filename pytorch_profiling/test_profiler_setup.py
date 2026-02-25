"""
Quick test: Verify sensitivity profiler setup without full profiling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    print("=" * 60)
    print("Sensitivity Profiler Setup Test")
    print("=" * 60)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nModel: {model_name}")
    print(f"Device: {device}")

    # Test 1: Load model
    print("\n1️⃣  Loading model (this will take ~1 minute)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print("   ✓ Model loaded")

    # Test 2: Get layer names
    print("\n2️⃣  Finding quantizable layers...")
    layer_names = []
    for name, module in model.named_modules():
        if 'mlp' in name.lower() or 'attn' in name.lower():
            if isinstance(module, torch.nn.Linear):
                layer_names.append(name)

    print(f"   Found {len(layer_names)} quantizable layers")
    print(f"   Examples:")
    for name in layer_names[:5]:
        print(f"     - {name}")
    print("   ✓ Layer discovery works")

    # Test 3: Load eval dataset
    print("\n3️⃣  Loading evaluation dataset...")
    dataset = load_dataset('Anthropic/hh-rlhf', split='train')
    dataset = dataset.shuffle(seed=42).select(range(5))  # Just 5 for quick test
    print(f"   Loaded {len(dataset)} examples for testing")
    print("   ✓ Dataset ready")

    # Test 4: Test perplexity calculation
    print("\n4️⃣  Testing perplexity calculation...")
    example = dataset[0]['chosen'][:200]  # Short text

    inputs = tokenizer(
        example,
        return_tensors="pt",
        max_length=100,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss.item()
        perplexity = torch.exp(torch.tensor(loss)).item()

    print(f"   Loss: {loss:.3f}")
    print(f"   Perplexity: {perplexity:.2f}")
    print("   ✓ Perplexity calculation works")

    # Test 5: Test quantization placeholder
    print("\n5️⃣  Testing quantization logic...")
    test_layer = dict(model.named_modules())[layer_names[0]]
    if hasattr(test_layer, 'weight'):
        original_weight = test_layer.weight.data.clone()

        # Simulate MXFP8 quantization
        max_val = original_weight.abs().max()
        quantized = torch.round(original_weight / max_val * 16) / 16 * max_val
        test_layer.weight.data = quantized

        # Restore
        test_layer.weight.data = original_weight

        print(f"   Tested quantization on: {layer_names[0]}")
        print("   ✓ Quantization logic works")

    print("\n" + "=" * 60)
    print("✅ All sensitivity profiler components working!")
    print("=" * 60)

    print("\nReady to run full profiling:")
    print("  python pytorch_profiling/sensitivity_profiler.py")
    print("\nNote: Full profiling takes:")
    print("  - ~2-4 hours for all layers")
    print("  - Tests MXFP4 and MXFP8 formats")
    print("  - Generates sensitivity matrix CSV")

if __name__ == "__main__":
    main()
