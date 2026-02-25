"""
Quick test: Load Qwen2.5-0.5B model and run a simple inference.
This verifies that the model downloads correctly and can generate text.

Expected runtime: 2-5 minutes (first run will download ~1GB model)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def main():
    print("=" * 60)
    print("Testing Qwen2.5-0.5B Model Loading")
    print("=" * 60)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"\n📥 Loading model: {model_name}")
    print("   (First run will download ~1GB - may take a few minutes)")

    start_time = time.time()

    # Load tokenizer
    print("\n1️⃣  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("   ✓ Tokenizer loaded")

    # Load model in FP16 (half precision) to save memory
    print("\n2️⃣  Loading model (FP16)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    if device == "cpu":
        model = model.to(device)

    print(f"   ✓ Model loaded ({time.time() - start_time:.1f}s)")

    # Test inference
    print("\n3️⃣  Running test inference...")
    test_prompt = "Hello! Can you help me understand reinforcement learning?"

    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    print(f"   Prompt: \"{test_prompt}\"")
    print(f"   Generating response...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False  # Greedy decoding for reproducibility
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n   Response:\n   {response}\n")

    # Model info
    print("=" * 60)
    print("Model Information:")
    print("=" * 60)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    model_size_mb = num_params * 2 / (1024**2)  # FP16 = 2 bytes per param
    print(f"Model size (FP16): ~{model_size_mb:.0f} MB")
    print(f"Total load time: {time.time() - start_time:.1f}s")

    print("\n✅ Model test successful! Ready for RLHF training.")
    print("=" * 60)

if __name__ == "__main__":
    main()
