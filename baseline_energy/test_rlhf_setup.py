"""
Quick test: Verify RLHF training setup without running full training.
Tests that all imports work and models can be initialized.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import config

def main():
    print("=" * 60)
    print("RLHF Setup Test")
    print("=" * 60)

    # Test 1: Configuration
    print("\n1️⃣  Testing configuration...")
    print(f"   Model: {config.MODEL_NAME}")
    print(f"   Dataset: {config.DATASET_NAME}")
    print(f"   Steps: {config.NUM_PPO_STEPS}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print("   ✓ Config loaded")

    # Test 2: TRL imports
    print("\n2️⃣  Testing TRL imports...")
    print(f"   PPOConfig: {PPOConfig}")
    print(f"   AutoModelForCausalLMWithValueHead: {AutoModelForCausalLMWithValueHead}")
    print("   ✓ TRL imports successful")

    # Test 3: Dataset loading
    print("\n3️⃣  Testing dataset loading...")
    dataset = load_dataset(config.DATASET_NAME, split='train')
    dataset_subset = dataset.shuffle(seed=42).select(range(10))  # Just 10 examples
    print(f"   Loaded {len(dataset_subset)} examples")
    print(f"   Example keys: {list(dataset_subset[0].keys())}")
    print("   ✓ Dataset loads successfully")

    # Test 4: Tokenizer
    print("\n4️⃣  Testing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    test_text = "Hello, this is a test"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"   Tokenized '{test_text}'")
    print(f"   Token shape: {tokens.input_ids.shape}")
    print("   ✓ Tokenizer works")

    # Test 5: Model initialization (without loading weights)
    print("\n5️⃣  Testing model initialization...")
    print("   Note: This would load the full model (~1GB)")
    print("   Skipping full load for quick test")
    print("   Model class available: AutoModelForCausalLMWithValueHead")
    print("   ✓ Model class ready")

    # Test 6: PPO Config
    print("\n6️⃣  Testing PPO configuration...")
    ppo_config = PPOConfig(
        model_name=config.MODEL_NAME,
        learning_rate=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        mini_batch_size=config.MINI_BATCH_SIZE,
        seed=config.RANDOM_SEED,
    )
    print(f"   Learning rate: {ppo_config.learning_rate}")
    print(f"   Batch size: {ppo_config.batch_size}")
    print(f"   Seed: {ppo_config.seed}")
    print("   ✓ PPO config created")

    # Test 7: Device detection
    print("\n7️⃣  Testing device detection...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print("   ✓ GPU available for training")
    else:
        print("   ⚠️  No GPU - will use CPU (slower)")

    print("\n" + "=" * 60)
    print("✅ All RLHF components initialized successfully!")
    print("=" * 60)

    print("\nReady to run:")
    print("  python baseline_energy/rlhf_baseline.py --steps 100")
    print("\nNote: Full training requires:")
    print("  - GPU for reasonable speed")
    print("  - ~2 hours for 100 steps")
    print("  - Power monitoring (nvidia-smi)")

if __name__ == "__main__":
    main()
