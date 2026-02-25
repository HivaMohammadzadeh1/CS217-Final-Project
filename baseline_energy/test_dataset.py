"""
Quick test: Load HH-RLHF dataset and prepare the 1000-pair subset.
This verifies dataset access and prepares the fixed benchmark set.

Expected runtime: 1-2 minutes (first run downloads dataset)
"""

from datasets import load_dataset
import json

def main():
    print("=" * 60)
    print("Testing HH-RLHF Dataset Loading")
    print("=" * 60)

    print("\n📥 Loading Anthropic HH-RLHF dataset...")
    print("   (First run will download ~50MB)")

    # Load full training set
    dataset = load_dataset('Anthropic/hh-rlhf', split='train')
    print(f"\n✓ Dataset loaded: {len(dataset):,} total examples")

    # Create our fixed 1000-pair subset with seed=42
    print("\n🎲 Creating fixed 1000-pair subset (seed=42)...")
    dataset_subset = dataset.shuffle(seed=42).select(range(1000))
    print(f"   Selected: {len(dataset_subset)} examples")

    # Split into train (800) and eval (200)
    train_dataset = dataset_subset.select(range(800))
    eval_dataset = dataset_subset.select(range(800, 1000))

    print(f"\n📊 Dataset Split:")
    print(f"   Training set:   {len(train_dataset)} examples")
    print(f"   Evaluation set: {len(eval_dataset)} examples")

    # Show example
    print("\n" + "=" * 60)
    print("Example Preference Pair:")
    print("=" * 60)
    example = train_dataset[0]
    print(f"\nChosen response (better):\n{example['chosen'][:200]}...")
    print(f"\nRejected response (worse):\n{example['rejected'][:200]}...")

    # Save dataset info
    dataset_info = {
        "total_examples": len(dataset),
        "subset_size": len(dataset_subset),
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "seed": 42,
        "dataset_name": "Anthropic/hh-rlhf"
    }

    output_path = "results/dataset_info.json"
    with open(output_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ Dataset ready!")
    print(f"   Info saved to: {output_path}")
    print("=" * 60)

    # Quick stats
    print("\nDataset Statistics:")
    sample_examples = [train_dataset[i] for i in range(min(100, len(train_dataset)))]
    avg_chosen_len = sum(len(ex['chosen']) for ex in sample_examples) / len(sample_examples)
    avg_rejected_len = sum(len(ex['rejected']) for ex in sample_examples) / len(sample_examples)
    print(f"   Avg chosen response length:   {avg_chosen_len:.0f} chars")
    print(f"   Avg rejected response length: {avg_rejected_len:.0f} chars")

if __name__ == "__main__":
    main()
