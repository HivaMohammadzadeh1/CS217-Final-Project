"""
Quick verification: Load the fixed dataset from HuggingFace Hub.
"""

from datasets import load_dataset

def main():
    print("=" * 60)
    print("Verifying HuggingFace Hub Dataset")
    print("=" * 60)

    hub_name = "hivamoh/cs217-rlhf-dataset"

    print(f"\nLoading dataset from: {hub_name}")
    print("This will download the dataset if not cached...")

    # Load from Hub
    dataset = load_dataset(hub_name)

    print("\n" + "=" * 60)
    print("Dataset Loaded Successfully!")
    print("=" * 60)

    # Verify splits
    print(f"\n✓ Available splits: {list(dataset.keys())}")
    print(f"✓ Train size: {len(dataset['train'])}")
    print(f"✓ Test size: {len(dataset['test'])}")

    # Show example
    print("\n" + "=" * 60)
    print("Example from Training Set:")
    print("=" * 60)
    example = dataset['train'][0]
    print(f"\nChosen: {example['chosen'][:150]}...")
    print(f"\nRejected: {example['rejected'][:150]}...")

    # Verify structure
    print("\n" + "=" * 60)
    print("Dataset Features:")
    print("=" * 60)
    print(dataset['train'].features)

    print("\n" + "=" * 60)
    print("✅ Dataset Verification Complete!")
    print("=" * 60)
    print(f"\nDataset URL: https://huggingface.co/datasets/{hub_name}")
    print("\nTo use in your experiments:")
    print(f"  from datasets import load_dataset")
    print(f"  dataset = load_dataset('{hub_name}')")
    print(f"  train_data = dataset['train']")
    print(f"  test_data = dataset['test']")

if __name__ == "__main__":
    main()
