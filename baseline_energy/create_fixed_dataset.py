"""
Create a fixed HH-RLHF dataset subset for reproducible experiments.

This script:
1. Loads Anthropic/hh-rlhf dataset (both train and test splits)
2. Samples 1000 examples from train split (seed=42)
3. Samples 200 examples from test split (seed=42)
4. Saves locally and optionally uploads to HuggingFace Hub

Usage:
    python baseline_energy/create_fixed_dataset.py [--push-to-hub] [--hub-name YOUR_USERNAME/cs217-rlhf-dataset]
"""

import argparse
from datasets import load_dataset, DatasetDict
from pathlib import Path
import json


def create_fixed_dataset(seed=42, train_size=1000, test_size=200):
    """
    Create fixed train/test split from HH-RLHF.
    Takes training samples from the train split and test samples from the test split.

    Args:
        seed: Random seed for reproducibility
        train_size: Number of training examples
        test_size: Number of test examples

    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    print("=" * 60)
    print("Creating Fixed HH-RLHF Dataset")
    print("=" * 60)

    total_size = train_size + test_size
    print(f"\nConfiguration:")
    print(f"  Total samples: {total_size}")
    print(f"  Train: {train_size} (from train split)")
    print(f"  Test: {test_size} (from test split)")
    print(f"  Random seed: {seed}")

    # Load train split
    print("\n1️⃣  Loading train split from Anthropic/hh-rlhf...")
    full_train = load_dataset('Anthropic/hh-rlhf', split='train')
    print(f"   Full train split size: {len(full_train):,} examples")

    # Load test split
    print("\n2️⃣  Loading test split from Anthropic/hh-rlhf...")
    full_test = load_dataset('Anthropic/hh-rlhf', split='test')
    print(f"   Full test split size: {len(full_test):,} examples")

    # Create fixed train subset with seed
    print(f"\n3️⃣  Selecting {train_size} samples from train split (seed={seed})...")
    train_dataset = full_train.shuffle(seed=seed).select(range(train_size))
    print(f"   ✓ Selected {len(train_dataset)} training examples")

    # Create fixed test subset with seed
    print(f"\n4️⃣  Selecting {test_size} samples from test split (seed={seed})...")
    test_dataset = full_test.shuffle(seed=seed).select(range(test_size))
    print(f"   ✓ Selected {len(test_dataset)} test examples")

    print(f"\n✓ Total dataset size: {len(train_dataset) + len(test_dataset)} examples")

    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    # Show examples
    print("\n" + "=" * 60)
    print("Example from Training Set:")
    print("=" * 60)
    train_example = train_dataset[0]
    print(f"\nChosen (better response):")
    print(f"{train_example['chosen'][:200]}...")
    print(f"\nRejected (worse response):")
    print(f"{train_example['rejected'][:200]}...")

    print("\n" + "=" * 60)
    print("Example from Test Set:")
    print("=" * 60)
    test_example = test_dataset[0]
    print(f"\nChosen (better response):")
    print(f"{test_example['chosen'][:200]}...")
    print(f"\nRejected (worse response):")
    print(f"{test_example['rejected'][:200]}...")

    return dataset_dict


def save_dataset_locally(dataset_dict, output_dir="data/cs217_rlhf_dataset"):
    """Save dataset to local directory."""
    print("\n" + "=" * 60)
    print("Saving Dataset Locally")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving to: {output_path}")
    dataset_dict.save_to_disk(str(output_path))
    print("   ✓ Dataset saved")

    # Save metadata
    metadata = {
        "dataset_name": "CS217 Fixed HH-RLHF Dataset",
        "source": "Anthropic/hh-rlhf",
        "train_size": len(dataset_dict['train']),
        "test_size": len(dataset_dict['test']),
        "total_size": len(dataset_dict['train']) + len(dataset_dict['test']),
        "seed": 42,
        "description": "Fixed dataset for CS217 final project RLHF experiments. Training samples from train split, test samples from test split."
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✓ Metadata saved to: {metadata_path}")

    return output_path


def push_to_hub(dataset_dict, hub_name, token=None):
    """
    Push dataset to HuggingFace Hub.

    Args:
        dataset_dict: DatasetDict to upload
        hub_name: Repository name (e.g., "username/cs217-rlhf-dataset")
        token: HuggingFace token (optional, will prompt if not provided)
    """
    print("\n" + "=" * 60)
    print("Pushing to HuggingFace Hub")
    print("=" * 60)

    print(f"\n📤 Uploading to: {hub_name}")
    print("\nNote: You'll need to authenticate with HuggingFace.")
    print("      Run: huggingface-cli login")
    print("      Or pass --token YOUR_TOKEN")

    try:
        dataset_dict.push_to_hub(
            hub_name,
            token=token,
            private=False  # Make it public so experiments can access it
        )
        print(f"\n✅ Dataset uploaded successfully!")
        print(f"   URL: https://huggingface.co/datasets/{hub_name}")
        print(f"\nTo use this dataset in experiments:")
        print(f"   from datasets import load_dataset")
        print(f"   dataset = load_dataset('{hub_name}')")

    except Exception as e:
        print(f"\n❌ Error uploading to Hub: {e}")
        print("\nTroubleshooting:")
        print("1. Run: huggingface-cli login")
        print("2. Or pass --token argument")
        print("3. Ensure you have write permissions")
        return False

    return True


def load_fixed_dataset(source="local", hub_name=None, local_path="data/cs217_rlhf_dataset"):
    """
    Load the fixed dataset from local or Hub.

    Args:
        source: 'local' or 'hub'
        hub_name: HuggingFace Hub name if source='hub'
        local_path: Local path if source='local'

    Returns:
        DatasetDict
    """
    if source == "hub":
        if hub_name is None:
            raise ValueError("hub_name required when source='hub'")
        print(f"Loading from HuggingFace Hub: {hub_name}")
        return load_dataset(hub_name)
    else:
        print(f"Loading from local: {local_path}")
        from datasets import load_from_disk
        return load_from_disk(local_path)


def main():
    parser = argparse.ArgumentParser(description="Create fixed HH-RLHF dataset")
    parser.add_argument(
        "--train-size",
        type=int,
        default=1000,
        help="Number of training examples (default: 1000)"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Number of test examples (default: 200)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/cs217_rlhf_dataset",
        help="Local output directory"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub-name",
        type=str,
        default=None,
        help="HuggingFace Hub repository name (e.g., username/cs217-rlhf-dataset)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional)"
    )

    args = parser.parse_args()

    # Create dataset
    dataset_dict = create_fixed_dataset(
        seed=args.seed,
        train_size=args.train_size,
        test_size=args.test_size
    )

    # Save locally
    output_path = save_dataset_locally(dataset_dict, args.output_dir)

    # Optionally push to Hub
    if args.push_to_hub:
        if args.hub_name is None:
            print("\n⚠️  --hub-name required when using --push-to-hub")
            print("   Example: --hub-name your-username/cs217-rlhf-dataset")
        else:
            push_to_hub(dataset_dict, args.hub_name, args.token)
    else:
        print("\n" + "=" * 60)
        print("Dataset saved locally only")
        print("=" * 60)
        print(f"\nLocal path: {output_path}")
        print("\nTo upload to HuggingFace Hub later, run:")
        print(f"  python baseline_energy/create_fixed_dataset.py \\")
        print(f"    --push-to-hub \\")
        print(f"    --hub-name YOUR_USERNAME/cs217-rlhf-dataset")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Created fixed dataset")
    print(f"✓ Train: {args.train_size} examples")
    print(f"✓ Test: {args.test_size} examples")
    print(f"✓ Seed: {args.seed}")
    print(f"✓ Saved to: {args.output_dir}")
    if args.push_to_hub and args.hub_name:
        print(f"✓ Uploaded to: {args.hub_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
