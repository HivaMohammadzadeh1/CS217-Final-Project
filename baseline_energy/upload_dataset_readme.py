"""
Upload README.md to the HuggingFace Hub dataset repository.
"""

from huggingface_hub import HfApi
from pathlib import Path

def main():
    repo_id = "hivamoh/cs217-rlhf-dataset"
    readme_path = "data/cs217_rlhf_dataset/README.md"

    print("=" * 60)
    print("Uploading README to HuggingFace Hub")
    print("=" * 60)
    print(f"\nRepository: {repo_id}")
    print(f"README path: {readme_path}")

    # Check if README exists
    if not Path(readme_path).exists():
        print(f"\n❌ README not found at: {readme_path}")
        return

    # Upload README
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add comprehensive README for dataset"
        )

        print("\n✅ README uploaded successfully!")
        print(f"\nView at: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"\n❌ Error uploading README: {e}")
        print("\nMake sure you're logged in:")
        print("  huggingface-cli login")
        return

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
