# Fixed HH-RLHF Dataset Created ✅

**Date**: February 25, 2026
**Status**: Dataset ready for experiments

## What Was Created

### Dataset Composition
- **Training Set**: 1,000 samples from `Anthropic/hh-rlhf` train split
- **Test Set**: 200 samples from `Anthropic/hh-rlhf` test split
- **Random Seed**: 42 (for reproducibility)
- **Total Size**: 1,200 examples

### Key Features
✅ Proper train/test separation (train samples from train split, test samples from test split)
✅ Fixed seed ensures reproducibility across all experiments
✅ Saved locally to `data/cs217_rlhf_dataset/`
✅ Metadata file included with dataset information
✅ Ready for HuggingFace Hub upload

---

## Dataset Structure

```
data/cs217_rlhf_dataset/
├── train/
│   └── data-00000-of-00001.arrow    # 1000 training examples
├── test/
│   └── data-00000-of-00001.arrow    # 200 test examples
├── dataset_info.json                # Dataset schema info
└── metadata.json                    # Custom metadata (source, seed, sizes)
```

---

## Metadata

```json
{
  "dataset_name": "CS217 Fixed HH-RLHF Dataset",
  "source": "Anthropic/hh-rlhf",
  "train_size": 1000,
  "test_size": 200,
  "total_size": 1200,
  "seed": 42,
  "description": "Fixed dataset for CS217 final project RLHF experiments. Training samples from train split, test samples from test split."
}
```

---

## How to Use the Dataset

### Option 1: Load Locally (Current Setup)

```python
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("data/cs217_rlhf_dataset")

# Access splits
train_data = dataset['train']
test_data = dataset['test']

print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")
```

### Option 2: Upload to HuggingFace Hub (Recommended for Sharing)

#### Step 1: Authenticate with HuggingFace
```bash
# Install HuggingFace CLI (if not already installed)
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login
```

#### Step 2: Upload the Dataset
```bash
python baseline_energy/create_fixed_dataset.py \
  --push-to-hub \
  --hub-name YOUR_USERNAME/cs217-rlhf-dataset
```

Replace `YOUR_USERNAME` with your HuggingFace username.

#### Step 3: Load from Hub (After Upload)
```python
from datasets import load_dataset

# Load from HuggingFace Hub
dataset = load_dataset("YOUR_USERNAME/cs217-rlhf-dataset")

train_data = dataset['train']
test_data = dataset['test']
```

---

## Example Data Format

### Training Example
```python
{
  'chosen': '\n\nH: Why did cells originally combine together to create life?\n\nA: Because their simple components -- chemicals -- interacted in particular ways...',
  'rejected': '\n\nH: Why did cells originally combine together to create life?\n\nA: Cells combine because they benefit from cooperation...'
}
```

### Test Example
```python
{
  'chosen': '\n\nH: Can you give me facts about jumping spiders?\n\nA: Sure, here are some fun facts about jumping spiders...',
  'rejected': '\n\nH: Can you give me facts about jumping spiders?\n\nA: Sure, here are some fun facts...'
}
```

---

## Integration with Experiments

### Update Config Files

All experiment scripts can now use the fixed dataset:

```python
# In config.py or script setup
from datasets import load_from_disk

# Load fixed dataset
dataset = load_from_disk("data/cs217_rlhf_dataset")
train_dataset = dataset['train']
test_dataset = dataset['test']

# No more random shuffling needed!
# All experiments will use the exact same data
```

### RLHF Baseline Script
The dataset is already compatible with `baseline_energy/rlhf_baseline.py`:

```bash
python baseline_energy/rlhf_baseline.py \
  --dataset-path data/cs217_rlhf_dataset \
  --output results/baseline_run1
```

### Sensitivity Profiler
The profiler can use the test set for evaluation:

```bash
python pytorch_profiling/sensitivity_profiler.py \
  --eval-dataset data/cs217_rlhf_dataset \
  --eval-split test \
  --output results/sensitivity_matrix.csv
```

---

## Benefits of Fixed Dataset

### Reproducibility ✅
- All experiments use the same data
- Results are directly comparable
- No variation from random shuffling

### Proper Evaluation ✅
- Test set is truly held-out (from separate split)
- No data leakage between train and test
- Follows best practices for ML experiments

### Easy Sharing ✅
- Can upload to HuggingFace Hub
- Others can reproduce your results
- Consistent baseline for comparisons

---

## Next Steps

### 1. Local Development (No Upload Needed)
If you're just running experiments locally:
- ✅ Dataset is ready to use
- ✅ Already saved in `data/cs217_rlhf_dataset/`
- ✅ Just use `load_from_disk()` in your scripts

### 2. Collaborative Work (Upload to Hub)
If you want to share with others or use on different machines:

```bash
# Login to HuggingFace
huggingface-cli login

# Upload the dataset
python baseline_energy/create_fixed_dataset.py \
  --push-to-hub \
  --hub-name YOUR_USERNAME/cs217-rlhf-dataset
```

Then anyone can access it with:
```python
dataset = load_dataset("YOUR_USERNAME/cs217-rlhf-dataset")
```

### 3. Run Experiments
Now that the dataset is fixed, you can run experiments:

```bash
source venv/bin/activate

# Run RLHF baseline
python baseline_energy/rlhf_baseline.py \
  --steps 100 \
  --output results/baseline_run1

# Run sensitivity profiling
python pytorch_profiling/sensitivity_profiler.py \
  --output results/sensitivity_matrix.csv
```

---

## Verification

To verify the dataset was created correctly:

```python
from datasets import load_from_disk

dataset = load_from_disk("data/cs217_rlhf_dataset")

# Check sizes
assert len(dataset['train']) == 1000, "Train set should have 1000 examples"
assert len(dataset['test']) == 200, "Test set should have 200 examples"

# Check structure
example = dataset['train'][0]
assert 'chosen' in example, "Should have 'chosen' field"
assert 'rejected' in example, "Should have 'rejected' field"

print("✅ Dataset verified successfully!")
print(f"Train: {len(dataset['train'])} examples")
print(f"Test: {len(dataset['test'])} examples")
```

---

## Summary

✅ **Dataset Created**: 1,000 train + 200 test samples
✅ **Seed Fixed**: Reproducible across all experiments
✅ **Proper Splits**: Train from train, test from test
✅ **Saved Locally**: `data/cs217_rlhf_dataset/`
✅ **Ready for Upload**: Can push to HuggingFace Hub
✅ **Milestone 2 Ready**: All scripts can use this dataset

---

**Status**: ✅ **Dataset ready for Milestone 2 experiments!**

All RLHF training and profiling can now use this fixed dataset for reproducible results.
