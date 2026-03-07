"""
Configuration for RLHF baseline experiments.
All hyperparameters and settings in one place.
"""

# Model Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
REWARD_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Same base, will add scalar head

# Dataset Configuration
# Use fixed dataset for reproducible experiments
DATASET_NAME = "hivamoh/cs217-rlhf-dataset"  # Fixed HH-RLHF dataset on HuggingFace Hub
USE_HUB_DATASET = True  # Set to False to use local dataset
LOCAL_DATASET_PATH = "data/cs217_rlhf_dataset"  # Path if using local dataset
DATASET_SPLIT = "train"  # Use train split
NUM_SAMPLES = 1000  # Total samples to use (800 train + 200 eval)
TRAIN_SIZE = 800  # Training samples
EVAL_SIZE = 200  # Evaluation samples
RANDOM_SEED = 42  # Fixed seed used to create the dataset

# Sequence Configuration
MAX_SEQ_LENGTH = 512  # Cap to reduce memory traffic
MAX_PROMPT_LENGTH = 256
MAX_RESPONSE_LENGTH = 64

# PPO Training Configuration
NUM_PPO_STEPS = 100  # For full baseline measurement
BATCH_SIZE = 8
MINI_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2

# PPO Hyperparameters
LEARNING_RATE = 1e-6
PPO_EPOCHS = 1
CLIP_RANGE = 0.2
VALUE_COEF = 0.1
KL_PENALTY = 0.02  # KL divergence penalty
GAMMA = 1.0  # Discount factor
LAM = 0.95  # GAE lambda

# Energy Measurement Configuration
POWER_LOG_FILE = "power_log_baseline.csv"
POWER_SAMPLING_INTERVAL_MS = 100  # 10 samples/second
RESULTS_DIR = "results/gpu_baseline"

# Device Configuration
USE_GPU = False  # f2.6xlarge has NO GPU - set to False
FP16 = False  # Disable FP16 when no GPU

# Logging Configuration
LOG_EVERY_N_STEPS = 10
SAVE_CHECKPOINTS = False  # Don't save model checkpoints for baseline run
VERBOSE = True

# Phase Timing (what to measure)
MEASURE_PHASES = True  # Track rollout/reward/gradient separately

# FPGA Offload Configuration
USE_FPGA_OFFLOAD = True  # DISABLE for pure CPU training (no mock FPGA)
USE_MOCK_FPGA = False  # Set to True for mock FPGA testing (software fallback)
USE_LAB1_FPGA = True  # Use Lab 1 FPGA hardware (16×16 matmul accelerator) when USE_FPGA_OFFLOAD=True and USE_MOCK_FPGA=False
FPGA_DEVICE_ID = 0  # FPGA device ID (for AWS F2 - typically slot 0)
FPGA_VERBOSE = False  # Print detailed FPGA offload info
FPGA_PRECISION_MODE = "INT8"  # INT8, MXFP8, or MXFP4
FPGA_GROUP_SIZE = 8  # Shared scaling group size for MX modes (8 or 16)
FPGA_POLICY_JSON = None  # Optional path to generated policy JSON for phase-aware precision
FPGA_POLICY_NAME = None  # Optional named policy to load from FPGA_POLICY_JSON (A/B/C/D)
FPGA_ALLOW_GRADIENT_OFFLOAD = False  # Current FPGA path is inference-oriented; keep gradients on native PyTorch by default
FPGA_POLICY_BLOCKS = [0, 23]  # Which transformer blocks to FPGA-offload in policy/ref models (first + last block)
FPGA_RESPONSE_LENGTH = 16  # Max new tokens during generation when FPGA is active on policy model

# Reward Model Pre-training
PRETRAIN_REWARD_STEPS = 100  # Steps to fine-tune reward head on preference pairs before RLHF
PRETRAIN_REWARD_LR = 5e-4  # Learning rate for reward head pre-training
