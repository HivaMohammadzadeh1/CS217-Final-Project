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
TRAIN_SPLIT = "train"  # 1000 samples
TEST_SPLIT = "test"  # 200 samples
RANDOM_SEED = 42  # Fixed seed used to create the dataset

# Sequence Configuration
MAX_SEQ_LENGTH = 512  # Cap to reduce memory traffic
MAX_PROMPT_LENGTH = 256
MAX_RESPONSE_LENGTH = 256

# PPO Training Configuration
NUM_PPO_STEPS = 100  # For full baseline measurement
BATCH_SIZE = 8
MINI_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2

# PPO Hyperparameters
LEARNING_RATE = 1.41e-5
PPO_EPOCHS = 4
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
USE_GPU = True  # Set to False for CPU-only testing
FP16 = True  # Use FP16 for baseline (standard practice)

# Logging Configuration
LOG_EVERY_N_STEPS = 10
SAVE_CHECKPOINTS = False  # Don't save model checkpoints for baseline run
VERBOSE = True

# Phase Timing (what to measure)
MEASURE_PHASES = True  # Track rollout/reward/gradient separately
