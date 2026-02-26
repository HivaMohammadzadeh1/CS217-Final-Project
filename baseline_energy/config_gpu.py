"""
GPU Configuration for RLHF training.
Use this on g4dn.xlarge or other GPU instances.

To use:
  cp config_gpu.py config.py
OR:
  python baseline_energy/rlhf_with_fpga.py --config config_gpu
"""

# Model Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
REWARD_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Dataset Configuration
DATASET_NAME = "hivamoh/cs217-rlhf-dataset"
USE_HUB_DATASET = True
LOCAL_DATASET_PATH = "data/cs217_rlhf_dataset"
DATASET_SPLIT = "train"
NUM_SAMPLES = 1000
TRAIN_SIZE = 800
EVAL_SIZE = 200
RANDOM_SEED = 42

# Sequence Configuration
MAX_SEQ_LENGTH = 512
MAX_PROMPT_LENGTH = 256
MAX_RESPONSE_LENGTH = 256

# PPO Training Configuration
NUM_PPO_STEPS = 100
BATCH_SIZE = 8
MINI_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2

# PPO Hyperparameters
LEARNING_RATE = 1.41e-5
PPO_EPOCHS = 4
CLIP_RANGE = 0.2
VALUE_COEF = 0.1
KL_PENALTY = 0.02
GAMMA = 1.0
LAM = 0.95

# Energy Measurement Configuration
POWER_LOG_FILE = "power_log_baseline.csv"
POWER_SAMPLING_INTERVAL_MS = 100
RESULTS_DIR = "results/gpu_baseline"

# Device Configuration - GPU MODE
USE_GPU = True  # Enable GPU (requires CUDA-capable GPU)
FP16 = True  # Use FP16 for faster training on GPU

# Logging Configuration
LOG_EVERY_N_STEPS = 10
SAVE_CHECKPOINTS = False
VERBOSE = True

# Phase Timing
MEASURE_PHASES = True

# FPGA Offload Configuration - DISABLED FOR GPU
USE_FPGA_OFFLOAD = False  # Disable FPGA offload when using GPU
USE_MOCK_FPGA = False  # Not needed for GPU
FPGA_DEVICE_ID = 0
FPGA_VERBOSE = False
