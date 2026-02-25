# Integration - Adaptive Controller & RLHF Loop

This directory contains the adaptive controller and RLHF training loop integration.

## Purpose

Connect the FPGA MX datapath to the RLHF training loop with layer-adaptive precision control.

## Key Components

1. **Adaptive Controller**: Selects MX format per layer and phase
2. **RLHF Training Loop**: PPO-based RLHF using TRL library
3. **FPGA Integration**: Host-FPGA communication for offloaded operations
4. **Experiment Runner**: Automated experiment execution across policies

## Files (to be added)

- `adaptive_controller.py`: Layer-adaptive precision controller
- `rlhf_loop.py`: RLHF training loop with FPGA integration
- `experiment_runner.py`: Run all policy experiments
- `config.py`: Configuration for policies and RLHF parameters
- `utils.py`: Helper functions

## Adaptive Controller API

```python
from adaptive_controller import AdaptiveController

# Initialize controller
controller = AdaptiveController(
    sensitivity_matrix='../results/sensitivity_matrix.csv',
    policy='D',  # A/B/C/D
    fpga_device=fpga
)

# Get precision for a specific layer and phase
precision = controller.get_precision(
    layer_name='transformer.h.0.attn',
    phase='rollout'  # rollout/reward/gradient
)
# Returns: 'MXFP4' or 'MXFP8' or 'FP16'

# Configure FPGA with selected precision
controller.configure_fpga(precision)

# Load pre-quantized weights
controller.load_quantized_weights(layer_name, precision)
```

## RLHF Training Loop

```python
from rlhf_loop import RLHFTrainer

# Initialize trainer
trainer = RLHFTrainer(
    model_name='Qwen/Qwen2.5-0.5B-Instruct',
    dataset='Anthropic/hh-rlhf',
    controller=controller,
    config=rlhf_config
)

# Run training
metrics = trainer.train(
    num_steps=100,
    log_energy=True,
    output_dir='../results/policy_D'
)
```

## Experiment Runner

```bash
# Run all policies
python experiment_runner.py --policies A B C D --output ../results/

# Run specific policy
python experiment_runner.py --policy D --num_steps 100 --seed 42
```

## Configuration

Edit `config.py` to define:
- PPO hyperparameters (learning rate, clip range, etc.)
- Batch size and sequence length
- Policy definitions (which layers use which format in each phase)
- Energy measurement settings

## Experiment Workflow

1. Load sensitivity matrix from PyTorch profiling
2. Initialize adaptive controller with policy
3. Load model and dataset
4. Start energy monitoring
5. Run RLHF training loop:
   - For each layer in each phase:
     - Query controller for precision
     - Configure FPGA
     - Load quantized weights
     - Execute forward/backward pass
6. Stop energy monitoring
7. Evaluate policy quality (win rate, KL divergence)
8. Log results to CSV

## Output Format

Results saved to `../results/<policy_name>/`:
- `energy.csv`: Energy breakdown by phase
- `metrics.csv`: Quality metrics (win rate, KL, reward)
- `training_log.txt`: Detailed training log
- `config.json`: Experiment configuration
