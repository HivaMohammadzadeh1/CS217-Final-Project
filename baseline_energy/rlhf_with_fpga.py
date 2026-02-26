"""
RLHF Training with FPGA Matmul Offload

This script extends rlhf_baseline.py to offload matrix multiplications to FPGA.

Key differences from baseline:
- Intercepts PyTorch matmul operations
- Tiles matmuls into 16x16 chunks
- Sends tiles to FPGA for computation
- Tracks FPGA vs CPU/GPU execution time

Usage:
    python baseline_energy/rlhf_with_fpga.py --steps 50 --output results/fpga_run1
"""

import torch
import torch.nn as nn
import time
import json
import argparse
import sys
from pathlib import Path

# Add integration directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "integration"))

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import config
from gpu_power_monitor import GPUPowerMonitor
from fpga_matmul_offload import FPGAMatmulOffload


class FPGALinearLayer(nn.Module):
    """
    Drop-in replacement for nn.Linear that offloads matmul to FPGA.
    """

    def __init__(self, original_linear, fpga_offloader):
        super().__init__()
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.fpga = fpga_offloader

    def forward(self, x):
        """
        Forward pass with FPGA offload.
        x @ W.T + b  becomes  FPGA(x, W.T) + b
        """
        # x: (batch, seq_len, in_features) or (batch, in_features)
        # W: (out_features, in_features)
        # Result: (batch, seq_len, out_features) or (batch, out_features)

        original_shape = x.shape
        if x.dim() == 3:
            # Flatten batch and sequence dimensions
            batch_size, seq_len, in_features = x.shape
            x_flat = x.reshape(-1, in_features)
        else:
            x_flat = x

        # Perform matmul with FPGA offload: x @ W.T
        result = self.fpga.matmul(x_flat, self.weight.T)

        # Reshape back if needed
        if len(original_shape) == 3:
            result = result.reshape(batch_size, seq_len, -1)

        # Add bias
        if self.bias is not None:
            result = result + self.bias

        return result


def replace_linear_with_fpga(model, fpga_offloader, verbose=False):
    """
    Replace all nn.Linear layers in the model with FPGA-offloaded versions.

    Args:
        model: PyTorch model
        fpga_offloader: FPGAMatmulOffload instance
        verbose: Print replacement information

    Returns:
        Number of layers replaced
    """
    count = 0

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace this linear layer
            setattr(model, name, FPGALinearLayer(module, fpga_offloader))
            count += 1
            if verbose:
                print(f"   Replaced {name} with FPGA offload")
        else:
            # Recursively replace in submodules
            count += replace_linear_with_fpga(module, fpga_offloader, verbose)

    return count


class RLHFWithFPGATrainer:
    """RLHF trainer with FPGA matmul offload."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
        print(f"\n{'='*60}")
        print("RLHF Training with FPGA Offload")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Steps: {args.steps}")
        print(f"Output: {args.output}")
        print(f"FPGA Mode: {'MOCK' if config.USE_MOCK_FPGA else 'REAL'}")
        print(f"{'='*60}\n")

        # Create output directory
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize FPGA offloader
        self.fpga_offloader = FPGAMatmulOffload(
            use_mock=config.USE_MOCK_FPGA,
            verbose=config.FPGA_VERBOSE
        )

        # Phase timing
        self.phase_times = {
            "rollout": [],
            "reward": [],
            "gradient": [],
        }

        # FPGA stats
        self.fpga_stats = {
            "total_matmuls": 0,
            "total_tiles": 0,
        }

    def load_models(self):
        """Load policy and reward models."""
        print("📥 Loading models...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load policy model
        print("  Loading policy model...")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.float16 if config.FP16 else torch.float32,
        )
        if not config.USE_GPU:
            self.model = self.model.to(self.device)

        # Replace linear layers with FPGA offload
        if config.USE_FPGA_OFFLOAD:
            print("  Replacing linear layers with FPGA offload...")
            num_replaced = replace_linear_with_fpga(
                self.model,
                self.fpga_offloader,
                verbose=True
            )
            print(f"  ✓ Replaced {num_replaced} linear layers")

        # Load reward model
        print("  Loading reward model...")
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.REWARD_MODEL_NAME,
            num_labels=1,
            torch_dtype=torch.float16 if config.FP16 else torch.float32,
        )
        self.reward_model = self.reward_model.to(self.device)
        self.reward_model.eval()

        # Replace linear layers in reward model
        if config.USE_FPGA_OFFLOAD:
            num_replaced = replace_linear_with_fpga(
                self.reward_model,
                self.fpga_offloader,
                verbose=False
            )
            print(f"  ✓ Replaced {num_replaced} linear layers in reward model")

        # Load reference model
        print("  Loading reference model...")
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.float16 if config.FP16 else torch.float32,
        )
        if not config.USE_GPU:
            self.ref_model = self.ref_model.to(self.device)
        self.ref_model.eval()

        # Replace linear layers in reference model
        if config.USE_FPGA_OFFLOAD:
            num_replaced = replace_linear_with_fpga(
                self.ref_model,
                self.fpga_offloader,
                verbose=False
            )
            print(f"  ✓ Replaced {num_replaced} linear layers in reference model")

        print("✓ Models loaded with FPGA offload enabled\n")

    def load_dataset(self):
        """Load and prepare HH-RLHF dataset."""
        print("📊 Loading dataset...")

        # Load full dataset
        dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)

        # Create fixed subset with seed (800 train + 200 eval = 1000 total)
        dataset_subset = dataset.shuffle(seed=config.RANDOM_SEED).select(
            range(config.NUM_SAMPLES)
        )

        # Split train/eval
        self.train_dataset = dataset_subset.select(range(config.TRAIN_SIZE))
        self.eval_dataset = dataset_subset.select(
            range(config.TRAIN_SIZE, config.NUM_SAMPLES)
        )

        print(f"  Train: {len(self.train_dataset)} examples")
        print(f"  Eval:  {len(self.eval_dataset)} examples")
        print("✓ Dataset loaded\n")

    def build_dataset_for_ppo(self):
        """Convert HH-RLHF preference pairs to prompts for PPO."""
        prompts = []
        for example in self.train_dataset:
            chosen_text = example["chosen"]
            prompt = chosen_text.split("Assistant:")[0] + "Assistant:"
            prompts.append(prompt)

        return prompts[:self.args.steps]

    def compute_reward(self, responses):
        """Compute rewards for generated responses using reward model."""
        start_time = time.time()

        rewards = []
        with torch.no_grad():
            for response in responses:
                inputs = self.tokenizer(
                    response,
                    return_tensors="pt",
                    max_length=config.MAX_SEQ_LENGTH,
                    truncation=True,
                ).to(self.device)

                reward_outputs = self.reward_model(**inputs)
                reward = reward_outputs.logits[0, 0].item()
                rewards.append(reward)

        elapsed = time.time() - start_time
        self.phase_times["reward"].append(elapsed)

        return torch.tensor(rewards)

    def train(self):
        """Main training loop with energy measurement."""
        print("🎯 Starting training with FPGA offload...\n")

        # PPO configuration
        ppo_config = PPOConfig(
            model_name=config.MODEL_NAME,
            learning_rate=config.LEARNING_RATE,
            batch_size=config.BATCH_SIZE,
            mini_batch_size=config.MINI_BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            ppo_epochs=config.PPO_EPOCHS,
            seed=config.RANDOM_SEED,
        )

        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

        # Get prompts
        prompts = self.build_dataset_for_ppo()

        # Training metrics
        training_stats = {
            "steps": [],
            "rewards": [],
            "kl_divergence": [],
        }

        # Start power monitoring
        power_log_path = self.output_dir / config.POWER_LOG_FILE
        with GPUPowerMonitor(str(power_log_path), config.POWER_SAMPLING_INTERVAL_MS):

            # Training loop
            for step, prompt in enumerate(prompts):
                print(f"\n📍 Step {step+1}/{len(prompts)}")

                # Reset FPGA stats for this step
                self.fpga_offloader.reset_stats()

                # Phase 1: Rollout
                rollout_start = time.time()
                prompt_tensors = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=config.MAX_PROMPT_LENGTH,
                    truncation=True,
                ).input_ids.to(self.device)

                with torch.no_grad():
                    response_tensors = ppo_trainer.generate(
                        prompt_tensors,
                        max_new_tokens=config.MAX_RESPONSE_LENGTH,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                    )

                responses = [
                    self.tokenizer.decode(r, skip_special_tokens=True)
                    for r in response_tensors
                ]
                rollout_time = time.time() - rollout_start
                self.phase_times["rollout"].append(rollout_time)

                # Phase 2: Reward computation
                rewards = self.compute_reward(responses)

                # Phase 3: Gradient update
                gradient_start = time.time()
                stats = ppo_trainer.step(
                    [prompt_tensors], [response_tensors], [rewards]
                )
                gradient_time = time.time() - gradient_start
                self.phase_times["gradient"].append(gradient_time)

                # Get FPGA stats for this step
                fpga_stats = self.fpga_offloader.get_stats()
                self.fpga_stats["total_matmuls"] += fpga_stats["num_calls"]
                self.fpga_stats["total_tiles"] += fpga_stats["total_tiles"]

                # Log stats
                training_stats["steps"].append(step)
                training_stats["rewards"].append(rewards.mean().item())
                training_stats["kl_divergence"].append(
                    stats.get("objective/kl", 0.0)
                )

                if (step + 1) % config.LOG_EVERY_N_STEPS == 0:
                    print(f"  Reward: {rewards.mean():.3f}")
                    print(f"  KL: {stats.get('objective/kl', 0.0):.4f}")
                    print(f"  Rollout: {rollout_time:.2f}s")
                    print(f"  Reward: {self.phase_times['reward'][-1]:.2f}s")
                    print(f"  Gradient: {gradient_time:.2f}s")
                    print(f"  FPGA tiles: {fpga_stats['total_tiles']}")

        print("\n✓ Training complete\n")

        # Save results
        self.save_results(training_stats)

    def save_results(self, training_stats):
        """Save training results and timing breakdown."""
        print("💾 Saving results...")

        # Calculate phase statistics
        phase_stats = {}
        for phase in ["rollout", "reward", "gradient"]:
            times = self.phase_times[phase]
            phase_stats[phase] = {
                "total_time_s": sum(times),
                "avg_time_s": sum(times) / len(times) if times else 0,
                "num_calls": len(times),
            }

        # Save training stats
        with open(self.output_dir / "training_stats.json", "w") as f:
            json.dump(training_stats, f, indent=2)

        # Save phase timing
        with open(self.output_dir / "phase_timing.json", "w") as f:
            json.dump(phase_stats, f, indent=2)

        # Save FPGA stats
        with open(self.output_dir / "fpga_stats.json", "w") as f:
            json.dump(self.fpga_stats, f, indent=2)

        print(f"✓ Results saved to {self.output_dir}/")

        # Print summary
        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        for phase in ["rollout", "reward", "gradient"]:
            stats = phase_stats[phase]
            print(f"{phase:15s}: {stats['total_time_s']:7.1f}s total, "
                  f"{stats['avg_time_s']:6.2f}s avg")
        print(f"\nFPGA Statistics:")
        print(f"  Total matmuls offloaded: {self.fpga_stats['total_matmuls']}")
        print(f"  Total tiles processed:   {self.fpga_stats['total_tiles']}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="RLHF Training with FPGA Offload")
    parser.add_argument(
        "--steps",
        type=int,
        default=50,  # Default to 50 for the 2-hour plan
        help="Number of PPO steps to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/fpga_baseline",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Initialize trainer
    trainer = RLHFWithFPGATrainer(args)

    # Load everything
    trainer.load_models()
    trainer.load_dataset()

    # Train with FPGA offload
    trainer.train()

    print("🎉 FPGA baseline measurement complete!")


if __name__ == "__main__":
    main()
