"""
RLHF Baseline Training Script using TRL (HuggingFace Transformers Reinforcement Learning)

This script:
1. Loads Qwen2.5-0.5B as policy and reward models
2. Trains using PPO on HH-RLHF dataset
3. Logs GPU power consumption during training
4. Measures energy breakdown by phase (rollout/reward/gradient)

Usage:
    python baseline_energy/rlhf_baseline.py [--steps NUM_STEPS] [--output OUTPUT_DIR]
"""

import torch
import time
import json
import argparse
from pathlib import Path
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


class RLHFBaselineTrainer:
    """RLHF baseline trainer with energy measurement."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
        print(f"\n{'='*60}")
        print("RLHF Baseline Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Steps: {args.steps}")
        print(f"Output: {args.output}")
        print(f"{'='*60}\n")

        # Create output directory
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Phase timing
        self.phase_times = {
            "rollout": [],
            "reward": [],
            "gradient": [],
        }

    def load_models(self):
        """Load policy and reward models."""
        print("📥 Loading models...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load policy model with value head (for PPO)
        print("  Loading policy model...")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.float16 if config.FP16 else torch.float32,
        )
        if not config.USE_GPU:
            self.model = self.model.to(self.device)

        # Load reward model (simple approach: use same model with classification head)
        print("  Loading reward model...")
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.REWARD_MODEL_NAME,
            num_labels=1,
            torch_dtype=torch.float16 if config.FP16 else torch.float32,
        )
        self.reward_model = self.reward_model.to(self.device)
        self.reward_model.eval()  # Frozen during PPO

        # Load reference model (frozen copy for KL penalty)
        print("  Loading reference model...")
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.float16 if config.FP16 else torch.float32,
        )
        if not config.USE_GPU:
            self.ref_model = self.ref_model.to(self.device)
        self.ref_model.eval()

        print("✓ Models loaded\n")

    def load_dataset(self):
        """Load and prepare HH-RLHF dataset."""
        print("📊 Loading dataset...")

        # Load full dataset
        dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)

        # Create fixed subset with seed
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
        """
        Convert HH-RLHF preference pairs to prompts for PPO.
        We'll use the 'chosen' responses as targets.
        """
        prompts = []
        for example in self.train_dataset:
            # Extract prompt from chosen response (everything before assistant's response)
            # This is a simplified approach - you may want to parse more carefully
            chosen_text = example["chosen"]
            # Simple heuristic: take first part as prompt
            prompt = chosen_text.split("Assistant:")[0] + "Assistant:"
            prompts.append(prompt)

        return prompts[:self.args.steps]  # Limit to num_steps for this run

    def compute_reward(self, responses):
        """Compute rewards for generated responses using reward model."""
        start_time = time.time()

        rewards = []
        with torch.no_grad():
            for response in responses:
                # Tokenize response
                inputs = self.tokenizer(
                    response,
                    return_tensors="pt",
                    max_length=config.MAX_SEQ_LENGTH,
                    truncation=True,
                ).to(self.device)

                # Get reward from reward model
                reward_outputs = self.reward_model(**inputs)
                reward = reward_outputs.logits[0, 0].item()
                rewards.append(reward)

        # Log phase timing
        elapsed = time.time() - start_time
        self.phase_times["reward"].append(elapsed)

        return torch.tensor(rewards)

    def train(self):
        """Main training loop with energy measurement."""
        print("🎯 Starting training...\n")

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

                # Phase 1: Rollout (generate responses)
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

        print(f"✓ Results saved to {self.output_dir}/")

        # Print summary
        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        for phase in ["rollout", "reward", "gradient"]:
            stats = phase_stats[phase]
            print(f"{phase:15s}: {stats['total_time_s']:7.1f}s total, "
                  f"{stats['avg_time_s']:6.2f}s avg")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="RLHF Baseline Training")
    parser.add_argument(
        "--steps",
        type=int,
        default=config.NUM_PPO_STEPS,
        help="Number of PPO steps to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.RESULTS_DIR,
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Initialize trainer
    trainer = RLHFBaselineTrainer(args)

    # Load everything
    trainer.load_models()
    trainer.load_dataset()

    # Train with energy measurement
    trainer.train()

    print("🎉 Baseline measurement complete!")


if __name__ == "__main__":
    main()
