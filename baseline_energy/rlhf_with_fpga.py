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
import torch.nn.functional as F
import time
import json
import argparse
import sys
import math
import numpy as np
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
            setattr(model, name, FPGALinearLayer(module, fpga_offloader))
            count += 1
            if verbose:
                print(f"   Replaced {name} with FPGA offload")
        else:
            count += replace_linear_with_fpga(module, fpga_offloader, verbose)

    return count


def restore_fpga_to_linear(model):
    """
    Convert all FPGALinearLayer modules back to standard nn.Linear so the
    model can be serialized with save_pretrained (which expects vanilla layers).

    Returns:
        Number of layers restored
    """
    count = 0

    for name, module in model.named_children():
        if isinstance(module, FPGALinearLayer):
            linear = nn.Linear(
                module.weight.shape[1],
                module.weight.shape[0],
                bias=module.bias is not None,
            )
            linear.weight = module.weight
            if module.bias is not None:
                linear.bias = module.bias
            setattr(model, name, linear)
            count += 1
        else:
            count += restore_fpga_to_linear(module)

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
            verbose=config.FPGA_VERBOSE,
            use_lab1=config.USE_LAB1_FPGA,
            device_id=config.FPGA_DEVICE_ID
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

            # Training loop - collect full batches before PPO step
            batch_num = 0
            num_batches = len(prompts) // config.BATCH_SIZE

            for batch_start in range(0, len(prompts), config.BATCH_SIZE):
                batch_prompts = prompts[batch_start:batch_start + config.BATCH_SIZE]
                if len(batch_prompts) < config.BATCH_SIZE:
                    break  # Skip incomplete final batch

                batch_num += 1
                print(f"\n📍 Batch {batch_num}/{num_batches} (steps {batch_start+1}-{batch_start+len(batch_prompts)})", flush=True)

                # Reset FPGA stats for this batch
                self.fpga_offloader.reset_stats()

                # Phase 1: Rollout - generate responses for entire batch
                rollout_start = time.time()
                query_tensors = []
                response_tensors = []

                for prompt in batch_prompts:
                    prompt_tensor = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=config.MAX_PROMPT_LENGTH,
                        truncation=True,
                    ).input_ids.squeeze(0).to(self.device)

                    with torch.no_grad():
                        response_tensor = ppo_trainer.generate(
                            prompt_tensor,
                            max_new_tokens=config.MAX_RESPONSE_LENGTH,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95,
                        )

                    query_tensors.append(prompt_tensor)
                    response_tensors.append(response_tensor.squeeze(0))
                    print(f"  Generated response {len(query_tensors)}/{config.BATCH_SIZE}", flush=True)

                rollout_time = time.time() - rollout_start
                self.phase_times["rollout"].append(rollout_time)

                # Phase 2: Reward computation for batch
                responses = [
                    self.tokenizer.decode(r, skip_special_tokens=True)
                    for r in response_tensors
                ]
                rewards = self.compute_reward(responses)
                rewards_list = [rewards[i] for i in range(len(rewards))]

                # Phase 3: Gradient update with full batch
                print(f"  Running PPO gradient update...", flush=True)
                gradient_start = time.time()
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards_list)
                gradient_time = time.time() - gradient_start
                self.phase_times["gradient"].append(gradient_time)

                # Get FPGA stats for this batch
                fpga_stats = self.fpga_offloader.get_stats()
                self.fpga_stats["total_matmuls"] += fpga_stats["num_calls"]
                self.fpga_stats["total_tiles"] += fpga_stats["total_tiles"]

                # Log stats
                training_stats["steps"].append(batch_num)
                training_stats["rewards"].append(rewards.mean().item())
                training_stats["kl_divergence"].append(stats.get("objective/kl", 0.0))

                if batch_num % config.LOG_EVERY_N_STEPS == 0:
                    print(f"  Reward: {rewards.mean():.3f}")
                    print(f"  KL: {stats.get('objective/kl', 0.0):.4f}")
                    print(f"  Rollout: {rollout_time:.2f}s")
                    print(f"  Reward time: {self.phase_times['reward'][-1]:.2f}s")
                    print(f"  Gradient: {gradient_time:.2f}s")
                    print(f"  FPGA tiles: {fpga_stats['total_tiles']}", flush=True)

        print("\n✓ Training complete\n")

        # Save results
        self.save_results(training_stats)

    def evaluate(self):
        """
        Post-training evaluation to measure quality degradation from FPGA quantization.

        Compares the FPGA-trained policy model against the frozen reference model
        on the held-out eval set. Tracks:
          - Reward scores (policy vs reference)
          - Perplexity (policy vs reference)
          - Win rate (how often policy is preferred over reference)
          - Sample generations for qualitative inspection
        """
        print(f"\n{'='*60}")
        print("Post-Training Evaluation: Quantization Quality Check")
        print(f"{'='*60}\n")

        eval_prompts = []
        for example in self.eval_dataset:
            chosen_text = example["chosen"]
            prompt = chosen_text.split("Assistant:")[0] + "Assistant:"
            eval_prompts.append(prompt)

        num_eval = min(len(eval_prompts), self.args.eval_samples)
        eval_prompts = eval_prompts[:num_eval]
        print(f"Evaluating on {num_eval} held-out examples...\n")

        policy_rewards = []
        ref_rewards = []
        policy_perplexities = []
        ref_perplexities = []
        wins, losses, ties = 0, 0, 0
        sample_outputs = []

        self.model.eval()
        self.ref_model.eval()

        for i, prompt in enumerate(eval_prompts):
            prompt_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=config.MAX_PROMPT_LENGTH,
                truncation=True,
            )
            prompt_ids = prompt_inputs.input_ids.to(self.device)

            with torch.no_grad():
                # --- Generate from policy (FPGA-quantized) model ---
                policy_output_ids = self.model.pretrained_model.generate(
                    prompt_ids,
                    max_new_tokens=config.MAX_RESPONSE_LENGTH,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                )
                policy_text = self.tokenizer.decode(
                    policy_output_ids[0], skip_special_tokens=True
                )

                # --- Generate from reference (unquantized) model ---
                ref_output_ids = self.ref_model.pretrained_model.generate(
                    prompt_ids,
                    max_new_tokens=config.MAX_RESPONSE_LENGTH,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                )
                ref_text = self.tokenizer.decode(
                    ref_output_ids[0], skip_special_tokens=True
                )

                # --- Reward scores ---
                policy_reward = self._score_text(policy_text)
                ref_reward = self._score_text(ref_text)
                policy_rewards.append(policy_reward)
                ref_rewards.append(ref_reward)

                # --- Perplexity ---
                policy_ppl = self._compute_perplexity(policy_output_ids)
                ref_ppl = self._compute_perplexity(ref_output_ids)
                policy_perplexities.append(policy_ppl)
                ref_perplexities.append(ref_ppl)

                # --- Win rate ---
                if policy_reward > ref_reward + 0.01:
                    wins += 1
                elif ref_reward > policy_reward + 0.01:
                    losses += 1
                else:
                    ties += 1

            # Collect a few sample outputs for qualitative review
            if i < 5:
                sample_outputs.append({
                    "prompt": prompt[:200],
                    "policy_response": policy_text[:300],
                    "ref_response": ref_text[:300],
                    "policy_reward": policy_reward,
                    "ref_reward": ref_reward,
                    "policy_ppl": policy_ppl,
                    "ref_ppl": ref_ppl,
                })

            if (i + 1) % 10 == 0 or (i + 1) == num_eval:
                print(f"  [{i+1}/{num_eval}] "
                      f"policy_reward={np.mean(policy_rewards):.3f}  "
                      f"ref_reward={np.mean(ref_rewards):.3f}  "
                      f"win_rate={wins/(wins+losses+ties):.1%}")

        # Aggregate metrics
        eval_results = {
            "num_eval_samples": num_eval,
            "policy_model": {
                "mean_reward": float(np.mean(policy_rewards)),
                "std_reward": float(np.std(policy_rewards)),
                "mean_perplexity": float(np.mean(policy_perplexities)),
                "median_perplexity": float(np.median(policy_perplexities)),
            },
            "reference_model": {
                "mean_reward": float(np.mean(ref_rewards)),
                "std_reward": float(np.std(ref_rewards)),
                "mean_perplexity": float(np.mean(ref_perplexities)),
                "median_perplexity": float(np.median(ref_perplexities)),
            },
            "quality_delta": {
                "reward_diff": float(np.mean(policy_rewards) - np.mean(ref_rewards)),
                "reward_diff_pct": float(
                    (np.mean(policy_rewards) - np.mean(ref_rewards))
                    / (abs(np.mean(ref_rewards)) + 1e-8) * 100
                ),
                "perplexity_diff": float(
                    np.mean(policy_perplexities) - np.mean(ref_perplexities)
                ),
                "perplexity_diff_pct": float(
                    (np.mean(policy_perplexities) - np.mean(ref_perplexities))
                    / (abs(np.mean(ref_perplexities)) + 1e-8) * 100
                ),
            },
            "win_rate": {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_pct": float(wins / (wins + losses + ties) * 100),
            },
            "sample_outputs": sample_outputs,
        }

        # Save eval results
        with open(self.output_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print("Evaluation Summary (FPGA-Quantized vs Reference)")
        print(f"{'='*60}")
        print(f"  {'Metric':<25s} {'Policy (FPGA)':>14s} {'Reference':>14s} {'Delta':>10s}")
        print(f"  {'-'*63}")
        print(f"  {'Mean Reward':<25s} "
              f"{eval_results['policy_model']['mean_reward']:>14.4f} "
              f"{eval_results['reference_model']['mean_reward']:>14.4f} "
              f"{eval_results['quality_delta']['reward_diff']:>+10.4f}")
        print(f"  {'Mean Perplexity':<25s} "
              f"{eval_results['policy_model']['mean_perplexity']:>14.2f} "
              f"{eval_results['reference_model']['mean_perplexity']:>14.2f} "
              f"{eval_results['quality_delta']['perplexity_diff']:>+10.2f}")
        print(f"  {'Win Rate':<25s} "
              f"{eval_results['win_rate']['win_pct']:>13.1f}% "
              f"{'':>14s} "
              f"({wins}W/{losses}L/{ties}T)")
        print(f"{'='*60}")

        reward_diff_pct = abs(eval_results["quality_delta"]["reward_diff_pct"])
        if reward_diff_pct < 2.0:
            print("  -> Quantization impact: NEGLIGIBLE (<2% reward change)")
        elif reward_diff_pct < 5.0:
            print("  -> Quantization impact: MINOR (2-5% reward change)")
        elif reward_diff_pct < 10.0:
            print("  -> Quantization impact: MODERATE (5-10% reward change)")
        else:
            print("  -> Quantization impact: SIGNIFICANT (>10% reward change)")
        print()

        return eval_results

    def _score_text(self, text):
        """Score a single text string with the reward model."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=config.MAX_SEQ_LENGTH,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
        return outputs.logits[0, 0].item()

    def _compute_perplexity(self, token_ids):
        """
        Compute perplexity of the policy model on the given token sequence.
        Uses the underlying causal LM (before the value head).
        """
        try:
            with torch.no_grad():
                outputs = self.model.pretrained_model(token_ids, labels=token_ids)
                loss = outputs.loss
                if loss is None:
                    return float("nan")
                return math.exp(min(loss.item(), 100))
        except Exception:
            return float("nan")

    def save_model(self):
        """
        Save the trained policy model, reward model, tokenizer, and training
        config so everything can be reloaded for future evaluations without
        re-training.

        Handles FPGALinearLayer -> nn.Linear conversion so that HuggingFace
        save_pretrained produces a valid, standard checkpoint.
        """
        print(f"\n{'='*60}")
        print("Saving Models & Artifacts")
        print(f"{'='*60}\n")

        # ── 1. Restore FPGA layers to standard nn.Linear before saving ──
        fpga_was_active = config.USE_FPGA_OFFLOAD
        if fpga_was_active:
            print("  Restoring FPGA layers to nn.Linear for serialization...")
            n_policy = restore_fpga_to_linear(self.model)
            n_reward = restore_fpga_to_linear(self.reward_model)
            n_ref = restore_fpga_to_linear(self.ref_model)
            print(f"    Restored: policy={n_policy}, reward={n_reward}, ref={n_ref}")

        # ── 2. Save policy model (causal-LM without value head) ──
        model_dir = self.output_dir / "trained_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.pretrained_model.save_pretrained(str(model_dir))
        self.tokenizer.save_pretrained(str(model_dir))
        print(f"  ✓ Policy model (causal-LM): {model_dir}/")

        # ── 3. Save full PPO model (with value head) for resume ──
        ppo_dir = self.output_dir / "trained_model_ppo"
        ppo_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(ppo_dir))
        self.tokenizer.save_pretrained(str(ppo_dir))
        print(f"  ✓ PPO model (with value head): {ppo_dir}/")

        # ── 4. Save reward model for reproducible future evals ──
        reward_dir = self.output_dir / "reward_model"
        reward_dir.mkdir(parents=True, exist_ok=True)
        self.reward_model.save_pretrained(str(reward_dir))
        self.tokenizer.save_pretrained(str(reward_dir))
        print(f"  ✓ Reward model: {reward_dir}/")

        # ── 5. Save reference model for future comparisons ──
        ref_dir = self.output_dir / "reference_model"
        ref_dir.mkdir(parents=True, exist_ok=True)
        self.ref_model.pretrained_model.save_pretrained(str(ref_dir))
        self.tokenizer.save_pretrained(str(ref_dir))
        print(f"  ✓ Reference model: {ref_dir}/")

        # ── 6. Save raw state dicts as .pt files (fail-safe backup) ──
        torch.save(
            self.model.pretrained_model.state_dict(),
            str(self.output_dir / "policy_state_dict.pt"),
        )
        torch.save(
            self.reward_model.state_dict(),
            str(self.output_dir / "reward_state_dict.pt"),
        )
        print(f"  ✓ State-dict backups (.pt): {self.output_dir}/")

        # ── 7. Training metadata for provenance ──
        meta = {
            "base_model": config.MODEL_NAME,
            "reward_model": config.REWARD_MODEL_NAME,
            "training_steps": self.args.steps,
            "fpga_offload": config.USE_FPGA_OFFLOAD,
            "fpga_mock": config.USE_MOCK_FPGA,
            "use_lab1_fpga": config.USE_LAB1_FPGA,
            "fp16": config.FP16,
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "ppo_epochs": config.PPO_EPOCHS,
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "max_prompt_length": config.MAX_PROMPT_LENGTH,
            "max_response_length": config.MAX_RESPONSE_LENGTH,
            "dataset": config.DATASET_NAME,
            "train_size": config.TRAIN_SIZE,
            "eval_size": config.EVAL_SIZE,
            "random_seed": config.RANDOM_SEED,
            "saved_artifacts": [
                "trained_model/",
                "trained_model_ppo/",
                "reward_model/",
                "reference_model/",
                "policy_state_dict.pt",
                "reward_state_dict.pt",
            ],
        }
        with open(self.output_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  ✓ Training metadata: {self.output_dir / 'training_meta.json'}")

        # ── 8. Re-apply FPGA layers if training/eval will continue ──
        if fpga_was_active:
            print("  Re-applying FPGA offload layers...")
            replace_linear_with_fpga(self.model, self.fpga_offloader)
            replace_linear_with_fpga(self.reward_model, self.fpga_offloader)
            replace_linear_with_fpga(self.ref_model, self.fpga_offloader)

        print(f"\n{'='*60}")
        print(f"All artifacts saved to: {self.output_dir}/")
        print(f"{'='*60}\n")

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
        default=50,
        help="Number of PPO steps to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/fpga_baseline",
        help="Output directory for results",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=50,
        dest="eval_samples",
        help="Number of held-out samples for post-training quality evaluation",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        dest="skip_eval",
        help="Skip post-training evaluation",
    )
    args = parser.parse_args()

    # Initialize trainer
    trainer = RLHFWithFPGATrainer(args)

    # Load everything
    trainer.load_models()
    trainer.load_dataset()

    # Train with FPGA offload
    trainer.train()

    # Save trained model for future evals
    trainer.save_model()

    # Evaluate quantization quality degradation
    if not args.skip_eval:
        trainer.evaluate()

    print("🎉 FPGA baseline measurement complete!")


if __name__ == "__main__":
    main()
