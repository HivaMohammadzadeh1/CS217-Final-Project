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
from contextlib import nullcontext
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
from adaptive_controller import AdaptivePrecisionController
from fpga_matmul_offload import FPGAMatmulOffload


def parse_int_list(raw_value):
    """Parse a comma-separated integer list."""
    if raw_value is None:
        return None
    values = []
    for item in str(raw_value).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def apply_runtime_overrides(args):
    """Apply CLI overrides directly onto the shared config module."""
    scalar_overrides = {
        "MODEL_NAME": args.model_name,
        "REWARD_MODEL_NAME": args.reward_model_name,
        "DATASET_NAME": args.dataset_name,
        "LOCAL_DATASET_PATH": args.local_dataset_path,
        "NUM_SAMPLES": args.num_samples,
        "TRAIN_SIZE": args.train_size,
        "EVAL_SIZE": args.eval_size,
        "BATCH_SIZE": args.batch_size,
        "MINI_BATCH_SIZE": args.mini_batch_size,
        "MAX_SEQ_LENGTH": args.max_seq_length,
        "MAX_PROMPT_LENGTH": args.max_prompt_length,
        "MAX_RESPONSE_LENGTH": args.max_response_length,
        "FPGA_RESPONSE_LENGTH": args.fpga_response_length,
        "PRETRAIN_REWARD_STEPS": args.pretrain_reward_steps,
        "FPGA_PRECISION_MODE": args.precision_mode,
        "FPGA_GROUP_SIZE": args.group_size,
        "FPGA_POLICY_JSON": args.policy_json,
        "FPGA_POLICY_NAME": args.policy_name,
    }

    for attr, value in scalar_overrides.items():
        if value is not None:
            setattr(config, attr, value)

    if args.use_mock_fpga:
        config.USE_FPGA_OFFLOAD = True
        config.USE_MOCK_FPGA = True
        config.USE_LAB1_FPGA = False

    if args.no_fpga_offload:
        config.USE_FPGA_OFFLOAD = False

    if args.use_hub_dataset:
        config.USE_HUB_DATASET = True

    if args.local_dataset_path:
        config.USE_HUB_DATASET = False

    if args.allow_gradient_offload:
        config.FPGA_ALLOW_GRADIENT_OFFLOAD = True

    if args.policy_blocks is not None:
        config.FPGA_POLICY_BLOCKS = parse_int_list(args.policy_blocks)
    if args.reward_policy_blocks is not None:
        config.FPGA_REWARD_BLOCKS = parse_int_list(args.reward_policy_blocks)

    if args.train_size is not None or args.eval_size is not None:
        if args.num_samples is None:
            config.NUM_SAMPLES = config.TRAIN_SIZE + config.EVAL_SIZE


def get_reward_head_modules(model):
    """Locate the sequence-classification head modules to train."""
    modules = []
    for attr in ("score", "classifier", "classification_head"):
        module = getattr(model, attr, None)
        if module is not None and any(True for _ in module.parameters()):
            modules.append(module)

    if modules:
        return modules

    linear_modules = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if linear_modules:
        return [linear_modules[-1]]

    raise RuntimeError("Could not locate a trainable reward head module.")


class FPGALinearLayer(nn.Module):
    """
    Phase-aware replacement for nn.Linear.

    If the adaptive controller selects FP16 for the current phase/layer, the
    original PyTorch linear path is used and autograd remains intact.
    """

    def __init__(self, original_linear, fpga_offloader,
                 layer_name, controller=None, model_role="unknown"):
        super().__init__()
        self.linear = original_linear
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.fpga = fpga_offloader
        self.layer_name = layer_name
        self.controller = controller
        self.model_role = model_role

    def forward(self, x):
        """
        Forward pass with optional FPGA offload.
        """
        if self.controller is not None:
            decision = self.controller.get_decision(
                self.layer_name,
                model_role=self.model_role,
            )
            if not decision.should_offload:
                return self.linear(x)
            self.fpga.configure_precision(
                decision.precision,
                group_size=decision.group_size,
                flush=True,
            )

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


FPGA_TARGET_LAYERS = {"q_proj", "k_proj", "v_proj", "o_proj"}

def replace_linear_with_fpga(model, fpga_offloader,
                             controller=None, model_role="unknown",
                             name_prefix="",
                             verbose=False):
    count = 0

    for name, module in model.named_children():
        full_name = f"{name_prefix}.{name}" if name_prefix else name
        if isinstance(module, nn.Linear) and name in FPGA_TARGET_LAYERS:
            setattr(
                model,
                name,
                FPGALinearLayer(
                    module,
                    fpga_offloader,
                    layer_name=full_name,
                    controller=controller,
                    model_role=model_role,
                )
            )
            count += 1
            if verbose:
                print(f"   Replaced {full_name} with FPGA offload")
        else:
            count += replace_linear_with_fpga(
                module,
                fpga_offloader,
                controller=controller,
                model_role=model_role,
                name_prefix=full_name,
                verbose=verbose,
            )

    return count


def replace_linear_with_fpga_selective(model, fpga_offloader, target_blocks,
                                       controller=None, model_role="unknown",
                                       verbose=False):
    """Replace attention projections with FPGA offload only in specific
    transformer blocks, leaving other blocks on CPU for speed.

    Works with Qwen2 naming: ``*.layers.{i}.self_attn.{q,k,v,o}_proj``
    and with the TRL value-head wrapper that nests under ``pretrained_model``.
    """
    count = 0
    target_prefixes = {f"layers.{i}." for i in target_blocks}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name not in FPGA_TARGET_LAYERS:
            continue
        if not any(prefix in name for prefix in target_prefixes):
            continue

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(
            parent,
            leaf_name,
            FPGALinearLayer(
                module,
                fpga_offloader,
                layer_name=name,
                controller=controller,
                model_role=model_role,
            )
        )
        count += 1
        if verbose:
            print(f"   Replaced {name} with FPGA offload")

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
            setattr(model, name, module.linear)
            count += 1
        else:
            count += restore_fpga_to_linear(module)

    return count


class RLHFWithFPGATrainer:
    """RLHF trainer with FPGA matmul offload."""

    def __init__(self, args):
        self.args = args
        self.policy_blocks = list(getattr(config, "FPGA_POLICY_BLOCKS", []))
        reward_blocks = getattr(config, "FPGA_REWARD_BLOCKS", None)
        self.reward_blocks = None if reward_blocks is None else list(reward_blocks)
        self.fpga_response_length = getattr(config, "FPGA_RESPONSE_LENGTH", config.MAX_RESPONSE_LENGTH)
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
        self.precision_controller = AdaptivePrecisionController(
            default_precision=args.precision_mode or getattr(config, "FPGA_PRECISION_MODE", "INT8"),
            default_group_size=args.group_size or getattr(config, "FPGA_GROUP_SIZE", 8),
            allow_gradient_offload=(
                args.allow_gradient_offload
                or getattr(config, "FPGA_ALLOW_GRADIENT_OFFLOAD", False)
            ),
            policy_name=args.policy_name or getattr(config, "FPGA_POLICY_NAME", None),
            policy_path=args.policy_json or getattr(config, "FPGA_POLICY_JSON", None),
        )
        print(f"\n{'='*60}")
        print("RLHF Training with FPGA Offload")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Steps: {args.steps}")
        print(f"Output: {args.output}")
        print(f"FPGA Mode: {'MOCK' if config.USE_MOCK_FPGA else 'REAL'}")
        print(f"FPGA Policy Blocks: {self.policy_blocks}")
        print(f"FPGA Reward Blocks: {'ALL' if self.reward_blocks is None else self.reward_blocks}")
        print(f"FPGA Response Length: {self.fpga_response_length}")
        if self.precision_controller.policy is not None:
            print(f"Precision Policy: {self.precision_controller.policy_name} from {self.precision_controller.policy_path}")
        else:
            print(f"Precision Policy: global {self.precision_controller.default_precision}")
        print(f"Gradient Offload: {'enabled' if self.precision_controller.allow_gradient_offload else 'disabled'}")
        print(f"{'='*60}\n")

        # Create output directory
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize FPGA offloader
        self.fpga_offloader = FPGAMatmulOffload(
            use_mock=config.USE_MOCK_FPGA,
            verbose=config.FPGA_VERBOSE,
            use_lab1=config.USE_LAB1_FPGA,
            device_id=config.FPGA_DEVICE_ID,
            precision_mode=self.precision_controller.default_precision,
            group_size=self.precision_controller.default_group_size,
        )

        # Phase timing
        self.phase_times = {
            "rollout": [],
            "reward": [],
            "gradient": [],
        }
        self.dataset_source = None

        # FPGA stats
        self.fpga_stats = {
            "total_matmuls": 0,
            "total_tiles": 0,
        }
        self.policy_device = self.device
        self.reward_device = self.device
        self.ref_device = self.device

    def _phase_scope(self, phase):
        if not config.USE_FPGA_OFFLOAD:
            return nullcontext()
        return self.precision_controller.phase_scope(phase)

    @staticmethod
    def _module_device(module):
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def load_models(self):
        """Load policy, reward, and reference models with FPGA offload.

        - Reward model: ALL attention layers replaced (single forward pass
          per response — fast enough for full FPGA coverage).
        - Policy & reference models: only the transformer blocks listed in
          ``config.FPGA_POLICY_BLOCKS`` are replaced so autoregressive
          generation stays feasible.  Generation length is capped at
          ``config.FPGA_RESPONSE_LENGTH`` to keep tile count manageable.
        """
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

        if config.USE_FPGA_OFFLOAD and self.policy_blocks:
            print(f"  Replacing policy model blocks {self.policy_blocks} with FPGA offload...")
            num_replaced = replace_linear_with_fpga_selective(
                self.model, self.fpga_offloader,
                target_blocks=self.policy_blocks,
                controller=self.precision_controller,
                model_role="policy",
                verbose=True,
            )
            print(f"  ✓ Replaced {num_replaced} linear layers in policy model")

        # Load reward model.
        print("  Loading reward model...")
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.REWARD_MODEL_NAME,
            num_labels=1,
            torch_dtype=torch.float16 if config.FP16 else torch.float32,
        )
        self.reward_model = self.reward_model.to(self.device)
        self.reward_model.eval()

        if config.USE_FPGA_OFFLOAD:
            if self.reward_blocks is None:
                print("  Replacing ALL reward model layers with FPGA offload...")
                num_replaced = replace_linear_with_fpga(
                    self.reward_model,
                    self.fpga_offloader,
                    controller=self.precision_controller,
                    model_role="reward",
                    verbose=True,
                )
            else:
                print(f"  Replacing reward model blocks {self.reward_blocks} with FPGA offload...")
                num_replaced = replace_linear_with_fpga_selective(
                    self.reward_model,
                    self.fpga_offloader,
                    target_blocks=self.reward_blocks,
                    controller=self.precision_controller,
                    model_role="reward",
                    verbose=True,
                )
            print(f"  ✓ Replaced {num_replaced} linear layers in reward model")

        # Load reference model (selective FPGA — used for KL in PPO, no grads)
        print("  Loading reference model...")
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.float16 if config.FP16 else torch.float32,
        )
        if not config.USE_GPU:
            self.ref_model = self.ref_model.to(self.device)
        self.ref_model.eval()

        if config.USE_FPGA_OFFLOAD and self.policy_blocks:
            print(f"  Replacing reference model blocks {self.policy_blocks} with FPGA offload...")
            num_replaced = replace_linear_with_fpga_selective(
                self.ref_model, self.fpga_offloader,
                target_blocks=self.policy_blocks,
                controller=self.precision_controller,
                model_role="reference",
                verbose=False,
            )
            print(f"  ✓ Replaced {num_replaced} linear layers in reference model")

        self.policy_device = self._module_device(self.model.pretrained_model)
        self.reward_device = self._module_device(self.reward_model)
        self.ref_device = self._module_device(self.ref_model.pretrained_model)
        policy_desc = self.policy_blocks if config.USE_FPGA_OFFLOAD else "none"
        reward_desc = "all" if self.reward_blocks is None else self.reward_blocks
        print(f"  Devices: policy={self.policy_device}, reward={self.reward_device}, reference={self.ref_device}")
        print(f"✓ Models loaded (reward blocks={reward_desc}, policy/ref blocks={policy_desc})\n")

    def load_dataset(self):
        """Load and prepare HH-RLHF dataset."""
        print("📊 Loading dataset...")

        if getattr(config, "USE_HUB_DATASET", True):
            dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)
            dataset_desc = config.DATASET_NAME
        else:
            dataset = load_dataset("json", data_files=config.LOCAL_DATASET_PATH, split="train")
            dataset_desc = config.LOCAL_DATASET_PATH

        available_samples = len(dataset)
        requested_samples = min(config.NUM_SAMPLES, available_samples)
        required_samples = config.TRAIN_SIZE + config.EVAL_SIZE
        if required_samples > requested_samples:
            raise ValueError(
                f"Dataset only provides {requested_samples} samples after selection, "
                f"but train_size + eval_size = {required_samples}."
            )

        # Create fixed subset with seed.
        dataset_subset = dataset.shuffle(seed=config.RANDOM_SEED).select(
            range(requested_samples)
        )

        # Split train/eval
        self.train_dataset = dataset_subset.select(range(config.TRAIN_SIZE))
        self.eval_dataset = dataset_subset.select(
            range(config.TRAIN_SIZE, config.TRAIN_SIZE + config.EVAL_SIZE)
        )

        self.dataset_source = dataset_desc
        print(f"  Source: {dataset_desc}")
        print(f"  Train: {len(self.train_dataset)} examples")
        print(f"  Eval:  {len(self.eval_dataset)} examples")
        print("✓ Dataset loaded\n")

    def pretrain_reward_model(self):
        """Fine-tune the reward model's scalar head on preference pairs.

        Uses the Bradley-Terry log-sigmoid loss (standard in RLHF):
            loss = -log(sigmoid(chosen_score - rejected_score))

        The transformer body is frozen; only the ``score`` head is trained.
        FPGA layers stay active since we only need gradients for the head.
        """
        steps = config.PRETRAIN_REWARD_STEPS
        lr = config.PRETRAIN_REWARD_LR
        print(f"🎓 Pre-training reward model head ({steps} steps, lr={lr})...")
        if config.USE_FPGA_OFFLOAD:
            print("  FPGA layers remain active during pre-training")

        self.reward_model.train()
        reward_head_modules = get_reward_head_modules(self.reward_model)

        # Only train the classification head, freeze the transformer body
        for param in self.reward_model.parameters():
            param.requires_grad = False
        reward_head_params = []
        for module in reward_head_modules:
            for param in module.parameters():
                param.requires_grad = True
                reward_head_params.append(param)

        optimizer = torch.optim.Adam(reward_head_params, lr=lr)
        self.fpga_offloader.reset_stats()

        total_loss = 0.0
        correct = 0
        for step in range(steps):
            idx = step % len(self.train_dataset)
            example = self.train_dataset[idx]

            chosen_inputs = self.tokenizer(
                example["chosen"], return_tensors="pt",
                max_length=config.MAX_SEQ_LENGTH, truncation=True,
            ).to(self.reward_device)
            rejected_inputs = self.tokenizer(
                example["rejected"], return_tensors="pt",
                max_length=config.MAX_SEQ_LENGTH, truncation=True,
            ).to(self.reward_device)

            with self._phase_scope("reward"):
                chosen_score = self.reward_model(**chosen_inputs).logits[0, 0]
                rejected_score = self.reward_model(**rejected_inputs).logits[0, 0]

            # Bradley-Terry log-sigmoid loss (standard RLHF reward model loss)
            loss = -F.logsigmoid(chosen_score - rejected_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if chosen_score.item() > rejected_score.item():
                correct += 1

            if (step + 1) % 5 == 0:
                pretrain_fpga = self.fpga_offloader.get_stats()
                print(f"  [{step+1}/{steps}] loss={total_loss/(step+1):.4f} "
                      f"accuracy={correct/(step+1):.1%} "
                      f"fpga_tiles={pretrain_fpga['total_tiles']}", flush=True)

        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False

        pretrain_fpga = self.fpga_offloader.get_stats()
        print(f"✓ Reward head pre-trained (accuracy={correct/steps:.1%}, "
              f"fpga_tiles={pretrain_fpga['total_tiles']})\n")

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
        with self._phase_scope("reward"):
            with torch.no_grad():
                for response in responses:
                    inputs = self.tokenizer(
                        response,
                        return_tensors="pt",
                        max_length=config.MAX_SEQ_LENGTH,
                        truncation=True,
                    ).to(self.reward_device)

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
        self.policy_device = self._module_device(ppo_trainer.model)
        self.ref_device = self._module_device(ppo_trainer.ref_model)

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
                    ).input_ids.squeeze(0).to(self.policy_device)

                    gen_length = (self.fpga_response_length
                                  if config.USE_FPGA_OFFLOAD and self.policy_blocks
                                  else config.MAX_RESPONSE_LENGTH)
                    with self._phase_scope("rollout"):
                        with torch.no_grad():
                            response_tensor = ppo_trainer.generate(
                                prompt_tensor,
                                max_new_tokens=gen_length,
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
                with self._phase_scope("gradient"):
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
            policy_prompt_ids = prompt_inputs.input_ids.to(self.policy_device)
            ref_prompt_ids = prompt_inputs.input_ids.to(self.ref_device)

            eval_gen_length = (self.fpga_response_length
                               if config.USE_FPGA_OFFLOAD and self.policy_blocks
                               else config.MAX_RESPONSE_LENGTH)
            with self._phase_scope("rollout"):
                with torch.no_grad():
                # --- Generate from policy (FPGA-quantized) model ---
                    policy_output_ids = self.model.pretrained_model.generate(
                        policy_prompt_ids,
                        max_new_tokens=eval_gen_length,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                    )
                    policy_text = self.tokenizer.decode(
                        policy_output_ids[0], skip_special_tokens=True
                    )

                    # --- Generate from reference model ---
                    ref_output_ids = self.ref_model.pretrained_model.generate(
                        ref_prompt_ids,
                        max_new_tokens=eval_gen_length,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                    )
                    ref_text = self.tokenizer.decode(
                        ref_output_ids[0], skip_special_tokens=True
                    )

                    # --- Perplexity ---
                    policy_ppl = self._compute_perplexity(policy_output_ids)
                    ref_ppl = self._compute_perplexity(ref_output_ids)
                    policy_perplexities.append(policy_ppl)
                    ref_perplexities.append(ref_ppl)

            # --- Reward scores ---
            policy_reward = self._score_text(policy_text)
            ref_reward = self._score_text(ref_text)
            policy_rewards.append(policy_reward)
            ref_rewards.append(ref_reward)

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

            print(f"  [{i+1}/{num_eval}] "
                  f"policy_reward={policy_reward:.3f}  "
                  f"ref_reward={ref_reward:.3f}  "
                  f"{'WIN' if policy_reward > ref_reward + 0.01 else 'LOSS' if ref_reward > policy_reward + 0.01 else 'TIE'}  "
                  f"(avg: {np.mean(policy_rewards):.3f} vs {np.mean(ref_rewards):.3f}, "
                  f"win_rate={wins/(wins+losses+ties):.1%})", flush=True)

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
        ).to(self.reward_device)
        with self._phase_scope("reward"):
            with torch.no_grad():
                outputs = self.reward_model(**inputs)
        return outputs.logits[0, 0].item()

    def _compute_perplexity(self, token_ids):
        """
        Compute perplexity of the policy model on the given token sequence.
        Uses the underlying causal LM (before the value head).
        """
        try:
            with self._phase_scope("rollout"):
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

        # ── 2-6. Save model checkpoints (skipped by default to save disk) ──
        if self.args.save_models:
            model_dir = self.output_dir / "trained_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            self.model.pretrained_model.save_pretrained(str(model_dir))
            self.tokenizer.save_pretrained(str(model_dir))
            print(f"  ✓ Policy model: {model_dir}/")

            ppo_dir = self.output_dir / "trained_model_ppo"
            ppo_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(ppo_dir))
            self.tokenizer.save_pretrained(str(ppo_dir))
            print(f"  ✓ PPO model: {ppo_dir}/")

            reward_dir = self.output_dir / "reward_model"
            reward_dir.mkdir(parents=True, exist_ok=True)
            self.reward_model.save_pretrained(str(reward_dir))
            self.tokenizer.save_pretrained(str(reward_dir))
            print(f"  ✓ Reward model: {reward_dir}/")

            ref_dir = self.output_dir / "reference_model"
            ref_dir.mkdir(parents=True, exist_ok=True)
            self.ref_model.pretrained_model.save_pretrained(str(ref_dir))
            self.tokenizer.save_pretrained(str(ref_dir))
            print(f"  ✓ Reference model: {ref_dir}/")
        else:
            print("  Skipping model checkpoints (use --save-models to enable)")

        # ── 7. Training metadata for provenance ──
        meta = {
            "base_model": config.MODEL_NAME,
            "reward_model": config.REWARD_MODEL_NAME,
            "training_steps": self.args.steps,
            "fpga_offload": config.USE_FPGA_OFFLOAD,
            "fpga_mock": config.USE_MOCK_FPGA,
            "use_lab1_fpga": config.USE_LAB1_FPGA,
            "fpga_policy_blocks": self.policy_blocks,
            "fpga_reward_blocks": self.reward_blocks,
            "fpga_response_length": self.fpga_response_length,
            "fpga_precision_default": self.precision_controller.default_precision,
            "fpga_group_size": self.precision_controller.default_group_size,
            "fpga_allow_gradient_offload": self.precision_controller.allow_gradient_offload,
            "fpga_policy_name": self.precision_controller.policy_name,
            "fpga_policy_path": self.precision_controller.policy_path,
            "fp16": config.FP16,
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "ppo_epochs": config.PPO_EPOCHS,
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "max_prompt_length": config.MAX_PROMPT_LENGTH,
            "max_response_length": config.MAX_RESPONSE_LENGTH,
            "dataset": self.dataset_source or config.DATASET_NAME,
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
            if self.reward_blocks is None:
                replace_linear_with_fpga(
                    self.reward_model,
                    self.fpga_offloader,
                    controller=self.precision_controller,
                    model_role="reward",
                )
            else:
                replace_linear_with_fpga_selective(
                    self.reward_model,
                    self.fpga_offloader,
                    target_blocks=self.reward_blocks,
                    controller=self.precision_controller,
                    model_role="reward",
                )
            if self.policy_blocks:
                replace_linear_with_fpga_selective(
                    self.model, self.fpga_offloader,
                    target_blocks=self.policy_blocks,
                    controller=self.precision_controller,
                    model_role="policy",
                )
                replace_linear_with_fpga_selective(
                    self.ref_model, self.fpga_offloader,
                    target_blocks=self.policy_blocks,
                    controller=self.precision_controller,
                    model_role="reference",
                )

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
        fpga_stats_payload = dict(self.fpga_stats)
        fpga_stats_payload["precision_controller"] = self.precision_controller.get_stats()
        with open(self.output_dir / "fpga_stats.json", "w") as f:
            json.dump(fpga_stats_payload, f, indent=2)

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
    parser.add_argument(
        "--save-models",
        action="store_true",
        default=False,
        dest="save_models",
        help="Save full model checkpoints (~4GB). Off by default to avoid disk overflow.",
    )
    parser.add_argument(
        "--policy-json",
        type=str,
        default=None,
        help="Path to a policy JSON file generated from pytorch_profiling/define_policies.py",
    )
    parser.add_argument(
        "--policy-name",
        type=str,
        default=None,
        help="Named policy to load from the policy JSON (for example: A, B, C, D).",
    )
    parser.add_argument(
        "--precision-mode",
        type=str,
        default=None,
        help="Global fallback precision when a policy does not specify a layer (INT8/MXFP8/MXFP4/FP16).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Default MX group size to use with policy-driven or global precision modes (8 or 16).",
    )
    parser.add_argument(
        "--allow-gradient-offload",
        action="store_true",
        default=False,
        help="Allow policy/controller to offload gradient-phase layers. Disabled by default because the current offload path is not autograd-safe.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override the policy/reference base model.",
    )
    parser.add_argument(
        "--reward-model-name",
        type=str,
        default=None,
        help="Override the reward model checkpoint.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Override the HuggingFace dataset name.",
    )
    parser.add_argument(
        "--use-hub-dataset",
        action="store_true",
        default=False,
        help="Force loading the dataset from HuggingFace Hub.",
    )
    parser.add_argument(
        "--local-dataset-path",
        type=str,
        default=None,
        help="Load a local JSON/JSONL dataset instead of the Hub dataset.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override total dataset samples to draw after shuffling.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Override the training subset size.",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=None,
        help="Override the held-out evaluation subset size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override PPO batch size.",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=None,
        help="Override PPO mini-batch size.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override max sequence length.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Override max prompt length.",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=None,
        help="Override max generated response length.",
    )
    parser.add_argument(
        "--fpga-response-length",
        type=int,
        default=None,
        help="Override rollout/eval response length when FPGA offload is active.",
    )
    parser.add_argument(
        "--pretrain-reward-steps",
        type=int,
        default=None,
        help="Override reward-head pretraining steps.",
    )
    parser.add_argument(
        "--policy-blocks",
        type=str,
        default=None,
        help="Comma-separated transformer block indices to offload on the policy/reference models.",
    )
    parser.add_argument(
        "--reward-policy-blocks",
        type=str,
        default=None,
        help="Comma-separated transformer block indices to offload on the reward model. Omit to offload all reward blocks.",
    )
    parser.add_argument(
        "--use-mock-fpga",
        action="store_true",
        default=False,
        help="Force mock-FPGA mode for local smoke runs.",
    )
    parser.add_argument(
        "--no-fpga-offload",
        action="store_true",
        default=False,
        help="Disable FPGA offload entirely.",
    )
    args = parser.parse_args()

    apply_runtime_overrides(args)

    # Initialize trainer
    trainer = RLHFWithFPGATrainer(args)

    # Load everything
    trainer.load_models()
    trainer.load_dataset()

    # Pre-train reward model head on preference pairs
    trainer.pretrain_reward_model()

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
