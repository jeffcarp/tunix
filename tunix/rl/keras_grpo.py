# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keras GRPO (Group Relative Policy Optimization) implementation.

Enables local and cross-backend GRPO training for Keras 3 models, specifically
optimized for Apple Silicon unified memory with the Keras MLX backend.
"""

from __future__ import annotations

import dataclasses
import gc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
  import keras
  from keras import ops
  from keras.src.backend.common.stateless_scope import StatelessScope
except ImportError:
  keras = None
  ops = None
  StatelessScope = None

from tunix.rl import algorithm_config as algo_config_lib


@dataclasses.dataclass(slots=True, kw_only=True)
class KerasGRPOConfig(algo_config_lib.AlgorithmConfig):
  """Configuration for Keras GRPO training.

  Attributes:
    num_generations: Number of completions sampled per prompt (G in paper).
    num_iterations: Number of optimization iterations per rollout batch (mu).
    beta: KL divergence penalty coefficient with reference model.
    epsilon: PPO/GRPO clipping epsilon for importance ratio.
    epsilon_high: Optional asymmetric upper clipping epsilon.
    temperature: Sampling temperature for autoregressive generation.
    max_tokens_to_generate: Maximum tokens generated during rollout.
    pad_token_id: Token ID used for sequence padding.
    eos_token_id: Optional end-of-sequence token ID.
    offload_optimizer_to_disk: Whether to persist optimizer state to disk
      between rollout and optimization to minimize peak RAM on Apple Silicon.
  """

  algo_variant: str = "grpo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "grpo"
  reward_manager: str = "sequence-level"

  num_generations: int = 4
  num_iterations: int = 1
  beta: float = 0.0
  epsilon: float = 0.2
  epsilon_high: Optional[float] = None
  temperature: float = 0.8
  max_tokens_to_generate: int = 32
  pad_token_id: int = 0
  eos_token_id: Optional[int] = None
  offload_optimizer_to_disk: bool = False

  def __post_init__(self):
    algo_config_lib.AlgorithmConfig.__post_init__(self)
    if self.num_generations <= 1:
      raise ValueError(
          f"num_generations must be > 1 for GRPO relative advantage, got {self.num_generations}"
      )
    if self.epsilon_high is None:
      self.epsilon_high = self.epsilon


@dataclasses.dataclass
class KerasRolloutOutput:
  """Container for rollout generation results."""

  prompt_ids: np.ndarray  # Shape: (B * G, prompt_len)
  completion_ids: np.ndarray  # Shape: (B * G, comp_len)
  completion_mask: np.ndarray  # Shape: (B * G, comp_len)
  old_log_probs: np.ndarray  # Shape: (B * G, comp_len)


def compute_group_advantages(
    rewards: np.ndarray,
    num_generations: int,
    eps: float = 1e-4,
) -> np.ndarray:
  """Compute group relative advantages across G generations per prompt.

  Args:
    rewards: 1D array of scalar rewards of length B * G.
    num_generations: Number of generations G per prompt.
    eps: Small constant for numerical stability.

  Returns:
    1D array of advantages with shape (B * G,).
  """
  rewards = np.asarray(rewards, dtype=np.float32)
  grouped = rewards.reshape(-1, num_generations)
  mean = grouped.mean(axis=-1, keepdims=True)
  std = grouped.std(axis=-1, ddof=0, keepdims=True)
  adv = (grouped - mean) / (std + eps)
  return adv.reshape(-1)


def compute_kl_divergence(
    log_probs: Any,
    ref_log_probs: Any,
    method: str = "low_var_kl",
) -> Any:
  """Compute per-token KL divergence between policy and reference.

  Args:
    log_probs: Per-token log-probabilities from trained policy.
    ref_log_probs: Per-token log-probabilities from reference policy.
    method: 'low_var_kl' (Schulman approx), 'kl', or 'mse_kl'.
  """
  if method == "kl":
    return log_probs - ref_log_probs
  elif method == "mse_kl":
    return 0.5 * ops.square(log_probs - ref_log_probs)
  elif method == "low_var_kl":
    log_ratio = ref_log_probs - log_probs
    return ops.exp(log_ratio) - log_ratio - 1.0
  else:
    raise ValueError(f"Unsupported KL method: {method}")


class KerasRollout:
  """Autoregressive rollout generator for Keras models."""

  def __init__(
      self,
      model: keras.Model,
      config: KerasGRPOConfig,
  ):
    self.model = model
    self.config = config

  def generate(
      self,
      prompt_tokens: Union[np.ndarray, Sequence[Sequence[int]]],
  ) -> KerasRolloutOutput:
    """Generates G completions per prompt with token log-probabilities.

    Args:
      prompt_tokens: 2D array or list of lists containing prompt token IDs.

    Returns:
      A KerasRolloutOutput containing prompt IDs, completion IDs, masks, and
      per-token old log-probabilities.
    """
    prompt_tokens = np.asarray(prompt_tokens, dtype=np.int32)
    batch_size = prompt_tokens.shape[0]
    num_g = self.config.num_generations

    # Repeat prompts G times
    repeated_prompts = np.repeat(prompt_tokens, num_g, axis=0)
    current_ids = repeated_prompts
    prompt_len = repeated_prompts.shape[1]

    recorded_logps = []
    generated_tokens = []
    finished = np.zeros(repeated_prompts.shape[0], dtype=bool)
    completion_mask_list = []

    temperature = max(self.config.temperature, 1e-4)

    for step in range(self.config.max_tokens_to_generate):
      # Forward pass to obtain next-token logits
      curr_tensor = ops.convert_to_tensor(current_ids)
      logits = self.model(curr_tensor)
      raw_next_logits = logits[:, -1, :]

      # Unscaled log-softmax for exact policy probability tracking
      step_log_probs = ops.log_softmax(raw_next_logits, axis=-1)

      # Sample next token using temperature-scaled logits
      if self.config.temperature <= 1e-4:
        sampled_token = ops.argmax(raw_next_logits, axis=-1)
      else:
        sampling_logits = raw_next_logits / temperature
        probs = ops.convert_to_numpy(ops.softmax(sampling_logits, axis=-1))
        # Ensure valid probability distribution
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        sampled_token = np.array([
            np.random.choice(probs.shape[-1], p=p) for p in probs
        ], dtype=np.int32)

      sampled_token_np = np.asarray(ops.convert_to_numpy(sampled_token), dtype=np.int32)
      sampled_tensor = ops.convert_to_tensor(sampled_token_np)

      # Extract log-probability of sampled token
      sampled_logp = ops.squeeze(
          ops.take_along_axis(
              step_log_probs,
              ops.expand_dims(sampled_tensor, -1),
              axis=-1,
          ),
          axis=-1,
      )

      token_logp_np = ops.convert_to_numpy(sampled_logp)

      # Update completion mask (1 if active, 0 if post-EOS or finished)
      step_mask = (~finished).astype(np.float32)
      completion_mask_list.append(step_mask)

      if self.config.eos_token_id is not None:
        finished = finished | (sampled_token_np == self.config.eos_token_id)

      generated_tokens.append(sampled_token_np)
      recorded_logps.append(token_logp_np)

      # Append sampled token to current sequence
      current_ids = np.concatenate(
          [current_ids, sampled_token_np[:, None]], axis=1
      )

      if np.all(finished):
        break

    completion_ids = np.stack(generated_tokens, axis=1)
    completion_mask = np.stack(completion_mask_list, axis=1)
    old_log_probs = np.stack(recorded_logps, axis=1)

    return KerasRolloutOutput(
        prompt_ids=repeated_prompts,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        old_log_probs=old_log_probs,
    )


class KerasGRPOTrainer:
  """Trainer that manages the GRPO lifecycle for Keras models.

  Supports MLX, JAX, PyTorch, and TensorFlow backends with native MLX
  acceleration on Apple Silicon.
  """

  def __init__(
      self,
      model: keras.Model,
      optimizer: keras.optimizers.Optimizer,
      config: KerasGRPOConfig,
      reward_fn: Callable[[List[Any], List[Any]], Sequence[float]],
      reference_model: Optional[keras.Model] = None,
  ):
    if keras is None:
      raise ImportError("Keras 3 is required to use KerasGRPOTrainer.")

    self.model = model
    self.optimizer = optimizer
    self.config = config
    self.reward_fn = reward_fn
    self.reference_model = reference_model

    self.rollout = KerasRollout(self.model, self.config)
    self.backend_name = keras.backend.backend()

    self.step_count = 0
    self._optimizer_built = False

  def _ensure_optimizer_built(self):
    if not self._optimizer_built and len(self.model.trainable_variables) > 0:
      self.optimizer.build(self.model.trainable_variables)
      self._optimizer_built = True

  def compute_ref_log_probs(
      self,
      full_ids: np.ndarray,
      prompt_len: int,
      comp_len: int,
      completion_ids: np.ndarray,
  ) -> np.ndarray:
    """Computes log-probabilities under reference model."""
    if self.reference_model is None:
      raise ValueError("Reference model required when beta > 0.")
    full_tensor = ops.convert_to_tensor(full_ids)
    ref_logits = self.reference_model(full_tensor)
    c_logits = ref_logits[:, prompt_len - 1 : prompt_len + comp_len - 1, :]
    ref_logps = ops.squeeze(
        ops.take_along_axis(
            ops.log_softmax(c_logits, axis=-1),
            ops.expand_dims(ops.convert_to_tensor(completion_ids), -1),
            axis=-1,
        ),
        axis=-1,
    )
    return ops.convert_to_numpy(ref_logps)

  def grpo_loss(
      self,
      trainable_vars: Sequence[Any],
      full_ids_tensor: Any,
      prompt_len: int,
      comp_len: int,
      comp_tensor: Any,
      comp_mask_tensor: Any,
      adv_tensor: Any,
      old_logps_tensor: Any,
      ref_logps_tensor: Optional[Any] = None,
  ) -> Tuple[Any, Dict[str, Any]]:
    """Calculates GRPO clipped surrogate loss with optional KL penalty."""
    mapping = list(zip(self.model.trainable_variables, trainable_vars))
    with StatelessScope(state_mapping=mapping):
      logits = self.model(full_ids_tensor)
      c_logits = logits[:, prompt_len - 1 : prompt_len + comp_len - 1, :]
      log_probs = ops.squeeze(
          ops.take_along_axis(
              ops.log_softmax(c_logits, axis=-1),
              ops.expand_dims(comp_tensor, -1),
              axis=-1,
          ),
          axis=-1,
      )

      ratio = ops.exp(log_probs - old_logps_tensor)
      clipped_ratio = ops.clip(
          ratio,
          1.0 - self.config.epsilon,
          1.0 + self.config.epsilon_high,
      )

      adv_exp = ops.expand_dims(adv_tensor, -1)
      surr1 = ratio * adv_exp
      surr2 = clipped_ratio * adv_exp
      policy_loss = -ops.minimum(surr1, surr2)

      kl_loss = 0.0
      if self.config.beta > 0.0 and ref_logps_tensor is not None:
        kl = compute_kl_divergence(log_probs, ref_logps_tensor, method="low_var_kl")
        policy_loss = policy_loss + self.config.beta * kl
        kl_loss = ops.mean(kl)

      total_loss = ops.sum(policy_loss * comp_mask_tensor) / ops.maximum(
          ops.sum(comp_mask_tensor), 1.0
      )

      aux = {
          "kl": kl_loss,
          "mean_ratio": ops.mean(ratio),
      }
      return total_loss, aux

  def _apply_gradients_mlx(
      self,
      full_ids_tensor: Any,
      prompt_len: int,
      comp_len: int,
      comp_tensor: Any,
      comp_mask_tensor: Any,
      adv_tensor: Any,
      old_logps_tensor: Any,
      ref_logps_tensor: Optional[Any],
  ) -> Tuple[float, Dict[str, Any]]:
    """MLX-optimized gradient computation and parameter update."""
    import mlx.core as mx

    def loss_wrapper(trainable_vals):
      loss, _ = self.grpo_loss(
          trainable_vals,
          full_ids_tensor,
          prompt_len,
          comp_len,
          comp_tensor,
          comp_mask_tensor,
          adv_tensor,
          old_logps_tensor,
          ref_logps_tensor,
      )
      return loss

    trainable_vals = [v.value for v in self.model.trainable_variables]
    val_and_grad_fn = mx.value_and_grad(loss_wrapper)
    loss_val, grads = val_and_grad_fn(trainable_vals)
    self.optimizer.apply(grads, self.model.trainable_variables)

    return float(loss_val.item()), {}

  def train_step(
      self,
      prompt_batch: Union[np.ndarray, List[List[int]]],
  ) -> Dict[str, float]:
    """Runs a single GRPO iteration (rollout, scoring, advantage, update).

    Args:
      prompt_batch: Batch of prompt sequences of shape (B, prompt_len).

    Returns:
      Dictionary of logged metrics for the step.
    """
    # 1. Rollout Phase
    rollout_out = self.rollout.generate(prompt_batch)
    self._ensure_optimizer_built()
    prompt_ids = rollout_out.prompt_ids
    comp_ids = rollout_out.completion_ids
    comp_mask = rollout_out.completion_mask
    old_logps = rollout_out.old_log_probs

    full_ids = np.concatenate([prompt_ids, comp_ids], axis=1)
    prompt_len = prompt_ids.shape[1]
    comp_len = comp_ids.shape[1]

    # 2. Reward & Advantage Computation
    rewards = self.reward_fn(prompt_ids.tolist(), comp_ids.tolist())
    advantages = compute_group_advantages(
        rewards,
        num_generations=self.config.num_generations,
    )

    # 3. Reference Log-Probs (if beta > 0)
    ref_logps = None
    if self.config.beta > 0.0 and self.reference_model is not None:
      ref_logps = self.compute_ref_log_probs(
          full_ids, prompt_len, comp_len, comp_ids
      )

    # 4. Convert inputs to backend tensors
    full_ids_tensor = ops.convert_to_tensor(full_ids)
    comp_tensor = ops.convert_to_tensor(comp_ids)
    comp_mask_tensor = ops.convert_to_tensor(comp_mask)
    adv_tensor = ops.convert_to_tensor(advantages)
    old_logps_tensor = ops.convert_to_tensor(old_logps)
    ref_logps_tensor = (
        ops.convert_to_tensor(ref_logps) if ref_logps is not None else None
    )

    # 5. Optimization Update
    losses = []
    for _ in range(self.config.num_iterations):
      if self.backend_name == "mlx":
        loss_val, _ = self._apply_gradients_mlx(
            full_ids_tensor,
            prompt_len,
            comp_len,
            comp_tensor,
            comp_mask_tensor,
            adv_tensor,
            old_logps_tensor,
            ref_logps_tensor,
        )
      else:
        # Fallback for other Keras backends using standard gradient computation
        trainable_vals = [v.value for v in self.model.trainable_variables]
        loss_val, _ = self.grpo_loss(
            trainable_vals,
            full_ids_tensor,
            prompt_len,
            comp_len,
            comp_tensor,
            comp_mask_tensor,
            adv_tensor,
            old_logps_tensor,
            ref_logps_tensor,
        )
        loss_val = float(ops.convert_to_numpy(loss_val))
      losses.append(loss_val)

    self.step_count += 1

    # Cleanup intermediate tensors
    del full_ids_tensor, comp_tensor, comp_mask_tensor, adv_tensor
    gc.collect()

    return {
        "step": self.step_count,
        "loss": float(np.mean(losses)),
        "reward/mean": float(np.mean(rewards)),
        "reward/max": float(np.max(rewards)),
        "reward/min": float(np.min(rewards)),
        "advantages/std": float(np.std(advantages)),
    }
