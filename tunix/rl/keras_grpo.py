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

Enables local and cross-backend RLVR / GRPO / DAPO / Dr.GRPO / GSPO training
for Keras 3 and KerasHub models, optimized for Apple Silicon unified memory
with the Keras MLX backend.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import inspect
import time
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


def _iter_model_layers(model: Any):
  """Yields all layers in a Keras model or backbone."""
  if hasattr(model, "_flatten_layers"):
    yield from model._flatten_layers(include_self=False)
    return
  stack = list(getattr(model, "layers", []))
  seen = set()
  while stack:
    layer = stack.pop(0)
    if id(layer) in seen:
      continue
    seen.add(id(layer))
    yield layer
    sublayers = getattr(layer, "layers", None)
    if sublayers:
      stack.extend(sublayers)


def has_lora_layers(model: Any) -> bool:
  """Returns True if any layer in `model` currently has LoRA enabled."""
  return any(
      bool(getattr(layer, "lora_enabled", False))
      for layer in _iter_model_layers(model)
  )


@contextlib.contextmanager
def disable_lora(model: Any):
  """Temporarily disables active LoRA adapters for zero-copy reference KL."""
  toggled_layers = []
  for layer in _iter_model_layers(model):
    if getattr(layer, "lora_enabled", False):
      layer.lora_enabled = False
      toggled_layers.append(layer)
  try:
    yield model
  finally:
    for layer in toggled_layers:
      layer.lora_enabled = True


def count_parameters(model: Any) -> Tuple[int, int]:
  """Returns `(total_params, trainable_params)` for a Keras model."""
  total = int(sum(np.prod(w.shape) for w in model.weights))
  trainable = int(sum(np.prod(w.shape) for w in model.trainable_variables))
  return total, trainable


@dataclasses.dataclass(slots=True, kw_only=True)
class KerasGRPOConfig(algo_config_lib.AlgorithmConfig):
  """Configuration for Keras GRPO / DAPO / Dr.GRPO / GSPO training.

  Attributes:
    algo_variant: One of `'grpo'`, `'drgrpo'`, `'dapo'`, or `'gspo'`.
    loss_agg_mode: Loss aggregation strategy (`'token-mean'`,
      `'sequence-mean-token-mean'`, or `'sequence-mean-token-scale'`). Defaults
      to `'sequence-mean-token-scale'` for `'drgrpo'`,
      `'sequence-mean-token-mean'` for `'grpo'`, and `'token-mean'` for
      `'dapo'` / `'gspo'`.
    num_generations: Number of completions sampled per prompt (G in paper).
    num_iterations: Number of optimization iterations per rollout batch (mu).
    beta: KL divergence penalty coefficient with reference model.
    epsilon: PPO/GRPO clipping epsilon for importance ratio.
    epsilon_high: Optional asymmetric upper clipping epsilon (defaults to
      `0.28` when `algo_variant == 'dapo'`, else `epsilon`).
    temperature: Sampling temperature for autoregressive generation.
    top_p: Nucleus sampling probability threshold in `(0, 1]`.
    max_tokens_to_generate: Maximum tokens generated during rollout.
    pad_token_id: Token ID used for sequence padding.
    eos_token_id: Optional end-of-sequence token ID.
    stop_token_ids: Optional extra stop token IDs terminating rollout.
    dynamic_sampling: Whether to skip zero-variance reward groups (defaults to
      `True` when `algo_variant == 'dapo'`, else `False`).
    train_micro_batch_size: Optional micro-batch size for memory-bounded
      gradient accumulation across unified RAM.
    max_grad_norm: Optional global L2 norm threshold for gradient clipping.
    use_kv_cache: Whether to use `model.call_with_cache` when available.
    max_turns: Maximum tool-interaction turns during rollout.
    offload_optimizer_to_disk: Whether to persist optimizer state to disk
      between rollout and optimization to minimize peak RAM on Apple Silicon.
    log_ref_kl: If True, evaluates reference KL via `disable_lora` /
      `reference_model` for telemetry even when `beta == 0.0`.
  """

  algo_variant: str = "grpo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "grpo"
  reward_manager: str = "sequence-level"
  loss_agg_mode: Optional[str] = None

  num_generations: int = 4
  num_iterations: int = 1
  beta: float = 0.0
  epsilon: float = 0.2
  epsilon_high: Optional[float] = None
  temperature: float = 0.8
  top_p: float = 1.0
  max_tokens_to_generate: int = 32
  pad_token_id: int = 0
  eos_token_id: Optional[int] = None
  stop_token_ids: Optional[Tuple[int, ...]] = None
  dynamic_sampling: Optional[bool] = None
  train_micro_batch_size: Optional[int] = None
  max_grad_norm: Optional[float] = 1.0
  use_kv_cache: bool = True
  max_turns: int = 1
  offload_optimizer_to_disk: bool = False
  log_ref_kl: bool = False

  def __post_init__(self):
    valid_algo_variants = ("grpo", "drgrpo", "dapo", "gspo")
    valid_loss_agg_modes = (
        "token-mean",
        "sequence-mean-token-mean",
        "sequence-mean-token-scale",
    )
    if self.algo_variant not in valid_algo_variants:
      raise ValueError(
          f"algo_variant must be one of {valid_algo_variants}. "
          f"Received: {self.algo_variant!r}"
      )
    if self.num_generations <= 1:
      raise ValueError(
          "num_generations must be > 1 for GRPO relative advantage, "
          f"got {self.num_generations}"
      )
    if self.epsilon_high is None:
      self.epsilon_high = (
          0.28 if self.algo_variant == "dapo" else self.epsilon
      )
    if self.epsilon_high < self.epsilon:
      raise ValueError(
          f"epsilon_high ({self.epsilon_high}) must be >= epsilon ({self.epsilon})."
      )
    if self.dynamic_sampling is None:
      self.dynamic_sampling = self.algo_variant == "dapo"
    if self.loss_agg_mode is None:
      if self.algo_variant == "drgrpo":
        self.loss_agg_mode = "sequence-mean-token-scale"
      elif self.algo_variant == "grpo":
        self.loss_agg_mode = "sequence-mean-token-mean"
      else:
        self.loss_agg_mode = "token-mean"
    if self.loss_agg_mode not in valid_loss_agg_modes:
      raise ValueError(
          f"loss_agg_mode must be one of {valid_loss_agg_modes}. "
          f"Received: {self.loss_agg_mode!r}"
      )
    if not (0.0 < self.top_p <= 1.0):
      raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
    if self.train_micro_batch_size is not None and self.train_micro_batch_size < 1:
      raise ValueError(
          "train_micro_batch_size must be >= 1 when set, "
          f"got {self.train_micro_batch_size}"
      )
    if self.max_turns < 1:
      raise ValueError(f"max_turns must be >= 1, got {self.max_turns}")

    self.print_config()


@dataclasses.dataclass
class KerasRolloutOutput:
  """Container for rollout generation results."""

  prompt_ids: np.ndarray  # Shape: (B * G, prompt_len)
  completion_ids: np.ndarray  # Shape: (B * G, comp_len)
  completion_mask: np.ndarray  # Shape: (B * G, comp_len) — loss mask
  old_log_probs: np.ndarray  # Shape: (B * G, comp_len)
  prompt_mask: Optional[np.ndarray] = None  # Shape: (B * G, prompt_len)
  attention_completion_mask: Optional[np.ndarray] = None  # Shape: (B * G, comp_len)


def compute_group_advantages(
    rewards: Sequence[float] | np.ndarray,
    num_generations: int,
    eps: float = 1e-4,
    algo_variant: str = "grpo",
) -> np.ndarray:
  """Compute group relative advantages across G generations per prompt.

  Args:
    rewards: 1D array of scalar rewards of length B * G.
    num_generations: Number of generations G per prompt.
    eps: Small constant for numerical stability.
    algo_variant: Algorithm variant; `'drgrpo'` omits std normalization to avoid
      difficulty/length bias.

  Returns:
    1D array of advantages with shape (B * G,).
  """
  rewards = np.asarray(rewards, dtype=np.float32)
  grouped = rewards.reshape(-1, num_generations)
  mean = grouped.mean(axis=-1, keepdims=True)
  if algo_variant == "drgrpo":
    adv = grouped - mean
  else:
    std = grouped.std(axis=-1, ddof=0, keepdims=True)
    adv = (grouped - mean) / (std + eps)
  return adv.reshape(-1)


def compute_kl_divergence(
    log_probs: Any,
    ref_log_probs: Any,
    method: str = "low_var_kl",
) -> Any:
  """Compute per-token KL divergence between policy and reference."""
  if method == "kl":
    return log_probs - ref_log_probs
  elif method == "mse_kl":
    return 0.5 * ops.square(log_probs - ref_log_probs)
  elif method == "low_var_kl":
    log_ratio = ops.clip(ref_log_probs - log_probs, -20.0, 20.0)
    return ops.exp(log_ratio) - log_ratio - 1.0
  else:
    raise ValueError(f"Unsupported KL method: {method}")


def _is_dict_input_model(model: Any) -> bool:
  """Returns True if `model` expects a dict `{'token_ids', 'padding_mask'}`."""
  return isinstance(getattr(model, "input", None), dict) or hasattr(
      model, "backbone"
  )


def _can_slice_reversible_lm_head(model: Any) -> bool:
  """Returns True if `model.backbone.token_embedding(..., reverse=True)` exists."""
  backbone = getattr(model, "backbone", None)
  if backbone is None:
    return False
  emb = getattr(backbone, "token_embedding", None)
  if emb is None or not hasattr(emb, "call"):
    return False
  return "reverse" in inspect.signature(emb.call).parameters


def _forward_completion_logits(
    model: Any,
    full_ids_tensor: Any,
    full_mask_tensor: Any,
    prompt_len: int,
    comp_len: int,
) -> Any:
  """Computes completion-token logits without projecting prompt tokens when possible."""
  start = prompt_len - 1
  end = prompt_len + comp_len - 1
  if _is_dict_input_model(model):
    inputs = {
        "token_ids": full_ids_tensor,
        "padding_mask": full_mask_tensor,
    }
    if _can_slice_reversible_lm_head(model):
      hidden_states = model.backbone(inputs)
      c_hidden = hidden_states[:, start:end, :]
      return model.backbone.token_embedding(c_hidden, reverse=True)
    logits = model(inputs)
    return logits[:, start:end, :]
  logits = model(full_ids_tensor)
  return logits[:, start:end, :]


def _forward_last_logits(model: Any, ids_np: np.ndarray) -> Any:
  """Runs an uncached forward pass and returns next-token logits `(B, V)`."""
  ids_tensor = ops.convert_to_tensor(ids_np)
  if _is_dict_input_model(model):
    mask_tensor = ops.ones_like(ids_tensor, dtype="int32")
    inputs = {"token_ids": ids_tensor, "padding_mask": mask_tensor}
    if _can_slice_reversible_lm_head(model):
      hidden_states = model.backbone(inputs)
      return model.backbone.token_embedding(hidden_states[:, -1:, :], reverse=True)[
          :, 0, :
      ]
    logits = model(inputs)
    return logits[:, -1, :]
  logits = model(ids_tensor)
  return logits[:, -1, :]


def _maybe_mlx_eval(*values: Any) -> None:
  """Evaluates MLX arrays eagerly to prevent unbounded lazy graphs."""
  if keras is not None and keras.backend.backend() == "mlx":
    import mlx.core as mx

    flat = []
    stack = list(values)
    while stack:
      item = stack.pop(0)
      if item is None:
        continue
      if isinstance(item, (list, tuple)):
        stack.extend(item)
      elif isinstance(item, mx.array):
        flat.append(item)
    if flat:
      mx.eval(*flat)


def _maybe_mlx_clear_cache() -> None:
  """Releases unused MLX Metal buffer cache between phases."""
  if keras is not None and keras.backend.backend() == "mlx":
    import mlx.core as mx

    if hasattr(mx, "clear_cache"):
      mx.clear_cache()


def _sample_next_token(
    raw_logits: Any,
    temperature: float,
    top_p: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
  """Samples next tokens and returns `(tokens_np, unscaled_log_probs_np)`."""
  logits_f32 = ops.cast(raw_logits, "float32")
  step_log_probs = ops.log_softmax(logits_f32, axis=-1)

  if temperature <= 1e-4:
    sampled_tensor = ops.cast(ops.argmax(logits_f32, axis=-1), "int32")
    sampled_np = np.asarray(ops.convert_to_numpy(sampled_tensor), dtype=np.int32)
  else:
    scaled = logits_f32 / max(float(temperature), 1e-4)
    probs = np.asarray(
        ops.convert_to_numpy(ops.softmax(scaled, axis=-1)), dtype=np.float64
    )
    probs = np.where(np.isfinite(probs), np.clip(probs, 0.0, None), 0.0)
    sampled_list = []
    for row in probs:
      row_sum = row.sum()
      if row_sum <= 0.0:
        row = np.ones_like(row) / len(row)
      else:
        row = row / row_sum
      if 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(row)[::-1]
        sorted_probs = row[sorted_idx]
        cum_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cum_probs, top_p, side="right") + 1
        keep_idx = sorted_idx[:cutoff]
        filtered = np.zeros_like(row)
        filtered[keep_idx] = row[keep_idx]
        f_sum = filtered.sum()
        row = filtered / f_sum if f_sum > 0.0 else row
      sampled_list.append(int(np.random.choice(len(row), p=row)))
    sampled_np = np.asarray(sampled_list, dtype=np.int32)
    sampled_tensor = ops.convert_to_tensor(sampled_np)

  sampled_logp = ops.squeeze(
      ops.take_along_axis(
          step_log_probs,
          ops.expand_dims(sampled_tensor, -1),
          axis=-1,
      ),
      axis=-1,
  )
  logp_np = np.asarray(ops.convert_to_numpy(sampled_logp), dtype=np.float32)
  return sampled_np, logp_np


class KerasRollout:
  """Autoregressive rollout generator for Keras and KerasHub models."""

  def __init__(
      self,
      model: keras.Model,
      config: KerasGRPOConfig,
  ):
    self.model = model
    self.config = config
    self._cache_uses_padding_mask: Optional[bool] = None
    self._cache_uses_tuple_layers: Optional[bool] = None

  def _effective_stop_ids(self) -> set[int]:
    stop_ids: set[int] = set()
    if self.config.eos_token_id is not None:
      stop_ids.add(int(self.config.eos_token_id))
    if self.config.stop_token_ids:
      stop_ids.update(int(x) for x in self.config.stop_token_ids)
    return stop_ids

  def _inspect_cache_signature(self) -> Tuple[bool, bool]:
    if (
        self._cache_uses_padding_mask is None
        or self._cache_uses_tuple_layers is None
    ):
      sig = inspect.signature(self.model.call_with_cache)
      uses_padding = "padding_mask" in sig.parameters
      uses_tuple = uses_padding and "cache_update_mask" in sig.parameters
      self._cache_uses_padding_mask = uses_padding
      self._cache_uses_tuple_layers = uses_tuple
    return self._cache_uses_padding_mask, self._cache_uses_tuple_layers

  def _build_empty_cache(self, batch_size: int, max_length: int) -> Any:
    backbone = self.model.backbone
    num_layers = int(backbone.num_layers)
    num_kv_heads = int(
        getattr(backbone, "num_key_value_heads", getattr(backbone, "num_query_heads"))
    )
    head_dim = getattr(backbone, "head_dim", None)
    if not head_dim:
      head_dim = int(backbone.hidden_dim) // int(backbone.num_query_heads)
    head_dim = int(head_dim)
    dtype = getattr(self.model, "compute_dtype", "float32")
    _, uses_tuple = self._inspect_cache_signature()
    if uses_tuple:
      layer_shape = (batch_size, max_length, num_kv_heads, head_dim)
      cache = tuple(
          (ops.zeros(layer_shape, dtype=dtype), ops.zeros(layer_shape, dtype=dtype))
          for _ in range(num_layers)
      )
    else:
      shape = (batch_size, num_layers, 2, max_length, num_kv_heads, head_dim)
      cache = ops.zeros(shape, dtype=dtype)
    _maybe_mlx_eval(cache)
    return cache

  def _kv_cache_step(
      self,
      token_ids_np: np.ndarray,
      cache: Any,
      cache_update_index: int,
      full_padding_mask: Optional[Any] = None,
  ) -> Tuple[Any, Any]:
    uses_padding, _ = self._inspect_cache_signature()
    token_tensor = ops.convert_to_tensor(token_ids_np)
    if uses_padding:
      logits, _, cache = self.model.call_with_cache(
          token_ids=token_tensor,
          cache=cache,
          cache_update_index=cache_update_index,
          padding_mask=full_padding_mask,
      )
    else:
      logits, _, cache = self.model.call_with_cache(
          token_tensor,
          cache,
          cache_update_index,
      )
    return logits, cache

  def _normalize_prompts(
      self,
      prompt_tokens: Union[np.ndarray, Sequence[Sequence[int]]],
  ) -> List[List[int]]:
    if isinstance(prompt_tokens, np.ndarray):
      if prompt_tokens.ndim != 2:
        raise ValueError(
            f"Expected 2D prompt_tokens array, got shape {prompt_tokens.shape}"
        )
      return [row.astype(np.int32).tolist() for row in prompt_tokens]
    return [[int(t) for t in row] for row in prompt_tokens]

  def _generate_group_single_turn(
      self,
      prompt_1d: List[int],
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates `num_generations` single-turn completions for one prompt."""
    num_g = self.config.num_generations
    max_new = self.config.max_tokens_to_generate
    stop_ids = self._effective_stop_ids()
    prompt_arr = np.asarray(prompt_1d, dtype=np.int32)
    repeated_prompts = np.repeat(prompt_arr[None, :], num_g, axis=0)
    prompt_len = repeated_prompts.shape[1]

    use_cache = (
        self.config.use_kv_cache
        and hasattr(self.model, "call_with_cache")
        and hasattr(self.model, "backbone")
    )

    generated_tokens: List[np.ndarray] = []
    recorded_logps: List[np.ndarray] = []
    mask_steps: List[np.ndarray] = []
    finished = np.zeros(num_g, dtype=bool)

    if use_cache:
      max_len = prompt_len + max_new
      cache = self._build_empty_cache(num_g, max_len)
      full_pad_mask = None
      if self._inspect_cache_signature()[0]:
        full_pad_mask = ops.ones((num_g, max_len), dtype="int32")

      logits, cache = self._kv_cache_step(
          repeated_prompts, cache, 0, full_pad_mask
      )
      next_logits = logits[:, -1, :]
      _maybe_mlx_eval(next_logits, cache)

      for step in range(max_new):
        tok_np, logp_np = _sample_next_token(
            next_logits, self.config.temperature, self.config.top_p
        )
        step_mask = (~finished).astype(np.float32)
        mask_steps.append(step_mask)
        if stop_ids:
          finished = finished | np.isin(tok_np, list(stop_ids))

        generated_tokens.append(tok_np)
        recorded_logps.append(logp_np * step_mask)
        if np.all(finished) or step == max_new - 1:
          break

        logits, cache = self._kv_cache_step(
            tok_np[:, None],
            cache,
            prompt_len + step,
            full_pad_mask,
        )
        next_logits = logits[:, -1, :]
        _maybe_mlx_eval(next_logits, cache)
    else:
      current_ids = repeated_prompts
      for _ in range(max_new):
        next_logits = _forward_last_logits(self.model, current_ids)
        _maybe_mlx_eval(next_logits)
        tok_np, logp_np = _sample_next_token(
            next_logits, self.config.temperature, self.config.top_p
        )
        step_mask = (~finished).astype(np.float32)
        mask_steps.append(step_mask)
        if stop_ids:
          finished = finished | np.isin(tok_np, list(stop_ids))

        generated_tokens.append(tok_np)
        recorded_logps.append(logp_np * step_mask)
        current_ids = np.concatenate([current_ids, tok_np[:, None]], axis=1)
        if np.all(finished):
          break

    comp_ids = np.stack(generated_tokens, axis=1)
    loss_mask = np.stack(mask_steps, axis=1)
    attn_mask = loss_mask.copy()
    old_logps = np.stack(recorded_logps, axis=1)
    return comp_ids, loss_mask, attn_mask, old_logps

  def _generate_single_with_tools(
      self,
      prompt_1d: List[int],
      tool_step_fn: Callable[[List[int]], Optional[List[int]]],
  ) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Generates one multi-turn trajectory, masking tool observations from loss."""
    max_new = self.config.max_tokens_to_generate
    stop_ids = self._effective_stop_ids()
    use_cache = (
        self.config.use_kv_cache
        and hasattr(self.model, "call_with_cache")
        and hasattr(self.model, "backbone")
    )

    comp_tokens: List[int] = []
    loss_mask: List[float] = []
    attn_mask: List[float] = []
    old_logps: List[float] = []
    turns_used = 1

    prompt_arr = np.asarray(prompt_1d, dtype=np.int32)[None, :]
    prompt_len = len(prompt_1d)

    if use_cache:
      max_len = prompt_len + max_new
      cache = self._build_empty_cache(1, max_len)
      full_pad_mask = None
      if self._inspect_cache_signature()[0]:
        full_pad_mask = ops.ones((1, max_len), dtype="int32")

      logits, cache = self._kv_cache_step(prompt_arr, cache, 0, full_pad_mask)
      next_logits = logits[:, -1, :]
      _maybe_mlx_eval(next_logits, cache)
      cur_idx = prompt_len
    else:
      current_seq = prompt_arr
      next_logits = _forward_last_logits(self.model, current_seq)
      _maybe_mlx_eval(next_logits)

    while len(comp_tokens) < max_new:
      tok_np, logp_np = _sample_next_token(
          next_logits, self.config.temperature, self.config.top_p
      )
      tok = int(tok_np[0])
      logp = float(logp_np[0])
      comp_tokens.append(tok)
      loss_mask.append(1.0)
      attn_mask.append(1.0)
      old_logps.append(logp)

      if tok in stop_ids or len(comp_tokens) >= max_new:
        break

      if use_cache:
        logits, cache = self._kv_cache_step(
            np.array([[tok]], dtype=np.int32), cache, cur_idx, full_pad_mask
        )
        cur_idx += 1
        next_logits = logits[:, -1, :]
        _maybe_mlx_eval(next_logits, cache)
      else:
        current_seq = np.concatenate(
            [current_seq, np.array([[tok]], dtype=np.int32)], axis=1
        )

      if turns_used < self.config.max_turns:
        obs_tokens = tool_step_fn(comp_tokens)
        if obs_tokens:
          remaining = max_new - len(comp_tokens)
          if remaining <= 0:
            break
          obs_chunk = [int(t) for t in obs_tokens[:remaining]]
          comp_tokens.extend(obs_chunk)
          loss_mask.extend([0.0] * len(obs_chunk))
          attn_mask.extend([1.0] * len(obs_chunk))
          old_logps.extend([0.0] * len(obs_chunk))
          turns_used += 1

          if len(comp_tokens) >= max_new:
            break
          if use_cache:
            obs_arr = np.asarray(obs_chunk, dtype=np.int32)[None, :]
            logits, cache = self._kv_cache_step(
                obs_arr, cache, cur_idx, full_pad_mask
            )
            cur_idx += len(obs_chunk)
            next_logits = logits[:, -1, :]
            _maybe_mlx_eval(next_logits, cache)
          else:
            obs_arr = np.asarray(obs_chunk, dtype=np.int32)[None, :]
            current_seq = np.concatenate([current_seq, obs_arr], axis=1)

      if not use_cache:
        next_logits = _forward_last_logits(self.model, current_seq)
        _maybe_mlx_eval(next_logits)

    return comp_tokens, loss_mask, attn_mask, old_logps

  def generate(
      self,
      prompt_tokens: Union[np.ndarray, Sequence[Sequence[int]]],
      tool_step_fn: Optional[Callable[[List[int]], Optional[List[int]]]] = None,
  ) -> KerasRolloutOutput:
    """Generates G completions per prompt with token log-probabilities."""
    prompt_list = self._normalize_prompts(prompt_tokens)
    num_g = self.config.num_generations
    pad_id = int(self.config.pad_token_id)

    all_prompts: List[List[int]] = []
    all_comps: List[List[int]] = []
    all_loss_masks: List[List[float]] = []
    all_attn_masks: List[List[float]] = []
    all_old_logps: List[List[float]] = []

    use_tools = tool_step_fn is not None and self.config.max_turns > 1

    for p_1d in prompt_list:
      if use_tools:
        for _ in range(num_g):
          c, lm, am, lp = self._generate_single_with_tools(p_1d, tool_step_fn)
          all_prompts.append(p_1d)
          all_comps.append(c)
          all_loss_masks.append(lm)
          all_attn_masks.append(am)
          all_old_logps.append(lp)
      else:
        comp_arr, lm_arr, am_arr, lp_arr = self._generate_group_single_turn(p_1d)
        for i in range(num_g):
          all_prompts.append(p_1d)
          all_comps.append(comp_arr[i].tolist())
          all_loss_masks.append(lm_arr[i].tolist())
          all_attn_masks.append(am_arr[i].tolist())
          all_old_logps.append(lp_arr[i].tolist())

    max_prompt_len = max(len(p) for p in all_prompts)
    max_comp_len = max(len(c) for c in all_comps)
    total_n = len(all_prompts)

    # Left-pad prompts, right-pad completions.
    padded_prompts = np.full((total_n, max_prompt_len), pad_id, dtype=np.int32)
    prompt_masks = np.zeros((total_n, max_prompt_len), dtype=np.float32)
    padded_comps = np.full((total_n, max_comp_len), pad_id, dtype=np.int32)
    padded_loss_masks = np.zeros((total_n, max_comp_len), dtype=np.float32)
    padded_attn_masks = np.zeros((total_n, max_comp_len), dtype=np.float32)
    padded_old_logps = np.zeros((total_n, max_comp_len), dtype=np.float32)

    for i in range(total_n):
      p = all_prompts[i]
      if p:
        padded_prompts[i, max_prompt_len - len(p) :] = p
        prompt_masks[i, max_prompt_len - len(p) :] = 1.0
      c = all_comps[i]
      c_len = len(c)
      if c_len:
        padded_comps[i, :c_len] = c
        padded_loss_masks[i, :c_len] = all_loss_masks[i]
        padded_attn_masks[i, :c_len] = all_attn_masks[i]
        padded_old_logps[i, :c_len] = all_old_logps[i]

    return KerasRolloutOutput(
        prompt_ids=padded_prompts,
        completion_ids=padded_comps,
        completion_mask=padded_loss_masks,
        old_log_probs=padded_old_logps,
        prompt_mask=prompt_masks,
        attention_completion_mask=padded_attn_masks,
    )


class KerasGRPOTrainer:
  """Trainer managing the GRPO/DAPO/Dr.GRPO/GSPO lifecycle for Keras models."""

  def __init__(
      self,
      model: keras.Model,
      optimizer: keras.optimizers.Optimizer,
      config: KerasGRPOConfig,
      reward_fn: Callable[..., Sequence[float]],
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

  def _compute_logps_for_model(
      self,
      model: keras.Model,
      full_ids: np.ndarray,
      full_mask: np.ndarray,
      prompt_len: int,
      comp_len: int,
      completion_ids: np.ndarray,
  ) -> np.ndarray:
    mb_size = self.config.train_micro_batch_size or full_ids.shape[0]
    chunks = []
    for start in range(0, full_ids.shape[0], mb_size):
      end = start + mb_size
      f_ids_t = ops.convert_to_tensor(full_ids[start:end])
      f_mask_t = ops.convert_to_tensor(full_mask[start:end])
      c_ids_t = ops.convert_to_tensor(completion_ids[start:end])
      c_logits = _forward_completion_logits(
          model, f_ids_t, f_mask_t, prompt_len, comp_len
      )
      c_logits_f32 = ops.cast(c_logits, "float32")
      logps = ops.squeeze(
          ops.take_along_axis(
              ops.log_softmax(c_logits_f32, axis=-1),
              ops.expand_dims(c_ids_t, -1),
              axis=-1,
          ),
          axis=-1,
      )
      _maybe_mlx_eval(logps)
      chunks.append(np.asarray(ops.convert_to_numpy(logps), dtype=np.float32))
    return np.concatenate(chunks, axis=0)

  def compute_ref_log_probs(
      self,
      full_ids: np.ndarray,
      prompt_len: int,
      comp_len: int,
      completion_ids: np.ndarray,
      full_mask: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """Computes log-probabilities under the reference policy."""
    if full_mask is None:
      full_mask = np.ones_like(full_ids, dtype=np.int32)
    if self.reference_model is not None:
      return self._compute_logps_for_model(
          self.reference_model,
          full_ids,
          full_mask,
          prompt_len,
          comp_len,
          completion_ids,
      )
    if has_lora_layers(self.model):
      with disable_lora(self.model):
        return self._compute_logps_for_model(
            self.model,
            full_ids,
            full_mask,
            prompt_len,
            comp_len,
            completion_ids,
        )
    raise ValueError(
        "beta > 0 requires either `reference_model` or a model with LoRA enabled."
    )

  def _aggregate_loss(
      self,
      per_token_loss: Any,
      loss_mask: Any,
      global_token_denom: float,
      global_num_seqs: int,
  ) -> Any:
    mode = self.config.loss_agg_mode
    if mode == "token-mean":
      return ops.sum(per_token_loss * loss_mask) / max(global_token_denom, 1.0)
    elif mode == "sequence-mean-token-scale":
      scale = max(
          float(global_num_seqs) * float(self.config.max_tokens_to_generate),
          1.0,
      )
      return ops.sum(per_token_loss * loss_mask) / scale
    elif mode == "sequence-mean-token-mean":
      seq_tokens = ops.maximum(ops.sum(loss_mask, axis=-1), 1.0)
      seq_means = ops.sum(per_token_loss * loss_mask, axis=-1) / seq_tokens
      return ops.sum(seq_means) / max(float(global_num_seqs), 1.0)
    raise ValueError(f"Unsupported loss_agg_mode: {mode}")

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
      full_mask_tensor: Optional[Any] = None,
      global_token_denom: Optional[float] = None,
      global_num_seqs: Optional[int] = None,
  ) -> Tuple[Any, Dict[str, Any]]:
    """Calculates GRPO / DAPO / Dr.GRPO / GSPO clipped surrogate loss."""
    if full_mask_tensor is None:
      full_mask_tensor = ops.ones_like(full_ids_tensor, dtype="int32")
    loss_mask = ops.cast(comp_mask_tensor, "float32")
    if global_token_denom is None:
      global_token_denom = float(
          np.maximum(np.sum(ops.convert_to_numpy(loss_mask)), 1.0)
      )
    if global_num_seqs is None:
      global_num_seqs = int(full_ids_tensor.shape[0])

    mapping = list(zip(self.model.trainable_variables, trainable_vars))
    with StatelessScope(state_mapping=mapping):
      c_logits = _forward_completion_logits(
          self.model,
          full_ids_tensor,
          full_mask_tensor,
          prompt_len,
          comp_len,
      )
      c_logits_f32 = ops.cast(c_logits, "float32")
      all_log_probs = ops.log_softmax(c_logits_f32, axis=-1)
      log_probs = ops.squeeze(
          ops.take_along_axis(
              all_log_probs,
              ops.expand_dims(comp_tensor, -1),
              axis=-1,
          ),
          axis=-1,
      )

      entropy_per_tok = -ops.sum(
          ops.exp(all_log_probs) * all_log_probs, axis=-1
      )
      mb_tokens = ops.maximum(ops.sum(loss_mask), 1.0)
      mean_entropy = ops.sum(entropy_per_tok * loss_mask) / mb_tokens

      old_logps_f32 = ops.cast(old_logps_tensor, "float32")
      log_ratio = log_probs - old_logps_f32

      if self.config.algo_variant == "gspo":
        seq_lens = ops.maximum(ops.sum(loss_mask, axis=-1, keepdims=True), 1.0)
        seq_log_ratio = ops.sum(log_ratio * loss_mask, axis=-1, keepdims=True) / seq_lens
        seq_log_ratio = ops.clip(seq_log_ratio, -10.0, 10.0)
        log_ratio = (
            log_probs
            - ops.stop_gradient(log_probs)
            + ops.stop_gradient(seq_log_ratio)
        )

      ratio = ops.exp(ops.clip(log_ratio, -20.0, 20.0))
      clipped_ratio = ops.clip(
          ratio,
          1.0 - self.config.epsilon,
          1.0 + self.config.epsilon_high,
      )

      adv_exp = ops.expand_dims(ops.cast(adv_tensor, "float32"), -1)
      pg_losses_1 = -ratio * adv_exp
      pg_losses_2 = -clipped_ratio * adv_exp
      policy_loss = ops.maximum(pg_losses_1, pg_losses_2)
      clipfrac = (
          ops.sum(ops.cast(pg_losses_2 > pg_losses_1, "float32") * loss_mask)
          / mb_tokens
      )

      kl_loss = ops.cast(0.0, "float32")
      if ref_logps_tensor is not None:
        kl = compute_kl_divergence(
            log_probs, ops.cast(ref_logps_tensor, "float32"), method="low_var_kl"
        )
        if self.config.beta > 0.0:
          policy_loss = policy_loss + self.config.beta * kl
        kl_loss = ops.sum(kl * loss_mask) / mb_tokens

      total_loss = self._aggregate_loss(
          policy_loss, loss_mask, global_token_denom, global_num_seqs
      )

      aux = {
          "kl": ops.stop_gradient(kl_loss),
          "entropy": ops.stop_gradient(mean_entropy),
          "pg_clipfrac": ops.stop_gradient(clipfrac),
          "mean_ratio": ops.stop_gradient(ops.mean(ratio)),
      }
      return total_loss, aux

  def _apply_gradients_mlx(
      self,
      full_ids_np: np.ndarray,
      full_mask_np: np.ndarray,
      prompt_len: int,
      comp_len: int,
      comp_np: np.ndarray,
      comp_mask_np: np.ndarray,
      adv_np: np.ndarray,
      old_logps_np: np.ndarray,
      ref_logps_np: Optional[np.ndarray],
  ) -> Tuple[float, Dict[str, float]]:
    """Micro-batched MLX gradient accumulation, clipping, and update."""
    import mlx.core as mx

    num_seqs = full_ids_np.shape[0]
    mb_size = self.config.train_micro_batch_size or num_seqs
    global_token_denom = float(np.maximum(np.sum(comp_mask_np), 1.0))

    trainable_vars = self.model.trainable_variables
    trainable_vals = [v.value for v in trainable_vars]
    acc_grads = [mx.zeros_like(v) for v in trainable_vals]

    total_loss_val = 0.0
    aux_accum: Dict[str, List[Tuple[float, float]]] = {
        "kl": [],
        "entropy": [],
        "pg_clipfrac": [],
    }

    for start in range(0, num_seqs, mb_size):
      end = min(start + mb_size, num_seqs)
      mb_mask = comp_mask_np[start:end]
      mb_weight = float(np.maximum(np.sum(mb_mask), 1.0))

      f_ids_t = ops.convert_to_tensor(full_ids_np[start:end])
      f_mask_t = ops.convert_to_tensor(full_mask_np[start:end])
      c_ids_t = ops.convert_to_tensor(comp_np[start:end])
      c_mask_t = ops.convert_to_tensor(mb_mask)
      adv_t = ops.convert_to_tensor(adv_np[start:end])
      old_lp_t = ops.convert_to_tensor(old_logps_np[start:end])
      ref_lp_t = (
          ops.convert_to_tensor(ref_logps_np[start:end])
          if ref_logps_np is not None
          else None
      )

      captured_aux: Dict[str, Any] = {}

      def loss_wrapper(curr_vals):
        loss, aux = self.grpo_loss(
            curr_vals,
            f_ids_t,
            prompt_len,
            comp_len,
            c_ids_t,
            c_mask_t,
            adv_t,
            old_lp_t,
            ref_lp_t,
            full_mask_tensor=f_mask_t,
            global_token_denom=global_token_denom,
            global_num_seqs=num_seqs,
        )
        captured_aux.update(aux)
        return loss

      val_and_grad_fn = mx.value_and_grad(loss_wrapper)
      loss_val, grads = val_and_grad_fn(trainable_vals)
      acc_grads = [a + g for a, g in zip(acc_grads, grads)]
      mx.eval(loss_val, *acc_grads, *captured_aux.values())

      total_loss_val += float(loss_val.item())
      for k in aux_accum:
        aux_accum[k].append((float(captured_aux[k].item()), mb_weight))

    # Global L2 norm and gradient clipping.
    sq_norm = mx.array(0.0, dtype=mx.float32)
    for g in acc_grads:
      g32 = g.astype(mx.float32)
      sq_norm = sq_norm + mx.sum(g32 * g32)
    mx.eval(sq_norm)
    grad_norm = float(np.sqrt(max(float(sq_norm.item()), 0.0)))

    if self.config.max_grad_norm is not None and grad_norm > self.config.max_grad_norm:
      scale = self.config.max_grad_norm / (grad_norm + 1e-6)
      acc_grads = [g * scale for g in acc_grads]

    # Match each variable's exact dtype before optimizer.apply.
    typed_grads = [
        g.astype(v.value.dtype) if g.dtype != v.value.dtype else g
        for g, v in zip(acc_grads, trainable_vars)
    ]
    self.optimizer.apply(typed_grads, trainable_vars)
    mx.eval(*[v.value for v in trainable_vars])
    _maybe_mlx_clear_cache()

    total_w = sum(w for _, w in aux_accum["entropy"]) or 1.0
    metrics = {
        k: sum(val * w for val, w in pairs) / total_w
        for k, pairs in aux_accum.items()
    }
    metrics["grad_norm"] = grad_norm
    return total_loss_val, metrics

  def train_step(
      self,
      prompt_batch: Union[np.ndarray, Sequence[Sequence[int]]],
      reward_kwargs: Optional[Dict[str, Any]] = None,
      tool_step_fn: Optional[Callable[[List[int]], Optional[List[int]]]] = None,
  ) -> Dict[str, Any]:
    """Runs a single RLVR iteration (rollout, scoring, advantage, update)."""
    t0 = time.perf_counter()
    rollout_out = self.rollout.generate(prompt_batch, tool_step_fn=tool_step_fn)
    self._ensure_optimizer_built()
    rollout_s = time.perf_counter() - t0

    prompt_ids = rollout_out.prompt_ids
    comp_ids = rollout_out.completion_ids
    loss_mask = rollout_out.completion_mask
    attn_comp_mask = (
        rollout_out.attention_completion_mask
        if rollout_out.attention_completion_mask is not None
        else loss_mask
    )
    prompt_mask = (
        rollout_out.prompt_mask
        if rollout_out.prompt_mask is not None
        else np.ones_like(prompt_ids, dtype=np.float32)
    )
    old_logps = rollout_out.old_log_probs

    # 2. Reward & Advantage Computation
    rewards_raw = self.reward_fn(
        prompt_ids.tolist(), comp_ids.tolist(), **(reward_kwargs or {})
    )
    rewards = np.asarray(rewards_raw, dtype=np.float32)
    advantages = compute_group_advantages(
        rewards,
        num_generations=self.config.num_generations,
        algo_variant=self.config.algo_variant,
    )

    response_len_mean = float(np.mean(np.sum(attn_comp_mask, axis=-1)))

    # Optional DAPO dynamic sampling: skip zero-variance reward groups.
    num_g = self.config.num_generations
    grouped_rewards = rewards.reshape(-1, num_g)
    group_stds = grouped_rewards.std(axis=-1)
    zero_var_groups = int(np.sum(group_stds <= 1e-6))

    if self.config.dynamic_sampling:
      active_group_mask = group_stds > 1e-6
      active_row_mask = np.repeat(active_group_mask, num_g)
      skipped_groups = zero_var_groups
    else:
      active_row_mask = np.ones(len(rewards), dtype=bool)
      skipped_groups = 0

    full_ids_all = np.concatenate([prompt_ids, comp_ids], axis=1)
    full_mask_all = np.concatenate(
        [prompt_mask, attn_comp_mask], axis=1
    ).astype(np.int32)
    prompt_len = prompt_ids.shape[1]
    comp_len = comp_ids.shape[1]

    if not np.any(active_row_mask):
      self.step_count += 1
      _maybe_mlx_clear_cache()
      peak_mem_gb = 0.0
      if self.backend_name == "mlx":
        import mlx.core as mx

        if hasattr(mx, "get_peak_memory"):
          peak_mem_gb = mx.get_peak_memory() / (1024**3)
      return {
          "step": self.step_count,
          "loss": 0.0,
          "reward/mean": float(np.mean(rewards)),
          "reward/max": float(np.max(rewards)),
          "reward/min": float(np.min(rewards)),
          "advantages/std": float(np.std(advantages)),
          "entropy": 0.0,
          "pg_clipfrac": 0.0,
          "kl": 0.0,
          "grad_norm": 0.0,
          "response_len/mean": response_len_mean,
          "skipped_zero_var_groups": skipped_groups,
          "rollout_s": rollout_s,
          "train_s": 0.0,
          "peak_mem_gb": peak_mem_gb,
          "rollout_output": rollout_out,
          "rewards": rewards,
      }

    active_full_ids = full_ids_all[active_row_mask]
    active_full_mask = full_mask_all[active_row_mask]
    active_comp_ids = comp_ids[active_row_mask]
    active_loss_mask = loss_mask[active_row_mask]
    active_advantages = advantages[active_row_mask]
    active_old_logps = old_logps[active_row_mask]

    # 3. Reference Log-Probs (if beta > 0 or log_ref_kl telemetry is enabled)
    active_ref_logps = None
    if self.config.beta > 0.0 or self.config.log_ref_kl:
      active_ref_logps = self.compute_ref_log_probs(
          active_full_ids,
          prompt_len,
          comp_len,
          active_comp_ids,
          full_mask=active_full_mask,
      )

    # 4. Optimization Update
    t1 = time.perf_counter()
    losses = []
    last_metrics: Dict[str, float] = {
        "kl": 0.0,
        "entropy": 0.0,
        "pg_clipfrac": 0.0,
        "grad_norm": 0.0,
    }
    for _ in range(self.config.num_iterations):
      if self.backend_name == "mlx":
        loss_val, step_metrics = self._apply_gradients_mlx(
            active_full_ids,
            active_full_mask,
            prompt_len,
            comp_len,
            active_comp_ids,
            active_loss_mask,
            active_advantages,
            active_old_logps,
            active_ref_logps,
        )
        last_metrics = step_metrics
      else:
        trainable_vals = [v.value for v in self.model.trainable_variables]
        loss_t, aux = self.grpo_loss(
            trainable_vals,
            ops.convert_to_tensor(active_full_ids),
            prompt_len,
            comp_len,
            ops.convert_to_tensor(active_comp_ids),
            ops.convert_to_tensor(active_loss_mask),
            ops.convert_to_tensor(active_advantages),
            ops.convert_to_tensor(active_old_logps),
            ops.convert_to_tensor(active_ref_logps)
            if active_ref_logps is not None
            else None,
            full_mask_tensor=ops.convert_to_tensor(active_full_mask),
        )
        loss_val = float(ops.convert_to_numpy(loss_t))
        last_metrics = {
            "kl": float(ops.convert_to_numpy(aux["kl"])),
            "entropy": float(ops.convert_to_numpy(aux["entropy"])),
            "pg_clipfrac": float(ops.convert_to_numpy(aux["pg_clipfrac"])),
            "grad_norm": 0.0,
        }
      losses.append(loss_val)
    train_s = time.perf_counter() - t1

    self.step_count += 1
    gc.collect()

    peak_mem_gb = 0.0
    if self.backend_name == "mlx":
      import mlx.core as mx

      if hasattr(mx, "get_peak_memory"):
        peak_mem_gb = mx.get_peak_memory() / (1024**3)

    return {
        "step": self.step_count,
        "loss": float(np.mean(losses)),
        "reward/mean": float(np.mean(rewards)),
        "reward/max": float(np.max(rewards)),
        "reward/min": float(np.min(rewards)),
        "advantages/std": float(np.std(advantages)),
        "entropy": float(last_metrics.get("entropy", 0.0)),
        "pg_clipfrac": float(last_metrics.get("pg_clipfrac", 0.0)),
        "kl": float(last_metrics.get("kl", 0.0)),
        "grad_norm": float(last_metrics.get("grad_norm", 0.0)),
        "response_len/mean": response_len_mean,
        "skipped_zero_var_groups": skipped_groups,
        "rollout_s": rollout_s,
        "train_s": train_s,
        "peak_mem_gb": peak_mem_gb,
        "rollout_output": rollout_out,
        "rewards": rewards,
    }
