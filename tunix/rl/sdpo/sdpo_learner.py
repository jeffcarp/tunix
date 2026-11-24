# Copyright 2026 Google LLC
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

"""SDPO (Self-Distillation Policy Optimization) learner.

Implements Self-Distillation Policy Optimization from Hübotter et al. (2026),
"Reinforcement Learning via Self-Distillation" (arXiv:2601.20802).
"""

from __future__ import annotations

import dataclasses
from functools import partial  # pylint: disable=g-importing-member
from typing import Callable, Iterable, List, Sequence, TypeVar

from absl import logging
import flax
from flax import nnx
from flax.nnx import filterlib
from flax.nnx import statelib
import jax
import jax.numpy as jnp
import numpy as np
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.generate import utils as gen_utils
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner
from tunix.rl import utils as rl_utils
from tunix.rl.grpo import grpo_learner as grpo_learner_lib
from tunix.sft import sharding_utils
from tunix.sft import utils as sft_utils

TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn
FeedbackFn = Callable[..., List[str | None]]
RepromptFormatterFn = Callable[..., str]


@flax.struct.dataclass(frozen=True)
class SDPOTrainExample(common.TrainExample):
  """Training batch element for SDPO."""

  teacher_topk_ids: jax.Array | None = None
  teacher_topk_logps: jax.Array | None = None
  teacher_tail_logp: jax.Array | None = None
  teacher_per_token_logps: jax.Array | None = None
  teacher_full_logps: jax.Array | None = None
  self_distillation_mask: jax.Array | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class SDPOConfig(grpo_learner_lib.GRPOConfig):
  """Configuration for Self-Distillation Policy Optimization (SDPO).

  Attributes:
    distillation_weight: Convex combination weight between GRPO policy gradient
      loss and SDPO self-distillation loss in [0.0, 1.0]. 1.0 uses pure SDPO;
      intermediate values blend `(1 - w) * pg_loss + w * distill_loss`.
    distillation_alpha: Divergence interpolation coefficient in [0.0, 1.0].
      0.0 = forward KL, 1.0 = reverse KL, (0, 1) = generalized JSD.
    distillation_mode: One of `"topk_logits"`, `"full_logits"`, or
      `"sampled_token"`.
    distillation_topk: Number of top student tokens to retain when
      `distillation_mode == "topk_logits"`.
    distillation_add_tail: Whether to include a tail probability bucket for the
      non-top-K vocabulary mass in `"topk_logits"` mode.
    is_clip: Upper clip bound for importance sampling ratio in `"sampled_token"`
      mode (None disables clipping).
    teacher_model_kind: Self-teacher weight strategy: `"ema"`, `"live"`,
      `"base"`, or `"trust_region"`.
    teacher_update_rate: EMA update rate (for `"ema"`) or live mixing weight
      (for `"trust_region"`).
    teacher_sync_steps: How often (in mini-batch steps) to update EMA teacher.
    use_successful_as_teacher: Whether to use successful rollouts in the same
      prompt group as demonstrations for failed rollouts.
    success_reward_threshold: Minimum scalar reward for a rollout to count as
      successful.
    dont_reprompt_on_self_success: If True, rollouts that already succeeded are
      not self-distilled (unless `include_environment_feedback` is True and
      feedback is explicitly provided).
    include_environment_feedback: Whether to include textual environment
      feedback (from `feedback_fn` or `privileged_context`/`feedback` in input).
    environment_feedback_only_without_solution: If True, only include
      environment feedback when no successful sibling rollout is available.
    max_reprompt_length: Maximum token length for reprompted teacher prompts
      (truncated from the left if exceeded).
    reprompt_template: Template for teacher reprompting when using raw text
      prompts.
    solution_template: Template for formatting a successful sibling rollout.
    feedback_template: Template for formatting environment feedback.
  """

  algo_variant: str = "sdpo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "sdpo"
  loss_agg_mode: str = "token-mean"
  num_generations: int = 4
  num_iterations: int = 1
  beta: float = 0.0
  epsilon: float = 0.2
  epsilon_high: float = 0.28

  distillation_weight: float = 1.0
  distillation_alpha: float = 0.5
  distillation_mode: str = "topk_logits"
  distillation_topk: int = 100
  distillation_add_tail: bool = True
  is_clip: float | None = 2.0

  teacher_model_kind: str = "ema"
  teacher_update_rate: float = 0.05
  teacher_sync_steps: int = 1

  use_successful_as_teacher: bool = True
  success_reward_threshold: float = 1.0
  dont_reprompt_on_self_success: bool = True
  include_environment_feedback: bool = True
  environment_feedback_only_without_solution: bool = False
  max_reprompt_length: int = 1024

  reprompt_template: str = (
      "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"
  )
  solution_template: str = (
      "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"
  )
  feedback_template: str = (
      "\nThe following is feedback from your unsuccessful earlier"
      " attempt:\n\n{feedback_raw}\n\n"
  )

  def __post_init__(self):
    if self.num_generations < 1:
      raise ValueError(
          f"num_generations must be >= 1. Received: {self.num_generations}"
      )
    if self.distillation_weight < 1.0 and self.num_generations < 2:
      raise ValueError(
          "num_generations must be >= 2 when distillation_weight < 1.0 "
          "so group-relative GRPO advantages are well-defined."
      )
    if not 0.0 <= self.distillation_weight <= 1.0:
      raise ValueError(
          "distillation_weight must be in [0.0, 1.0]. Received: "
          f"{self.distillation_weight}"
      )
    if not 0.0 <= self.distillation_alpha <= 1.0:
      raise ValueError(
          "distillation_alpha must be in [0.0, 1.0]. Received: "
          f"{self.distillation_alpha}"
      )
    valid_modes = ("topk_logits", "full_logits", "sampled_token")
    if self.distillation_mode not in valid_modes:
      raise ValueError(
          f"distillation_mode must be one of {valid_modes}. Received: "
          f"{self.distillation_mode!r}"
      )
    if self.distillation_mode == "topk_logits" and self.distillation_topk < 1:
      raise ValueError(
          "distillation_topk must be >= 1 for topk_logits mode. Received: "
          f"{self.distillation_topk}"
      )
    valid_teachers = ("ema", "live", "base", "trust_region")
    if self.teacher_model_kind not in valid_teachers:
      raise ValueError(
          f"teacher_model_kind must be one of {valid_teachers}. Received: "
          f"{self.teacher_model_kind!r}"
      )
    if not 0.0 <= self.teacher_update_rate <= 1.0:
      raise ValueError(
          "teacher_update_rate must be in [0.0, 1.0]. Received: "
          f"{self.teacher_update_rate}"
      )
    if self.teacher_sync_steps < 1:
      raise ValueError(
          f"teacher_sync_steps must be >= 1. Received: {self.teacher_sync_steps}"
      )
    if self.loss_algo not in ("grpo", "gspo-token"):
      raise ValueError(
          "loss_algo should be either grpo or gspo-token. Received: "
          f"{self.loss_algo}"
      )


TSdpoConfig = TypeVar("TSdpoConfig", bound=SDPOConfig)


@partial(
    jax.jit,
    static_argnames=("pad_id", "eos_id", "topk"),
)
def _compute_student_topk_ids(
    graphdef,
    state,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    completion_mask: jax.Array | None,
    topk: int,
) -> jax.Array:
  """Computes the top-K token indices under the current student logits."""
  model = nnx.merge(graphdef, state)
  input_tokens, positions, attn_mask = common.process_ids(
      prompt_tokens, completion_tokens, pad_id, eos_id, completion_mask
  )
  logits, _ = model(
      input_tokens, positions=positions, attention_mask=attn_mask, cache=None
  )
  logits_to_keep = completion_tokens.shape[1]
  logits = logits[:, -logits_to_keep - 1 : -1, :].astype(jnp.float32)
  k = min(topk, logits.shape[-1])
  _, topk_ids = jax.lax.top_k(logits, k=k)
  return jax.lax.stop_gradient(topk_ids)


@partial(
    jax.jit,
    static_argnames=("pad_id", "eos_id", "return_full_logps", "mix_rate"),
)
def _compute_teacher_targets(
    graphdef,
    teacher_state,
    live_state,
    teacher_prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    completion_mask: jax.Array | None,
    topk_ids: jax.Array | None,
    return_full_logps: bool,
    mix_rate: float | None,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, jax.Array | None]:
  """Computes stop-gradient teacher log-probabilities on reprompted inputs."""
  input_tokens, positions, attn_mask = common.process_ids(
      teacher_prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      completion_mask,
  )
  logits_to_keep = completion_tokens.shape[1]

  teacher_model = nnx.merge(graphdef, teacher_state)
  t_logits, _ = teacher_model(
      input_tokens, positions=positions, attention_mask=attn_mask, cache=None
  )
  t_logits = t_logits[:, -logits_to_keep - 1 : -1, :].astype(jnp.float32)
  t_log_probs = jax.nn.log_softmax(t_logits, axis=-1)

  if mix_rate is not None and live_state is not None:
    live_model = nnx.merge(graphdef, live_state)
    l_logits, _ = live_model(
        input_tokens, positions=positions, attention_mask=attn_mask, cache=None
    )
    l_logits = l_logits[:, -logits_to_keep - 1 : -1, :].astype(jnp.float32)
    l_log_probs = jax.nn.log_softmax(l_logits, axis=-1)
    # Trust-region teacher: (1 - alpha) * base + alpha * live in probability space
    if mix_rate <= 0.0:
      pass
    elif mix_rate >= 1.0:
      t_log_probs = l_log_probs
    else:
      t_log_probs = jnp.logaddexp(
          jnp.log(1.0 - mix_rate) + t_log_probs,
          jnp.log(mix_rate) + l_log_probs,
      )

  teacher_per_token_logps = jnp.take_along_axis(
      t_log_probs, completion_tokens[..., None], axis=-1
  )[..., 0]

  teacher_topk_logps = None
  teacher_tail_logp = None
  if topk_ids is not None:
    teacher_topk_logps = jnp.take_along_axis(t_log_probs, topk_ids, axis=-1)
    head_prob = jnp.sum(jnp.exp(teacher_topk_logps), axis=-1, keepdims=True)
    teacher_tail_logp = jnp.log(jnp.clip(1.0 - head_prob, min=1e-7))

  full_logps = t_log_probs if return_full_logps else None
  return (
      jax.lax.stop_gradient(teacher_per_token_logps),
      jax.lax.stop_gradient(teacher_topk_logps)
      if teacher_topk_logps is not None
      else None,
      jax.lax.stop_gradient(teacher_tail_logp)
      if teacher_tail_logp is not None
      else None,
      jax.lax.stop_gradient(full_logps) if full_logps is not None else None,
  )


def _compute_alpha_divergence(
    log_p_s: jax.Array,
    log_p_t: jax.Array,
    alpha: float,
) -> jax.Array:
  """Computes per-token KL or generalized JSD between student and teacher."""
  log_p_t = jax.lax.stop_gradient(log_p_t)
  p_s = jnp.exp(log_p_s)
  p_t = jnp.exp(log_p_t)

  if alpha <= 0.0:
    return jnp.sum(p_t * (log_p_t - log_p_s), axis=-1)
  if alpha >= 1.0:
    return jnp.sum(p_s * (log_p_s - log_p_t), axis=-1)

  log_m = jnp.logaddexp(
      jnp.log(alpha) + log_p_s,
      jnp.log(1.0 - alpha) + log_p_t,
  )
  kl_s_m = jnp.sum(p_s * (log_p_s - log_m), axis=-1)
  kl_t_m = jnp.sum(p_t * (log_p_t - log_m), axis=-1)
  return alpha * kl_s_m + (1.0 - alpha) * kl_t_m


@function_registry.register_policy_loss_fn("sdpo")
def sdpo_loss_fn(
    model,
    train_example: SDPOTrainExample,
    algo_config: SDPOConfig,
    pad_id: int,
    eos_id: int,
):
  """Computes the SDPO self-distillation (+ optional GRPO) loss."""
  beta = algo_config.beta
  epsilon = algo_config.epsilon
  epsilon_high = getattr(algo_config, "epsilon_high", epsilon)
  loss_agg_mode = algo_config.loss_agg_mode
  distill_mode = algo_config.distillation_mode
  distill_weight = algo_config.distillation_weight
  distill_alpha = algo_config.distillation_alpha

  completion_ids = train_example.completion_ids
  completion_mask = train_example.completion_mask

  need_logits = distill_mode in ("topk_logits", "full_logits")
  graphdef, state = nnx.split(model)
  out = common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=train_example.prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      completion_mask=completion_mask,
      stop_gradient=False,
      return_logits=need_logits,
  )
  if need_logits:
    per_token_logps, student_logits = out
  else:
    per_token_logps, student_logits = out, None

  if train_example.old_per_token_logps is None:
    old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
  else:
    old_per_token_logps = train_example.old_per_token_logps

  # 1. Self-distillation loss
  if distill_mode == "topk_logits":
    student_log_probs = jax.nn.log_softmax(
        student_logits.astype(jnp.float32), axis=-1
    )
    student_topk_logps = jnp.take_along_axis(
        student_log_probs, train_example.teacher_topk_ids, axis=-1
    )
    teacher_topk_logps = train_example.teacher_topk_logps.astype(jnp.float32)

    if algo_config.distillation_add_tail:
      s_head_prob = jnp.sum(
          jnp.exp(student_topk_logps), axis=-1, keepdims=True
      )
      s_tail_logp = jnp.log(jnp.clip(1.0 - s_head_prob, min=1e-7))
      log_p_s = jnp.concatenate([student_topk_logps, s_tail_logp], axis=-1)
      log_p_t = jnp.concatenate(
          [
              teacher_topk_logps,
              train_example.teacher_tail_logp.astype(jnp.float32),
          ],
          axis=-1,
      )
    else:
      log_p_s = student_topk_logps
      log_p_t = teacher_topk_logps

    # Normalize log-probs strictly onto the simplex for numerical stability.
    log_p_s = log_p_s - jax.nn.logsumexp(log_p_s, axis=-1, keepdims=True)
    log_p_t = log_p_t - jax.nn.logsumexp(log_p_t, axis=-1, keepdims=True)
    per_token_distill_loss = _compute_alpha_divergence(
        log_p_s, log_p_t, distill_alpha
    )
  elif distill_mode == "full_logits":
    log_p_s = jax.nn.log_softmax(student_logits.astype(jnp.float32), axis=-1)
    log_p_t = train_example.teacher_full_logps.astype(jnp.float32)
    per_token_distill_loss = _compute_alpha_divergence(
        log_p_s, log_p_t, distill_alpha
    )
  else:  # "sampled_token"
    log_ratio = per_token_logps.astype(jnp.float32) - old_per_token_logps.astype(
        jnp.float32
    )
    ratio = jnp.exp(log_ratio)
    if algo_config.is_clip is not None:
      ratio = jnp.clip(ratio, 0.0, algo_config.is_clip)
    sdpo_adv = jax.lax.stop_gradient(
        train_example.teacher_per_token_logps.astype(jnp.float32)
        - old_per_token_logps.astype(jnp.float32)
    )
    per_token_distill_loss = -ratio * sdpo_adv

  sd_mask = (
      train_example.self_distillation_mask[:, None].astype(jnp.float32)
      if train_example.self_distillation_mask is not None
      else jnp.ones((completion_mask.shape[0], 1), dtype=jnp.float32)
  )
  effective_distill_mask = completion_mask.astype(jnp.float32) * sd_mask
  distill_loss = common.aggregate_loss(
      per_token_distill_loss, effective_distill_mask, loss_agg_mode
  )

  # 2. Optional GRPO policy gradient loss (hybrid SDPO + GRPO)
  pg_loss = jnp.asarray(0.0, dtype=jnp.float32)
  pg_clipfrac = jnp.asarray(0.0, dtype=jnp.float32)
  if distill_weight < 1.0:
    advantages = train_example.advantages
    seq_importance_ratio = per_token_logps.astype(
        jnp.float32
    ) - old_per_token_logps.astype(jnp.float32)
    if algo_config.loss_algo == "gspo-token":
      seq_importance_ratio = (seq_importance_ratio * completion_mask).sum(
          axis=-1
      ) / jnp.clip(completion_mask.sum(-1), min=1)
      seq_importance_ratio = (
          per_token_logps
          - jax.lax.stop_gradient(per_token_logps)
          + jnp.expand_dims(
              jax.lax.stop_gradient(seq_importance_ratio), axis=-1
          )
      )
      seq_importance_ratio = jnp.clip(seq_importance_ratio, max=10.0)
    coef_1 = jnp.exp(seq_importance_ratio)
    coef_2 = jnp.clip(coef_1, 1.0 - epsilon, 1.0 + epsilon_high)
    pg_losses_1 = -coef_1 * jnp.expand_dims(advantages, 1)
    pg_losses_2 = -coef_2 * jnp.expand_dims(advantages, 1)
    pg_clipfrac = jnp.sum(
        (pg_losses_2 > pg_losses_1) * completion_mask
    ) / jnp.clip(jnp.sum(completion_mask), min=1)
    per_token_pg_loss = -jnp.minimum(
        coef_1 * jnp.expand_dims(advantages, 1),
        coef_2 * jnp.expand_dims(advantages, 1),
    )
    pg_loss = common.aggregate_loss(
        per_token_pg_loss, completion_mask, loss_agg_mode
    )

  total_loss = (
      1.0 - distill_weight
  ) * pg_loss + distill_weight * distill_loss

  # 3. Optional reference-model KL regularization
  mean_kl = jnp.asarray(0.0, dtype=jnp.float32)
  if (
      beta is not None
      and beta != 0.0
      and train_example.ref_per_token_logps is not None
  ):
    kl = common.compute_kl_divergence(
        per_token_logps, train_example.ref_per_token_logps
    )
    total_loss = total_loss + beta * common.aggregate_loss(
        kl, completion_mask, loss_agg_mode
    )
    mean_kl = (kl * completion_mask).sum() / jnp.clip(
        completion_mask.sum(), min=1
    )

  teacher_kl_sampled = (
      (
          per_token_logps.astype(jnp.float32)
          - train_example.teacher_per_token_logps.astype(jnp.float32)
      )
      * effective_distill_mask
  ).sum() / jnp.clip(effective_distill_mask.sum(), min=1.0)

  aux = {
      "distill_loss": distill_loss,
      "pg_loss": pg_loss,
      "kl": mean_kl,
      "teacher_kl_sampled": teacher_kl_sampled,
      "pg_clipfrac": pg_clipfrac,
      "reprompt_frac": jnp.mean(sd_mask),
  }
  return total_loss, aux


class SDPOLearner(rl_learner.RLLearner[TSdpoConfig]):
  """Self-Distillation Policy Optimization (SDPO) learner."""

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: TSdpoConfig,
      reward_fns: RewardFn | List[RewardFn],
      feedback_fn: FeedbackFn | None = None,
      reprompt_formatter: RepromptFormatterFn | None = None,
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `SDPOLearner`.

    Args:
      rl_cluster: RL cluster containing actor, rollout, and optional reference.
      algo_config: An `SDPOConfig` instance.
      reward_fns: Scalar verifier reward function(s).
      feedback_fn: Optional callable accepting `(prompts, completions, rewards,
        **kwargs)` and returning `list[str | None]` with textual environment
        feedback for each rollout.
      reprompt_formatter: Optional callable accepting `(prompt, solution,
        feedback, tokenizer, **row_kwargs)` and returning the formatted teacher
        prompt string (useful for chat-templated models).
      metric_fns: Optional metric callables.
      data_shuffle_seed: Optional random seed for shuffling data batches.
    """
    super().__init__(
        rl_cluster=rl_cluster,
        algo_config=algo_config,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )
    self.feedback_fn = feedback_fn
    self.reprompt_formatter = reprompt_formatter
    self._sdpo_step_counter = 0

    actor_model = self.rl_cluster.actor_trainer.model
    self._lora_enabled = sft_utils.is_lora_enabled(actor_model)
    self._teacher_param_filter = (
        nnx.LoRAParam if self._lora_enabled else nnx.Param
    )

    self._teacher_params = None
    if self.algo_config.teacher_model_kind in ("ema", "base", "trust_region"):
      if (
          not self._lora_enabled
          and "reference" in getattr(self.rl_cluster.inference_worker, "_models", {})
          and self.algo_config.teacher_model_kind in ("base", "trust_region")
      ):
        ref_model = self.rl_cluster.inference_worker.get_model("reference")
        self._teacher_params = nnx.state(ref_model, self._teacher_param_filter)
      else:
        self._teacher_params = jax.tree.map(
            jnp.copy, nnx.state(actor_model, self._teacher_param_filter)
        )

    self._tokenizer_adapter = self._resolve_tokenizer_adapter()

    policy_loss_fn = function_registry.get_policy_loss_fn(
        self.algo_config.policy_loss_fn
    )
    loss_fn = lambda model, train_example, algo_config: policy_loss_fn(
        model,
        train_example,
        algo_config=self.algo_config,
        pad_id=self.rl_cluster.rollout.pad_id(),
        eos_id=self.rl_cluster.rollout.eos_id(),
    )
    self.rl_cluster.actor_trainer.with_loss_fn(loss_fn, has_aux=True)
    self.rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "algo_config": self.algo_config,
        }
    )
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log({
        "distill_loss": np.mean,
        "pg_loss": np.mean,
        "kl": np.mean,
        "teacher_kl_sampled": np.mean,
        "pg_clipfrac": np.mean,
        "reprompt_frac": np.mean,
    })
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display([
        "distill_loss",
        "reprompt_frac",
    ])

  def _resolve_tokenizer_adapter(self) -> tok_adapter.TokenizerAdapter | None:
    if hasattr(self.rl_cluster.rollout, "_sampler"):
      return self.rl_cluster.rollout._sampler.tokenizer  # pylint: disable=protected-access
    if self.rl_cluster.tokenizer is not None:
      return tok_adapter.TokenizerAdapter(self.rl_cluster.tokenizer)
    return None

  def _tokenize_prompt(self, text: str) -> np.ndarray:
    if hasattr(self.rl_cluster.rollout, "_sampler"):
      return np.asarray(
          self.rl_cluster.rollout._sampler.tokenize(text), dtype=np.int32  # pylint: disable=protected-access
      )
    tok = self._tokenizer_adapter
    if tok is None:
      raise ValueError("Tokenizer is required for SDPO teacher reprompting.")
    input_ids = tok.encode(text)
    bos_id = tok.bos_id()
    bos_tok = [bos_id] if bos_id else []
    return np.array(tok.dedup_bos_ids(bos_tok + input_ids), dtype=np.int32)

  def _get_teacher_full_state(self) -> statelib.State:
    actor_model = self.rl_cluster.actor_trainer.model
    if self.algo_config.teacher_model_kind == "live":
      return nnx.state(actor_model)
    if self._lora_enabled:
      base_state = statelib.filter_state(
          nnx.state(actor_model), filterlib.Not(nnx.LoRAParam)
      )
      return statelib.merge_state(base_state, self._teacher_params)
    return self._teacher_params

  def _maybe_update_ema_teacher(self) -> None:
    if self.algo_config.teacher_model_kind != "ema":
      return
    self._sdpo_step_counter += 1
    if self._sdpo_step_counter % self.algo_config.teacher_sync_steps != 0:
      return
    rate = self.algo_config.teacher_update_rate
    actor_params = nnx.state(
        self.rl_cluster.actor_trainer.model, self._teacher_param_filter
    )
    self._teacher_params = jax.tree.map(
        lambda t, a: ((1.0 - rate) * t.astype(jnp.float32) + rate * a.astype(jnp.float32)).astype(t.dtype),
        self._teacher_params,
        actor_params,
    )

  def _build_teacher_prompts(
      self,
      training_input: TrainingInputT,
      completions: list[str],
      rewards: np.ndarray,
  ) -> tuple[list[str], np.ndarray]:
    prompts = list(training_input["prompts"])
    num_total = len(prompts)
    num_gens = self.algo_config.num_generations
    extra_kwargs = {k: v for k, v in training_input.items() if k != "prompts"}

    if self.algo_config.include_environment_feedback:
      if self.feedback_fn is not None:
        env_feedbacks = self.feedback_fn(
            prompts=prompts,
            completions=completions,
            rewards=rewards,
            **extra_kwargs,
        )
      elif "privileged_context" in training_input:
        env_feedbacks = list(training_input["privileged_context"])
      elif "feedback" in training_input:
        env_feedbacks = list(training_input["feedback"])
      else:
        env_feedbacks = [None] * num_total
    else:
      env_feedbacks = [None] * num_total

    teacher_prompts: list[str] = []
    sd_mask = np.zeros((num_total,), dtype=np.float32)

    num_groups = num_total // num_gens
    for group_idx in range(num_groups):
      start = group_idx * num_gens
      end = start + num_gens
      success_indices = [
          idx
          for idx in range(start, end)
          if rewards[idx] >= self.algo_config.success_reward_threshold
      ]

      for i in range(start, end):
        is_self_success = (
            rewards[i] >= self.algo_config.success_reward_threshold
        )
        fb_raw = env_feedbacks[i]
        fb_raw = fb_raw if (fb_raw is not None and str(fb_raw).strip()) else None

        if is_self_success and self.algo_config.dont_reprompt_on_self_success:
          teacher_prompts.append(prompts[i])
          sd_mask[i] = 0.0
          continue

        # Select a successful sibling demonstration if available.
        demo_text = None
        if self.algo_config.use_successful_as_teacher and success_indices:
          candidate_indices = [
              idx
              for idx in success_indices
              if idx != i or not self.algo_config.dont_reprompt_on_self_success
          ]
          if candidate_indices:
            best_idx = max(candidate_indices, key=lambda idx: rewards[idx])
            demo_text = completions[best_idx]

        if (
            self.algo_config.environment_feedback_only_without_solution
            and demo_text is not None
        ):
          fb_raw = None

        if demo_text is None and fb_raw is None:
          teacher_prompts.append(prompts[i])
          sd_mask[i] = 0.0
          continue

        if self.reprompt_formatter is not None:
          row_kwargs = {
              k: (v[i] if hasattr(v, "__len__") and len(v) == num_total else v)
              for k, v in extra_kwargs.items()
          }
          formatted = self.reprompt_formatter(
              prompt=prompts[i],
              solution=demo_text,
              feedback=fb_raw,
              tokenizer=self.rl_cluster.tokenizer,
              **row_kwargs,
          )
        else:
          solution_block = (
              self.algo_config.solution_template.format(
                  successful_previous_attempt=demo_text
              )
              if demo_text is not None
              else ""
          )
          feedback_block = (
              self.algo_config.feedback_template.format(feedback_raw=fb_raw)
              if fb_raw is not None
              else ""
          )
          formatted = self.algo_config.reprompt_template.format(
              prompt=prompts[i],
              solution=solution_block,
              feedback=feedback_block,
          )
        teacher_prompts.append(formatted)
        sd_mask[i] = 1.0

    return teacher_prompts, sd_mask

  def _tokenize_teacher_prompts(
      self,
      teacher_prompts: list[str],
      pad_value: int,
      min_prompt_len: int,
  ) -> jax.Array:
    tokenized = [self._tokenize_prompt(tp) for tp in teacher_prompts]
    max_len = self.algo_config.max_reprompt_length
    tokenized = [
        ids[-max_len:] if len(ids) > max_len else ids for ids in tokenized
    ]
    longest = max(max(len(ids) for ids in tokenized), min_prompt_len)
    target_len = min(gen_utils.next_power_of_2(longest), max(max_len, min_prompt_len))
    padded = np.array(
        [
            gen_utils.pad_to_length(
                ids, target_length=target_len, pad_value=pad_value, left=True
            )
            for ids in tokenized
        ],
        dtype=np.int32,
    )
    return jnp.array(padded)

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> SDPOTrainExample:
    rollout_config = self.rl_cluster.cluster_config.rollout_config
    if isinstance(rollout_config, dict):
      rollout_config = rollout_config[mode]

    training_input["prompts"] = list(training_input["prompts"])
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()

    rollout_output = self.rl_cluster.generate(
        prompts=training_input["prompts"],
        mode=mode,
        micro_batch_size=(
            self._rollout_micro_batch_size * self.algo_config.num_generations
        ),
    )

    padded_completion_ids = np.array([
        gen_utils.pad_to_length(
            completion_ids,
            target_length=rollout_config.max_tokens_to_generate,
            pad_value=pad_value,
            left=False,
        )
        for completion_ids in rollout_output.tokens
    ])
    prompt_ids = jnp.array(rollout_output.left_padded_prompt_tokens)
    prompt_mask = prompt_ids != pad_value
    completion_mask = np.not_equal(padded_completion_ids, pad_value)
    jax_completion_ids = jnp.array(padded_completion_ids)
    jax_completion_mask = jnp.array(completion_mask)

    if self.algo_config.beta != 0.0 and "reference" in getattr(
        self.rl_cluster.inference_worker, "_models", {}
    ):
      ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=jax_completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
          micro_batch_size=(
              self._compute_logps_micro_batch_size
              * self.algo_config.num_generations
          ),
      )
    else:
      ref_per_token_logps = None

    if self.algo_config.num_iterations > 1 or (
        self.algo_config.distillation_mode == "sampled_token"
        and self.algo_config.is_clip is not None
    ):
      old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=jax_completion_ids,
          micro_batch_size=(
              self._compute_logps_micro_batch_size
              * self.algo_config.num_generations
          ),
      )
    else:
      old_per_token_logps = None

    # Compute scalar rewards and optional GRPO advantages
    rewards = self._compute_rewards(
        prompts=training_input["prompts"],
        completions=rollout_output.text,
        mode=mode,
        **{k: v for k, v in training_input.items() if k != "prompts"},
    )
    if self.algo_config.num_generations >= 2:
      advantage_estimator = function_registry.get_advantage_estimator(
          self.algo_config.advantage_estimator
      )
      advantages = advantage_estimator(
          rewards=rewards, num_generations=self.algo_config.num_generations
      )
    else:
      advantages = np.zeros_like(rewards, dtype=np.float32)

    # Build reprompted teacher inputs and self-distillation mask
    teacher_prompts, sd_mask_np = self._build_teacher_prompts(
        training_input=training_input,
        completions=rollout_output.text,
        rewards=rewards,
    )
    teacher_prompt_ids = self._tokenize_teacher_prompts(
        teacher_prompts,
        pad_value=pad_value,
        min_prompt_len=prompt_ids.shape[1],
    )

    # Compute student top-K (when topk_logits) and teacher log-probs
    batch_size = prompt_ids.shape[0]
    micro_bs = (
        self._compute_logps_micro_batch_size * self.algo_config.num_generations
    )
    data_sharding_axis = (
        self.rl_cluster.cluster_config.training_config.data_sharding_axis
    )

    with self.rl_cluster._get_mesh_and_logical_axis_rules_cm(  # pylint: disable=protected-access
        rl_cluster_lib.Role.ACTOR
    ):
      actor_model = self.rl_cluster.actor_trainer.model
      self.rl_cluster._maybe_load_model_from_cpu(  # pylint: disable=protected-access
          actor_model, rl_cluster_lib.Role.ACTOR
      )
      actor_graphdef, actor_state = nnx.split(actor_model)
      teacher_state = self._get_teacher_full_state()
      live_state = (
          actor_state
          if self.algo_config.teacher_model_kind == "trust_region"
          else None
      )
      mix_rate = (
          self.algo_config.teacher_update_rate
          if self.algo_config.teacher_model_kind == "trust_region"
          else None
      )

      s_prompt_sharded = sharding_utils.shard_input(
          prompt_ids, data_sharding_axis
      )
      t_prompt_sharded = sharding_utils.shard_input(
          teacher_prompt_ids, data_sharding_axis
      )
      comp_sharded = sharding_utils.shard_input(
          jax_completion_ids, data_sharding_axis
      )
      mask_sharded = sharding_utils.shard_input(
          jax_completion_mask, data_sharding_axis
      )

      need_topk = self.algo_config.distillation_mode == "topk_logits"
      need_full = self.algo_config.distillation_mode == "full_logits"

      topk_ids_list = []
      t_per_tok_list = []
      t_topk_list = []
      t_tail_list = []
      t_full_list = []

      for b_slice in rl_utils.chunk_slices_by_size(
          stop=batch_size, step=micro_bs
      ):
        if need_topk:
          topk_ids = _compute_student_topk_ids(
              actor_graphdef,
              actor_state,
              s_prompt_sharded[b_slice],
              comp_sharded[b_slice],
              pad_value,
              eos_value,
              mask_sharded[b_slice],
              self.algo_config.distillation_topk,
          )
          topk_ids_list.append(topk_ids)
        else:
          topk_ids = None

        t_per_tok, t_topk, t_tail, t_full = _compute_teacher_targets(
            actor_graphdef,
            teacher_state,
            live_state,
            t_prompt_sharded[b_slice],
            comp_sharded[b_slice],
            pad_value,
            eos_value,
            mask_sharded[b_slice],
            topk_ids,
            need_full,
            mix_rate,
        )
        t_per_tok_list.append(t_per_tok)
        if t_topk is not None:
          t_topk_list.append(t_topk)
        if t_tail is not None:
          t_tail_list.append(t_tail)
        if t_full is not None:
          t_full_list.append(t_full)

      teacher_per_token_logps = jnp.concatenate(t_per_tok_list, axis=0)
      teacher_topk_ids = (
          jnp.concatenate(topk_ids_list, axis=0) if topk_ids_list else None
      )
      teacher_topk_logps = (
          jnp.concatenate(t_topk_list, axis=0) if t_topk_list else None
      )
      teacher_tail_logp = (
          jnp.concatenate(t_tail_list, axis=0) if t_tail_list else None
      )
      teacher_full_logps = (
          jnp.concatenate(t_full_list, axis=0) if t_full_list else None
      )

      self.rl_cluster._maybe_offload_model_to_cpu(  # pylint: disable=protected-access
          actor_model, rl_cluster_lib.Role.ACTOR
      )

    self.rl_cluster.buffer_metrics(
        {
            "rewards/score/mean": (np.mean(rewards), np.mean),
            "rewards/score/max": (np.max(rewards), np.max),
            "rewards/score/min": (np.min(rewards), np.min),
            "sdpo/reprompt_fraction": (np.mean(sd_mask_np), np.mean),
        },
        mode=mode,
    )

    agg_completion_mask = completion_mask.sum(axis=-1)
    self.rl_cluster.buffer_metrics(
        {
            "completions/mean_length": (
                np.mean(agg_completion_mask),
                np.mean,
            ),
            "completions/max_length": (
                np.max(agg_completion_mask),
                np.max,
            ),
            "completions/min_length": (
                np.min(agg_completion_mask),
                np.min,
            ),
        },
        mode=mode,
    )

    for m_fn in self.metric_fns:
      user_defined_metric = m_fn(
          prompts=training_input["prompts"],
          completions=rollout_output.text,
          advantages=advantages,
          rewards=rewards,
          **{k: v for k, v in training_input.items() if k != "prompts"},
      )
      self.rl_cluster.buffer_metrics(user_defined_metric, mode=mode)

    return SDPOTrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=jax_completion_ids,
        completion_mask=jax_completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=jax.device_put(advantages),
        old_per_token_logps=old_per_token_logps,
        teacher_topk_ids=teacher_topk_ids,
        teacher_topk_logps=teacher_topk_logps,
        teacher_tail_logp=teacher_tail_logp,
        teacher_per_token_logps=teacher_per_token_logps,
        teacher_full_logps=teacher_full_logps,
        self_distillation_mask=jax.device_put(sd_mask_np),
    )

  def _run_mini_batch_step(
      self,
      initial_steps: int,
      service_target_batch_size: int,
      grad_acc_steps: int,
      train_iterator: Iterable[TrainingInputT],
      eval_ds: Iterable[TrainingInputT] | None,
      skip_jit: bool,
  ) -> None:
    super()._run_mini_batch_step(
        initial_steps,
        service_target_batch_size,
        grad_acc_steps,
        train_iterator,
        eval_ds,
        skip_jit,
    )
    self._maybe_update_ema_teacher()

  def _compute_trajectory_ids(
      self, example: TrainingInputT, steps: int
  ) -> List[str]:
    batch_size = len(example["prompts"]) // self.algo_config.num_generations
    row_offset = steps * batch_size
    row_offsets = np.repeat(
        np.arange(row_offset, row_offset + batch_size),
        self.algo_config.num_generations,
        axis=0,
    )
    group_offsets = np.tile(
        np.arange(self.algo_config.num_generations),
        batch_size,
    )
    return [
        f"{r_off}_{g_off}" for r_off, g_off in zip(row_offsets, group_offsets)
    ]

  def _num_iterations(self) -> int:
    return self.algo_config.num_iterations

  def _num_generations(self) -> int:
    return self.algo_config.num_generations

  def train(  # pylint: disable=useless-parent-delegation
      self,
      train_ds: Iterable[TrainingInputT],
      eval_ds: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """Runs the SDPO training loop."""
    super().train(train_ds, eval_ds, skip_jit)
