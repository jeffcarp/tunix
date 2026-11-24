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
"""Tests for SDPOLearner."""

import os
from typing import Any, Dict, Optional

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from grain import python as grain
import jax
from jax import sharding
from jax.interpreters import pxla
import jax.numpy as jnp
import numpy as np
import optax
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.sdpo import sdpo_learner as sdpo_lib
from tunix.tests import test_common as tc

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

Mesh = sharding.Mesh

_DUMMY_DATA = [
    "input string",
    "hello world",
    "My name",
    "hello there",
]


class _MySource(grain.RandomAccessDataSource):

  def __init__(self, data=None, repeat=1):
    if data is None:
      data = _DUMMY_DATA
    self._data = data * repeat

  def __getitem__(self, idx):
    return self._data[idx]

  def __len__(self):
    return len(self._data)


def _dummy_dataset(source=_MySource(), batch_size: int = 2):
  return (
      grain.MapDataset.source(source)
      .batch(batch_size)
      .map(lambda x: {"prompts": x, "answer": x})
  )


def _setup_cluster(
    kwargs: Optional[Dict[str, Any]] = None, with_ref: bool = False
):
  if kwargs is None:
    kwargs = {}
  vocab = tc.MockVocab()
  model = tc.ToyTransformer(
      config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
      rngs=nnx.Rngs(0),
  )
  original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
  ref_model = (
      tc.ToyTransformer(
          config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
          rngs=nnx.Rngs(0),
      )
      if with_ref
      else None
  )

  mesh = pxla.thread_resources.env.physical_mesh
  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: mesh,
          rl_cluster_lib.Role.REFERENCE: mesh,
          rl_cluster_lib.Role.ROLLOUT: mesh,
      },
      rollout_engine="vanilla",
      offload_to_cpu=False,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optax.sgd(1e-2),
          eval_every_n_steps=kwargs.get("eval_every_n_steps", 10),
          max_steps=kwargs.get("max_steps", 2),
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=6,
          max_prompt_length=32,
          kv_cache_size=128,
      ),
  )
  rl_cluster = rl_cluster_lib.RLCluster(
      actor=model,
      reference=ref_model,
      tokenizer=vocab,
      cluster_config=cluster_config,
  )
  return rl_cluster, model, original_variables


def _alternating_reward(completions, **kwargs):
  del kwargs
  # First item in each pair succeeds (1.0), second fails (0.0)
  return [1.0 if i % 2 == 0 else 0.0 for i in range(len(completions))]


def _all_fail_reward(completions, **kwargs):
  del kwargs
  return [0.0 for _ in completions]


def _simple_feedback(prompts, completions, rewards, **kwargs):
  del prompts, completions, kwargs
  return [
      None if r >= 1.0 else "Error: output did not match expected format."
      for r in rewards
  ]


class SDPOLearnerTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    jax.config.update("jax_platform_name", "cpu")

  def test_config_validation(self):
    with self.assertRaises(ValueError):
      sdpo_lib.SDPOConfig(distillation_weight=1.5)
    with self.assertRaises(ValueError):
      sdpo_lib.SDPOConfig(distillation_alpha=-0.1)
    with self.assertRaises(ValueError):
      sdpo_lib.SDPOConfig(distillation_mode="invalid")
    with self.assertRaises(ValueError):
      sdpo_lib.SDPOConfig(teacher_model_kind="unknown")
    with self.assertRaises(ValueError):
      sdpo_lib.SDPOConfig(num_generations=1, distillation_weight=0.5)

    # num_generations=1 is valid for pure SDPO with feedback
    cfg = sdpo_lib.SDPOConfig(num_generations=1, distillation_weight=1.0)
    self.assertEqual(cfg.num_generations, 1)

  def test_alpha_divergence_properties(self):
    logits_a = jnp.array([[2.0, 0.5, -1.0, 0.0]], dtype=jnp.float32)
    logits_b = jnp.array([[0.0, 1.5, 0.5, -0.5]], dtype=jnp.float32)
    log_pa = jax.nn.log_softmax(logits_a, axis=-1)
    log_pb = jax.nn.log_softmax(logits_b, axis=-1)

    for alpha in (0.0, 0.5, 1.0):
      same = sdpo_lib._compute_alpha_divergence(log_pa, log_pa, alpha)
      diff = sdpo_lib._compute_alpha_divergence(log_pa, log_pb, alpha)
      self.assertAlmostEqual(float(same[0]), 0.0, places=5)
      self.assertGreater(float(diff[0]), 0.0)

    # JSD at alpha=0.5 should be symmetric
    jsd_ab = sdpo_lib._compute_alpha_divergence(log_pa, log_pb, 0.5)
    jsd_ba = sdpo_lib._compute_alpha_divergence(log_pb, log_pa, 0.5)
    self.assertAlmostEqual(float(jsd_ab[0]), float(jsd_ba[0]), places=5)

  def test_build_teacher_prompts_sibling_and_feedback(self):
    rl_cluster, _, _ = _setup_cluster()
    cfg = sdpo_lib.SDPOConfig(
        num_generations=2,
        dont_reprompt_on_self_success=True,
        include_environment_feedback=True,
    )
    learner = sdpo_lib.SDPOLearner(
        rl_cluster=rl_cluster,
        algo_config=cfg,
        reward_fns=_alternating_reward,
        feedback_fn=_simple_feedback,
    )

    training_input = {
        "prompts": ["input string", "input string", "hello world", "hello world"]
    }
    completions = ["good_1", "bad_1", "good_2", "bad_2"]
    rewards = np.array([1.0, 0.0, 1.0, 0.0])

    teacher_prompts, sd_mask = learner._build_teacher_prompts(
        training_input, completions, rewards
    )
    np.testing.assert_array_equal(sd_mask, np.array([0.0, 1.0, 0.0, 1.0]))
    self.assertEqual(teacher_prompts[0], "input string")
    self.assertIn("good_1", teacher_prompts[1])
    self.assertIn("Error: output did not match", teacher_prompts[1])

    # All fail with no feedback -> sd_mask is all zeros
    learner_no_fb = sdpo_lib.SDPOLearner(
        rl_cluster=rl_cluster,
        algo_config=cfg,
        reward_fns=_all_fail_reward,
        feedback_fn=None,
    )
    _, sd_mask_zero = learner_no_fb._build_teacher_prompts(
        training_input, completions, np.zeros(4)
    )
    np.testing.assert_array_equal(sd_mask_zero, np.zeros(4))

  @parameterized.named_parameters(
      dict(
          testcase_name="topk_ema_jsd",
          distillation_mode="topk_logits",
          teacher_model_kind="ema",
          distillation_alpha=0.5,
          distillation_weight=1.0,
      ),
      dict(
          testcase_name="topk_live_reverse_kl",
          distillation_mode="topk_logits",
          teacher_model_kind="live",
          distillation_alpha=1.0,
          distillation_weight=1.0,
      ),
      dict(
          testcase_name="full_logits_trust_region",
          distillation_mode="full_logits",
          teacher_model_kind="trust_region",
          distillation_alpha=0.5,
          distillation_weight=1.0,
      ),
      dict(
          testcase_name="sampled_token_hybrid_grpo",
          distillation_mode="sampled_token",
          teacher_model_kind="base",
          distillation_alpha=1.0,
          distillation_weight=0.5,
      ),
  )
  def test_sdpo_learner_train(
      self,
      distillation_mode: str,
      teacher_model_kind: str,
      distillation_alpha: float,
      distillation_weight: float,
  ):
    rl_cluster, model, original_variables = _setup_cluster(
        kwargs={"max_steps": 2, "eval_every_n_steps": 10},
        with_ref=(teacher_model_kind == "base"),
    )
    cfg = sdpo_lib.SDPOConfig(
        num_generations=2,
        distillation_mode=distillation_mode,
        distillation_topk=16,
        teacher_model_kind=teacher_model_kind,
        distillation_alpha=distillation_alpha,
        distillation_weight=distillation_weight,
        max_reprompt_length=64,
    )
    learner = sdpo_lib.SDPOLearner(
        rl_cluster=rl_cluster,
        algo_config=cfg,
        reward_fns=_alternating_reward,
        feedback_fn=_simple_feedback,
    )
    ds = _dummy_dataset(batch_size=2)
    learner.train(ds)

    updated_variables = nnx.state(model, nnx.Param)
    diffs = jax.tree.leaves(
        jax.tree.map(lambda a, b: bool(jnp.any(a != b)), original_variables, updated_variables)
    )
    self.assertTrue(any(diffs), "Expected actor weights to update after SDPO training.")


if __name__ == "__main__":
  absltest.main()
