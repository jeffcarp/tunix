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

"""Tests for Keras GRPO implementation."""

import unittest
import numpy as np

try:
  import keras
  from keras import ops
  from tunix.rl import keras_grpo
except ImportError:
  keras = None
  keras_grpo = None


class KerasGRPOTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    if keras is None or keras_grpo is None:
      self.skipTest("Keras 3 or keras_grpo dependencies not available.")

  def _build_toy_model(self, vocab_size=16, embed_dim=16):
    class ToyCausalModel(keras.Model):

      def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = keras.layers.Embedding(vocab_size, embed_dim)
        self.dense = keras.layers.Dense(embed_dim, activation="gelu")
        self.head = keras.layers.Dense(vocab_size)

      def call(self, x):
        return self.head(self.dense(self.embed(x)))

    return ToyCausalModel(vocab_size, embed_dim)

  def test_compute_group_advantages(self):
    # Two prompts, G = 3 generations each
    rewards = np.array([1.0, 0.0, 0.0, 2.0, 2.0, 2.0], dtype=np.float32)
    adv = keras_grpo.compute_group_advantages(rewards, num_generations=3)

    self.assertEqual(adv.shape, (6,))
    # First group: mean = 0.333, std > 0 -> reward 1.0 has positive advantage
    self.assertGreater(adv[0], 0.0)
    self.assertLess(adv[1], 0.0)
    self.assertAlmostEqual(adv[1], adv[2])

    # Second group: all rewards equal -> advantages should be 0.0
    np.testing.assert_allclose(adv[3:6], 0.0, atol=1e-3)

  def test_compute_kl_divergence(self):
    log_p = ops.convert_to_tensor(np.array([[-1.0, -0.5]], dtype=np.float32))
    ref_log_p = ops.convert_to_tensor(np.array([[-1.2, -0.5]], dtype=np.float32))

    kl_low_var = keras_grpo.compute_kl_divergence(log_p, ref_log_p, method="low_var_kl")
    kl_exact = keras_grpo.compute_kl_divergence(log_p, ref_log_p, method="kl")

    # Identical log probabilities should yield 0 KL
    self.assertAlmostEqual(float(ops.convert_to_numpy(kl_low_var)[0, 1]), 0.0, places=4)
    self.assertAlmostEqual(float(ops.convert_to_numpy(kl_exact)[0, 1]), 0.0, places=4)

  def test_keras_rollout_generate(self):
    vocab_size = 16
    model = self._build_toy_model(vocab_size=vocab_size)
    config = keras_grpo.KerasGRPOConfig(
        num_generations=2,
        max_tokens_to_generate=4,
        temperature=0.0,  # greedy
        eos_token_id=15,
    )

    rollout = keras_grpo.KerasRollout(model, config)
    prompts = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out = rollout.generate(prompts)

    # 2 prompts * 2 generations = 4 sequences
    self.assertEqual(out.prompt_ids.shape, (4, 2))
    self.assertEqual(out.completion_ids.shape, (4, 4))
    self.assertEqual(out.completion_mask.shape, (4, 4))
    self.assertEqual(out.old_log_probs.shape, (4, 4))

  def test_grpo_train_step(self):
    vocab_size = 16
    model = self._build_toy_model(vocab_size=vocab_size)
    optimizer = keras.optimizers.AdamW(learning_rate=0.01)

    config = keras_grpo.KerasGRPOConfig(
        num_generations=2,
        max_tokens_to_generate=2,
        temperature=0.8,
    )

    def dummy_reward(prompts, completions):
      # Give reward 1.0 if first completion token is even, else 0.0
      return [1.0 if c[0] % 2 == 0 else 0.0 for c in completions]

    trainer = keras_grpo.KerasGRPOTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        reward_fn=dummy_reward,
    )

    prompts = np.array([[1, 2], [3, 4]], dtype=np.int32)
    metrics = trainer.train_step(prompts)

    self.assertIn("loss", metrics)
    self.assertIn("reward/mean", metrics)
    self.assertEqual(metrics["step"], 1)

  def test_grpo_toy_task_convergence(self):
    np.random.seed(42)
    vocab_size = 8
    model = self._build_toy_model(vocab_size=vocab_size, embed_dim=32)
    optimizer = keras.optimizers.AdamW(learning_rate=0.05)

    config = keras_grpo.KerasGRPOConfig(
        num_generations=4,
        max_tokens_to_generate=1,
        temperature=0.7,
        epsilon=0.2,
    )

    # Task: Prompt [X] -> Completion must be [X]
    def match_reward(prompts, completions):
      return [1.0 if p[0] == c[0] else 0.0 for p, c in zip(prompts, completions)]

    trainer = keras_grpo.KerasGRPOTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        reward_fn=match_reward,
    )

    train_prompts = np.array([[1], [2], [3], [4]], dtype=np.int32)
    initial_rewards = []
    final_rewards = []

    for step in range(12):
      m = trainer.train_step(train_prompts)
      if step < 2:
        initial_rewards.append(m["reward/mean"])
      if step >= 10:
        final_rewards.append(m["reward/mean"])

    avg_initial = np.mean(initial_rewards)
    avg_final = np.mean(final_rewards)

    # Model should have learned and improved significantly
    self.assertGreater(avg_final, avg_initial)

  def test_top_level_lazy_exports(self):
    import tunix

    self.assertIs(tunix.KerasGRPOConfig, keras_grpo.KerasGRPOConfig)
    self.assertIs(tunix.KerasGRPOTrainer, keras_grpo.KerasGRPOTrainer)
    self.assertIs(tunix.KerasRollout, keras_grpo.KerasRollout)
    exported = dir(tunix)
    self.assertIn("GRPOLearner", exported)
    self.assertIn("PeftTrainer", exported)
    self.assertIn("KerasGRPOTrainer", exported)
    with self.assertRaises(AttributeError):
      _ = tunix.NonExistentTunixSymbol

  def test_rollout_old_logps_unscaled(self):
    np.random.seed(0)
    model = self._build_toy_model(vocab_size=8, embed_dim=8)
    config = keras_grpo.KerasGRPOConfig(
        num_generations=2,
        max_tokens_to_generate=1,
        temperature=0.5,
    )
    rollout = keras_grpo.KerasRollout(model, config)
    prompts = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out = rollout.generate(prompts)

    raw_logits = model(ops.convert_to_tensor(out.prompt_ids))[:, -1, :]
    expected_logps = ops.convert_to_numpy(ops.log_softmax(raw_logits, axis=-1))
    chosen = out.completion_ids[:, 0]
    np.testing.assert_allclose(
        out.old_log_probs[:, 0],
        expected_logps[np.arange(len(chosen)), chosen],
        atol=1e-5,
    )


if __name__ == "__main__":
  unittest.main()
