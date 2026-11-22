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

  def test_drgrpo_advantages_and_loss(self):
    rewards = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    adv_dr = keras_grpo.compute_group_advantages(
        rewards, num_generations=4, algo_variant="drgrpo"
    )
    # Dr.GRPO omits std division: R - mean(R) = [0.75, -0.25, -0.25, -0.25]
    np.testing.assert_allclose(adv_dr, [0.75, -0.25, -0.25, -0.25], atol=1e-5)

    model = self._build_toy_model(vocab_size=8, embed_dim=16)
    optimizer = keras.optimizers.AdamW(learning_rate=0.01)
    config = keras_grpo.KerasGRPOConfig(
        algo_variant="drgrpo",
        num_generations=4,
        max_tokens_to_generate=2,
        temperature=0.8,
    )
    self.assertEqual(config.loss_agg_mode, "sequence-mean-token-scale")
    trainer = keras_grpo.KerasGRPOTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        reward_fn=lambda ps, cs: [1.0 if c[0] % 2 == 0 else 0.0 for c in cs],
    )
    metrics = trainer.train_step(np.array([[1, 2]], dtype=np.int32))
    self.assertTrue(np.isfinite(metrics["loss"]))
    self.assertIn("entropy", metrics)

  def test_dapo_dynamic_sampling_skips_zero_variance(self):
    model = self._build_toy_model(vocab_size=8, embed_dim=16)
    optimizer = keras.optimizers.AdamW(learning_rate=0.01)
    config = keras_grpo.KerasGRPOConfig(
        algo_variant="dapo",
        num_generations=3,
        max_tokens_to_generate=1,
        temperature=0.0,
    )
    self.assertTrue(config.dynamic_sampling)
    self.assertAlmostEqual(config.epsilon_high, 0.28)

    # Constant rewards for all prompts -> all groups skipped, weights unchanged
    trainer = keras_grpo.KerasGRPOTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        reward_fn=lambda ps, cs: [1.0] * len(cs),
    )
    prompts = np.array([[1], [2]], dtype=np.int32)
    _ = model(ops.convert_to_tensor(prompts))
    before = [np.array(ops.convert_to_numpy(v.value)) for v in model.trainable_variables]
    metrics = trainer.train_step(prompts)
    after = [np.array(ops.convert_to_numpy(v.value)) for v in model.trainable_variables]
    self.assertEqual(metrics["skipped_zero_var_groups"], 2)
    self.assertEqual(metrics["loss"], 0.0)
    for b, a in zip(before, after):
      np.testing.assert_allclose(b, a)

  def test_gspo_train_step(self):
    model = self._build_toy_model(vocab_size=8, embed_dim=16)
    optimizer = keras.optimizers.AdamW(learning_rate=0.02)
    config = keras_grpo.KerasGRPOConfig(
        algo_variant="gspo",
        num_generations=4,
        max_tokens_to_generate=3,
        temperature=0.9,
    )
    trainer = keras_grpo.KerasGRPOTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        reward_fn=lambda ps, cs: [float(c[0] % 3 == 0) for c in cs],
    )
    metrics = trainer.train_step(np.array([[1, 2], [3, 4]], dtype=np.int32))
    self.assertTrue(np.isfinite(metrics["loss"]))
    self.assertTrue(np.isfinite(metrics["pg_clipfrac"]))

  def test_zero_copy_lora_kl(self):
    model = self._build_toy_model(vocab_size=8, embed_dim=16)
    sample_in = ops.convert_to_tensor(np.array([[1, 2]], dtype=np.int32))
    base_logits = np.array(ops.convert_to_numpy(model(sample_in)))

    model.dense.trainable = True
    model.dense.enable_lora(rank=2)
    model.embed.trainable = False
    model.head.trainable = False

    self.assertTrue(keras_grpo.has_lora_layers(model))
    # Immediately after enable_lora (B=0), base & lora outputs match; perturb B
    model.dense.lora_kernel_b.assign(
        np.ones_like(ops.convert_to_numpy(model.dense.lora_kernel_b)) * 0.25
    )
    lora_logits = np.array(ops.convert_to_numpy(model(sample_in)))
    self.assertFalse(np.allclose(base_logits, lora_logits, atol=1e-4))

    with keras_grpo.disable_lora(model):
      disabled_logits = np.array(ops.convert_to_numpy(model(sample_in)))
    np.testing.assert_allclose(disabled_logits, base_logits, atol=1e-5)

    optimizer = keras.optimizers.AdamW(learning_rate=0.01)
    config = keras_grpo.KerasGRPOConfig(
        num_generations=3,
        max_tokens_to_generate=2,
        temperature=0.9,
        beta=0.05,
        train_micro_batch_size=2,
    )
    trainer = keras_grpo.KerasGRPOTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        reward_fn=lambda ps, cs: [float(sum(c) % 2 == 0) for c in cs],
    )
    metrics = trainer.train_step(np.array([[1, 2], [3, 4]], dtype=np.int32))
    self.assertGreaterEqual(metrics["kl"], 0.0)
    self.assertTrue(np.isfinite(metrics["kl"]))

  def test_micro_batch_gradient_accumulation(self):
    def _run_one(mb_size):
      np.random.seed(123)
      keras.utils.set_random_seed(123)
      model = self._build_toy_model(vocab_size=8, embed_dim=16)
      opt = keras.optimizers.AdamW(learning_rate=0.02)
      cfg = keras_grpo.KerasGRPOConfig(
          num_generations=2,
          max_tokens_to_generate=2,
          temperature=0.0,
          train_micro_batch_size=mb_size,
      )
      trainer = keras_grpo.KerasGRPOTrainer(
          model=model,
          optimizer=opt,
          config=cfg,
          reward_fn=lambda ps, cs: [1.0 if p[0] == 1 else 0.0 for p in ps],
      )
      # Use differing rewards across the two prompts via group-dependent score
      trainer.reward_fn = lambda ps, cs: [1.0, 0.0, 1.0, 0.0]
      metrics = trainer.train_step(np.array([[1, 2], [3, 4]], dtype=np.int32))
      weights = [
          np.array(ops.convert_to_numpy(v.value)) for v in model.trainable_variables
      ]
      return metrics["loss"], weights

    loss_full, w_full = _run_one(None)
    loss_mb, w_mb = _run_one(1)
    self.assertAlmostEqual(loss_full, loss_mb, places=5)
    for wf, wm in zip(w_full, w_mb):
      np.testing.assert_allclose(wf, wm, atol=1e-4)

  def test_dict_input_and_kv_cache_rollout(self):
    class _MockBackbone(keras.layers.Layer):

      def __init__(self, vocab_size=8, hidden_dim=8):
        super().__init__()
        self.num_layers = 1
        self.num_query_heads = 2
        self.num_key_value_heads = 2
        self.head_dim = 4
        self.hidden_dim = hidden_dim
        self.token_embedding = keras.layers.ReversibleEmbedding(
            input_dim=vocab_size, output_dim=hidden_dim, tie_weights=True
        )
        self.proj = keras.layers.Dense(hidden_dim)

      def call(self, inputs):
        x = self.token_embedding(inputs["token_ids"])
        return self.proj(x)

    class _MockDictCausalLM(keras.Model):

      def __init__(self, vocab_size=8, hidden_dim=8):
        super().__init__()
        self.backbone = _MockBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim)

      def call(self, inputs):
        hidden = self.backbone(inputs)
        return self.backbone.token_embedding(hidden, reverse=True)

      def call_with_cache(self, token_ids, cache, cache_update_index):
        # Position-wise token embedding -> proj -> reverse matches uncached call
        hidden = self.backbone(
            {"token_ids": token_ids, "padding_mask": ops.ones_like(token_ids)}
        )
        logits = self.backbone.token_embedding(hidden, reverse=True)
        return logits, hidden, cache

    model = _MockDictCausalLM(vocab_size=8, hidden_dim=8)
    prompts = np.array([[1, 2]], dtype=np.int32)
    _ = model({"token_ids": ops.convert_to_tensor(prompts), "padding_mask": ops.ones((1, 2), dtype="int32")})

    cfg = keras_grpo.KerasGRPOConfig(
        num_generations=2,
        max_tokens_to_generate=3,
        temperature=0.0,
        use_kv_cache=True,
    )
    rollout = keras_grpo.KerasRollout(model, cfg)
    out = rollout.generate(prompts)
    self.assertEqual(out.completion_ids.shape, (2, 3))

    # Compare KV-cached logps vs direct uncached forward logps
    trainer = keras_grpo.KerasGRPOTrainer(
        model=model,
        optimizer=keras.optimizers.AdamW(learning_rate=0.01),
        config=cfg,
        reward_fn=lambda ps, cs: [1.0, 0.0],
    )
    full_ids = np.concatenate([out.prompt_ids, out.completion_ids], axis=1)
    uncached_logps = trainer._compute_logps_for_model(
        model,
        full_ids,
        np.ones_like(full_ids, dtype=np.int32),
        out.prompt_ids.shape[1],
        out.completion_ids.shape[1],
        out.completion_ids,
    )
    np.testing.assert_allclose(out.old_log_probs, uncached_logps, atol=1e-4)

    metrics = trainer.train_step(prompts)
    self.assertTrue(np.isfinite(metrics["loss"]))

  def test_multi_turn_tool_observation_masking(self):
    model = self._build_toy_model(vocab_size=12, embed_dim=16)
    cfg = keras_grpo.KerasGRPOConfig(
        num_generations=2,
        max_tokens_to_generate=5,
        temperature=0.0,
        max_turns=2,
        use_kv_cache=False,
    )
    rollout = keras_grpo.KerasRollout(model, cfg)

    calls = [0]

    def fake_tool_step(gen_tokens):
      # Trigger observation injection once after the very first generated token
      if len(gen_tokens) == 1 and calls[0] < 2:
        calls[0] += 1
        return [10, 11]
      return None

    out = rollout.generate([[1, 2]], tool_step_fn=fake_tool_step)
    self.assertEqual(out.completion_ids.shape, (2, 5))
    # Tokens at indices 1 and 2 are injected observation tokens [10, 11]
    np.testing.assert_array_equal(out.completion_ids[:, 1:3], [[10, 11], [10, 11]])
    # Loss mask must be 0.0 on observation tokens and 1.0 on model tokens
    np.testing.assert_array_equal(out.completion_mask[:, 0], [1.0, 1.0])
    np.testing.assert_array_equal(out.completion_mask[:, 1:3], [[0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(out.completion_mask[:, 3:5], [[1.0, 1.0], [1.0, 1.0]])
    # Attention mask must be 1.0 on BOTH model tokens and observation tokens
    np.testing.assert_array_equal(
        out.attention_completion_mask, np.ones((2, 5), dtype=np.float32)
    )


if __name__ == "__main__":
  unittest.main()
