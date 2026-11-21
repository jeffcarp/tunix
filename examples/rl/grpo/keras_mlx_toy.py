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

"""Toy end-to-end GRPO training example using Keras MLX on Apple Silicon.

Demonstrates:
1. Defining a causal language model using Keras 3.
2. Running autoregressive rollout with Keras MLX backend.
3. Scoring rollouts with a programmatic reward function.
4. Computing group relative advantages (Tunix GRPO logic).
5. Performing clipped surrogate policy updates directly on Apple Silicon GPU.

Usage:
  See docs/proposals/keras_mlx_grpo.md for full setup instructions, then run:
  KERAS_BACKEND=mlx python examples/rl/grpo/keras_mlx_toy.py
"""

import os
import time
import numpy as np

# Ensure MLX backend is selected
if "KERAS_BACKEND" not in os.environ:
  os.environ["KERAS_BACKEND"] = "mlx"

import keras
from keras import ops
from tunix.rl.keras_grpo import KerasGRPOConfig, KerasGRPOTrainer


def greedy_predict(model: keras.Model, prompts: np.ndarray) -> np.ndarray:
  """Runs single-step greedy decoding for each prompt."""
  logits = model(ops.convert_to_tensor(prompts))
  next_tokens = ops.argmax(logits[:, -1, :], axis=-1)
  return np.asarray(ops.convert_to_numpy(next_tokens), dtype=np.int32)[:, None]


class MiniCausalLM(keras.Model):
  """A lightweight causal language model for local experimentation."""

  def __init__(self, vocab_size: int = 32, embed_dim: int = 64, num_heads: int = 2):
    super().__init__()
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.token_embedding = keras.layers.Embedding(vocab_size, embed_dim)
    self.pos_embedding = keras.layers.Embedding(128, embed_dim)
    self.attn = keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        use_bias=False,
    )
    self.norm1 = keras.layers.LayerNormalization()
    self.mlp = keras.Sequential([
        keras.layers.Dense(embed_dim * 2, activation="gelu"),
        keras.layers.Dense(embed_dim),
    ])
    self.norm2 = keras.layers.LayerNormalization()
    self.lm_head = keras.layers.Dense(vocab_size, use_bias=False)

  def call(self, inputs):
    seq_len = ops.shape(inputs)[1]
    positions = ops.arange(0, seq_len)
    x = self.token_embedding(inputs) + self.pos_embedding(positions)

    # Causal attention mask: lower triangular
    causal_mask = ops.triu(ops.ones((seq_len, seq_len)), k=1)
    causal_mask = ops.cast(causal_mask, "bool")

    # MHA self-attention
    attn_out = self.attn(
        query=x,
        value=x,
        key=x,
        attention_mask=ops.logical_not(causal_mask),
    )
    x = self.norm1(x + attn_out)
    x = self.norm2(x + self.mlp(x))
    return self.lm_head(x)


def create_arithmetic_task():
  """Creates a synthetic single-digit addition task:

  Prompt: [StartToken, A, B]
  Target Completion: [A + B]
  """
  start_tok = 15
  # Digits 1 to 5 so A + B <= 10
  pairs = [(1, 2), (2, 3), (3, 1), (4, 2), (1, 4), (2, 2)]
  prompts = [[start_tok, a, b] for a, b in pairs]
  targets = {tuple(p): [a + b] for p, (a, b) in zip(prompts, pairs)}

  def reward_fn(prompts_batch, completions_batch):
    rewards = []
    for p, c in zip(prompts_batch, completions_batch):
      target = targets.get(tuple(p), None)
      if target is None:
        rewards.append(0.0)
        continue

      pred = list(c[:len(target)])
      rewards.append(1.0 if pred == target else 0.0)
    return rewards

  return np.array(prompts, dtype=np.int32), targets, reward_fn


def main():
  np.random.seed(42)
  keras.utils.set_random_seed(42)

  print("=" * 60)
  print(f"Tunix Keras GRPO on Apple Silicon (Backend: {keras.backend.backend()})")
  print("=" * 60)

  vocab_size = 16
  model = MiniCausalLM(vocab_size=vocab_size, embed_dim=48, num_heads=2)
  optimizer = keras.optimizers.AdamW(learning_rate=0.005, weight_decay=1e-4)

  config = KerasGRPOConfig(
      num_generations=8,
      max_tokens_to_generate=1,
      temperature=1.2,
      epsilon=0.2,
      beta=0.0,  # Critic-free, pure rule-based GRPO
  )

  prompts, targets, reward_fn = create_arithmetic_task()

  trainer = KerasGRPOTrainer(
      model=model,
      optimizer=optimizer,
      config=config,
      reward_fn=reward_fn,
  )

  print(f"Dataset: {len(prompts)} unique prompts, G={config.num_generations} rollouts per prompt.")
  print("Task: Arithmetic addition [Start, A, B] -> [A + B]")
  print("-" * 60)

  # Initial evaluation before training
  print("Initial greedy predictions:")
  initial_preds = greedy_predict(model, prompts)
  for p, c in zip(prompts, initial_preds):
    print(f"  Prompt: {p.tolist()} -> Completion: {c.tolist()} (Expected: {targets[tuple(p.tolist())]})")
  print("-" * 60)

  start_time = time.time()
  num_steps = 80

  for step in range(1, num_steps + 1):
    step_start = time.time()
    metrics = trainer.train_step(prompts)
    step_latency = (time.time() - step_start) * 1000

    if step == 1 or step % 10 == 0 or step == num_steps:
      print(
          f"Step {step:2d}/{num_steps:2d} | "
          f"Mean Reward: {metrics['reward/mean']:.2f} | "
          f"Max Reward: {metrics['reward/max']:.2f} | "
          f"Loss: {metrics['loss']:.4f} | "
          f"Step Time: {step_latency:.1f}ms"
      )

  total_elapsed = time.time() - start_time
  print("-" * 60)
  print(f"Training completed in {total_elapsed:.2f}s ({total_elapsed/num_steps:.2f}s per step)")
  print("\nFinal greedy predictions after GRPO post-training:")

  final_preds = greedy_predict(model, prompts)
  successes = 0
  for p, c in zip(prompts, final_preds):
    expected = targets[tuple(p.tolist())]
    pred = c.tolist()
    is_correct = pred == expected
    successes += int(is_correct)
    status = "✓ CORRECT" if is_correct else "✗ MISMATCH"
    print(f"  Prompt: {p.tolist()} -> Completion: {pred} (Expected: {expected}) [{status}]")

  print(f"\nFinal Accuracy: {successes}/{len(prompts)} ({100.0 * successes / len(prompts):.1f}%)")
  print("=" * 60)


if __name__ == "__main__":
  main()
