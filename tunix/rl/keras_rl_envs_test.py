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

"""Tests for `tunix.rl.keras_rl_envs`."""

import fractions
import unittest

from tunix.rl import keras_rl_envs


class _MockTokenizer:

  def __init__(self, vocab):
    self.vocabulary = dict(vocab)
    self._inv = {v: k for k, v in self.vocabulary.items()}

  def tokenize(self, text):
    # Split character-by-character or by known special strings for testing
    tokens = []
    i = 0
    keys = sorted(self.vocabulary.keys(), key=len, reverse=True)
    while i < len(text):
      matched = False
      for k in keys:
        if text.startswith(k, i):
          tokens.append(self.vocabulary[k])
          i += len(k)
          matched = True
          break
      if not matched:
        tokens.append(ord(text[i]) % 256)
        i += 1
    return tokens

  def detokenize(self, ids):
    parts = []
    for idx in ids:
      if idx in self._inv:
        parts.append(self._inv[idx])
      else:
        parts.append(chr(idx))
    return "".join(parts)


class KerasRLEnvsTest(unittest.TestCase):

  def test_safe_eval_arithmetic(self):
    self.assertEqual(
        keras_rl_envs.safe_eval_arithmetic("(12 + 8) * 3 / 4"),
        fractions.Fraction(15),
    )
    self.assertEqual(
        keras_rl_envs.safe_eval_arithmetic("-5 + 1,000"),
        fractions.Fraction(995),
    )
    for bad in ("", "2 ** 8", "__import__('os')", "x + 1", "5 / 0", "abs(-1)"):
      with self.assertRaises(ValueError):
        keras_rl_envs.safe_eval_arithmetic(bad)

  def test_countdown_env_solvable_and_grading(self):
    for diff in ("easy", "medium", "hard"):
      env = keras_rl_envs.CountdownEnv(difficulty=diff, seed=7)
      problems = env.sample_batch(5)
      self.assertEqual(len(problems), 5)
      for p in problems:
        self.assertNotIn(p.target, p.numbers)
        solved_text = f"<think>Checking</think><answer>{p.reference_solution}</answer>"
        reward, info = env.grade(p, solved_text)
        self.assertEqual(reward, 1.0)
        self.assertTrue(info["solved"])

    env = keras_rl_envs.CountdownEnv(difficulty="easy", seed=0)
    problem = keras_rl_envs.CountdownProblem(
        numbers=(2, 5, 7), target=17, reference_solution="(2 * 5) + 7"
    )
    # Rejects reusing a number beyond multiplicity (5 + 5 + 7)
    r_reuse, info_reuse = env.grade(
        problem, "<think>x</think><answer>5 + 5 + 7</answer>"
    )
    self.assertLess(r_reuse, 1.0)
    self.assertFalse(info_reuse["solved"])

    # Rejects inventing unlisted numbers (<answer>17</answer>)
    r_cheat, info_cheat = env.grade(
        problem, "<think>x</think><answer>17</answer>"
    )
    self.assertLess(r_cheat, 1.0)
    self.assertFalse(info_cheat["solved"])

    # Format bonus awarded for valid format + valid numbers even when wrong value
    r_fmt, info_fmt = env.grade(
        problem, "<think>try</think><answer>2 + 5 + 7</answer>"
    )
    self.assertEqual(r_fmt, env.format_bonus)
    self.assertTrue(info_fmt["has_format"])

  def test_tool_math_env_and_callback(self):
    env = keras_rl_envs.ToolMathEnv(difficulty="easy", seed=123)
    problem = env.sample_problem()

    tok = _MockTokenizer({"<|im_start|>": 200})
    tool_fn = env.make_tool_step_fn(tok)

    partial_tokens = tok.tokenize("<think>Need mul: <calc>12 * 10</calc>")
    obs_tokens = tool_fn(partial_tokens)
    self.assertIsNotNone(obs_tokens)
    obs_text = tok.detokenize(obs_tokens)
    self.assertIn("<result>120</result>", obs_text)

    # Should not re-trigger once <result> follows </calc>
    combined = partial_tokens + obs_tokens
    self.assertIsNone(tool_fn(combined))

    full_text = f"<think><calc>1</calc><result>1</result></think><answer>{problem.target}</answer>"
    r_ok, info_ok = env.grade(problem, full_text)
    self.assertEqual(r_ok, 1.0)
    self.assertTrue(info_ok["solved"])
    self.assertTrue(info_ok["used_tool"])

  def test_format_chat_prompt(self):
    qwen_tok = _MockTokenizer({"<|im_start|>": 1})
    q_prompt = keras_rl_envs.format_chat_prompt("sys", "usr", tokenizer=qwen_tok)
    self.assertIn("<|im_start|>system", q_prompt)
    self.assertTrue(q_prompt.endswith("<|im_start|>assistant\n"))

    gemma_tok = _MockTokenizer({"<start_of_turn>": 1})
    g_prompt = keras_rl_envs.format_chat_prompt("sys", "usr", tokenizer=gemma_tok)
    self.assertIn("<start_of_turn>user", g_prompt)
    self.assertTrue(g_prompt.endswith("<start_of_turn>model\n"))

    plain = keras_rl_envs.format_chat_prompt("sys", "usr", tokenizer=None)
    self.assertIn("System: sys", plain)


if __name__ == "__main__":
  unittest.main()
