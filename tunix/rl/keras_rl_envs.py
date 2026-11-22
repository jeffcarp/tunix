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

"""Procedural 2026 SOTA RLVR and Multi-Turn Tool-Use environments.

Zero-download verifiable reasoning environments designed to work with KerasHub
tokenizers and `tunix.rl.keras_grpo` on local Apple Silicon machines:
- `CountdownEnv`: Canonical R1-Zero / TinyZero Countdown arithmetic puzzle.
- `ToolMathEnv`: Multi-turn calculator tool-integrated reasoning (TIR).
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import fractions
import operator
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_arithmetic(expr: str) -> fractions.Fraction:
  """Safely evaluates an arithmetic expression to an exact Fraction.

  Allows only integer/float literals, unary `+`/`-`, and binary `+`, `-`, `*`,
  `/` operations via `ast.parse`. Rejects names, function calls, attribute
  access, exponentiation (`**`), and division by zero.
  """
  cleaned = expr.strip().replace(",", "")
  if not cleaned:
    raise ValueError("Empty arithmetic expression.")
  try:
    tree = ast.parse(cleaned, mode="eval")
  except SyntaxError as exc:
    raise ValueError(f"Invalid arithmetic syntax: {expr!r}") from exc

  def _eval(node: ast.AST) -> fractions.Fraction:
    if isinstance(node, ast.Expression):
      return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
      if isinstance(node.value, bool):
        raise ValueError("Boolean constants are not allowed.")
      return fractions.Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
      return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
      left = _eval(node.left)
      right = _eval(node.right)
      if isinstance(node.op, ast.Div) and right == 0:
        raise ValueError("Division by zero.")
      return _BIN_OPS[type(node.op)](left, right)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")

  return _eval(tree)


def _extract_positive_int_literals(expr: str) -> List[int]:
  """Extracts all numeric integer literals from an AST-parsed expression."""
  tree = ast.parse(expr.strip().replace(",", ""), mode="eval")
  nums: List[int] = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Constant):
      if isinstance(node.value, bool) or not isinstance(node.value, int):
        raise ValueError(f"Only integer literals allowed, got {node.value!r}")
      nums.append(int(node.value))
  return nums


def extract_tag(text: str, tag: str) -> Optional[str]:
  """Extracts the content inside the last `<tag>...</tag>` block."""
  pattern = rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
  matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
  if not matches:
    return None
  return matches[-1].strip()


def _tokenizer_has_token(tokenizer: Any, token: str) -> bool:
  if tokenizer is None:
    return False
  vocab = getattr(tokenizer, "vocabulary", None)
  if isinstance(vocab, dict) and token in vocab:
    return True
  try:
    if hasattr(tokenizer, "get_vocabulary") and token in tokenizer.get_vocabulary():
      return True
  except Exception:
    pass
  try:
    if hasattr(tokenizer, "token_to_id"):
      tid = tokenizer.token_to_id(token)
      return tid is not None and int(tid) >= 0
  except Exception:
    return False
  return False


def format_chat_prompt(
    system_prompt: str,
    user_prompt: str,
    tokenizer: Any = None,
    think_mode: str = "think",
) -> str:
  """Formats a system + user prompt using Qwen/ChatML, Gemma, or plain format."""
  if _tokenizer_has_token(tokenizer, "<|im_start|>"):
    user_body = (
        f"{user_prompt}\n/no_think" if think_mode == "no_think" else user_prompt
    )
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_body}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
  if _tokenizer_has_token(tokenizer, "<start_of_turn>"):
    return (
        f"<start_of_turn>user\n{system_prompt}\n\n{user_prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
  return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:\n"


@dataclasses.dataclass(frozen=True)
class CountdownProblem:
  """One procedurally generated Countdown numbers puzzle."""

  numbers: Tuple[int, ...]
  target: int
  reference_solution: str


class CountdownEnv:
  """Procedural R1-Zero / TinyZero Countdown arithmetic environment."""

  _DIFFICULTY_SPECS = {
      "easy": {"count": 3, "min_num": 1, "max_num": 12, "min_t": 10, "max_t": 40},
      "medium": {"count": 4, "min_num": 1, "max_num": 25, "min_t": 15, "max_t": 99},
      "hard": {"count": 4, "min_num": 2, "max_num": 50, "min_t": 50, "max_t": 300},
  }

  def __init__(
      self,
      difficulty: str = "easy",
      seed: int = 42,
      format_bonus: float = 0.1,
  ):
    if difficulty not in self._DIFFICULTY_SPECS:
      raise ValueError(
          f"difficulty must be one of {tuple(self._DIFFICULTY_SPECS)}, "
          f"got {difficulty!r}"
      )
    self.difficulty = difficulty
    self.rng = random.Random(seed)
    self.format_bonus = format_bonus

  def sample_problem(self) -> CountdownProblem:
    spec = self._DIFFICULTY_SPECS[self.difficulty]
    for _ in range(1000):
      nums = [
          self.rng.randint(spec["min_num"], spec["max_num"])
          for _ in range(spec["count"])
      ]
      items: List[Tuple[fractions.Fraction, str]] = [
          (fractions.Fraction(n), str(n)) for n in nums
      ]
      while len(items) > 1:
        i = self.rng.randrange(len(items))
        val_a, expr_a = items.pop(i)
        j = self.rng.randrange(len(items))
        val_b, expr_b = items.pop(j)
        op_choices = ["+", "-", "*"]
        if val_b != 0 and val_a % val_b == 0:
          op_choices.append("/")
        op = self.rng.choice(op_choices)
        if op == "+":
          new_val = val_a + val_b
        elif op == "-":
          new_val = val_a - val_b
        elif op == "*":
          new_val = val_a * val_b
        else:
          new_val = val_a / val_b
        new_expr = f"({expr_a} {op} {expr_b})"
        items.append((new_val, new_expr))

      final_val, final_expr = items[0]
      if final_val.denominator == 1:
        target = int(final_val)
        if (
            spec["min_t"] <= target <= spec["max_t"]
            and target not in nums
        ):
          return CountdownProblem(
              numbers=tuple(nums),
              target=target,
              reference_solution=final_expr,
          )
    # Deterministic fallback puzzle guaranteed valid
    return CountdownProblem(
        numbers=(2, 4, 5),
        target=14,
        reference_solution="((2 * 5) + 4)",
    )

  def sample_batch(self, batch_size: int) -> List[CountdownProblem]:
    return [self.sample_problem() for _ in range(batch_size)]

  def build_prompt(
      self,
      problem: CountdownProblem,
      tokenizer: Any = None,
      think_mode: str = "think",
  ) -> str:
    system_prompt = (
        "You are a math assistant. Use only the given numbers (each at most "
        "once) and +, -, *, / to equal the target. Wrap your final arithmetic "
        "expression in <answer>...</answer>."
    )
    nums_str = ", ".join(str(n) for n in problem.numbers)
    user_prompt = (
        f"Numbers: [{nums_str}]\n"
        f"Target: {problem.target}\n"
        "Respond with <answer>expression</answer> (for example, "
        "<answer>a * b + c</answer>)."
    )
    return format_chat_prompt(
        system_prompt, user_prompt, tokenizer=tokenizer, think_mode=think_mode
    )

  def grade(
      self, problem: CountdownProblem, completion_text: str
  ) -> Tuple[float, Dict[str, Any]]:
    answer = extract_tag(completion_text, "answer")
    has_think = "</think>" in completion_text.lower()
    if answer is None:
      return 0.0, {"solved": False, "has_format": False}

    # Strip optional '= target' suffix if model wrote `a + b = target`
    expr = answer.split("=")[0].strip()
    try:
      used_nums = _extract_positive_int_literals(expr)
      value = safe_eval_arithmetic(expr)
    except ValueError:
      partial = 0.5 * self.format_bonus if has_think else 0.0
      return partial, {"solved": False, "has_format": has_think}

    available_counts = collections.Counter(problem.numbers)
    used_counts = collections.Counter(used_nums)
    valid_numbers = bool(used_nums) and all(
        used_counts[n] <= available_counts[n] for n in used_counts
    )
    solved = valid_numbers and value == fractions.Fraction(problem.target)
    if solved:
      return 1.0, {"solved": True, "has_format": True}
    reward = self.format_bonus if (valid_numbers and has_think) else 0.5 * self.format_bonus
    return reward, {"solved": False, "has_format": True}


@dataclasses.dataclass(frozen=True)
class ToolMathProblem:
  """Multi-step arithmetic word/expression problem for Tool-Integrated RL."""

  question: str
  target: int


class ToolMathEnv:
  """Multi-turn Tool-Integrated Reasoning (TIR) environment with `<calc>`."""

  def __init__(
      self,
      difficulty: str = "easy",
      seed: int = 42,
      format_bonus: float = 0.1,
  ):
    if difficulty not in ("easy", "medium", "hard"):
      raise ValueError(
          f"difficulty must be 'easy', 'medium', or 'hard', got {difficulty!r}"
      )
    self.difficulty = difficulty
    self.rng = random.Random(seed)
    self.format_bonus = format_bonus

  def sample_problem(self) -> ToolMathProblem:
    if self.difficulty == "easy":
      a = self.rng.randint(12, 49)
      b = self.rng.randint(12, 49)
      c = self.rng.randint(10, 99)
      if self.rng.random() < 0.5:
        expr = f"{a} * {b} + {c}"
        target = a * b + c
      else:
        expr = f"{a} * {b} - {c}"
        target = a * b - c
    elif self.difficulty == "medium":
      a, b = self.rng.randint(15, 79), self.rng.randint(15, 79)
      c, d = self.rng.randint(12, 49), self.rng.randint(12, 49)
      e = self.rng.randint(10, 99)
      expr = f"({a} * {b}) + ({c} * {d}) - {e}"
      target = (a * b) + (c * d) - e
    else:
      a, b = self.rng.randint(25, 199), self.rng.randint(25, 199)
      c = self.rng.randint(50, 400)
      d = self.rng.randint(11, 39)
      e = self.rng.randint(100, 999)
      expr = f"(({a} * {b}) - {c}) * {d} + {e}"
      target = ((a * b) - c) * d + e
    return ToolMathProblem(question=f"Compute the exact value of: {expr}", target=target)

  def sample_batch(self, batch_size: int) -> List[ToolMathProblem]:
    return [self.sample_problem() for _ in range(batch_size)]

  def build_prompt(
      self,
      problem: ToolMathProblem,
      tokenizer: Any = None,
      think_mode: str = "think",
  ) -> str:
    system_prompt = (
        "You have a calculator tool. To evaluate an expression, emit "
        "<calc>expression</calc> and wait for <result>value</result>. "
        "Then output the final integer inside <answer>...</answer>."
    )
    user_prompt = (
        f"{problem.question}\n"
        "First call <calc>...</calc> to evaluate intermediate terms, then "
        "respond with <answer>integer</answer>."
    )
    return format_chat_prompt(
        system_prompt, user_prompt, tokenizer=tokenizer, think_mode=think_mode
    )

  def make_tool_step_fn(
      self, tokenizer: Any
  ) -> Callable[[List[int]], Optional[List[int]]]:
    """Returns a `KerasRollout` callback that intercepts `<calc>...</calc>`."""
    calc_re = re.compile(r"<calc>(.*?)</calc>", re.IGNORECASE | re.DOTALL)
    result_re = re.compile(r"<result>.*?</result>", re.IGNORECASE | re.DOTALL)

    def _tokenize(text: str) -> List[int]:
      out = tokenizer.tokenize(text)
      if hasattr(out, "tolist"):
        return [int(t) for t in out.tolist()]
      return [int(t) for t in out]

    def _detokenize(ids: Sequence[int]) -> str:
      out = tokenizer.detokenize(list(ids))
      if isinstance(out, (bytes, bytearray)):
        return out.decode("utf-8", errors="replace")
      return str(out)

    def tool_step_fn(generated_ids: List[int]) -> Optional[List[int]]:
      text = _detokenize(generated_ids)
      if not text.rstrip().endswith("</calc>"):
        return None
      calc_matches = list(calc_re.finditer(text))
      if not calc_matches:
        return None
      last_calc = calc_matches[-1]
      result_matches = list(result_re.finditer(text))
      if result_matches and result_matches[-1].end() > last_calc.end():
        return None

      expr = last_calc.group(1)
      try:
        val = safe_eval_arithmetic(expr)
        val_str = str(int(val)) if val.denominator == 1 else f"{float(val):.6g}"
      except ValueError:
        val_str = "ERROR"
      obs_text = f"\n<result>{val_str}</result>\n"
      return _tokenize(obs_text)

    return tool_step_fn

  def grade(
      self, problem: ToolMathProblem, completion_text: str
  ) -> Tuple[float, Dict[str, Any]]:
    used_tool = "<result>" in completion_text.lower()
    answer = extract_tag(completion_text, "answer")
    if answer is None:
      return 0.0, {
          "solved": False,
          "has_format": False,
          "used_tool": used_tool,
      }
    try:
      val = safe_eval_arithmetic(answer.split("=")[-1].strip())
    except ValueError:
      return 0.5 * self.format_bonus, {
          "solved": False,
          "has_format": True,
          "used_tool": used_tool,
      }
    solved = val == fractions.Fraction(problem.target)
    if solved:
      return 1.0, {"solved": True, "has_format": True, "used_tool": used_tool}
    return self.format_bonus, {
        "solved": False,
        "has_format": True,
        "used_tool": used_tool,
    }
