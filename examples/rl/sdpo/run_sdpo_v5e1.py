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

"""Prototype SDPO training run on a single TPU v5e-1 using Qwen3-0.6B on GSM8K."""

from __future__ import annotations

import argparse
import re
import time

from datasets import load_dataset
from flax import nnx
from grain import python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import qwix
from transformers import AutoTokenizer
from tunix.models import automodel
from tunix.rl import reshard
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.sdpo import sdpo_learner as sdpo_lib
from tunix.sft import metrics_logger


SYSTEM_PROMPT = (
    "You are a concise math solver. Show your reasoning briefly, then put "
    "your final numeric answer inside <answer>...</answer> tags, e.g. "
    "<answer>42</answer>."
)

_ANSWER_RE = re.compile(r"<answer>\s*([^<]+?)\s*</answer>", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_answer(text: str) -> str | None:
  match = _ANSWER_RE.search(text)
  if match:
    nums = _NUMBER_RE.findall(match.group(1))
    if nums:
      return nums[-1].replace(",", "")
    return match.group(1).strip()
  return None


def _is_numeric_match(guess: str | None, truth: str) -> bool:
  if guess is None:
    return False
  try:
    return abs(float(guess.replace(",", "")) - float(truth.replace(",", ""))) < 1e-4
  except ValueError:
    return guess.strip() == truth.strip()


def gsm8k_reward_fn(prompts, completions, answer, **kwargs):
  del prompts, kwargs
  rewards = []
  for comp, truth in zip(completions, answer):
    guess = extract_answer(comp)
    if _is_numeric_match(guess, str(truth)):
      rewards.append(1.0)
    elif guess is not None:
      rewards.append(0.1)
    else:
      rewards.append(0.0)
  return rewards


def gsm8k_feedback_fn(prompts, completions, rewards, answer, **kwargs):
  del prompts, answer, kwargs
  feedbacks = []
  for comp, r in zip(completions, rewards):
    if r >= 1.0:
      feedbacks.append(None)
      continue
    guess = extract_answer(comp)
    if guess is None:
      feedbacks.append(
          "Format error: your response was missing <answer>NUMBER</answer>. "
          "Write step-by-step reasoning first, then conclude with <answer>NUMBER</answer>."
      )
    else:
      feedbacks.append(
          f"Incorrect answer: your earlier attempt gave <answer>{guess}</answer>, "
          "which is wrong. Show corrected step-by-step reasoning first, verify "
          "each calculation, and then end with <answer>NUMBER</answer>."
      )
  return feedbacks


def format_chat_prompt(tokenizer, question: str) -> str:
  messages = [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": question},
  ]
  return tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=False,
  )


def chat_reprompt_formatter(
    prompt: str,
    solution: str | None,
    feedback: str | None,
    tokenizer,
    question: str = "",
    **kwargs,
) -> str:
  del prompt, kwargs
  parts = [question]
  if solution:
    parts.append(f"\nCorrect solution:\n\n{solution}")
  if feedback:
    parts.append(
        "\nThe following is feedback from your unsuccessful earlier "
        f"attempt:\n\n{feedback}"
    )
  parts.append(
      "\nShow brief step-by-step reasoning first, then put the final numeric "
      "answer inside <answer>...</answer>."
  )
  user_content = "\n".join(parts)
  return format_chat_prompt(tokenizer, user_content)


def build_gsm8k_examples(tokenizer, split: str, limit: int):
  ds = load_dataset("openai/gsm8k", "main", split=split)
  rows = []
  for item in ds.select(range(min(limit, len(ds)))):
    raw_ans = item["answer"].split("####")[-1].strip().replace(",", "")
    rows.append({
        "prompts": format_chat_prompt(tokenizer, item["question"]),
        "question": item["question"],
        "answer": raw_ans,
    })
  return rows


class ListDataSource(grain.RandomAccessDataSource):

  def __init__(self, rows):
    self._rows = rows

  def __getitem__(self, idx):
    return self._rows[idx]

  def __len__(self):
    return len(self._rows)


def make_batched_dataset(rows, batch_size: int):
  return grain.MapDataset.source(ListDataSource(rows)).batch(batch_size)


def evaluate(rl_cluster, eval_rows, label: str):
  prompts = [r["prompts"] for r in eval_rows]
  answers = [r["answer"] for r in eval_rows]
  out = rl_cluster.generate(
      prompts=prompts,
      mode=rl_cluster_lib.Mode.EVAL,
      micro_batch_size=8,
  )
  correct = 0
  formatted = 0
  lengths = []
  for text, toks, truth in zip(out.text, out.tokens, answers):
    guess = extract_answer(text)
    if guess is not None:
      formatted += 1
    if _is_numeric_match(guess, truth):
      correct += 1
    lengths.append(len(toks))
  n = len(eval_rows)
  print(
      f"[Eval {label}] accuracy={correct}/{n} ({correct / n:.1%}) | "
      f"format={formatted}/{n} ({formatted / n:.1%}) | "
      f"mean_tokens={np.mean(lengths):.1f}",
      flush=True,
  )
  if out.text:
    sample_preview = out.text[0].replace("\n", " ")[:240]
    print(f"[Eval {label} sample0 (truth={answers[0]})]: {sample_preview}", flush=True)
  return correct / n, formatted / n, float(np.mean(lengths))


def apply_qwen3_lora(base_model, mesh, rank: int = 16, alpha: float = 32.0):
  lora_provider = qwix.LoraProvider(
      module_path=(
          ".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
          ".*gate_proj|.*up_proj|.*down_proj"
      ),
      rank=rank,
      alpha=alpha,
  )
  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, rngs=nnx.Rngs(0), **model_input
  )
  return reshard.reshard_model_to_mesh(lora_model, mesh)


def main():
  from absl import logging as absl_logging
  absl_logging.set_verbosity(absl_logging.INFO)
  absl_logging.use_python_logging()

  parser = argparse.ArgumentParser()
  parser.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
  parser.add_argument("--max_steps", type=int, default=20)
  parser.add_argument("--batch_size", type=int, default=2)
  parser.add_argument("--num_generations", type=int, default=4)
  parser.add_argument("--eval_size", type=int, default=16)
  parser.add_argument("--max_prompt_length", type=int, default=256)
  parser.add_argument("--max_reprompt_length", type=int, default=512)
  parser.add_argument("--max_tokens_to_generate", type=int, default=256)
  parser.add_argument("--distillation_weight", type=float, default=0.5)
  parser.add_argument("--distillation_alpha", type=float, default=1.0)
  parser.add_argument("--distillation_topk", type=int, default=100)
  parser.add_argument("--learning_rate", type=float, default=1e-5)
  args, _ = parser.parse_known_args()

  print(f"Devices: {jax.devices()}")
  mesh = jax.make_mesh(
      (1, 1),
      ("fsdp", "tp"),
      axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
  )

  t0 = time.perf_counter()
  base_model, local_path = automodel.AutoModel.from_pretrained(
      model_id=args.model_id,
      mesh=mesh,
      model_source=automodel.ModelSource.HUGGINGFACE,
      model_download_path="/tmp/models",
  )
  # Cast floating-point weights to bfloat16 for TPU efficiency
  graphdef, state = nnx.split(base_model)
  state = jax.tree.map(
      lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
      state,
  )
  base_model = nnx.merge(graphdef, state)

  actor_model = apply_qwen3_lora(base_model, mesh, rank=16, alpha=32.0)
  tokenizer = AutoTokenizer.from_pretrained(local_path)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
  print(f"Model + LoRA initialized in {time.perf_counter() - t0:.1f}s")

  train_rows = build_gsm8k_examples(
      tokenizer, "train", limit=args.max_steps * args.batch_size
  )
  eval_rows = build_gsm8k_examples(tokenizer, "test", limit=args.eval_size)
  train_ds = make_batched_dataset(train_rows, batch_size=args.batch_size)

  kv_cache_size = args.max_prompt_length + args.max_tokens_to_generate + 64
  train_rollout_cfg = base_rollout.RolloutConfig(
      max_tokens_to_generate=args.max_tokens_to_generate,
      max_prompt_length=args.max_prompt_length,
      kv_cache_size=kv_cache_size,
      temperature=0.8,
      top_p=0.95,
      top_k=50,
  )
  eval_rollout_cfg = base_rollout.RolloutConfig(
      max_tokens_to_generate=args.max_tokens_to_generate,
      max_prompt_length=args.max_prompt_length,
      kv_cache_size=kv_cache_size,
      temperature=0.0,
      top_p=None,
      top_k=None,
  )

  optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(learning_rate=args.learning_rate, weight_decay=0.01),
  )

  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: mesh,
          rl_cluster_lib.Role.REFERENCE: mesh,
          rl_cluster_lib.Role.ROLLOUT: mesh,
      },
      rollout_engine="vanilla",
      offload_to_cpu=False,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optimizer,
          eval_every_n_steps=args.max_steps + 10,
          max_steps=args.max_steps,
          mini_batch_size=args.batch_size,
          train_micro_batch_size=args.batch_size,
          rollout_micro_batch_size=args.batch_size,
          compute_logps_micro_batch_size=args.batch_size,
          metrics_logging_options=metrics_logger.MetricsLoggerOptions(
              log_dir="/tmp/sdpo_logs",
              flush_every_n_steps=1,
          ),
      ),
      rollout_config={
          rl_cluster_lib.Mode.TRAIN: train_rollout_cfg,
          rl_cluster_lib.Mode.EVAL: eval_rollout_cfg,
      },
  )

  rl_cluster = rl_cluster_lib.RLCluster(
      actor=actor_model,
      reference=None,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )

  step_history = []

  def external_logger(buf):
    if str(buf.mode) != "train":
      return
    summary = {"step": buf.global_steps}
    for k, (vals, op) in buf.metrics.items():
      if k in (
          "rewards/score/mean",
          "sdpo/reprompt_fraction",
          "completions/mean_length",
      ):
        arr = np.array(vals)
        summary[k] = float(op(arr)) if op is not None and arr.size > 0 else float(arr.mean())
    step_history.append(summary)
    print(
        f"[Train step {summary['step']:02d}] "
        f"reward={summary.get('rewards/score/mean', 0.0):.3f} | "
        f"reprompt_frac={summary.get('sdpo/reprompt_fraction', 0.0):.2f} | "
        f"mean_len={summary.get('completions/mean_length', 0.0):.1f}"
    )

  rl_cluster.with_external_metrics_logger(external_logger)

  sdpo_config = sdpo_lib.SDPOConfig(
      num_generations=args.num_generations,
      num_iterations=1,
      distillation_weight=args.distillation_weight,
      distillation_alpha=args.distillation_alpha,
      distillation_mode="topk_logits",
      distillation_topk=args.distillation_topk,
      distillation_add_tail=True,
      teacher_model_kind="ema",
      teacher_update_rate=0.05,
      use_successful_as_teacher=True,
      success_reward_threshold=1.0,
      dont_reprompt_on_self_success=True,
      include_environment_feedback=True,
      max_reprompt_length=args.max_reprompt_length,
  )

  learner = sdpo_lib.SDPOLearner(
      rl_cluster=rl_cluster,
      algo_config=sdpo_config,
      reward_fns=gsm8k_reward_fn,
      feedback_fn=gsm8k_feedback_fn,
      reprompt_formatter=chat_reprompt_formatter,
  )

  acc_before, fmt_before, len_before = evaluate(rl_cluster, eval_rows, "before")
  t_train = time.perf_counter()
  learner.train(train_ds)
  train_elapsed = time.perf_counter() - t_train
  print(f"SDPO training finished {args.max_steps} steps in {train_elapsed:.1f}s")
  acc_after, fmt_after, len_after = evaluate(rl_cluster, eval_rows, "after")

  print(
      "=== SDPO Summary ===\n"
      f"  Eval accuracy : {acc_before:.1%} -> {acc_after:.1%}\n"
      f"  Eval format   : {fmt_before:.1%} -> {fmt_after:.1%}\n"
      f"  Eval mean len : {len_before:.1f} -> {len_after:.1f} tokens"
  )


if __name__ == "__main__":
  main()
