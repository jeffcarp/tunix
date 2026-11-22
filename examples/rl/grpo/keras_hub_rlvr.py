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

"""2026 SOTA RLVR & Multi-Turn Tool-Use RL on Apple Silicon with KerasHub + MLX.

Runs Group Relative Policy Optimization variants (`dapo`, `drgrpo`, `gspo`,
`grpo`) on pretrained KerasHub CausalLMs (`qwen3_0.6b_en`, `qwen3_1.7b_en`,
`qwen2.5_instruct_0.5b_en`, `gemma3_instruct_1b`, or `hf://...` handles)
using verifiable procedural RL environments:
- `countdown`: Canonical R1-Zero / TinyZero Countdown arithmetic puzzle.
- `tool_math`: Multi-turn calculator Tool-Integrated Reasoning (TIR) with
  observation token loss masking (`loss_mask = 0.0` on `<result>...</result>`).

See `docs/proposals/keras_mlx_grpo.md` for full setup instructions.

Example:
  KERAS_BACKEND=mlx python examples/rl/grpo/keras_hub_rlvr.py \
      --preset qwen3_0.6b_en --task countdown --algo dapo --steps 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, List, Sequence, Set

# Select MLX backend before importing Keras / KerasHub.
os.environ.setdefault("KERAS_BACKEND", "mlx")

import keras
import keras_hub
import numpy as np
from tunix.rl.keras_grpo import (
    KerasGRPOConfig,
    KerasGRPOTrainer,
    count_parameters,
)
from tunix.rl.keras_rl_envs import CountdownEnv, ToolMathEnv


def encode_text(tokenizer: Any, text: str) -> List[int]:
  """Encodes a prompt string to token IDs, prepending BOS if used by tokenizer."""
  out = tokenizer.tokenize(text)
  tokens = [int(t) for t in (out.tolist() if hasattr(out, "tolist") else out)]
  bos_id = getattr(tokenizer, "start_token_id", None)
  if bos_id is not None and (not tokens or tokens[0] != int(bos_id)):
    tokens = [int(bos_id)] + tokens
  return tokens


def decode_completion(
    tokenizer: Any,
    token_ids: Sequence[int],
    stop_ids: Set[int],
    pad_id: int,
) -> str:
  """Decodes completion token IDs up to the first stop/pad token."""
  clean_ids: List[int] = []
  for tok in token_ids:
    t = int(tok)
    if t in stop_ids or (t == pad_id and pad_id not in stop_ids):
      break
    clean_ids.append(t)
  if not clean_ids:
    return ""
  decoded = tokenizer.detokenize(clean_ids)
  if isinstance(decoded, (bytes, bytearray)):
    return decoded.decode("utf-8", errors="replace")
  return str(decoded)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run SOTA RLVR / Tool-Use GRPO locally on Apple Silicon via KerasHub + MLX."
  )
  parser.add_argument(
      "--preset",
      type=str,
      default="qwen3_0.6b_en",
      help=(
          "KerasHub preset or hf:// handle. Recommended for 24 GB M4: "
          "qwen3_0.6b_en (default), qwen3_1.7b_en, qwen2.5_instruct_0.5b_en, "
          "qwen2.5_math_instruct_1.5b_en, gemma3_instruct_270m, "
          "gemma3_instruct_1b, or hf://Qwen/Qwen3-0.6B."
      ),
  )
  parser.add_argument(
      "--task",
      type=str,
      choices=["countdown", "tool_math"],
      default="countdown",
      help="Verifiable RL environment: 'countdown' or multi-turn 'tool_math'.",
  )
  parser.add_argument(
      "--difficulty",
      type=str,
      choices=["easy", "medium", "hard"],
      default="easy",
  )
  parser.add_argument(
      "--algo",
      type=str,
      choices=["dapo", "drgrpo", "grpo", "gspo"],
      default="dapo",
      help="RLVR policy optimization variant.",
  )
  parser.add_argument("--dtype", type=str, default="bfloat16")
  parser.add_argument(
      "--quantize",
      type=str,
      choices=["none", "int4", "int8"],
      default="none",
      help="Optional post-load weight quantization before enabling LoRA.",
  )
  parser.add_argument(
      "--lora-rank",
      type=int,
      default=8,
      help="LoRA rank (0 disables LoRA and trains full weights).",
  )
  parser.add_argument(
      "--lora-targets",
      type=str,
      default="auto",
      help="Comma-separated layer names for LoRA, or 'auto' for backbone defaults.",
  )
  parser.add_argument("--steps", type=int, default=20)
  parser.add_argument("--prompts-per-step", type=int, default=2)
  parser.add_argument("--num-generations", type=int, default=4)
  parser.add_argument(
      "--num-iterations",
      type=int,
      default=1,
      help=(
          "Number of optimization iterations per rollout batch (mu). Set >1 to "
          "activate off-policy clipping."
      ),
  )
  parser.add_argument(
      "--loss-agg-mode",
      type=str,
      default=None,
      choices=["token-mean", "sequence-mean-token-mean", "sequence-mean-token-scale"],
      help="Optional override for loss aggregation mode.",
  )
  parser.add_argument("--train-micro-batch-size", type=int, default=1)
  parser.add_argument("--max-new-tokens", type=int, default=256)
  parser.add_argument("--max-turns", type=int, default=3)
  parser.add_argument("--temperature", type=float, default=0.9)
  parser.add_argument("--top-p", type=float, default=0.95)
  parser.add_argument(
      "--beta",
      type=float,
      default=0.0,
      help="KL penalty coefficient (uses zero-copy LoRA toggle when LoRA > 0).",
  )
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument(
      "--think-mode",
      type=str,
      choices=["think", "no_think"],
      default="no_think",
      help=(
          "Qwen3 thinking mode: 'no_think' appends /no_think for fast local "
          "episodes (default); 'think' allows long <think>...</think> CoT."
      ),
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
      "--print-every",
      type=int,
      default=5,
      help="Print sample generations every N steps (0 disables samples).",
  )
  parser.add_argument(
      "--save-lora",
      type=str,
      default=None,
      help="Optional path ending in .lora.h5 to save trained LoRA weights.",
  )
  parser.add_argument(
      "--metrics-json",
      type=str,
      default=None,
      help="Optional path to save per-step metrics JSON.",
  )
  parser.add_argument(
      "--device",
      type=str,
      choices=["auto", "gpu", "cpu"],
      default="auto",
      help="MLX device override ('auto', 'gpu', or 'cpu').",
  )
  parser.add_argument(
      "--log-ref-kl",
      action="store_true",
      help="Log reference KL via zero-copy LoRA toggle even when --beta 0.0.",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  np.random.seed(args.seed)
  keras.utils.set_random_seed(args.seed)

  device_str = "default"
  if keras.backend.backend() == "mlx":
    import mlx.core as mx

    if args.device == "gpu":
      mx.set_default_device(mx.gpu)
    elif args.device == "cpu":
      mx.set_default_device(mx.cpu)
    device_str = str(mx.default_device())

  print("=" * 72)
  print(
      f"Tunix x KerasHub RLVR | Backend: {keras.backend.backend()} | "
      f"Device: {device_str} | Preset: {args.preset} ({args.dtype})"
  )
  print(
      f"Algo: {args.algo.upper()} | Task: {args.task} ({args.difficulty}) | "
      f"LoRA rank: {args.lora_rank} | Beta KL: {args.beta}"
  )
  print("=" * 72)

  load_t0 = time.perf_counter()
  model = keras_hub.models.CausalLM.from_preset(args.preset, dtype=args.dtype)
  if args.quantize != "none":
    model.quantize(args.quantize)
  if args.lora_rank > 0:
    targets = (
        None
        if args.lora_targets == "auto"
        else [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    )
    model.backbone.enable_lora(rank=args.lora_rank, target_layer_names=targets)

  total_params, trainable_params = count_parameters(model)
  load_s = time.perf_counter() - load_t0
  print(
      f"Model loaded in {load_s:.2f}s | Total params: {total_params:,} | "
      f"Trainable params: {trainable_params:,} "
      f"({100.0 * trainable_params / max(total_params, 1):.2f}%)"
  )

  tokenizer = model.preprocessor.tokenizer
  eos_id = getattr(tokenizer, "end_token_id", None)
  extra_stops = []
  if getattr(tokenizer, "end_token2_id", None) is not None:
    extra_stops.append(int(tokenizer.end_token2_id))
  pad_id = getattr(tokenizer, "pad_token_id", 0)
  if pad_id is None:
    pad_id = int(eos_id) if eos_id is not None else 0

  stop_set: Set[int] = set(extra_stops)
  if eos_id is not None:
    stop_set.add(int(eos_id))

  if args.task == "countdown":
    env = CountdownEnv(difficulty=args.difficulty, seed=args.seed)
    tool_step_fn = None
    max_turns = 1
  else:
    env = ToolMathEnv(difficulty=args.difficulty, seed=args.seed)
    tool_step_fn = env.make_tool_step_fn(tokenizer)
    max_turns = args.max_turns

  config = KerasGRPOConfig(
      algo_variant=args.algo,
      loss_agg_mode=args.loss_agg_mode,
      num_generations=args.num_generations,
      num_iterations=args.num_iterations,
      max_tokens_to_generate=args.max_new_tokens,
      temperature=args.temperature,
      top_p=args.top_p,
      beta=args.beta,
      log_ref_kl=args.log_ref_kl,
      pad_token_id=int(pad_id),
      eos_token_id=int(eos_id) if eos_id is not None else None,
      stop_token_ids=tuple(extra_stops) if extra_stops else None,
      train_micro_batch_size=args.train_micro_batch_size,
      max_turns=max_turns,
  )
  optimizer = keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=0.0)

  current_problems: List[Any] = []
  last_infos: List[dict] = []
  last_texts: List[str] = []
  history: List[dict] = []

  def reward_fn(prompts_batch, completions_batch):
    nonlocal last_infos, last_texts
    rewards: List[float] = []
    infos: List[dict] = []
    texts: List[str] = []
    for idx, comp_ids in enumerate(completions_batch):
      prob = current_problems[idx // args.num_generations]
      text = decode_completion(tokenizer, comp_ids, stop_set, int(pad_id))
      r, info = env.grade(prob, text)
      rewards.append(float(r))
      infos.append(info)
      texts.append(text)
    last_infos = infos
    last_texts = texts
    return rewards

  trainer = KerasGRPOTrainer(
      model=model,
      optimizer=optimizer,
      config=config,
      reward_fn=reward_fn,
  )

  print("-" * 72)
  for step in range(1, args.steps + 1):
    current_problems = env.sample_batch(args.prompts_per_step)
    prompt_strs = [
        env.build_prompt(p, tokenizer=tokenizer, think_mode=args.think_mode)
        for p in current_problems
    ]
    prompt_token_batch = [encode_text(tokenizer, s) for s in prompt_strs]

    metrics = trainer.train_step(prompt_token_batch, tool_step_fn=tool_step_fn)

    solve_pct = float(100.0 * np.mean([float(x["solved"]) for x in last_infos]))
    fmt_pct = float(100.0 * np.mean([float(x["has_format"]) for x in last_infos]))
    tool_use_pct = float(
        100.0 * np.mean([float(x.get("used_tool", False)) for x in last_infos])
    )
    seq_lens = np.sum(metrics["rollout_output"].completion_mask, axis=-1)
    solved_lens = [float(l) for l, info in zip(seq_lens, last_infos) if info["solved"]]
    wrong_lens = [float(l) for l, info in zip(seq_lens, last_infos) if not info["solved"]]
    len_solved_mean = float(np.mean(solved_lens)) if solved_lens else None
    len_wrong_mean = float(np.mean(wrong_lens)) if wrong_lens else None
    truncated_pct = float(100.0 * np.mean(seq_lens >= args.max_new_tokens))

    history.append(
        {
            "step": int(step),
            "reward_mean": float(metrics["reward/mean"]),
            "reward_max": float(metrics["reward/max"]),
            "solve_pct": solve_pct,
            "fmt_pct": fmt_pct,
            "tool_use_pct": tool_use_pct,
            "entropy": float(metrics["entropy"]),
            "pg_clipfrac": float(metrics["pg_clipfrac"]),
            "kl": float(metrics["kl"]),
            "response_len_mean": float(metrics["response_len/mean"]),
            "len_solved_mean": len_solved_mean,
            "len_wrong_mean": len_wrong_mean,
            "truncated_pct": truncated_pct,
            "skipped_zero_var_groups": int(metrics["skipped_zero_var_groups"]),
            "grad_norm": float(metrics["grad_norm"]),
            "rollout_s": float(metrics["rollout_s"]),
            "train_s": float(metrics["train_s"]),
            "peak_mem_gb": float(metrics["peak_mem_gb"]),
        }
    )

    tool_str = f" Tool={tool_use_pct:3.0f}%" if args.task == "tool_math" else ""
    print(
        f"Step {step:2d}/{args.steps:2d} | "
        f"R={metrics['reward/mean']:.2f} (max {metrics['reward/max']:.2f}) | "
        f"Solve={solve_pct:4.0f}% Fmt={fmt_pct:4.0f}%{tool_str} | "
        f"Ent={metrics['entropy']:.2f} Clip={100*metrics['pg_clipfrac']:4.1f}% "
        f"KL={metrics['kl']:.4f} | "
        f"Len={metrics['response_len/mean']:.1f} Skip={metrics['skipped_zero_var_groups']} | "
        f"Grad={metrics['grad_norm']:.2f} | "
        f"t={metrics['rollout_s']:.2f}+{metrics['train_s']:.2f}s | "
        f"PeakMem={metrics['peak_mem_gb']:.2f}GB"
    )

    if args.print_every > 0 and (step == 1 or step % args.print_every == 0 or step == args.steps):
      best_idx = int(np.argmax(metrics["rewards"]))
      sample_prob = current_problems[best_idx // args.num_generations]
      snippet = last_texts[best_idx].replace("\n", "\\n")
      if len(snippet) > 260:
        snippet = snippet[:260] + "..."
      print(f"  [Sample] Problem: {sample_prob} | Best Reward: {metrics['rewards'][best_idx]:.2f}")
      print(f"  [Output] {snippet}")

  if args.metrics_json:
    metrics_dir = os.path.dirname(args.metrics_json)
    if metrics_dir:
      os.makedirs(metrics_dir, exist_ok=True)
    with open(args.metrics_json, "w", encoding="utf-8") as f:
      json.dump({"args": vars(args), "steps": history}, f, indent=2)
    print(f"Saved metrics JSON to {args.metrics_json}")

  if args.save_lora and args.lora_rank > 0:
    model.backbone.save_lora_weights(args.save_lora)
    print(f"Saved LoRA adapter weights to {args.save_lora}")


if __name__ == "__main__":
  main()
