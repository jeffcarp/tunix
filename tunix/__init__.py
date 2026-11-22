# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tunix API."""

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
  __version__ = version("google-tunix")  # match the name in pyproject.toml
except PackageNotFoundError:
  __version__ = "0.0.0.dev0"  # fallback for editable installs


# Lazy public exports avoid eagerly importing JAX/Flax/Metrax/Perfetto when
# using lightweight or alternative backends such as Keras MLX.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AgenticGRPOConfig": (
        "tunix.rl.agentic.agentic_grpo_learner",
        "GRPOConfig",
    ),
    "AgenticGRPOLearner": (
        "tunix.rl.agentic.agentic_grpo_learner",
        "GRPOLearner",
    ),
    "CacheConfig": ("tunix.generate.sampler", "CacheConfig"),
    "CheckpointManager": ("tunix.sft.checkpoint_manager", "CheckpointManager"),
    "ClusterConfig": ("tunix.rl.rl_cluster", "ClusterConfig"),
    "CountdownEnv": ("tunix.rl.keras_rl_envs", "CountdownEnv"),
    "DPOTrainer": ("tunix.sft.dpo.dpo_trainer", "DPOTrainer"),
    "DPOTrainingConfig": ("tunix.sft.dpo.dpo_trainer", "DPOTrainingConfig"),
    "DistillationTrainer": (
        "tunix.distillation.distillation_trainer",
        "DistillationTrainer",
    ),
    "DistillationTrainingConfig": (
        "tunix.distillation.distillation_trainer",
        "TrainingConfig",
    ),
    "DpoTrainer": ("tunix.sft.dpo.dpo_trainer", "DpoTrainer"),
    "DpoTrainingConfig": ("tunix.sft.dpo.dpo_trainer", "DpoTrainingConfig"),
    "GRPOConfig": ("tunix.rl.grpo.grpo_learner", "GRPOConfig"),
    "GRPOLearner": ("tunix.rl.grpo.grpo_learner", "GRPOLearner"),
    "GrpoConfig": ("tunix.rl.grpo.grpo_learner", "GrpoConfig"),
    "GrpoLearner": ("tunix.rl.grpo.grpo_learner", "GrpoLearner"),
    "KerasGRPOConfig": ("tunix.rl.keras_grpo", "KerasGRPOConfig"),
    "KerasGRPOTrainer": ("tunix.rl.keras_grpo", "KerasGRPOTrainer"),
    "KerasRollout": ("tunix.rl.keras_grpo", "KerasRollout"),
    "MetricsBuffer": ("tunix.rl.rl_cluster", "MetricsBuffer"),
    "MetricsLogger": ("tunix.sft.metrics_logger", "MetricsLogger"),
    "MetricsLoggerOptions": (
        "tunix.sft.metrics_logger",
        "MetricsLoggerOptions",
    ),
    "ORPOTrainer": ("tunix.sft.dpo.dpo_trainer", "ORPOTrainer"),
    "ORPOTrainingConfig": ("tunix.sft.dpo.dpo_trainer", "ORPOTrainingConfig"),
    "OrpoTrainer": ("tunix.sft.dpo.dpo_trainer", "OrpoTrainer"),
    "OrpoTrainingConfig": ("tunix.sft.dpo.dpo_trainer", "OrpoTrainingConfig"),
    "PPOConfig": ("tunix.rl.ppo.ppo_learner", "PPOConfig"),
    "PPOLearner": ("tunix.rl.ppo.ppo_learner", "PPOLearner"),
    "PeftTrainer": ("tunix.sft.peft_trainer", "PeftTrainer"),
    "PerfMetricsConfig": ("tunix.perf.metrics", "PerfMetricsConfig"),
    "PerfMetricsExport": ("tunix.perf.export", "PerfMetricsExport"),
    "PerfSpanQuery": ("tunix.perf.metrics", "PerfSpanQuery"),
    "PpoConfig": ("tunix.rl.ppo.ppo_learner", "PpoConfig"),
    "PpoLearner": ("tunix.rl.ppo.ppo_learner", "PpoLearner"),
    "RLCluster": ("tunix.rl.rl_cluster", "RLCluster"),
    "RLTrainingConfig": ("tunix.rl.rl_cluster", "RLTrainingConfig"),
    "RewardFn": ("tunix.rl.grpo.grpo_learner", "RewardFn"),
    "Role": ("tunix.rl.rl_cluster", "Role"),
    "RolloutConfig": ("tunix.rl.rollout.base_rollout", "RolloutConfig"),
    "Sampler": ("tunix.generate.sampler", "Sampler"),
    "Tokenizer": ("tunix.generate.tokenizer_adapter", "Tokenizer"),
    "TokenizerAdapter": (
        "tunix.generate.tokenizer_adapter",
        "TokenizerAdapter",
    ),
    "ToolMathEnv": ("tunix.rl.keras_rl_envs", "ToolMathEnv"),
    "TrainingConfig": ("tunix.sft.peft_trainer", "TrainingConfig"),
    "disable_lora": ("tunix.rl.keras_grpo", "disable_lora"),
}

__all__ = sorted(_LAZY_IMPORTS)


def __getattr__(name: str):
  if name not in _LAZY_IMPORTS:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  module_name, attr_name = _LAZY_IMPORTS[name]
  module = importlib.import_module(module_name)
  value = getattr(module, attr_name)
  globals()[name] = value
  return value


def __dir__() -> list[str]:
  return sorted(set(globals()) | set(_LAZY_IMPORTS))
