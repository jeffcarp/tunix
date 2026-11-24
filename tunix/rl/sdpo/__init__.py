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
"""SDPO (Self-Distillation Policy Optimization) module."""

from tunix.rl.sdpo.sdpo_learner import SDPOConfig
from tunix.rl.sdpo.sdpo_learner import SDPOLearner
from tunix.rl.sdpo.sdpo_learner import SDPOTrainExample
from tunix.rl.sdpo.sdpo_learner import sdpo_loss_fn

__all__ = [
    "SDPOConfig",
    "SDPOLearner",
    "SDPOTrainExample",
    "sdpo_loss_fn",
]
