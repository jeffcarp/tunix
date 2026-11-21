# RFC: Local GRPO Training with Keras Models and Apple MLX in Tunix

**Status**: Experimental Prototype (maintained on `jeffcarp/tunix` fork, branch `keras-mlx-grpo-proposal`)  
**Author**: Jeff Carpenter & Antigravity  
**Target Subsystems**: `tunix.rl`, `tunix.generate`, `tunix.sft`  

---

## 1. Executive Summary

This proposal outlines an architectural path to enable **local reinforcement learning (RL) post-training on Apple Silicon** by integrating **Keras 3** with the **Keras MLX** backend into **Tunix**.

By leveraging **Group Relative Policy Optimization (GRPO)** and Apple Silicon's Unified Memory Architecture (UMA), this design enables developers to run end-to-end RL training loops for small reasoning models directly on a Mac, without requiring remote TPU or GPU clusters. 

To achieve this on consumer hardware with bounded memory (e.g., 16 GB to 64 GB RAM), we propose an alternating phase lifecycle between rollout and parameter optimization, combined with reference model eviction and Parameter-Efficient Fine-Tuning (LoRA).

---

## 2. Running Keras MLX with Tunix (Apple Silicon Quickstart)

> **Note**: This feature lives on the `keras-mlx-grpo-proposal` branch of `jeffcarp/tunix` and is not intended to be merged into upstream `google/tunix` yet. `tunix/__init__.py` uses lazy imports so `tunix.rl.keras_grpo` runs without installing JAX, Flax, or TPU dependencies.

### 2.1 Prerequisites

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4)
- **Python**: 3.11 or 3.12 (`>=3.11`)
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/)
- **Keras 3**: `keras-team/keras` `master` (`>=3.16.0` with pluggable backend support; PyPI `keras<=3.15.1` does not yet support `KERAS_BACKEND=mlx`)
- **Keras MLX**: `keras-team/keras-mlx` (automatically installs `mlx`)

### 2.2 Setup Option A — Install directly from GitHub with `uv`

Use this option if you do not already have local clones of `keras` and `keras-mlx`:

```bash
git clone -b keras-mlx-grpo-proposal https://github.com/jeffcarp/tunix.git
cd tunix

uv venv --python 3.12
source .venv/bin/activate

# Install Keras (master with pluggable backends) and keras-mlx (pulls in mlx)
uv pip install \
  "keras @ git+https://github.com/keras-team/keras.git" \
  "keras-mlx @ git+https://github.com/keras-team/keras-mlx.git"

# Install Tunix without JAX/TPU dependencies
uv pip install --no-deps -e .
```

### 2.3 Setup Option B — Install from local editable checkouts

Use this option when co-developing `keras`, `keras-mlx`, and `tunix` side-by-side (assuming sibling directories `../keras` on `master` and `../keras-mlx` on `main`):

```bash
cd tunix

uv venv --python 3.12
source .venv/bin/activate

uv pip install -e ../keras -e ../keras-mlx
uv pip install --no-deps -e .
```

### 2.4 Verify Backend & Metal GPU

```bash
KERAS_BACKEND=mlx python -c "import keras, mlx.core as mx; print('Backend:', keras.backend.backend(), '| Device:', mx.default_device())"
```

Expected output:

```text
Backend: mlx | Device: Device(gpu, 0)
```

### 2.5 Run the Toy GRPO Example

```bash
KERAS_BACKEND=mlx python examples/rl/grpo/keras_mlx_toy.py
```

### 2.6 Run Unit Tests

```bash
KERAS_BACKEND=mlx python -m unittest -v tunix/rl/keras_grpo_test.py
```

### 2.7 Minimal Usage Example

```python
import os

os.environ["KERAS_BACKEND"] = "mlx"

import keras
import numpy as np
from tunix.rl.keras_grpo import KerasGRPOConfig, KerasGRPOTrainer

# Build any causal LM returning logits of shape (batch, seq_len, vocab_size)
model = build_my_keras_causal_lm()
optimizer = keras.optimizers.AdamW(learning_rate=1e-3)

config = KerasGRPOConfig(
    num_generations=4,
    max_tokens_to_generate=16,
    temperature=0.8,
    epsilon=0.2,
    beta=0.0,
)

def reward_fn(prompts_batch, completions_batch):
  return [score_completion(p, c) for p, c in zip(prompts_batch, completions_batch)]

trainer = KerasGRPOTrainer(
    model=model,
    optimizer=optimizer,
    config=config,
    reward_fn=reward_fn,
)

prompts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
metrics = trainer.train_step(prompts)
print(metrics)
```

### 2.8 Troubleshooting

- **Why `--no-deps` on `tunix`?** Tunix's `pyproject.toml` targets JAX/TPU environments (`flax`, `google-metrax`, `perfetto`, `tensorflow_datasets`, `transformers<=4.57.1` which can trigger old `tokenizers` Rust builds on macOS). Because `tunix/__init__.py` resolves exports lazily, `tunix.rl.keras_grpo` only needs `keras`, `keras-mlx`, `mlx`, and `numpy`.
- **`ValueError: Unable to import backend : mlx`**: An older release of `keras` (`<=3.15.1`) is installed instead of `keras-team/keras` `master` (which added pluggable backend loading). Re-run the `uv pip install` command from Option A or Option B above.

---

## 3. Background and Motivation

### 3.1 Tunix Post-Training
Tunix provides state-of-the-art post-training capabilities for Large Language Models, including Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Reinforcement Learning (PPO, GRPO, GSPO, DAPO). Tunix's execution engine is currently built on JAX, Flax NNX, and Optax, targeting Google Cloud TPUs and multi-node GPU clusters.

### 3.2 Keras MLX
Keras 3 supports a pluggable backend architecture. The `keras-mlx` backend maps Keras operations directly to Apple's **MLX** (`mlx.core`), unlocking native Metal GPU acceleration, lazy computation graphs, and unified memory access on Apple Silicon.

### 3.3 The Opportunity
Post-training research and rapid prototyping often suffer from long iteration cycles due to remote cluster provisioning and environment setup. Enabling Tunix to accept a `keras.Model` running on MLX creates a zero-cloud-cost, local-first experimentation environment for reasoning algorithms.

---

## 4. Why GRPO is Uniquely Suited for Local Mac Execution

Traditional RLHF algorithms such as PPO (Proximal Policy Optimization) are prohibitively memory-intensive for local training. GRPO drastically reduces this overhead:

1. **Elimination of the Critic Model**:
   - In PPO, an Actor model ($\pi_\theta$) and a Critic/Value model ($V_\phi$) must be co-located or alternated in memory, along with their respective Adam optimizer states and activation buffers.
   - GRPO eliminates the value network entirely. Instead, it computes advantages by normalizing rewards across a group of $G$ generations sampled from the same prompt:
     $$\hat{A}_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G) + \epsilon}$$
   - This eliminates approximately 50% of the active parameter and optimizer memory footprint.

2. **Zero-Parameter / Programmatic Rewards**:
   - For mathematical reasoning, coding, countdown puzzles, or schema adherence, rewards are deterministic Python functions (e.g., regex matching, sympy equivalence in `tunix.utils.math_rewards`).
   - No neural reward model needs to be loaded into unified memory.

3. **No Reference Model in Memory During Parameter Optimization**:
   - If the KL divergence penalty coefficient is zero ($\beta = 0$, as in DAPO or DeepSeek-R1-Zero style setups), no reference model ($\pi_{ref}$) is loaded at all.
   - If $\beta > 0$, $\pi_{ref}$ is completely static. The reference model is needed only to evaluate log-probabilities $\log \pi_{ref}(y_i|x)$ over the completed rollout trajectories. Once evaluated, the scalar log-probabilities are cached and $\pi_{ref}$ can be immediately evicted from RAM before gradient backpropagation begins.

4. **Single-Pass Old Policy Log-Probs ($\pi_{old}$)**:
   - When training with `num_iterations = 1` (a single optimization epoch per rollout batch), $\pi_{old} = \pi_{rollout}$. The token log-probabilities are recorded during the autoregressive generation pass, requiring zero extra forward passes.

---

## 5. Memory Lifecycle & Model Swapping

### 5.1 Apple Silicon Unified Memory Considerations
Apple Silicon features unified memory shared dynamically between the CPU and Metal GPU. While memory bandwidth is extremely high (up to 800+ GB/s on Max/Ultra chips), capacity is strictly bounded:

- **16 GB – 24 GB**: Typical MacBook Air / Pro base configurations.
- **36 GB – 48 GB**: Mid-tier Pro / Max configurations.
- **64 GB – 128 GB+**: High-end Studio / Max / Ultra configurations.

A 1B parameter model in 16-bit precision requires ~2 GB for weights. However, a standard full-parameter AdamW optimizer requires $2 \times 4\text{ bytes} \times \text{trainable params} = 8\text{ GB}$ in fp32. Combined with activations and KV caches, full-parameter training can quickly saturate a 16 GB machine.

### 5.2 The Alternating Phase Strategy

To guarantee execution within tight memory constraints, Tunix will operate in discrete, non-overlapping phases:

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: ROLLOUT                         │
│  - Load Policy Model (pi_theta) in eval mode               │
│  - KV Cache allocated                                       │
│  - Optimizer states: UNLOADED / ON DISK                     │
│  - Generate G completions per prompt                        │
│  - Cache generated tokens and log pi_old                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               PHASE 2: EVALUATE & SCORE                     │
│  - Free KV Cache                                            │
│  - Evaluate programmatic reward functions r(x, y)          │
│  - Compute group relative advantages A_i                    │
│  - If beta > 0: score log pi_ref and evict reference model  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 3: OPTIMIZATION                    │
│  - Reload / Restore Optimizer States                        │
│  - Forward pass: compute log pi_theta(y|x)                  │
│  - Compute GRPO clipped surrogate loss                      │
│  - Backward pass: compute gradients                         │
│  - Optimizer step: update policy weights                    │
│  - (Optional) Persist optimizer state to disk               │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Full-Weight Fine-Tuning vs. LoRA (PEFT)

| Aspect | Full Weights Fine-Tuning | LoRA (PEFT) |
| :--- | :--- | :--- |
| **Trainable Parameters** | 100% of backbone (~0.5B–2B) | ~0.5%–2% of backbone (~5M–20M) |
| **AdamW Optimizer Memory** | 4–16 GB | **< 50 MB** |
| **Disk Swapping Required** | Recommended for $\le$ 24 GB RAM | **None required** |
| **Reference Model** | Requires swapping base weights | **Zero memory**: Disable LoRA adapter ($\Delta W = 0$) |
| **Target Hardware** | 36 GB+ Macs (or with disk swap) | **Any Mac (16 GB+)** |

With LoRA, the reference model is intrinsically present inside the same model instance (by executing a forward pass with the LoRA scaling factor set to 0), and optimizer state memory becomes negligible.

---

## 6. Architectural Design

Tunix's architecture can be decomposed into an algorithm layer and an execution backend:

```
                       Tunix Algorithmic Core
          ┌──────────────────────────────────────────────┐
          │  - GRPOConfig                                │
          │  - compute_advantages (group normalization)  │
          │  - Reward Registry & Math Rewards            │
          │  - Clipped Surrogate Loss Formulations       │
          └──────────────────────┬───────────────────────┘
                                 │
                     Execution Engine Abstraction
                     ┌───────────┴───────────┐
                     ▼                       ▼
            JAX / NNX Cluster        Keras / MLX Cluster
          (TPU/GPU, Multi-host)     (Apple Silicon Local)
          ┌─────────────────────┐   ┌──────────────────────┐
          │ - RLCluster         │   │ - LocalKerasCluster  │
          │ - PeftTrainer (NNX) │   │ - KerasGRPOTrainer   │
          │ - Vanilla / vLLM    │   │ - KerasRollout (MLX) │
          └─────────────────────┘   └──────────────────────┘
```

### 6.1 Proposed Components

1. **`KerasRollout`** (aligning with `tunix.rl.rollout.base_rollout.BaseRollout`):
   - Accepts a `keras.Model` running with `KERAS_BACKEND=mlx`.
   - Executes batched autoregressive sampling (with future KV cache support).
   - Returns completions and per-token old log-probabilities.

2. **`KerasGRPOTrainer`**:
   - Manages the policy loss calculation using `keras.ops`:
     ```python
     def grpo_loss(model, prompt_ids, completion_ids, completion_mask, advantages, old_logps, beta=0.04, ref_logps=None):
         full_ids = keras.ops.concatenate([prompt_ids, completion_ids], axis=1)
         logits = model(full_ids)
         
         # Slice completion logits
         prompt_len = prompt_ids.shape[1]
         comp_logits = logits[:, prompt_len - 1 : -1, :]
         
         # Compute per-token log probabilities
         log_probs = keras.ops.squeeze(
             keras.ops.take_along_axis(
                 keras.ops.log_softmax(comp_logits, axis=-1),
                 keras.ops.expand_dims(completion_ids, -1),
                 axis=-1,
             ),
             axis=-1,
         )
         
         # Importance sampling ratio
         ratio = keras.ops.exp(log_probs - old_logps)
         clipped_ratio = keras.ops.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
         
         surr1 = ratio * keras.ops.expand_dims(advantages, 1)
         surr2 = clipped_ratio * keras.ops.expand_dims(advantages, 1)
         policy_loss = -keras.ops.minimum(surr1, surr2)
         
         if beta > 0.0 and ref_logps is not None:
             kl = keras.ops.exp(ref_logps - log_probs) - (ref_logps - log_probs) - 1.0
             policy_loss = policy_loss + beta * kl
             
         loss = keras.ops.sum(policy_loss * completion_mask) / keras.ops.maximum(keras.ops.sum(completion_mask), 1.0)
         return loss
     ```
   - Drives model weight updates via standard Keras optimizers (`keras.optimizers.AdamW`).

3. **`LocalKerasCluster`**:
   - Replaces distributed mesh orchestration (`jax.sharding.Mesh`) with a single-device orchestrator managing the memory lifecycle, optional weight serialization (`mx.savez` / `safetensors`), and garbage collection.

---

## 7. Implementation Roadmap

### Phase 1: Minimal End-to-End Proof of Concept (Completed)
- [x] Implement standalone example [examples/rl/grpo/keras_mlx_toy.py](../../examples/rl/grpo/keras_mlx_toy.py) and core trainer [tunix/rl/keras_grpo.py](../../tunix/rl/keras_grpo.py).
- [x] Define a miniature Causal Transformer model in Keras.
- [x] Configure `KERAS_BACKEND=mlx` without requiring JAX/Flax dependencies.
- [x] Implement a toy arithmetic reasoning task (`[Start, A, B] -> [A + B]`).
- [x] Validate the rollout $\to$ reward evaluation $\to$ advantage calculation $\to$ gradient update cycle with unit tests ([tunix/rl/keras_grpo_test.py](../../tunix/rl/keras_grpo_test.py)).

### Phase 2: LoRA Integration & KerasHub Support
- [ ] Leverage Keras 3 LoRA layers to demonstrate PEFT GRPO training.
- [ ] Verify zero-copy reference log-prob calculation by disabling LoRA scaling.
- [ ] Test with lightweight open models from KerasHub (e.g., Qwen-2.5-0.5B or Gemma-2-2B).

### Phase 3: Formal Tunix Integration
- [ ] Extract the backend-agnostic interfaces in Tunix:
  - Abstract `RLCluster` into a base protocol supported by both `JaxRLCluster` and `KerasCluster`.
  - Provide `KerasRollout` alongside `VanillaRollout` and `VllmRollout`.
- [ ] Allow the Tunix CLI (`tunix.cli.grpo_main`) to target Keras MLX backends through configuration flags.

---

## 8. Open Questions

1. **KV Cache in Keras MLX**: Does standard KerasHub `generate()` provide sufficient fine-grained control to return step-wise logits/logprobs on MLX, or is a dedicated Tunix MLX sampler preferable?
2. **Gradient Compilation (`mx.compile`)**: How does `mx.compile` interact with variable-length sequences and masks during the GRPO loss computation? Static padding vs. bucketed sequences.
3. **Multi-device Mac Studio Clusters**: Can MLX distributed (`mx.distributed`) be exposed through Tunix for multi-Mac or multi-GPU M-series configurations in the future?
