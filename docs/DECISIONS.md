# Engineering Decisions

## D-001: From-scratch spec decoding instead of using transformers' built-in

**Status:** Accepted

**Context.** HuggingFace `transformers` ships `model.generate(assistant_model=draft)` for speculative decoding. The 50-line version would just call that.

**Decision.** Implement the Leviathan et al. acceptance loop directly.

**Why.** The point of the experiment is to understand the algorithm. Calling `model.generate` hides exactly the part I want to measure — the acceptance probabilities at each step, where rejections happen, how the draft model's distribution diverges from the target's. With the custom implementation I can log all of it.

**Tradeoff.** Slower than the HF version. Doesn't matter for the experiment; matters a lot if anyone tries to use this for production. Don't.

---

## D-002: Custom benchmark harness instead of pytest-benchmark or timeit

**Status:** Accepted

**Context.** Lots of existing benchmark tools. Why another one?

**Decision.** Roll a small one with explicit warmup, percentiles, and environment metadata.

**Why.**
- `timeit` only gives means, not distributions. Means hide tail-latency stories.
- `pytest-benchmark` is great but assumes you're inside pytest. The experiments are scripts.
- `airspeed` and similar are too heavy.
- I need to attach environment metadata (GPU, CUDA version, model id) to every result. Building that in is cleaner than hacking onto an external tool.

The harness is ~150 lines and does exactly what's needed.

---

## D-003: Simulated forward functions for CPU-runnable experiments

**Status:** Accepted

**Context.** Experiments 02, 03, and 05 study scheduling and allocation. Running them on a real model would require a GPU and add measurement noise unrelated to what we're studying.

**Decision.** Use fake forward functions that sleep for realistic durations.

**Why.** The point of these experiments is the scheduling shape, not absolute throughput. A fake model with `time.sleep(per_token_seconds * batch_size^0.6)` captures the sub-linear batch scaling we care about, while letting the experiments run in CI and on laptops.

For absolute numbers on real models, use vLLM benchmarks. For "does this scheduling policy beat that one," the fake model is sufficient and reproducible.

**Tradeoff.** The numbers are not directly comparable to production. We make this loud in each README.

---

## D-004: No automated GPU experiments in CI

**Status:** Accepted

**Context.** The GPU experiments (01, 04) need ~16GB VRAM and download multi-GB weights. CI runners don't have GPUs.

**Decision.** CI runs CPU-only experiments and the unit tests. GPU experiments are manual; their READMEs include checked-in result snapshots.

**Why.** Self-hosted GPU runners are expensive and fragile. The CPU experiments + unit tests give enough coverage to catch most regressions. The GPU experiments are infrequent enough (re-run when model versions change, new hardware) that running them manually is fine.

---

## R-001: Reverted — JAX implementation

I started experiment 01 in JAX because the functional style maps well to spec decoding. The implementation was clean but:
- Most readers expect PyTorch/HuggingFace.
- JAX's eager mode is significantly slower than PyTorch for these shapes (we don't benefit from JIT for variable-length generation).
- HuggingFace's draft model interface is PyTorch-native.

Reverted to PyTorch. The JAX version sits in git history for anyone curious.

## R-002: Reverted — vLLM as the throughput baseline

Initial plan: compare each experiment against vLLM as the gold-standard baseline. Built it; quickly hit problems:
- vLLM versions drift; reproducing numbers across versions is painful.
- The point of these experiments is to understand the techniques, not to claim "I beat vLLM."
- Including vLLM made the dependency surface huge.

The TL;DR snippets in each README now reference vLLM/TGI by name where relevant but don't try to outperform them.
