# InferenceLab

A series of LLM inference optimization experiments, each with reproducible code, a writeup, and honest numbers.

This isn't a serving framework. It's a learning lab — I built minimal versions of speculative decoding, continuous batching, KV-cache management, and quantization comparison so I could understand *why* production frameworks like vLLM and TGI are built the way they are.

## Experiments

| # | Topic | What it measures | Needs GPU? |
|---|---|---|---|
| 01 | Speculative Decoding | Throughput speedup and acceptance rate vs k | Yes |
| 02 | Continuous Batching | Static vs continuous batching across concurrency | No (CPU sim) |
| 03 | KV Cache (PagedAttention) | Memory savings: paged vs naive allocator | No (sim) |
| 04 | Quantization | INT8/INT4/BF16 quality, speed, memory | Yes |
| 05 | Scheduler Policies | Admission policy effects on tail latency | No (CPU sim) |

Each experiment is self-contained under `experiments/` with its own README, runnable `run.py`, and saved results. The synthesis writeup is in [`writeups/01_what_i_learned.md`](writeups/01_what_i_learned.md).

## Quick start

```bash
pip install -e ".[dev]"

# CPU-only experiments
inferencelab run 02_continuous_batching
inferencelab run 03_kv_cache
inferencelab run 05_scheduler

# GPU experiments (need ~16-24GB VRAM and a HuggingFace login)
pip install -e ".[gpu]"
huggingface-cli login
inferencelab run 01_speculative_decoding
inferencelab run 04_quantization
```

## Project structure

```
inferencelab/
├── inferencelab/
│   ├── bench/        # benchmark harness with percentiles and CIs
│   ├── models/       # HF wrapper with quantization options
│   ├── serving/
│   │   ├── speculative.py    # spec decoding from scratch
│   │   └── scheduler.py      # continuous-batching scheduler
│   ├── viz/          # consistent matplotlib helpers
│   └── cli.py
├── experiments/
│   ├── 01_speculative_decoding/
│   ├── 02_continuous_batching/
│   ├── 03_kv_cache/
│   ├── 04_quantization/
│   └── 05_scheduler/
├── writeups/
└── tests/
```

## What's here vs what's not

**What's here**: minimal, readable implementations of each technique. The speculative decoding is the actual Leviathan et al. algorithm. The CB scheduler does FCFS/shortest-first/longest-first admission. The paged allocator does block-based allocation with a free pool. Each one is small enough to read in one sitting.

**What's not here**: production-grade kernels. PagedAttention's attention kernel is in vLLM, not here. FlashAttention's CUDA tricks are in `flash-attn`, not here. CUDA graph capture, tensor parallelism, etc. — all out of scope. This is a learning project, not a serving stack.

## Reproducing the numbers

Every experiment has a `run.py` that produces the JSON and plots referenced in its README. Numbers in the READMEs are from runs on:

- RTX 4090 (24GB VRAM), AMD Ryzen 9 7950X, 64GB RAM
- Ubuntu 22.04, CUDA 12.4, PyTorch 2.5.0, Transformers 4.46

If you can't reproduce within ~15% on similar hardware, something is off. File an issue.

## Why I built this

I kept reading papers (Spec Decoding, vLLM, etc.) and skipping the details that made the techniques actually work. Implementing each one from scratch — even badly — forced me to face the engineering choices the papers gloss over. The writeup at [`writeups/01_what_i_learned.md`](writeups/01_what_i_learned.md) is the synthesis of what stuck.

## License

Apache 2.0.
