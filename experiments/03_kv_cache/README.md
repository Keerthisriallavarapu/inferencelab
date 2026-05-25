# Experiment 03: KV Cache Strategies

**Question**: How much do PagedAttention-style block-based KV caches actually save vs. naive contiguous allocation? When is the win biggest?

## TL;DR

- For uniform request lengths: naive caching is fine; paging adds maybe 5-10% memory savings.
- For high prompt-length variance + long generations: paging saves 40-60% memory at the same throughput.
- The savings translate into batch-size headroom: with paging you can fit 2-3x more concurrent requests in the same VRAM.

This is the headline win of vLLM and the reason PagedAttention is now standard. The experiment quantifies it.

## Why naive KV caches are wasteful

A naive implementation allocates `(batch, max_seq, n_heads, head_dim)` upfront. If `max_seq` is conservatively large (say, 4096), every request burns that much memory even if it only generates 50 tokens.

Block-based caches allocate fixed-size blocks (typically 16 tokens) on demand. A 50-token request uses ~4 blocks. A 4000-token request uses 250.

Fragmentation: naive allocation also can't reuse freed memory from short requests for new long ones without compaction. Block-based caches naturally support this — blocks are returned to a free pool.

## Method

Simulate a batch of N requests with mixed lengths. Measure:
1. Peak VRAM usage for a naive contiguous allocator.
2. Peak VRAM usage for a block-based paged allocator.
3. Max number of requests we can fit in a fixed VRAM budget.

We don't run a real model — that's vLLM's job. This experiment isolates the *allocator* behavior, which is the part you can reason about and verify without GPUs.

## Files

- `run.py` — runs the simulator
- `paged_allocator.py` — minimal block-based allocator implementation
- `results/memory_comparison.png`
