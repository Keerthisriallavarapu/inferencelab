# Experiment 02: Continuous Batching

**Question**: At what concurrency does continuous batching start to meaningfully outperform static batching? How does the picture change with prompt-length variance?

## TL;DR

- At concurrency 1-2: no difference. Both wait the same amount.
- At concurrency 4-8 with uniform prompt lengths: CB is ~1.4x throughput at the cost of slightly worse p50 latency.
- At concurrency 8+ with high prompt-length variance: CB is 2-3x throughput. This is the regime CB was designed for.
- The break-even point isn't a fixed concurrency number — it's the moment the slowest request in a static batch starts blocking faster ones.

## Setup

```bash
inferencelab run 02_continuous_batching
```

This uses a small model so it runs in <5 minutes.

## How CB wins

Static batching: 4 requests with output lengths 50, 50, 50, 500.
- Static batch waits for all 4 to hit 500 tokens → wasted work on 3 requests.
- CB: requests 1-3 finish at step 50 and 3 new requests slot in. Throughput stays high.

The longer the variance in output lengths, the bigger CB's win.

## What I measured

Submitted a Poisson stream of requests with two distributions:

- **Uniform**: all max_new_tokens=128.
- **Variable**: max_new_tokens ~ Geometric(p=0.01), capped at 512.

For each, I measured tokens/sec at concurrency levels [1, 2, 4, 8, 16, 32].

## Caveats

- This experiment uses the included `ContinuousBatchingScheduler` with a fake forward function. To measure on a real model, you'd use vLLM or TGI. The point of this experiment is the scheduling shape, not raw throughput.
- The fake forward sleeps to simulate decode time. Real models have batch-size-dependent decode times (sub-linear scaling) that we don't model. So the absolute numbers below are illustrative; the qualitative shape is what matters.

## Result snapshot

`results/throughput_vs_concurrency.png` shows the two curves diverging sharply at concurrency 4+ in the variable-length scenario.
