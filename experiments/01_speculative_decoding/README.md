# Experiment 01: Speculative Decoding

**Question**: At what value of `k` (draft tokens per round) does speculative decoding give the best throughput? Does the acceptance rate vs. k curve match the geometric prediction?

## TL;DR

For Llama 3.2 1B drafting for Llama 3.1 8B on HumanEval:

- Best throughput at **k=4**, ~1.86x speedup over no-spec baseline.
- Acceptance rate decays from 0.71 at k=2 to 0.24 at k=16.
- The geometric model `acceptance(k) = α^k` with α ≈ 0.85 fits well up to k=6, then deviates as the draft model loses thread.

Numbers below are from an RTX 4090 (24GB), batch size 1, prompt 512 / output 256.

## Setup

```bash
pip install -e ".[gpu]"
huggingface-cli login
inferencelab run 01_speculative_decoding
```

This downloads ~16GB of model weights on first run.

## Results

| k | Tokens/sec | Speedup | Acceptance rate |
|---|---|---|---|
| 1 (baseline) | 28.4 | 1.00x | — |
| 2 | 41.1 | 1.45x | 0.71 |
| 4 | **52.7** | **1.86x** | 0.58 |
| 6 | 51.0 | 1.80x | 0.47 |
| 8 | 49.3 | 1.74x | 0.41 |
| 16 | 38.2 | 1.34x | 0.24 |

Acceptance curve: see `results/acceptance_curve.png`.

## Interpretation

The classic tradeoff: larger k amortizes the target-model forward pass over more draft tokens, but each additional draft token is less likely to be accepted (the draft model gets further from where the target would go). At k=4 we hit the sweet spot.

Above k=8, the cost of running the draft model autoregressively starts to dominate, AND the acceptance rate drops below what the speedup math needs to stay positive. By k=16 the speedup is half what it was at k=4.

## Caveats and notes

- **Workload-dependent**. On chat data the optimum shifted to k=6 because chat token distributions are more predictable (more repeated tokens, less branching). For HumanEval (code), branching is high — every variable name is a fresh sample.
- **The geometric model is approximate.** Real acceptance isn't independent across positions: a rejection early in a round invalidates the rest, which makes empirical rates fall faster than α^k for large k. This shows in the chart — the deviation grows at k=10+.
- **Batch size matters a lot.** At batch=1 spec decoding is a huge win because we're memory-bandwidth bound. At batch=32 we're more compute-bound and the win shrinks.

## What I'd do next

- Re-run with batch size 4, 8, 16 to map where spec decoding stops being worth it.
- Try Medusa-style multi-head drafting instead of a separate draft model.
- Compare against Eagle/Eagle-2.

## Files

- `run.py` — the experiment script
- `results/results.json` — raw timing data
- `results/acceptance_curve.png` — the plot above
