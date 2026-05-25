"""Experiment 05: admission policy comparison.

Submits a mixed workload to the CB scheduler under three admission policies
and measures per-request latency, segmented by short vs long requests.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path

from inferencelab.serving import ContinuousBatchingScheduler, SchedulerConfig
from inferencelab.viz import PALETTE, style

log = logging.getLogger(__name__)


class FakeForward:
    def __init__(self, per_token_seconds: float = 0.002):
        self.per_token_seconds = per_token_seconds

    def step(self, batch):
        cost = self.per_token_seconds * (len(batch) ** 0.6)
        time.sleep(cost)
        return {req.id: random.randint(0, 100_000) for req in batch}


def make_mixed_workload(n: int, seed: int = 0):
    """80% short (max_new=32), 20% long (max_new=256)."""
    rng = random.Random(seed)
    workload = []
    for i in range(n):
        if rng.random() < 0.8:
            workload.append(("short", 32))
        else:
            workload.append(("long", 256))
    return workload


async def run_policy(policy: str, workload, max_batch_size: int = 4):
    sched = ContinuousBatchingScheduler(
        SchedulerConfig(
            max_batch_size=max_batch_size,
            max_total_tokens=10_000,
            admission_policy=policy,
        ),
        forward_fn=FakeForward(0.002).step,
        eos_token_id=-1,
    )

    # Submit in random arrival order; spread arrivals over a small window
    futures = []
    metadata = []
    for kind, max_new in workload:
        req = sched.submit(prompt=kind, max_new_tokens=max_new,
                           prompt_tokens=list(range(20)))
        futures.append(req.future)
        metadata.append((kind, req))
        await asyncio.sleep(0.005)  # 200 RPS arrival rate

    run_task = asyncio.create_task(sched.run_forever())
    await asyncio.gather(*futures)
    sched.stop()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    short_latencies = []
    long_latencies = []
    for kind, req in metadata:
        latency = (req.finished_at - req.arrived_at) * 1000  # ms
        if kind == "short":
            short_latencies.append(latency)
        else:
            long_latencies.append(latency)
    return short_latencies, long_latencies


def percentile(samples, p):
    s = sorted(samples)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


async def main_async():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    workload = make_mixed_workload(60, seed=42)
    results = {}

    for policy in ["fcfs", "shortest_first", "longest_first"]:
        log.info("--- Policy: %s ---", policy)
        short, long_ = await run_policy(policy, workload)
        results[policy] = {
            "short_p50": percentile(short, 0.5),
            "short_p99": percentile(short, 0.99),
            "long_p50": percentile(long_, 0.5),
            "long_p99": percentile(long_, 0.99),
            "n_short": len(short),
            "n_long": len(long_),
        }
        log.info(
            "  short p50=%.0fms p99=%.0fms | long p50=%.0fms p99=%.0fms",
            results[policy]["short_p50"], results[policy]["short_p99"],
            results[policy]["long_p50"], results[policy]["long_p99"],
        )

    (results_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Plot
    import matplotlib.pyplot as plt
    style()
    fig, ax = plt.subplots(figsize=(9, 5))
    policies = list(results.keys())
    x = range(len(policies))
    w = 0.2
    metrics = ["short_p50", "short_p99", "long_p50", "long_p99"]
    for i, m in enumerate(metrics):
        vals = [results[p][m] for p in policies]
        ax.bar([xi + (i - 1.5) * w for xi in x], vals, w, label=m, color=PALETTE[i])
    ax.set_xticks(list(x))
    ax.set_xticklabels(policies)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Admission policies: short vs long request latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "policy_comparison.png", dpi=150, bbox_inches="tight")
    log.info("Saved results to %s", results_dir)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
