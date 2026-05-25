"""Experiment 02: continuous batching vs static batching.

Uses a fake forward function so it runs on CPU without GPUs. The shape of
the result (CB pulls ahead with prompt-length variance) is preserved even
with the fake model — what matters is the scheduling logic.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from inferencelab.serving import ContinuousBatchingScheduler, Request, SchedulerConfig
from inferencelab.viz import throughput_vs_concurrency

log = logging.getLogger(__name__)


@dataclass
class FakeForwardModel:
    """Pretends to be an LLM. Per-token cost depends on batch size in a
    realistic-ish way: amortizes over the batch (sub-linear)."""
    per_token_seconds: float = 0.005  # base cost at batch=1
    eos_token: int = -1

    def step(self, batch: list[Request]) -> dict[str, int]:
        # Simulate batched decode: cost scales sub-linearly with batch size
        cost = self.per_token_seconds * (len(batch) ** 0.6)
        time.sleep(cost)
        # Each request gets a random non-EOS token; EOS is signaled when the
        # request hits its max_new_tokens (scheduler handles that).
        return {req.id: random.randint(0, 100_000) for req in batch}


async def static_batching(
    workload: list[tuple[str, int]],
    batch_size: int,
    forward_model: FakeForwardModel,
) -> tuple[float, int]:
    """All requests in a batch run lock-step. Slowest determines duration.
    Returns (elapsed, total_tokens_generated).
    """
    total_tokens = 0
    start = time.perf_counter()

    # Process in chunks of batch_size
    for i in range(0, len(workload), batch_size):
        chunk = workload[i:i + batch_size]
        max_len = max(t for _, t in chunk)
        # Each step is one decode step at this batch size
        for _ in range(max_len):
            cost = forward_model.per_token_seconds * (len(chunk) ** 0.6)
            await asyncio.sleep(cost)
            total_tokens += len(chunk)  # we don't model EOS for static
        # In real static batching, padding would be wasted compute; we count
        # it here for fairness vs CB.

    return time.perf_counter() - start, total_tokens


async def continuous_batching(
    workload: list[tuple[str, int]],
    max_batch_size: int,
    forward_model: FakeForwardModel,
) -> tuple[float, int]:
    """Use the CB scheduler. Submit all at once, run until drained."""
    sched = ContinuousBatchingScheduler(
        SchedulerConfig(max_batch_size=max_batch_size, max_total_tokens=10_000),
        forward_fn=forward_model.step,
        eos_token_id=-1,
    )
    futures = []
    for prompt, max_new in workload:
        req = sched.submit(prompt=prompt, max_new_tokens=max_new,
                           prompt_tokens=list(range(len(prompt) // 4)))
        futures.append(req.future)

    run_task = asyncio.create_task(sched.run_forever())
    start = time.perf_counter()
    completed = await asyncio.gather(*futures)
    elapsed = time.perf_counter() - start
    sched.stop()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    total = sum(len(r.generated_tokens) for r in completed)
    return elapsed, total


def make_workload(n: int, variable: bool, seed: int = 0) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if variable:
            # Geometric-ish: most are short, some long
            max_new = min(512, int(rng.expovariate(1 / 80)) + 20)
        else:
            max_new = 128
        out.append((f"prompt_{i}", max_new))
    return out


async def main_async() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    fwd = FakeForwardModel(per_token_seconds=0.001)
    concurrencies = [1, 2, 4, 8, 16]
    n_requests = 32

    results = {
        "static_uniform": [],
        "continuous_uniform": [],
        "static_variable": [],
        "continuous_variable": [],
    }

    for c in concurrencies:
        log.info("--- Concurrency %d ---", c)
        # Uniform
        wl = make_workload(n_requests, variable=False, seed=42)
        elapsed, toks = await static_batching(wl, c, fwd)
        results["static_uniform"].append((c, toks / elapsed))
        log.info("static uniform: %.0f tok/s", toks / elapsed)

        elapsed, toks = await continuous_batching(wl, c, fwd)
        results["continuous_uniform"].append((c, toks / elapsed))
        log.info("continuous uniform: %.0f tok/s", toks / elapsed)

        # Variable
        wl = make_workload(n_requests, variable=True, seed=42)
        elapsed, toks = await static_batching(wl, c, fwd)
        results["static_variable"].append((c, toks / elapsed))
        log.info("static variable: %.0f tok/s", toks / elapsed)

        elapsed, toks = await continuous_batching(wl, c, fwd)
        results["continuous_variable"].append((c, toks / elapsed))
        log.info("continuous variable: %.0f tok/s", toks / elapsed)

    # Save raw + plot
    (results_dir / "raw.json").write_text(json.dumps(results, indent=2))

    throughput_vs_concurrency(
        results,
        title="Throughput: Static vs Continuous Batching",
        out=results_dir / "throughput_vs_concurrency.png",
    )
    log.info("Saved results to %s", results_dir)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
