"""Tests for the continuous-batching scheduler."""
from __future__ import annotations

import asyncio

import pytest

from inferencelab.serving import ContinuousBatchingScheduler, SchedulerConfig


def _fake_forward_factory(token_id: int = 1, eos_after: int | None = None):
    """Returns a forward fn that always emits `token_id`, except after
    `eos_after` calls per request when it emits the EOS token."""
    call_counts: dict[str, int] = {}

    def forward(batch):
        out = {}
        for req in batch:
            call_counts[req.id] = call_counts.get(req.id, 0) + 1
            if eos_after is not None and call_counts[req.id] > eos_after:
                out[req.id] = -1  # EOS
            else:
                out[req.id] = token_id
        return out

    return forward


async def test_request_finishes_at_max_new_tokens():
    sched = ContinuousBatchingScheduler(
        SchedulerConfig(max_batch_size=4, max_total_tokens=1000),
        forward_fn=_fake_forward_factory(),
        eos_token_id=-1,
    )
    req = sched.submit(prompt="hi", max_new_tokens=5, prompt_tokens=[1, 2, 3])

    task = asyncio.create_task(sched.run_forever())
    finished = await asyncio.wait_for(req.future, timeout=2.0)
    sched.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(finished.generated_tokens) == 5


async def test_request_finishes_at_eos():
    sched = ContinuousBatchingScheduler(
        SchedulerConfig(max_batch_size=4, max_total_tokens=1000),
        forward_fn=_fake_forward_factory(eos_after=3),
        eos_token_id=-1,
    )
    req = sched.submit(prompt="hi", max_new_tokens=100, prompt_tokens=[1, 2])

    task = asyncio.create_task(sched.run_forever())
    finished = await asyncio.wait_for(req.future, timeout=2.0)
    sched.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # 3 normal tokens + 1 EOS = 4 generated
    assert len(finished.generated_tokens) == 4
    assert finished.generated_tokens[-1] == -1


async def test_admission_respects_batch_size():
    """If max_batch_size=2 and we submit 5 requests, at most 2 should be
    decoding simultaneously."""
    max_concurrent_seen = [0]
    real_batch_sizes = []

    def forward(batch):
        real_batch_sizes.append(len(batch))
        max_concurrent_seen[0] = max(max_concurrent_seen[0], len(batch))
        return {req.id: 1 for req in batch}

    sched = ContinuousBatchingScheduler(
        SchedulerConfig(max_batch_size=2, max_total_tokens=1000),
        forward_fn=forward,
        eos_token_id=-1,
    )
    futures = [
        sched.submit(prompt=f"p{i}", max_new_tokens=3, prompt_tokens=[1, 2]).future
        for i in range(5)
    ]
    task = asyncio.create_task(sched.run_forever())
    await asyncio.wait_for(asyncio.gather(*futures), timeout=2.0)
    sched.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert max_concurrent_seen[0] <= 2
