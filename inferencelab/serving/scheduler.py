"""A minimal continuous-batching scheduler.

Continuous batching (sometimes called "in-flight batching") is the dominant
serving optimization for LLMs. Static batching forces every request in a
batch to wait for the slowest to finish; CB swaps finished requests out
mid-batch and slots new ones in.

This implementation is simplified vs. vLLM/TGI:
- No PagedAttention (we use the model's native KV cache).
- No prefix caching, no chunked prefill.
- Single-GPU, single-process.

But the scheduling loop is the same shape: a request queue, an active
batch, a forward pass per decode step, and policy decisions about when to
admit new requests vs. continue decoding.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class RequestState(str, Enum):
    QUEUED = "queued"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class Request:
    id: str
    prompt: str
    max_new_tokens: int
    arrived_at: float
    state: RequestState = RequestState.QUEUED
    prompt_tokens: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    started_at: float | None = None
    finished_at: float | None = None
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())

    @property
    def is_done(self) -> bool:
        return self.state in (RequestState.FINISHED, RequestState.CANCELLED)


@dataclass
class SchedulerConfig:
    max_batch_size: int = 16
    max_total_tokens: int = 4096  # sum of (prompt + generated) across active batch
    admission_policy: str = "fcfs"  # "fcfs" | "longest_first" | "shortest_first"
    prefill_chunk_size: int | None = None  # if set, do chunked prefill


class ContinuousBatchingScheduler:
    """Simple continuous-batching scheduler.

    Unit-testable: the actual model forward pass is delegated to an injected
    callable. Tests use a fake model to verify scheduling logic without GPUs.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        forward_fn,  # callable: (list[Request]) -> dict[req_id, next_token]
        eos_token_id: int = -1,
    ):
        self._config = config
        self._forward = forward_fn
        self._eos = eos_token_id
        self._queue: list[Request] = []
        self._active: list[Request] = []
        self._running = False

    def submit(self, prompt: str, max_new_tokens: int, prompt_tokens: list[int]) -> Request:
        req = Request(
            id=f"req_{uuid.uuid4().hex[:8]}",
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            arrived_at=time.time(),
            prompt_tokens=prompt_tokens,
        )
        self._queue.append(req)
        return req

    async def run_forever(self) -> None:
        """The decode loop. Each iteration: admit new requests if capacity
        allows, then one decode step over the active batch."""
        self._running = True
        while self._running:
            self._admit()
            if not self._active:
                await asyncio.sleep(0.001)  # idle backoff
                continue

            # One decode step
            next_tokens = self._forward(self._active)
            now = time.time()

            for req in list(self._active):
                tok = next_tokens.get(req.id)
                if tok is None:
                    continue
                req.generated_tokens.append(tok)

                if (
                    tok == self._eos
                    or len(req.generated_tokens) >= req.max_new_tokens
                ):
                    req.state = RequestState.FINISHED
                    req.finished_at = now
                    self._active.remove(req)
                    if not req.future.done():
                        req.future.set_result(req)

            await asyncio.sleep(0)  # yield to other tasks

    def stop(self) -> None:
        self._running = False

    def _admit(self) -> None:
        """Move requests from queue to active batch if capacity allows."""
        if not self._queue:
            return

        if self._config.admission_policy == "longest_first":
            self._queue.sort(key=lambda r: -len(r.prompt_tokens))
        elif self._config.admission_policy == "shortest_first":
            self._queue.sort(key=lambda r: len(r.prompt_tokens))
        # else FCFS (insertion order)

        used_tokens = sum(len(r.prompt_tokens) + len(r.generated_tokens) for r in self._active)

        while (
            self._queue
            and len(self._active) < self._config.max_batch_size
        ):
            head = self._queue[0]
            tokens_needed = len(head.prompt_tokens) + head.max_new_tokens
            if used_tokens + tokens_needed > self._config.max_total_tokens:
                break  # admitting this request would blow the token budget

            head = self._queue.pop(0)
            head.state = RequestState.DECODING
            head.started_at = time.time()
            self._active.append(head)
            used_tokens += tokens_needed
            log.debug("Admitted %s (active=%d, used_tokens=%d)",
                      head.id, len(self._active), used_tokens)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "queue_depth": len(self._queue),
            "active_batch": len(self._active),
        }
