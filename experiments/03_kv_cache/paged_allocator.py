"""Minimal block-based KV cache allocator simulating PagedAttention.

Each request's KV cache is split into fixed-size blocks (default 16 tokens).
Blocks are allocated from a global pool on demand. When a request finishes,
its blocks return to the pool.

This is the allocator, not the attention kernel. The kernel changes are
substantial (you need to scatter-gather across non-contiguous blocks);
but the memory savings come from the allocator alone, which is what we
measure here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class PagedAllocator:
    block_size: int  # tokens per block
    total_blocks: int

    _free_blocks: list[int] = field(default_factory=list)
    _allocated: dict[str, list[int]] = field(default_factory=dict)  # req_id -> block IDs

    def __post_init__(self):
        self._free_blocks = list(range(self.total_blocks))

    @property
    def free_count(self) -> int:
        return len(self._free_blocks)

    @property
    def used_count(self) -> int:
        return self.total_blocks - self.free_count

    def allocate_for(self, req_id: str, n_tokens: int) -> bool:
        """Reserve blocks for n_tokens. Returns True if successful, False if OOM."""
        n_blocks_needed = (n_tokens + self.block_size - 1) // self.block_size
        already_allocated = len(self._allocated.get(req_id, []))
        delta = n_blocks_needed - already_allocated
        if delta <= 0:
            return True
        if delta > self.free_count:
            return False
        new_blocks = [self._free_blocks.pop() for _ in range(delta)]
        self._allocated.setdefault(req_id, []).extend(new_blocks)
        return True

    def free(self, req_id: str) -> None:
        blocks = self._allocated.pop(req_id, [])
        self._free_blocks.extend(blocks)


@dataclass
class NaiveAllocator:
    """Allocates `max_seq * block_size` for each request upfront. This models
    a naive implementation that doesn't know how long the request will be
    and pre-allocates to the worst case."""
    max_seq_tokens: int
    block_size: int  # included for parity; unused in calc

    _allocated: dict[str, int] = field(default_factory=dict)
    _total_allocated_tokens: int = 0
    peak_allocated_tokens: int = 0

    def allocate_for(self, req_id: str, n_tokens: int) -> bool:
        if req_id not in self._allocated:
            self._allocated[req_id] = self.max_seq_tokens
            self._total_allocated_tokens += self.max_seq_tokens
            self.peak_allocated_tokens = max(
                self.peak_allocated_tokens, self._total_allocated_tokens
            )
        return True

    def free(self, req_id: str) -> None:
        tokens = self._allocated.pop(req_id, 0)
        self._total_allocated_tokens -= tokens

    @property
    def used_tokens(self) -> int:
        return self._total_allocated_tokens
