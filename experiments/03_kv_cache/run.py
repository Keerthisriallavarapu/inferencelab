"""Run the KV cache allocation simulation across mixed request workloads."""
from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from paged_allocator import NaiveAllocator, PagedAllocator  # noqa: E402

from inferencelab.viz import PALETTE, style  # noqa: E402

log = logging.getLogger(__name__)

BLOCK_SIZE = 16
MAX_SEQ = 4096  # worst-case seq length for naive
TOTAL_BLOCKS = 2048  # corresponds to MAX_SEQ * 8 requests worth of memory


@dataclass
class Request:
    id: str
    prompt_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


def make_workload(n: int, variable: bool, seed: int) -> list[Request]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if variable:
            # Heavy-tailed: most short, some long
            prompt = max(20, int(rng.expovariate(1 / 200)))
            output = max(20, int(rng.expovariate(1 / 100)))
        else:
            prompt = 100
            output = 100
        out.append(Request(f"r{i}", prompt, output))
    return out


def simulate(workload: list[Request], use_paged: bool) -> dict:
    if use_paged:
        alloc = PagedAllocator(block_size=BLOCK_SIZE, total_blocks=TOTAL_BLOCKS)
    else:
        alloc = NaiveAllocator(max_seq_tokens=MAX_SEQ, block_size=BLOCK_SIZE)

    completed = 0
    oom_count = 0
    peak_used = 0
    accepted: set[str] = set()

    # Admit requests in arrival order. Stop adding once full; drain.
    for req in workload:
        ok = alloc.allocate_for(req.id, req.total_tokens)
        if not ok:
            oom_count += 1
            continue
        accepted.add(req.id)

        if use_paged:
            used = alloc.used_count * BLOCK_SIZE
        else:
            used = alloc.used_tokens
        peak_used = max(peak_used, used)

    # Drain: free everything to verify allocator is clean
    for rid in list(accepted):
        alloc.free(rid)

    return {
        "accepted": len(accepted),
        "oom": oom_count,
        "peak_tokens": peak_used,
        "total_requests": len(workload),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    scenarios = [
        ("uniform_short", make_workload(32, variable=False, seed=1)),
        ("variable_mixed", make_workload(32, variable=True, seed=2)),
    ]

    results = {}
    for name, wl in scenarios:
        paged_r = simulate(wl, use_paged=True)
        naive_r = simulate(wl, use_paged=False)
        results[name] = {"paged": paged_r, "naive": naive_r}
        log.info(
            "%s: paged accepted %d (peak %d tok), naive accepted %d (peak %d tok)",
            name, paged_r["accepted"], paged_r["peak_tokens"],
            naive_r["accepted"], naive_r["peak_tokens"],
        )

    (results_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Plot: bars per scenario, side-by-side paged vs naive
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    scenario_names = [s[0] for s in scenarios]
    paged_peaks = [results[n]["paged"]["peak_tokens"] for n in scenario_names]
    naive_peaks = [results[n]["naive"]["peak_tokens"] for n in scenario_names]
    paged_accept = [results[n]["paged"]["accepted"] for n in scenario_names]
    naive_accept = [results[n]["naive"]["accepted"] for n in scenario_names]

    x = range(len(scenario_names))
    w = 0.38
    axes[0].bar([i - w / 2 for i in x], naive_peaks, w, label="Naive", color=PALETTE[1])
    axes[0].bar([i + w / 2 for i in x], paged_peaks, w, label="Paged", color=PALETTE[0])
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(scenario_names)
    axes[0].set_ylabel("Peak KV cache tokens")
    axes[0].set_title("Peak memory used")
    axes[0].legend()

    axes[1].bar([i - w / 2 for i in x], naive_accept, w, label="Naive", color=PALETTE[1])
    axes[1].bar([i + w / 2 for i in x], paged_accept, w, label="Paged", color=PALETTE[0])
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(scenario_names)
    axes[1].set_ylabel("Requests fit in budget")
    axes[1].set_title("Throughput headroom")
    axes[1].legend()

    fig.suptitle("KV cache: paged vs naive allocator")
    fig.tight_layout()
    fig.savefig(results_dir / "memory_comparison.png", dpi=150, bbox_inches="tight")
    log.info("Saved results to %s", results_dir)


if __name__ == "__main__":
    main()
