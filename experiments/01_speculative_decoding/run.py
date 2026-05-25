"""Experiment 01: speculative decoding acceptance and speedup vs k.

Designed to run on a single GPU with ~24GB. Models can be swapped via
the constants below. For machines without GPUs, see the README — the
results in there are from a real run; this script just lets you reproduce.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from inferencelab import BenchConfig, benchmark, compare
from inferencelab.viz import acceptance_curve

log = logging.getLogger(__name__)

# Models. Smaller versions if you don't have 24GB.
TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Toy fallbacks if the above gate-protected. Comment out the above and
# uncomment these to run end-to-end on a smaller box:
# TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

K_VALUES = [1, 2, 4, 6, 8, 16]
MAX_NEW_TOKENS = 256

# A few short coding-style prompts. Real run would use HumanEval; this
# subset keeps the experiment runnable in <10 minutes.
PROMPTS = [
    "def fibonacci(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is prime.\"\"\"\n",
    "def quicksort(arr):\n    \"\"\"In-place quicksort.\"\"\"\n",
    "def binary_search(arr, target):\n    \"\"\"Return the index of target in arr or -1.\"\"\"\n",
    "def longest_palindrome(s: str) -> str:\n    \"\"\"Find the longest palindromic substring.\"\"\"\n",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Imports are deferred so users without GPU can still read the experiment
    try:
        from inferencelab.models import HFModelWrapper
        from inferencelab.serving import speculative_decode
    except ImportError as e:
        print(f"GPU deps not installed: {e}")
        print("Install with: pip install -e '.[gpu]'")
        return

    log.info("Loading target: %s", TARGET_MODEL)
    target = HFModelWrapper(TARGET_MODEL, dtype="bfloat16")
    log.info("Loading draft: %s", DRAFT_MODEL)
    draft = HFModelWrapper(DRAFT_MODEL, dtype="bfloat16")

    # 1) Baseline: target model without speculation
    log.info("--- Baseline (no spec) ---")
    def run_baseline():
        total_tokens = 0
        from inferencelab import timer
        with timer() as elapsed:
            for p in PROMPTS:
                r = target.generate(p, max_new_tokens=MAX_NEW_TOKENS)
                total_tokens += r.output_tokens
        return total_tokens / elapsed()  # tokens/sec

    baseline = benchmark(
        run_baseline,
        BenchConfig(name="baseline_no_spec", warmup=1, repeats=3),
        metric="tokens_per_second",
        metadata={"target": TARGET_MODEL, "max_new_tokens": MAX_NEW_TOKENS},
    )
    baseline.save(results_dir / "baseline.json")

    all_results = [baseline]
    acceptance_data: dict[int, float] = {}
    throughput_data: dict[int, float] = {}

    # 2) Speculative decode at various k
    for k in K_VALUES:
        if k == 1:
            continue  # k=1 is essentially baseline
        log.info("--- Spec decode k=%d ---", k)

        # Use a holder so we can report acceptance after the bench
        acc_total = [0]
        prop_total = [0]

        def run_spec(_k=k):
            local_acc, local_prop = 0, 0
            from inferencelab import timer
            tok_count = 0
            with timer() as elapsed:
                for p in PROMPTS:
                    r = speculative_decode(
                        target.model, draft.model, target.tokenizer,
                        prompt=p, max_new_tokens=MAX_NEW_TOKENS, k=_k,
                    )
                    tok_count += len(r.tokens)
                    local_acc += r.accepted_total
                    local_prop += r.proposed_total
            acc_total[0] += local_acc
            prop_total[0] += local_prop
            return tok_count / elapsed()

        result = benchmark(
            run_spec,
            BenchConfig(name=f"spec_k{k}", warmup=1, repeats=3),
            metric="tokens_per_second",
            metadata={
                "k": k,
                "target": TARGET_MODEL,
                "draft": DRAFT_MODEL,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
        )
        result.save(results_dir / f"spec_k{k}.json")
        all_results.append(result)

        rate = acc_total[0] / max(prop_total[0], 1)
        acceptance_data[k] = rate
        throughput_data[k] = result.mean

    # 3) Summarize
    print()
    print(compare(all_results, baseline="baseline_no_spec"))
    print()

    # 4) Plot
    ks = sorted(acceptance_data.keys())
    rates = [acceptance_data[k] for k in ks]
    acceptance_curve(
        ks, rates,
        title="Speculative decoding: acceptance rate vs k",
        theoretical_alpha=0.85,
        out=results_dir / "acceptance_curve.png",
    )

    # Save summary
    (results_dir / "summary.json").write_text(json.dumps({
        "target_model": TARGET_MODEL,
        "draft_model": DRAFT_MODEL,
        "baseline_tps": baseline.mean,
        "by_k": {
            str(k): {
                "tokens_per_sec": throughput_data[k],
                "acceptance_rate": acceptance_data[k],
                "speedup": throughput_data[k] / baseline.mean,
            }
            for k in ks
        },
    }, indent=2))
    log.info("Saved results to %s", results_dir)


if __name__ == "__main__":
    main()
