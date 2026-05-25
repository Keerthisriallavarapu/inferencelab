"""Experiment 04: quantization quality/speed/memory comparison.

Runs the same model under BF16, INT8, and INT4 and reports tokens/sec,
peak memory, and accuracy on a small MMLU subset.

Requires .[gpu] extras and a GPU with ~24GB VRAM for BF16. INT4 will
fit in 8GB.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from inferencelab import BenchConfig, benchmark

log = logging.getLogger(__name__)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# Smaller fallback:
# MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# A handful of MMLU-style prompts to score. Real run would pull HF datasets.
MMLU_SAMPLES = [
    {
        "prompt": (
            "The following is a multiple choice question.\n"
            "Question: What is the chemical symbol for gold?\n"
            "A) Au B) Ag C) Gd D) Go\nAnswer:"
        ),
        "answer": "A",
    },
    {
        "prompt": (
            "Question: Which planet is known as the Red Planet?\n"
            "A) Venus B) Mars C) Jupiter D) Saturn\nAnswer:"
        ),
        "answer": "B",
    },
    {
        "prompt": (
            "Question: The derivative of x^2 with respect to x is:\n"
            "A) x B) 2x C) x^2 D) 2\nAnswer:"
        ),
        "answer": "B",
    },
    # In a real run, load 50-500 questions from cais/mmlu via datasets.
]

GEN_PROMPTS = [
    "Write a one-paragraph summary of the French Revolution:",
    "Explain how a transformer model works to an engineer:",
    "Write a Python function to compute Fibonacci numbers:",
]

FORMATS = [
    ("bf16", {"dtype": "bfloat16", "quantization": None}),
    ("int8", {"dtype": "bfloat16", "quantization": "int8"}),
    ("int4", {"dtype": "bfloat16", "quantization": "int4"}),
]


def grade(prediction: str, expected: str) -> bool:
    """Loose grader: check if the first non-space character matches."""
    stripped = prediction.strip()
    return bool(stripped) and stripped[0].upper() == expected


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    try:
        import torch
        from inferencelab.models import HFModelWrapper
    except ImportError as e:
        print(f"GPU deps not installed: {e}")
        return

    all_results = {}

    for fmt_name, kwargs in FORMATS:
        log.info("=== Format: %s ===", fmt_name)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        wrapper = HFModelWrapper(MODEL_ID, **kwargs)
        mem_after_load = torch.cuda.max_memory_allocated() / 1e9

        # Throughput
        def run():
            total = 0
            from inferencelab import timer
            with timer() as elapsed:
                for p in GEN_PROMPTS:
                    r = wrapper.generate(p, max_new_tokens=128)
                    total += r.output_tokens
            return total / elapsed()

        speed = benchmark(
            run,
            BenchConfig(name=f"throughput_{fmt_name}", warmup=1, repeats=3),
            metric="tokens_per_second",
            metadata={"format": fmt_name},
        )
        speed.save(results_dir / f"throughput_{fmt_name}.json")

        # Quality: MMLU subset
        correct = 0
        for sample in MMLU_SAMPLES:
            r = wrapper.generate(sample["prompt"], max_new_tokens=3)
            if grade(r.text, sample["answer"]):
                correct += 1
        accuracy = correct / max(len(MMLU_SAMPLES), 1)

        all_results[fmt_name] = {
            "memory_gb": mem_after_load,
            "tokens_per_sec_mean": speed.mean,
            "tokens_per_sec_p99": speed.percentile(0.99),
            "mmlu_subset_accuracy": accuracy,
        }
        log.info("%s: mem=%.1fGB  speed=%.1f tok/s  acc=%.3f",
                 fmt_name, mem_after_load, speed.mean, accuracy)

        del wrapper
        torch.cuda.empty_cache()

    (results_dir / "summary.json").write_text(json.dumps(all_results, indent=2))
    log.info("Done. Results in %s", results_dir)


if __name__ == "__main__":
    main()
