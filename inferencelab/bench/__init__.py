"""Benchmark harness.

What makes a benchmark trustworthy:
1. Warmup runs (first few are slower due to cache, JIT, allocation).
2. Multiple repeats — single runs are noisy.
3. Report percentiles, not just means. p50, p99, p99.9 tell different stories.
4. Confidence intervals. If repeats are too few, say so.
5. Pin the random seed and (where possible) the device.
6. Record environment metadata: hardware, library versions, batch size, etc.

Reproducibility lives or dies in those details. We keep them in the BenchResult.
"""
from __future__ import annotations

import gc
import json
import logging
import platform
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

log = logging.getLogger(__name__)


@dataclass
class BenchResult:
    """A single benchmark's results.

    Saved to JSON so they're diffable and version-controllable.
    """
    name: str
    metric: str  # "tokens_per_second" | "latency_ms" | etc
    samples: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    warmup_samples: int = 0

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    def percentile(self, p: float) -> float:
        """p as fraction (0.99 -> p99). Linear interpolation between samples."""
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        k = (len(s) - 1) * p
        f = int(k)
        c = min(f + 1, len(s) - 1)
        if f == c:
            return s[f]
        return s[f] + (s[c] - s[f]) * (k - f)

    @property
    def ci_95(self) -> tuple[float, float]:
        """95% confidence interval for the mean, normal approximation."""
        if len(self.samples) < 2:
            return (self.mean, self.mean)
        margin = 1.96 * self.stdev / (len(self.samples) ** 0.5)
        return (self.mean - margin, self.mean + margin)

    def summary(self) -> dict[str, Any]:
        lo, hi = self.ci_95
        return {
            "name": self.name,
            "metric": self.metric,
            "n_samples": len(self.samples),
            "mean": self.mean,
            "median": self.median,
            "stdev": self.stdev,
            "p50": self.percentile(0.5),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
            "p99_9": self.percentile(0.999),
            "ci_95_low": lo,
            "ci_95_high": hi,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            **self.summary(),
            "samples": self.samples,
        }, indent=2, default=str))


@dataclass
class BenchConfig:
    name: str
    warmup: int = 3
    repeats: int = 10
    cooldown_seconds: float = 0.0
    gc_between_runs: bool = True


def _env_metadata() -> dict[str, Any]:
    md: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    try:
        import torch
        md["torch"] = torch.__version__
        if torch.cuda.is_available():
            md["cuda"] = torch.version.cuda
            md["gpu"] = torch.cuda.get_device_name(0)
            md["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except ImportError:
        pass
    try:
        import transformers
        md["transformers"] = transformers.__version__
    except ImportError:
        pass
    return md


def benchmark(
    fn: Callable[[], float],
    config: BenchConfig,
    metric: str = "seconds",
    metadata: dict[str, Any] | None = None,
) -> BenchResult:
    """Run `fn` warmup+repeats times. fn returns the measured value (typically
    elapsed seconds, but can be tokens/sec or whatever you're measuring).
    """
    samples: list[float] = []

    log.info("[bench] %s: warmup x%d", config.name, config.warmup)
    for _ in range(config.warmup):
        try:
            fn()
        except Exception as e:
            log.warning("warmup run raised: %s", e)
        if config.gc_between_runs:
            gc.collect()

    log.info("[bench] %s: measuring x%d", config.name, config.repeats)
    for i in range(config.repeats):
        if config.cooldown_seconds:
            time.sleep(config.cooldown_seconds)
        if config.gc_between_runs:
            gc.collect()
        value = fn()
        samples.append(float(value))
        log.debug("  run %d/%d -> %.4f", i + 1, config.repeats, value)

    md = {**_env_metadata(), **(metadata or {})}
    result = BenchResult(
        name=config.name,
        metric=metric,
        samples=samples,
        metadata=md,
        warmup_samples=config.warmup,
    )
    log.info(
        "[bench] %s: mean=%.4f median=%.4f p99=%.4f n=%d",
        config.name, result.mean, result.median, result.percentile(0.99), len(samples),
    )
    return result


@contextmanager
def timer() -> Iterator[Callable[[], float]]:
    """Context manager yielding a function that returns elapsed seconds.

    Usage:
        with timer() as elapsed:
            do_work()
        return elapsed()
    """
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def compare(results: list[BenchResult], baseline: str | None = None) -> str:
    """Render a comparison table of BenchResults.

    If baseline is given, show speedup vs that result.
    """
    lines = []
    header = f"{'name':<40} {'p50':>10} {'p99':>10} {'mean':>10} {'speedup':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    base_mean = None
    if baseline:
        for r in results:
            if r.name == baseline:
                base_mean = r.mean
                break
    for r in results:
        speedup = (base_mean / r.mean) if (base_mean and r.mean) else 1.0
        speedup_str = f"{speedup:.2f}x" if base_mean else "—"
        lines.append(
            f"{r.name:<40} {r.percentile(0.5):>10.4f} {r.percentile(0.99):>10.4f} "
            f"{r.mean:>10.4f} {speedup_str:>10}"
        )
    return "\n".join(lines)
