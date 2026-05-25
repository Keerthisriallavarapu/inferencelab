"""Tests for the benchmark harness."""
from __future__ import annotations

import time

from inferencelab.bench import BenchConfig, BenchResult, benchmark, compare, timer


def test_benchmark_basic_stats():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    counter = iter(samples)

    def fn():
        return next(counter)

    cfg = BenchConfig(name="t", warmup=0, repeats=5)
    result = benchmark(fn, cfg, metric="seconds")
    assert result.mean == 3.0
    assert result.median == 3.0
    assert result.percentile(0.5) == 3.0
    assert result.percentile(1.0) == 5.0
    assert result.percentile(0.0) == 1.0


def test_warmup_excluded_from_samples():
    counter = iter([100.0, 100.0, 100.0, 1.0, 2.0])  # first 3 warmups, then 2 measures

    def fn():
        return next(counter)

    cfg = BenchConfig(name="t", warmup=3, repeats=2)
    result = benchmark(fn, cfg)
    assert result.samples == [1.0, 2.0]


def test_bench_result_save_and_summary(tmp_path):
    r = BenchResult(name="x", metric="ms", samples=[1.0, 2.0, 3.0])
    out = tmp_path / "r.json"
    r.save(out)
    assert out.exists()
    text = out.read_text()
    assert "x" in text


def test_compare_renders_speedup():
    baseline = BenchResult(name="b", metric="s", samples=[2.0, 2.0, 2.0])
    fast = BenchResult(name="f", metric="s", samples=[1.0, 1.0, 1.0])
    table = compare([baseline, fast], baseline="b")
    assert "2.00x" in table


def test_timer_context_returns_elapsed():
    with timer() as elapsed:
        time.sleep(0.01)
    e = elapsed()
    assert 0.005 < e < 0.5  # generous bounds for CI variance


def test_confidence_interval_widens_with_low_n():
    high_n = BenchResult(name="x", metric="s", samples=[1.0] * 100)
    low_n = BenchResult(name="y", metric="s", samples=[1.0, 1.0])
    # Both have stdev=0 so CIs collapse — use varied samples
    varied_high = BenchResult(name="x", metric="s", samples=[i for i in range(100)])
    varied_low = BenchResult(name="y", metric="s", samples=[0.0, 99.0])
    lo_h, hi_h = varied_high.ci_95
    lo_l, hi_l = varied_low.ci_95
    assert (hi_l - lo_l) > (hi_h - lo_h)
