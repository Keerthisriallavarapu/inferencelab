"""Plotting helpers. Each experiment uses these to render its results.

We keep plot styling consistent across experiments so the writeups look like
they're from the same lab notebook, not five different ones.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Color palette - colorblind-safe, screenshots cleanly in light and dark modes
PALETTE = [
    "#0173B2",  # blue
    "#DE8F05",  # orange
    "#029E73",  # green
    "#CC78BC",  # purple
    "#D55E00",  # red
    "#56B4E9",  # light blue
]


def style() -> None:
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 110,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
    })


def latency_distribution(
    samples_by_method: dict[str, list[float]],
    title: str,
    out: str | Path | None = None,
    unit_label: str = "Latency (ms)",
) -> plt.Figure:
    """Box + scatter plot comparing latency distributions across methods."""
    style()
    fig, ax = plt.subplots()
    methods = list(samples_by_method.keys())
    data = [samples_by_method[m] for m in methods]

    bp = ax.boxplot(data, labels=methods, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], PALETTE, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay jittered scatter
    for i, samples in enumerate(data, start=1):
        jitter = np.random.normal(0, 0.04, size=len(samples))
        ax.scatter([i + j for j in jitter], samples, s=8, alpha=0.4, color="#333")

    ax.set_ylabel(unit_label)
    ax.set_title(title)
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def throughput_vs_concurrency(
    data: dict[str, list[tuple[int, float]]],
    title: str,
    out: str | Path | None = None,
) -> plt.Figure:
    """x: concurrency, y: throughput. Multiple lines per method."""
    style()
    fig, ax = plt.subplots()
    for (label, points), color in zip(data.items(), PALETTE, strict=False):
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, "-o", label=label, color=color, linewidth=2, markersize=6)
    ax.set_xlabel("Concurrency (requests in flight)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def acceptance_curve(
    k_values: list[int],
    acceptance_rates: list[float],
    title: str,
    theoretical_alpha: float | None = None,
    out: str | Path | None = None,
) -> plt.Figure:
    """Speculative-decoding acceptance rate vs k.

    If theoretical_alpha given, overlay the geometric decay alpha^i for
    comparison."""
    style()
    fig, ax = plt.subplots()
    ax.plot(k_values, acceptance_rates, "-o", color=PALETTE[0], linewidth=2,
            markersize=8, label="Measured")
    if theoretical_alpha is not None:
        theoretical = [theoretical_alpha ** k for k in k_values]
        ax.plot(k_values, theoretical, "--", color=PALETTE[1], linewidth=2,
                label=f"Geometric (α={theoretical_alpha:.2f})")
        ax.legend()
    ax.set_xlabel("k (draft tokens per round)")
    ax.set_ylabel("Acceptance rate")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig
