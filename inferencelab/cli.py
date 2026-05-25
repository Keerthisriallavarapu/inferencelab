"""CLI for running experiments. Each experiment is a script under experiments/
that exports a `main()` function. The CLI just dispatches by name."""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

EXPERIMENTS_ROOT = Path(__file__).parent.parent / "experiments"


@app.command()
def list_experiments():
    """List available experiments."""
    for d in sorted(EXPERIMENTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "run.py").exists():
            continue
        readme = d / "README.md"
        first_line = "(no README)"
        if readme.exists():
            for ln in readme.read_text().splitlines():
                if ln.strip().startswith("#"):
                    first_line = ln.lstrip("# ").strip()
                    break
        console.print(f"[bold]{d.name}[/bold] — {first_line}")


@app.command()
def run(
    experiment: str = typer.Argument(..., help="e.g. '01_speculative_decoding'"),
    verbose: bool = typer.Option(False, "-v"),
):
    """Run a single experiment by name."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s :: %(message)s",
    )
    exp_dir = EXPERIMENTS_ROOT / experiment
    if not exp_dir.exists():
        console.print(f"[red]Experiment not found:[/red] {experiment}")
        sys.exit(1)

    # Add to path and import
    sys.path.insert(0, str(exp_dir))
    try:
        mod = importlib.import_module("run")
        if not hasattr(mod, "main"):
            console.print(f"[red]Experiment {experiment} has no main()[/red]")
            sys.exit(1)
        mod.main()
    finally:
        sys.path.pop(0)


if __name__ == "__main__":
    app()
