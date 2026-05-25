"""InferenceLab — LLM inference optimization experiments."""
__version__ = "0.1.0"

from .bench import BenchConfig, BenchResult, benchmark, compare, timer

__all__ = ["BenchConfig", "BenchResult", "benchmark", "compare", "timer"]
