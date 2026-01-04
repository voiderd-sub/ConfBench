"""
ConfBench - Conformational Change Benchmark for Structure Prediction

A benchmark for evaluating how well structure prediction models can predict
holo structures from apo structures.
"""

from .evaluator import Evaluator
from .benchmark import ConfBenchmark

__version__ = "0.1.0"

__all__ = [
    "Evaluator",
    "ConfBenchmark",
]
