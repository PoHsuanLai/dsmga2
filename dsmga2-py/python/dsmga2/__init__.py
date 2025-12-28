"""
DSMGA2 - Dependency Structure Matrix Genetic Algorithm II

A fast genetic algorithm with automatic linkage learning using a two-edge
graphical model. Efficiently solves optimization problems by discovering
and exploiting problem structure.

Example:
    >>> from dsmga2 import Dsmga2, OneMax
    >>>
    >>> # Create optimizer for 100-bit OneMax problem
    >>> optimizer = Dsmga2(problem_size=100, fitness_function=OneMax())
    >>> optimizer.population_size = 200
    >>> optimizer.max_generations = 1000
    >>> optimizer.seed = 42
    >>>
    >>> # Run optimization
    >>> result = optimizer.run()
    >>> print(f"Best fitness: {result.best_fitness}")
    >>> print(f"Generations: {result.generation}")
"""

from ._dsmga2 import (
    Dsmga2,
    OneMax,
    MkTrap,
    OptimizationResult,
)

__version__ = "0.1.0"

__all__ = [
    "Dsmga2",
    "OneMax",
    "MkTrap",
    "OptimizationResult",
]
