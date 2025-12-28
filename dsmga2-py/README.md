# DSMGA2 Python Bindings

Python bindings for DSMGA2 (Dependency Structure Matrix Genetic Algorithm II), a fast genetic algorithm with automatic linkage learning.

## Installation

```bash
pip install dsmga2
```

Or build from source:

```bash
pip install maturin
maturin develop --release
```

## Quick Start

```python
from dsmga2 import Dsmga2, OneMax

# Create optimizer for 100-bit OneMax problem
optimizer = Dsmga2(problem_size=100, fitness_function=OneMax())
optimizer.population_size = 200
optimizer.max_generations = 1000
optimizer.seed = 42

# Run optimization
result = optimizer.run()
print(f"Best fitness: {result.best_fitness}")
print(f"Generations: {result.generation}")
print(f"Evaluations: {result.num_evaluations}")
```

## Built-in Fitness Functions

- `OneMax()`: Counts the number of 1s in the chromosome
- `MkTrap(k=5)`: m-k Trap function with deceptive blocks

## Custom Fitness Functions

You can implement custom fitness functions in Python:

```python
class CustomFitness:
    def evaluate(self, genes):
        """
        Evaluate fitness of a chromosome.

        Args:
            genes: List of booleans representing the chromosome

        Returns:
            float: Fitness value (higher is better)
        """
        # Your fitness evaluation logic here
        return sum(genes)

    def optimum(self, length):
        """Return the optimal fitness value for this problem size."""
        return float(length)

# Use custom fitness
optimizer = Dsmga2(problem_size=100, fitness_function=CustomFitness())
result = optimizer.run()
```

## API Reference

### Dsmga2

Main optimizer class.

**Constructor:**
```python
Dsmga2(problem_size: int, fitness_function: FitnessFunction)
```

**Attributes:**
- `population_size: int` - Population size (default: problem_size)
- `max_generations: int` - Maximum generations (default: -1, unlimited)
- `seed: int` - Random seed (default: 42)

**Methods:**
- `run() -> OptimizationResult` - Run optimization until convergence or max generations
- `step() -> bool` - Run one generation, returns True if not converged
- `best_fitness() -> float` - Get current best fitness
- `generation() -> int` - Get current generation number

### OptimizationResult

Result object returned by `run()`.

**Attributes:**
- `best_fitness: float` - Best fitness found
- `generation: int` - Number of generations run
- `num_evaluations: int` - Total fitness evaluations
- `mean_fitness: float` - Mean population fitness
- `converged: bool` - Whether algorithm converged

## License

MIT OR Apache-2.0
