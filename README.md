# DSMGA2

[![Crates.io](https://img.shields.io/crates/v/dsmga2.svg)](https://crates.io/crates/dsmga2)
[![Documentation](https://docs.rs/dsmga2/badge.svg)](https://docs.rs/dsmga2)
[![PyPI](https://img.shields.io/pypi/v/dsmga2.svg)](https://pypi.org/project/dsmga2/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/PoHsuanLai/dsmga2#license)

A genetic algorithm that learns problem structure. **75-91% fewer evaluations** than standard GAs on hard problems.

## Installation

```bash
pip install dsmga2
```

## Quick Start

```python
import dsmga2

# Built-in problem
ga = dsmga2.Dsmga2(100, dsmga2.OneMax())
result = ga.run()
print(f"Fitness: {result.best_fitness}")

# Custom problem (NumPy compatible)
class MyProblem:
    def evaluate(self, solution):
        return solution.sum()

    def optimum(self, length):
        return length

ga = dsmga2.Dsmga2(100, MyProblem())
result = ga.run()
```

## Why DSMGA2?

DSMGA2 automatically discovers which variables interact, allowing it to solve problems more efficiently than standard genetic algorithms.

**Benchmark: MAX-SAT (NP-complete problem)**

![MAX-SAT Benchmark](benchmarks/maxsat_combined.png)

| Problem Size | DSMGA2 | PyGAD | DEAP |
| ------------ | ------ | ----- | ---- |
| 20 variables | 17K    | 202K  | 201K |
| 30 variables | 127K   | 500K  | 500K |
| 40 variables | 50K    | 459K  | 500K |

*Values = function evaluations to solution (lower is better)*

> This compares DSMGA2 against popular Python GA libraries. For academic comparisons with linkage-learning algorithms (LT-GOMEA, hBOA), see the [original paper](https://doi.org/10.1145/3071178.3071236).

## Rust Usage

```rust
use dsmga2::{Dsmga2, fitness::OneMax};

let mut ga = Dsmga2::new(100, &OneMax)
    .population_size(200)
    .build();

ga.run();
println!("Best: {}", ga.best_fitness());
```

## License

MIT OR Apache-2.0
