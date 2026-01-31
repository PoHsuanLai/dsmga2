# DSMGA-II

High-performance Rust implementation of DSMGA-II with automatic linkage learning for discrete optimization.

## Quick Start

```rust
use dsmga2::{Dsmga2, fitness::OneMax};

let fitness_fn = OneMax;
let mut ga = Dsmga2::new(100, &fitness_fn)
    .population_size(200)
    .max_generations(1000)
    .build();

ga.run();
println!("Best fitness: {}", ga.best_fitness());
```

## Features

- Automatic linkage learning with two-edge model
- Parallel fitness evaluation and linkage building
- Optional greedy hill climbing
- Checkpoint save/resume support
- Iterator and callback interfaces

## Usage

### Configuration

```rust
let mut ga = Dsmga2::new(problem_size, &fitness_fn)
    .population_size(200)
    .max_generations(1000)
    .use_ghc(true)
    .seed(42)
    .build();
```

### Iteration

```rust
for state in ga.take(100) {
    println!("Gen {}: {:.2}", state.generation, state.best_fitness);
}
```

### Callbacks

```rust
ga.run_with(|state| {
    println!("Generation {}: {:.2}", state.generation, state.best_fitness);
});
```

### Checkpoints

```rust
ga.save_checkpoint("checkpoint.bin")?;
let checkpoint = Dsmga2::load_checkpoint("checkpoint.bin")?;
let mut ga = Dsmga2::from_checkpoint(checkpoint, &fitness_fn);
```

## Custom Fitness Functions

```rust
use dsmga2::FitnessFunction;

struct MyProblem;

impl FitnessFunction for MyProblem {
    fn evaluate(&self, genes: &[bool]) -> f64 {
        genes.iter().filter(|&&g| g).count() as f64
    }

    fn optimum(&self, problem_size: usize) -> f64 {
        problem_size as f64
    }
}
```

## Architecture

```
src/dsmga2/
├── mod.rs          # Main algorithm
├── config.rs       # Configuration
├── population.rs   # Population & GHC
├── linkage.rs      # Linkage learning
├── selection.rs    # Tournament selection
├── mixing.rs       # Restricted mixing
└── convergence.rs  # Convergence detection
```

## Examples & Tools

Run examples:
```bash
cargo run --example basic_usage
cargo run --example custom_fitness
```

CLI tools:
```bash
cargo run --bin dsmga2_cli -- --help
cargo run --bin dsmga2_sweep -- --help
```

## Testing

```bash
cargo test
cargo bench
```
