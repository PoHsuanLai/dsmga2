//! DSMGA-II: Dependency Structure Matrix Genetic Algorithm II
//!
//! This crate implements DSMGA-II with a two-edge graphical linkage model for solving
//! discrete optimization problems, particularly binary optimization problems.
//!
//! # Quick Start
//!
//! ```rust
//! use dsmga2::{Dsmga2, fitness::OneMax};
//!
//! let fitness_fn = OneMax;
//! let mut ga = Dsmga2::new(100, &fitness_fn)
//!     .population_size(200)
//!     .max_generations(1000)
//!     .seed(42)
//!     .build();
//!
//! let solution = ga.run();
//! println!("Best fitness: {}", ga.best_fitness());
//! ```
//!
//! # Iterator-based Usage
//!
//! ```rust
//! use dsmga2::{Dsmga2, fitness::OneMax};
//!
//! let fitness_fn = OneMax;
//! let mut ga = Dsmga2::new(100, &fitness_fn).build();
//!
//! for state in ga.take(100) {
//!     if state.generation % 10 == 0 {
//!         println!("Gen {}: fitness = {:.2}", state.generation, state.best_fitness);
//!     }
//! }
//! ```
//!
//! # With Callbacks
//!
//! ```rust
//! use dsmga2::{Dsmga2, fitness::OneMax};
//!
//! let fitness_fn = OneMax;
//! let mut ga = Dsmga2::new(100, &fitness_fn).build();
//!
//! ga.run_with(|state| {
//!     println!("Generation {}: {:.2}", state.generation, state.best_fitness);
//! });
//! ```

mod api;
pub mod checkpoint;
pub mod chromosome;
#[doc(hidden)]
pub mod dsmga2; // Public for testing, but hidden from docs
pub mod fitness;
pub mod structures;
pub mod utils;

pub use api::{Dsmga2, Dsmga2Builder, State};
pub use checkpoint::Checkpoint;
pub use chromosome::Chromosome;
pub use fitness::FitnessFunction;
