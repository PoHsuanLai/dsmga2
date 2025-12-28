mod onemax;
mod trap;

pub use onemax::OneMax;
pub use trap::{CyclicTrap, FoldedTrap, MkTrap};

use crate::chromosome::Chromosome;

/// Trait for fitness functions
/// Must be Sync for parallel evaluation
pub trait FitnessFunction: Sync {
    /// Evaluate the fitness of a chromosome
    fn evaluate(&self, chromosome: &Chromosome) -> f64;

    /// Get the optimal fitness value for a given problem size
    fn optimum(&self, length: usize) -> f64;
}

/// Custom fitness function using a closure
pub struct CustomFitness<F>
where
    F: Fn(&Chromosome) -> f64,
{
    function: F,
    optimum: f64,
}

impl<F> CustomFitness<F>
where
    F: Fn(&Chromosome) -> f64,
{
    pub fn new(function: F, optimum: f64) -> Self {
        Self { function, optimum }
    }
}

impl<F> FitnessFunction for CustomFitness<F>
where
    F: Fn(&Chromosome) -> f64 + Sync,
{
    fn evaluate(&self, chromosome: &Chromosome) -> f64 {
        (self.function)(chromosome)
    }

    fn optimum(&self, _length: usize) -> f64 {
        self.optimum
    }
}
