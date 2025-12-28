use crate::checkpoint::Checkpoint;
use crate::chromosome::Chromosome;
use crate::dsmga2::Dsmga2 as Dsmga2Core;
use crate::fitness::FitnessFunction;
use std::path::Path;

/// Builder for configuring DSMGA2
///
/// # Example
/// ```rust
/// use dsmga2::{Dsmga2, fitness::OneMax};
///
/// let fitness_fn = OneMax;
/// let mut ga = Dsmga2::new(100, &fitness_fn)
///     .population_size(200)
///     .max_generations(1000)
///     .seed(42)
///     .build();
///
/// // Run to completion
/// let solution = ga.run();
/// ```
pub struct Dsmga2Builder<'a> {
    problem_size: usize,
    fitness_fn: &'a dyn FitnessFunction,
    population_size: usize,
    max_generations: Option<usize>,
    max_evaluations: Option<usize>,
    seed: Option<u64>,
    use_ghc: bool,
    ghc_patience: Option<usize>,
    fitness_threshold: Option<f64>,
    patience: Option<usize>,
}

impl<'a> Dsmga2Builder<'a> {
    /// Create a new DSMGA2 builder
    pub fn new(problem_size: usize, fitness_fn: &'a dyn FitnessFunction) -> Self {
        Self {
            problem_size,
            fitness_fn,
            population_size: problem_size, // Default: same as problem size
            max_generations: Some(1000),
            max_evaluations: None,
            seed: None,
            use_ghc: true,           // Default: enabled (matches C++ behavior)
            ghc_patience: Some(500), // Default: stop after 500 consecutive non-improvements
            fitness_threshold: None,
            patience: None,
        }
    }

    /// Set population size (default: problem_size)
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Set maximum generations (default: 1000)
    pub fn max_generations(mut self, generations: usize) -> Self {
        self.max_generations = Some(generations);
        self
    }

    /// Set maximum fitness evaluations (default: None)
    pub fn max_evaluations(mut self, evaluations: usize) -> Self {
        self.max_evaluations = Some(evaluations);
        self
    }

    /// Set random seed for reproducibility
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable or disable Greedy Hill Climbing (default: true)
    /// GHC applies local search to the initial population
    pub fn use_ghc(mut self, use_ghc: bool) -> Self {
        self.use_ghc = use_ghc;
        self
    }

    /// Set GHC early stopping patience (default: Some(500))
    /// GHC will stop after this many consecutive non-improvements.
    /// Set to None to try all bits (no early stopping).
    pub fn ghc_patience(mut self, patience: Option<usize>) -> Self {
        self.ghc_patience = patience;
        self
    }

    /// Stop when fitness reaches or exceeds this threshold
    pub fn fitness_threshold(mut self, threshold: f64) -> Self {
        self.fitness_threshold = Some(threshold);
        self
    }

    /// Stop if best fitness doesn't improve for this many generations
    pub fn patience(mut self, generations: usize) -> Self {
        self.patience = Some(generations);
        self
    }

    /// Build the DSMGA2 instance
    pub fn build(self) -> Dsmga2<'a> {
        let inner = Dsmga2Core::new_with_ghc_and_patience(
            self.problem_size,
            self.population_size,
            self.fitness_fn,
            self.max_generations,
            self.max_evaluations,
            self.seed,
            self.use_ghc,
            self.ghc_patience,
        );

        Dsmga2 {
            inner,
            fitness_threshold: self.fitness_threshold,
            patience: self.patience,
            generations_without_improvement: 0,
            best_fitness_so_far: f64::NEG_INFINITY,
        }
    }
}

/// DSMGA2: Dependency Structure Matrix Genetic Algorithm II
///
/// Provides both iterator-based and convenience APIs for optimization.
pub struct Dsmga2<'a> {
    inner: Dsmga2Core<'a>,
    fitness_threshold: Option<f64>,
    patience: Option<usize>,
    generations_without_improvement: usize,
    best_fitness_so_far: f64,
}

impl<'a> Dsmga2<'a> {
    /// Create a new DSMGA2 instance with default settings
    ///
    /// # Example
    /// ```rust
    /// use dsmga2::{Dsmga2, fitness::OneMax};
    ///
    /// let fitness_fn = OneMax;
    /// let mut ga = Dsmga2::new(100, &fitness_fn)
    ///     .population_size(200)
    ///     .build();
    /// ```
    #[allow(clippy::new_ret_no_self)]
    pub fn new(problem_size: usize, fitness_fn: &'a dyn FitnessFunction) -> Dsmga2Builder<'a> {
        Dsmga2Builder::new(problem_size, fitness_fn)
    }

    /// Perform a single optimization step
    ///
    /// Returns `Some(state)` if optimization should continue, `None` if terminated.
    pub fn step(&mut self) -> Option<State> {
        if !self.inner.step() {
            return None;
        }

        let current_fitness = self.inner.best_fitness();

        // Check fitness threshold
        if let Some(threshold) = self.fitness_threshold {
            if current_fitness >= threshold {
                return None;
            }
        }

        // Check patience (early stopping)
        if let Some(patience) = self.patience {
            if current_fitness > self.best_fitness_so_far {
                self.best_fitness_so_far = current_fitness;
                self.generations_without_improvement = 0;
            } else {
                self.generations_without_improvement += 1;
                if self.generations_without_improvement >= patience {
                    return None;
                }
            }
        } else {
            self.best_fitness_so_far = current_fitness;
        }

        Some(State {
            generation: self.inner.generation(),
            num_evaluations: self.inner.num_evaluations(),
            best_fitness: current_fitness,
            mean_fitness: self.inner.statistics().mean(),
            min_fitness: self.inner.statistics().min(),
        })
    }

    /// Run optimization to completion
    ///
    /// Returns the best solution found.
    pub fn run(&mut self) -> &Chromosome {
        self.inner.run()
    }

    /// Run with a callback function called after each generation
    ///
    /// # Example
    /// ```rust
    /// use dsmga2::{Dsmga2, fitness::OneMax};
    ///
    /// let fitness_fn = OneMax;
    /// let mut ga = Dsmga2::new(100, &fitness_fn).build();
    ///
    /// ga.run_with(|state| {
    ///     println!("Gen {}: fitness = {:.2}", state.generation, state.best_fitness);
    /// });
    /// ```
    pub fn run_with<F>(&mut self, mut callback: F) -> &Chromosome
    where
        F: FnMut(&State),
    {
        while let Some(state) = self.step() {
            callback(&state);
        }
        self.best_solution()
    }

    /// Get current generation number
    pub fn generation(&self) -> usize {
        self.inner.generation()
    }

    /// Get total number of fitness evaluations
    pub fn num_evaluations(&self) -> usize {
        self.inner.num_evaluations()
    }

    /// Get best fitness found so far
    pub fn best_fitness(&self) -> f64 {
        self.inner.best_fitness()
    }

    /// Get the best solution found
    pub fn best_solution(&self) -> &Chromosome {
        self.inner.best_solution()
    }

    /// Get the learned linkage structure
    ///
    /// Returns a vector of (gene_i, gene_j, weight) tuples representing
    /// dependencies between genes. Higher weights indicate stronger
    /// statistical dependencies in the population.
    pub fn linkage(&self) -> Vec<(usize, usize, f64)> {
        self.inner.linkage()
    }

    /// Get current state
    pub fn state(&self) -> State {
        State {
            generation: self.inner.generation(),
            num_evaluations: self.inner.num_evaluations(),
            best_fitness: self.inner.best_fitness(),
            mean_fitness: self.inner.statistics().mean(),
            min_fitness: self.inner.statistics().min(),
        }
    }

    /// Get population fitness values for debugging
    pub fn get_population_fitness(&self) -> Vec<(usize, f64)> {
        self.inner.get_population_fitness()
    }

    /// Get chromosome bits as string for debugging
    pub fn get_chromosome_bits(&self, index: usize) -> Option<String> {
        self.inner.get_chromosome_bits(index)
    }

    /// Save current state to a checkpoint file
    ///
    /// Saves the entire population and state, allowing optimization to be resumed.
    /// Similar to PyGAD's save() method.
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        // Get population fitness
        let pop_fitness = self.inner.get_population_fitness();

        // Convert population to bit vectors
        let population: Vec<Vec<bool>> = pop_fitness
            .iter()
            .map(|(idx, _)| {
                (0..self.inner.problem_size())
                    .map(|i| {
                        self.inner
                            .get_chromosome_bits(*idx)
                            .and_then(|s| s.chars().nth(i))
                            .map(|c| c == '1')
                            .unwrap_or(false)
                    })
                    .collect()
            })
            .collect();

        let fitness_values: Vec<f64> = pop_fitness.iter().map(|(_, f)| *f).collect();

        let checkpoint = Checkpoint {
            problem_size: self.inner.problem_size(),
            population_size: self.inner.population_size(),
            generation: self.generation(),
            num_evaluations: self.num_evaluations(),
            population,
            fitness_values,
            seed: self.inner.seed(),
            max_generations: self.inner.max_generations(),
            max_evaluations: self.inner.max_evaluations(),
            use_ghc: self.inner.use_ghc(),
        };

        checkpoint.save(path)
    }

    /// Load state from a checkpoint file
    ///
    /// Returns the checkpoint data which can be inspected or used with `from_checkpoint()`.
    pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> std::io::Result<Checkpoint> {
        Checkpoint::load(path)
    }

    /// Resume optimization from a checkpoint
    ///
    /// Creates a new DSMGA2 instance with state restored from a checkpoint.
    /// The population, generation count, and evaluation count are all restored.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dsmga2::{Dsmga2, fitness::OneMax};
    ///
    /// let fitness_fn = OneMax;
    ///
    /// // Load checkpoint
    /// let checkpoint = Dsmga2::load_checkpoint("/tmp/checkpoint.bin").unwrap();
    ///
    /// // Resume from checkpoint
    /// let mut ga = Dsmga2::from_checkpoint(checkpoint, &fitness_fn);
    ///
    /// // Continue optimization
    /// ga.run();
    /// ```
    pub fn from_checkpoint(checkpoint: Checkpoint, fitness_fn: &'a dyn FitnessFunction) -> Self {
        let inner = Dsmga2Core::from_checkpoint(&checkpoint, fitness_fn);

        Dsmga2 {
            inner,
            fitness_threshold: None,
            patience: None,
            generations_without_improvement: 0,
            best_fitness_so_far: f64::NEG_INFINITY,
        }
    }
}

/// State information about the DSMGA2 run
#[derive(Debug, Clone, Copy)]
pub struct State {
    pub generation: usize,
    pub num_evaluations: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub min_fitness: f64,
}

/// Iterator adapter for step-by-step optimization
impl<'a> Iterator for Dsmga2<'a> {
    type Item = State;

    fn next(&mut self) -> Option<Self::Item> {
        self.step()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::OneMax;

    #[test]
    fn test_dsmga2_builder() {
        let fitness_fn = OneMax;
        let mut ga = Dsmga2::new(50, &fitness_fn)
            .population_size(100)
            .max_generations(10)
            .seed(42)
            .build();

        let _solution = ga.run();
        assert!(ga.generation() > 0);
    }

    #[test]
    fn test_dsmga2_iterator() {
        // Use a harder problem to ensure it doesn't converge immediately
        let fitness_fn = crate::fitness::MkTrap::new(5);
        let ga = Dsmga2::new(50, &fitness_fn)
            .max_generations(5)
            .population_size(50)
            .seed(42)
            .build();

        let generations: Vec<State> = ga.take(5).collect();
        // Should get at least some generations before stopping
        assert!(!generations.is_empty());
    }

    #[test]
    fn test_dsmga2_callback() {
        // Use a harder problem to ensure multiple generations
        let fitness_fn = crate::fitness::MkTrap::new(5);
        let mut ga = Dsmga2::new(50, &fitness_fn)
            .max_generations(10)
            .population_size(50)
            .seed(42)
            .build();

        let mut callback_count = 0;
        ga.run_with(|_state| {
            callback_count += 1;
        });

        // Should have at least one callback
        assert!(callback_count > 0);
    }
}
