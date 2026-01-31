pub(crate) mod config;
pub(crate) mod convergence;
pub(crate) mod linkage;
pub(crate) mod mixing;
pub(crate) mod population;
pub(crate) mod selection;

use crate::chromosome::Chromosome;
use crate::fitness::FitnessFunction;
use crate::utils::{RngExt, Statistics, ZobristKeys};
use rand::seq::SliceRandom;
use rand_mt::Mt;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use config::Config;
use convergence::ConvergenceTracker;
use linkage::LinkageModel;
use population::Population;
use selection::SelectionBuffers;

pub(crate) const EPSILON: f64 = 1e-8;

/// DSMGA-II with two-edge graphical linkage model
pub struct Dsmga2<'a> {
    config: Config,
    fitness_fn: &'a dyn FitnessFunction,

    population: Population,
    linkage: LinkageModel,

    generation: usize,
    num_evaluations: AtomicUsize,
    statistics: Statistics,

    convergence: ConvergenceTracker,
    buffers: SelectionBuffers,

    zobrist: ZobristKeys,
    rng: Mt,
}

impl<'a> Dsmga2<'a> {
    /// Create a new DSMGA-II instance (internal use only - use builder pattern via api)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_ghc_and_patience(
        problem_size: usize,
        population_size: usize,
        fitness_fn: &'a dyn FitnessFunction,
        max_generations: Option<usize>,
        max_evaluations: Option<usize>,
        seed: Option<u64>,
        use_ghc: bool,
        ghc_patience: Option<usize>,
    ) -> Self {
        // Ensure even population size
        let population_size = (population_size / 2) * 2;

        // Initialize zobrist keys and RNG
        let zobrist = ZobristKeys::new(problem_size, seed.unwrap_or(42));
        let mut rng = Mt::new(seed.unwrap_or(42) as u32);

        // Initialize population
        let (population, initial_evaluations) = Population::new(
            population_size,
            problem_size,
            &zobrist,
            &mut rng,
            fitness_fn,
        );

        let num_evaluations = AtomicUsize::new(initial_evaluations);

        // Initialize linkage model
        let linkage = LinkageModel::new(problem_size, population_size);

        // Initialize buffers
        let buffers = SelectionBuffers::new(population_size);

        // Create config
        let config = Config::new(problem_size, max_generations, max_evaluations, use_ghc);

        let mut instance = Self {
            config,
            fitness_fn,
            population,
            linkage,
            generation: 0,
            num_evaluations,
            statistics: Statistics::new(),
            convergence: ConvergenceTracker::new(),
            buffers,
            zobrist,
            rng,
        };

        // Update statistics BEFORE GHC to see initial population
        instance.statistics = instance.population.update_statistics(instance.fitness_fn);

        // Apply Greedy Hill Climbing to initial population (like C++ version)
        if use_ghc {
            instance.population.apply_ghc(
                &instance.zobrist,
                instance.fitness_fn,
                &mut instance.rng,
                ghc_patience,
            );
            instance.statistics = instance.population.update_statistics(instance.fitness_fn);
        }

        instance
    }

    /// Restore DSMGA-II from a checkpoint (internal use only - use builder pattern via api)
    pub(crate) fn from_checkpoint(
        checkpoint: &crate::checkpoint::Checkpoint,
        fitness_fn: &'a dyn FitnessFunction,
    ) -> Self {
        let problem_size = checkpoint.problem_size;
        let population_size = checkpoint.population_size;

        // Initialize zobrist keys and RNG with same seed
        let seed = checkpoint.seed.unwrap_or(42);
        let zobrist = ZobristKeys::new(problem_size, seed);
        let rng = Mt::new(seed as u32);

        // Restore population from checkpoint
        let population = Population::from_checkpoint(
            &checkpoint.population,
            &checkpoint.fitness_values,
            &zobrist,
        );

        // Initialize linkage model (will be rebuilt on next step)
        let linkage = LinkageModel::new(problem_size, population_size);

        // Initialize buffers
        let buffers = SelectionBuffers::new(population_size);

        // Create config
        let config = Config::new(
            problem_size,
            checkpoint.max_generations,
            checkpoint.max_evaluations,
            checkpoint.use_ghc,
        );

        let mut instance = Self {
            config,
            fitness_fn,
            population,
            linkage,
            generation: checkpoint.generation,
            num_evaluations: AtomicUsize::new(checkpoint.num_evaluations),
            statistics: Statistics::new(),
            convergence: ConvergenceTracker::new(),
            buffers,
            zobrist,
            rng,
        };

        // Update statistics for restored population
        instance.statistics = instance.population.update_statistics(instance.fitness_fn);

        instance
    }

    /// Set whether to use Greedy Hill Climbing (default: true)
    pub fn set_ghc(&mut self, use_ghc: bool) {
        self.config.use_ghc = use_ghc;
    }

    /// Run one generation
    pub fn step(&mut self) -> bool {
        self.mixing();
        self.statistics = self.population.update_statistics(self.fitness_fn);
        self.generation += 1;

        !self.should_terminate()
    }

    /// Run until termination
    pub fn run(&mut self) -> &Chromosome {
        while self.step() {}
        self.population.best()
    }

    /// Check if algorithm should terminate
    fn should_terminate(&self) -> bool {
        // Max evaluations reached
        if let Some(max_eval) = self.config.max_evaluations {
            if self.num_evaluations.load(Ordering::Relaxed) > max_eval {
                return true;
            }
        }

        // Max generations reached
        if let Some(max_gen) = self.config.max_generations {
            if self.generation > max_gen {
                return true;
            }
        }

        // Found optimum
        if self.statistics.max() >= self.fitness_fn.optimum(self.config.problem_size) - EPSILON {
            return true;
        }

        // Terminate if population has converged (fitness variance near zero)
        if self.statistics.max() - EPSILON <= self.statistics.mean() {
            return true;
        }

        // Terminate if fitness statistics remain unchanged for many generations
        if self.convergence.check(
            self.statistics.max(),
            self.statistics.mean(),
            self.statistics.min(),
        ) {
            return true;
        }

        false
    }

    /// Perform mixing operations
    fn mixing(&mut self) {
        // Tournament selection
        selection::tournament_selection(
            &mut self.population.chromosomes,
            self.config.selection_pressure,
            self.fitness_fn,
            &mut self.buffers,
            &mut self.rng,
        );

        // Build fast counting from selected population
        selection::build_fast_counting(
            &self.population.chromosomes,
            &self.buffers.indices,
            &mut self.linkage.fast_counting,
        );

        // Build linkage graphs
        self.linkage
            .build_graph(self.population.size(), self.config.problem_size);
        self.linkage
            .build_graph_size(self.population.size(), self.config.problem_size);

        // Determine number of mixing rounds
        let rounds = if self.config.problem_size > 50 {
            self.config.problem_size / 50
        } else {
            1
        };

        // Apply restricted mixing
        for _round in 0..rounds {
            // Shuffle population order
            self.buffers.population_order.shuffle(&mut self.rng);

            // Generate seeds for parallel mixing
            let seeds: Vec<u32> = (0..self.buffers.population_order.len())
                .map(|_| self.rng.next_u32())
                .collect();

            // Parallel: compute all mixing results
            let results: Vec<mixing::MixingResult> = self
                .buffers
                .population_order
                .par_iter()
                .zip(seeds.par_iter())
                .map(|(&idx, &seed)| {
                    let mut local_rng = Mt::new(seed);
                    self.compute_mixing(idx, &mut local_rng)
                })
                .collect();

            // Sequential: apply results and update state
            for result in results {
                // Update evaluation counter
                self.num_evaluations
                    .fetch_add(result.evaluations, Ordering::Relaxed);

                // Apply successful mixing
                if let Some(new_chromosome) = result.new_chromosome {
                    // Remove old key from hash
                    if let Some(old_key) = result.old_key {
                        self.population.hash.remove(&old_key);
                    }

                    // Insert new key with fitness
                    if let Some(new_fitness) = result.new_fitness {
                        self.population
                            .hash
                            .insert(new_chromosome.key(), new_fitness);
                    }

                    // Update population
                    self.population.chromosomes[result.chromosome_idx] = new_chromosome;
                }
            }
        }
    }

    /// Restricted mixing: compute mixing result without modifying state
    fn compute_mixing(&self, chromosome_idx: usize, rng: &mut Mt) -> mixing::MixingResult {
        let start_gene = rng.uniform_int(0, self.config.problem_size - 1);

        // Build mask
        let chromosome = self.population.chromosomes[chromosome_idx].clone();
        let mut mask = mixing::find_mask(
            &chromosome,
            start_gene,
            self.config.problem_size,
            &self.linkage.graph,
            rng,
        );

        // Calculate sizes
        let size = mixing::calculate_mask_size(&chromosome, &mask, &self.population.chromosomes);
        let size_bound = mixing::find_size_bound(
            &chromosome,
            start_gene,
            size,
            self.config.problem_size,
            &self.linkage.graph_size,
        );

        let final_size = size.min(size_bound);
        mask.truncate(final_size);

        // Compute mixing result
        mixing::compute_mixing_with_mask(
            chromosome_idx,
            &mask,
            &self.population.chromosomes,
            &self.population.hash,
            &self.zobrist,
            self.fitness_fn,
        )
    }

    // Public getters
    pub fn generation(&self) -> usize {
        self.generation
    }

    pub fn num_evaluations(&self) -> usize {
        self.num_evaluations.load(Ordering::Relaxed)
    }

    pub fn population_size(&self) -> usize {
        self.population.size()
    }

    pub fn seed(&self) -> Option<u64> {
        None // Seed is not stored after initialization
    }

    pub fn problem_size(&self) -> usize {
        self.config.problem_size
    }

    pub fn max_generations(&self) -> Option<usize> {
        self.config.max_generations
    }

    pub fn max_evaluations(&self) -> Option<usize> {
        self.config.max_evaluations
    }

    pub fn use_ghc(&self) -> bool {
        self.config.use_ghc
    }

    pub fn best_fitness(&self) -> f64 {
        self.statistics.max()
    }

    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }

    pub fn linkage(&self) -> Vec<(usize, usize, f64)> {
        self.linkage.get_linkage_edges(self.config.problem_size)
    }

    pub fn best_solution(&self) -> &Chromosome {
        self.population.best()
    }

    // Internal methods used by api.rs for checkpointing
    pub(crate) fn get_population_fitness(&self) -> Vec<(usize, f64)> {
        self.population.get_fitness_values()
    }

    pub(crate) fn get_chromosome_bits(&self, index: usize) -> Option<String> {
        self.population.get_chromosome_bits(index)
    }
}
