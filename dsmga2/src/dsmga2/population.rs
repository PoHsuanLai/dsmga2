use crate::chromosome::Chromosome;
use crate::fitness::FitnessFunction;
use crate::utils::{Statistics, ZobristKeys};
use dashmap::DashMap;
use rand_mt::Mt;
use rayon::prelude::*;

/// Population and fitness tracking
pub(crate) struct Population {
    pub(crate) chromosomes: Vec<Chromosome>,
    pub(crate) hash: DashMap<u64, f64>,
    pub(crate) best_index: usize,
}

impl Population {
    /// Create a new population
    pub(crate) fn new(
        population_size: usize,
        problem_size: usize,
        zobrist: &ZobristKeys,
        rng: &mut Mt,
        fitness_fn: &dyn FitnessFunction,
    ) -> (Self, usize) {
        let mut chromosomes = Vec::with_capacity(population_size);
        let hash = DashMap::new();

        for _ in 0..population_size {
            let mut chromosome = Chromosome::new_random(problem_size, zobrist, rng);
            let fitness = chromosome.evaluate(fitness_fn);
            hash.insert(chromosome.key(), fitness);
            chromosomes.push(chromosome);
        }

        let evaluations = population_size;

        (
            Self {
                chromosomes,
                hash,
                best_index: 0,
            },
            evaluations,
        )
    }

    /// Restore population from checkpoint
    pub(crate) fn from_checkpoint(
        population_bits: &[Vec<bool>],
        fitness_values: &[f64],
        zobrist: &ZobristKeys,
    ) -> Self {
        let mut chromosomes = Vec::with_capacity(population_bits.len());
        let hash = DashMap::new();

        for (bits, &fitness) in population_bits.iter().zip(fitness_values.iter()) {
            let chromosome = Chromosome::from_bits(bits, zobrist);
            hash.insert(chromosome.key(), fitness);
            chromosomes.push(chromosome);
        }

        // Find best individual
        let best_index = fitness_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Self {
            chromosomes,
            hash,
            best_index,
        }
    }

    /// Get population size
    pub(crate) fn size(&self) -> usize {
        self.chromosomes.len()
    }

    /// Apply GHC to entire population (parallelized)
    pub(crate) fn apply_ghc(
        &mut self,
        zobrist: &ZobristKeys,
        fitness_fn: &dyn FitnessFunction,
        rng: &mut Mt,
        patience: Option<usize>,
    ) {
        // Generate seeds for each chromosome
        let seeds: Vec<u32> = (0..self.chromosomes.len())
            .map(|_| rng.next_u32())
            .collect();

        // Apply GHC in parallel
        self.chromosomes
            .par_iter_mut()
            .zip(seeds.par_iter())
            .for_each(|(chromosome, &seed)| {
                let mut local_rng = Mt::new(seed);
                chromosome.greedy_hill_climb(zobrist, fitness_fn, &mut local_rng, patience);
            });

        // Update hash after GHC
        self.hash.clear();
        for chromosome in &self.chromosomes {
            if let Some(fitness) = chromosome.fitness() {
                self.hash.insert(chromosome.key(), fitness);
            }
        }
    }

    /// Update statistics and find best individual (parallelized)
    pub(crate) fn update_statistics(&mut self, fitness_fn: &dyn FitnessFunction) -> Statistics {
        let mut stats = Statistics::new();

        // Evaluate all fitnesses in parallel
        let results: Vec<(usize, f64)> = self
            .chromosomes
            .par_iter_mut()
            .enumerate()
            .map(|(i, chromosome)| {
                let fitness = chromosome.evaluate(fitness_fn);
                (i, fitness)
            })
            .collect();

        // Record statistics and find best
        let mut max_fitness = f64::NEG_INFINITY;
        let mut best_idx = 0;
        for (i, fitness) in results {
            stats.record(fitness);
            if fitness > max_fitness {
                max_fitness = fitness;
                best_idx = i;
            }
        }

        self.best_index = best_idx;
        stats
    }

    /// Get best chromosome
    pub(crate) fn best(&self) -> &Chromosome {
        &self.chromosomes[self.best_index]
    }

    /// Get fitness values for debugging
    pub(crate) fn get_fitness_values(&self) -> Vec<(usize, f64)> {
        self.chromosomes
            .iter()
            .enumerate()
            .filter_map(|(i, chr)| chr.fitness().map(|f| (i, f)))
            .collect()
    }

    /// Get chromosome bits as string for debugging
    pub(crate) fn get_chromosome_bits(&self, index: usize) -> Option<String> {
        if index >= self.chromosomes.len() {
            return None;
        }

        let chr = &self.chromosomes[index];
        let bits: String = (0..chr.length())
            .map(|i| if chr.get_gene(i) { '1' } else { '0' })
            .collect();
        Some(bits)
    }
}
