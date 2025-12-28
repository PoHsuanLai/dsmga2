use crate::chromosome::Chromosome;
use crate::fitness::FitnessFunction;
use crate::structures::{FastCounting, SimpleSet, TriMatrix};
use crate::utils::{RngExt, Statistics, ZobristKeys};
use dashmap::DashMap;
use rand::seq::SliceRandom;
use rand_mt::Mt;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const EPSILON: f64 = 1e-8;

/// Result of a mixing operation
struct MixingResult {
    chromosome_idx: usize,
    new_chromosome: Option<Chromosome>,
    new_fitness: Option<f64>,
    old_key: Option<u64>,
    evaluations: usize,
}

/// DSMGA-II with two-edge graphical linkage model
pub struct Dsmga2<'a> {
    // Problem configuration (public for API access)
    problem_size: usize,
    fitness_fn: &'a dyn FitnessFunction,

    // Population (private internals)
    population: Vec<Chromosome>,
    population_size: usize,

    // Algorithm state (private internals)
    generation: usize,
    num_evaluations: AtomicUsize,
    best_index: usize,

    // Configuration (private internals)
    max_generations: Option<usize>,
    max_evaluations: Option<usize>,
    selection_pressure: usize,
    use_ghc: bool,
    ghc_patience: Option<usize>,

    // Linkage learning structures (private internals)
    linkage_graph: TriMatrix,
    linkage_graph_size: TriMatrix,
    fast_counting: Vec<FastCounting>,

    // Utilities (private internals)
    zobrist: ZobristKeys,
    rng: Mt,
    population_hash: DashMap<u64, f64>,
    statistics: Statistics,

    // Selection indices (private internals)
    selection_indices: Vec<usize>,
    population_order: Vec<usize>,

    // Convergence tracking (private internals, atomic for thread-safety)
    converge_count: AtomicUsize,
    last_max: AtomicU64,
    last_mean: AtomicU64,
    last_min: AtomicU64,
}

impl<'a> Dsmga2<'a> {
    /// Create a new DSMGA-II instance
    pub fn new(
        problem_size: usize,
        population_size: usize,
        fitness_fn: &'a dyn FitnessFunction,
        max_generations: Option<usize>,
        max_evaluations: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        Self::new_with_ghc(
            problem_size,
            population_size,
            fitness_fn,
            max_generations,
            max_evaluations,
            seed,
            true, // GHC enabled by default
        )
    }

    /// Create a new DSMGA-II instance with configurable GHC
    pub fn new_with_ghc(
        problem_size: usize,
        population_size: usize,
        fitness_fn: &'a dyn FitnessFunction,
        max_generations: Option<usize>,
        max_evaluations: Option<usize>,
        seed: Option<u64>,
        use_ghc: bool,
    ) -> Self {
        Self::new_with_ghc_and_patience(
            problem_size,
            population_size,
            fitness_fn,
            max_generations,
            max_evaluations,
            seed,
            use_ghc,
            None, // Default: no early stopping
        )
    }

    /// Create a new DSMGA-II instance with configurable GHC and patience
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_ghc_and_patience(
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
        let mut population = Vec::with_capacity(population_size);
        let population_hash = DashMap::new();

        for _ in 0..population_size {
            let mut chromosome = Chromosome::new_random(problem_size, &zobrist, &mut rng);
            let fitness = chromosome.evaluate(fitness_fn);
            population_hash.insert(chromosome.key(), fitness);
            population.push(chromosome);
        }

        let num_evaluations = AtomicUsize::new(population_size);

        // Initialize linkage graphs
        let linkage_graph = TriMatrix::new(problem_size);
        let linkage_graph_size = TriMatrix::new(problem_size);

        // Initialize fast counting
        let fast_counting = (0..problem_size)
            .map(|_| FastCounting::new(population_size))
            .collect();

        let selection_indices = vec![0; population_size];
        let population_order = (0..population_size).collect();

        let mut instance = Self {
            problem_size,
            fitness_fn,
            population,
            population_size,
            generation: 0,
            num_evaluations,
            best_index: 0,
            max_generations,
            max_evaluations,
            selection_pressure: 2,
            use_ghc,
            ghc_patience,
            linkage_graph,
            linkage_graph_size,
            fast_counting,
            zobrist,
            rng,
            population_hash,
            statistics: Statistics::new(),
            selection_indices,
            population_order,
            converge_count: AtomicUsize::new(0),
            last_max: AtomicU64::new(0),
            last_mean: AtomicU64::new(0),
            last_min: AtomicU64::new(0),
        };

        // Update statistics BEFORE GHC to see initial population
        instance.update_statistics();

        // Apply Greedy Hill Climbing to initial population (like C++ version)
        if use_ghc {
            instance.apply_ghc_to_population();
            instance.update_statistics();
        }

        instance
    }

    /// Restore DSMGA-II from a checkpoint
    ///
    /// Creates a new instance with state restored from checkpoint.
    /// This allows resuming optimization from a saved state.
    pub fn from_checkpoint(
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
        let mut population = Vec::with_capacity(population_size);
        let population_hash = DashMap::new();

        for (bits, &fitness) in checkpoint.population.iter().zip(&checkpoint.fitness_values) {
            let chromosome = Chromosome::from_bits(bits, &zobrist);
            population_hash.insert(chromosome.key(), fitness);
            population.push(chromosome);
        }

        // Initialize linkage graphs (will be rebuilt on next step)
        let linkage_graph = TriMatrix::new(problem_size);
        let linkage_graph_size = TriMatrix::new(problem_size);

        // Initialize fast counting
        let fast_counting = (0..problem_size)
            .map(|_| FastCounting::new(population_size))
            .collect();

        let selection_indices = vec![0; population_size];
        let population_order = (0..population_size).collect();

        // Find best individual
        let best_index = checkpoint
            .fitness_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let mut instance = Self {
            problem_size,
            fitness_fn,
            population,
            population_size,
            generation: checkpoint.generation,
            num_evaluations: AtomicUsize::new(checkpoint.num_evaluations),
            best_index,
            max_generations: checkpoint.max_generations,
            max_evaluations: checkpoint.max_evaluations,
            selection_pressure: 2,
            use_ghc: checkpoint.use_ghc,
            ghc_patience: Some(500), // Default patience
            linkage_graph,
            linkage_graph_size,
            fast_counting,
            zobrist,
            rng,
            population_hash,
            statistics: Statistics::new(),
            selection_indices,
            population_order,
            converge_count: AtomicUsize::new(0),
            last_max: AtomicU64::new(0),
            last_mean: AtomicU64::new(0),
            last_min: AtomicU64::new(0),
        };

        // Update statistics for restored population
        instance.update_statistics();

        instance
    }

    /// Apply Greedy Hill Climbing to entire population
    /// Optimizes initial population with local search
    /// Apply GHC to population (parallelized)
    fn apply_ghc_to_population(&mut self) {
        let patience = self.ghc_patience;

        // Generate seeds for each chromosome using main RNG
        let seeds: Vec<u32> = (0..self.population_size)
            .map(|_| self.rng.next_u32())
            .collect();

        // Apply GHC in parallel with thread-local RNGs
        self.population
            .par_iter_mut()
            .zip(seeds.par_iter())
            .for_each(|(chromosome, &seed)| {
                let mut local_rng = Mt::new(seed);
                chromosome.greedy_hill_climb(
                    &self.zobrist,
                    self.fitness_fn,
                    &mut local_rng,
                    patience,
                );
            });

        // NOTE: Different from C++ - we update pHash after GHC
        // C++ has a bug where it does not update pHash here, leading to stale hash entries
        // We do the correct thing and update the hash after chromosomes are modified by GHC
        self.population_hash.clear();
        for chromosome in &self.population {
            if let Some(fitness) = chromosome.fitness() {
                self.population_hash.insert(chromosome.key(), fitness);
            }
        }

        // NOTE: Same as C++ - GHC evaluations are NOT counted in main nfe
        // C++ tracks them separately in lsnfe
    }

    /// Set whether to use Greedy Hill Climbing (default: true)
    pub fn set_ghc(&mut self, use_ghc: bool) {
        self.use_ghc = use_ghc;
    }

    /// Run one generation
    pub fn step(&mut self) -> bool {
        self.mixing();
        self.update_statistics();
        self.generation += 1;

        !self.should_terminate()
    }

    /// Run until termination
    pub fn run(&mut self) -> &Chromosome {
        while self.step() {}
        &self.population[self.best_index]
    }

    /// Check if algorithm should terminate
    fn should_terminate(&self) -> bool {
        // Max evaluations reached
        if let Some(max_eval) = self.max_evaluations {
            if self.num_evaluations.load(Ordering::Relaxed) > max_eval {
                return true;
            }
        }

        // Max generations reached
        if let Some(max_gen) = self.max_generations {
            if self.generation > max_gen {
                return true;
            }
        }

        // Found optimum
        if self.statistics.max() >= self.fitness_fn.optimum(self.problem_size) - EPSILON {
            return true;
        }

        // Population converged (all individuals have similar fitness)
        // C++: if (stFitness.getMax() - EPSILON <= stFitness.getMean()) termination = true;
        if self.statistics.max() - EPSILON <= self.statistics.mean() {
            return true;
        }

        // Slow convergence check (matching C++ logic: max/mean/min unchanged for >300 generations)
        // Note: This is rarely reached due to the faster check above
        if self.converged() {
            return true;
        }

        false
    }

    /// Check convergence (matching C++ implementation)
    fn converged(&self) -> bool {
        let current_max = self.statistics.max();
        let current_mean = self.statistics.mean();
        let current_min = self.statistics.min();

        let last_max = f64::from_bits(self.last_max.load(Ordering::Relaxed));
        let last_mean = f64::from_bits(self.last_mean.load(Ordering::Relaxed));
        let last_min = f64::from_bits(self.last_min.load(Ordering::Relaxed));

        if current_max == last_max && current_mean == last_mean && current_min == last_min {
            self.converge_count.fetch_add(1, Ordering::Relaxed);
        } else {
            self.converge_count.store(0, Ordering::Relaxed);
        }

        self.last_max
            .store(current_max.to_bits(), Ordering::Relaxed);
        self.last_mean
            .store(current_mean.to_bits(), Ordering::Relaxed);
        self.last_min
            .store(current_min.to_bits(), Ordering::Relaxed);

        self.converge_count.load(Ordering::Relaxed) > 300
    }

    /// Update fitness statistics (parallelized)
    fn update_statistics(&mut self) {
        self.statistics.reset();

        // Evaluate all fitnesses in parallel and find max
        let results: Vec<(usize, f64)> = self
            .population
            .par_iter_mut()
            .enumerate()
            .map(|(i, chromosome)| {
                let fitness = chromosome.evaluate(self.fitness_fn);
                (i, fitness)
            })
            .collect();

        // Record statistics and find best
        let mut max_fitness = f64::NEG_INFINITY;
        let mut best_idx = 0;
        for (i, fitness) in results {
            self.statistics.record(fitness);
            if fitness > max_fitness {
                max_fitness = fitness;
                best_idx = i;
            }
        }

        self.best_index = best_idx;
    }

    /// Tournament selection
    /// Matches C++ implementation exactly (sequential, not parallel)
    pub fn tournament_selection(&mut self) {
        // Ensure all fitness values are cached
        for chromosome in self.population.iter_mut() {
            chromosome.evaluate(self.fitness_fn);
        }

        let mut candidates = vec![0; self.selection_pressure * self.population_size];

        // Generate random indices for tournament (without replacement per pressure level)
        for i in 0..self.selection_pressure {
            let start = i * self.population_size;
            // Create range [0, 1, ..., population_size-1] and shuffle
            let mut indices: Vec<usize> = (0..self.population_size).collect();
            indices.shuffle(&mut self.rng);
            candidates[start..start + self.population_size].copy_from_slice(&indices);
        }

        // Run tournaments sequentially to match C++ exactly
        self.selection_indices.clear();
        for i in 0..self.population_size {
            let mut winner = 0;
            let mut winner_fitness = f64::NEG_INFINITY;

            for j in 0..self.selection_pressure {
                let candidate = candidates[self.selection_pressure * i + j];
                let fitness = self.population[candidate].fitness().unwrap();

                if fitness > winner_fitness {
                    winner = candidate;
                    winner_fitness = fitness;
                }
            }

            self.selection_indices.push(winner);
        }
    }

    /// Wrapper for RNG uniform_int
    pub fn uniform_int(&mut self, a: usize, b: usize) -> usize {
        self.rng.uniform_int(a, b)
    }

    /// Build fast counting structure from population (parallelized)
    pub fn build_fast_counting(&mut self) {
        // Process each gene position in parallel
        self.fast_counting
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, fast_count)| {
                for i in 0..self.population_size {
                    let idx = self.selection_indices[i];
                    let value = self.population[idx].get_gene(j);
                    fast_count.set(i, value);
                }
            });
    }

    /// Build linkage graph using mutual information
    /// Build linkage graph using two-edge model (parallelized)
    pub fn build_graph(&mut self) {
        // Count ones for each gene (can be parallelized)
        let ones_count: Vec<usize> = (0..self.problem_size)
            .into_par_iter()
            .map(|i| self.fast_counting[i].count_ones())
            .collect();

        // Compute all (i,j) pairs to process
        let pairs: Vec<(usize, usize)> = (0..self.problem_size)
            .flat_map(|i| ((i + 1)..self.problem_size).map(move |j| (i, j)))
            .collect();

        // Compute pairwise linkage in parallel
        let results: Vec<((usize, usize), (f64, f64))> = pairs
            .par_iter()
            .map(|&(i, j)| {
                let xor_count = self.fast_counting[i].count_xor(&self.fast_counting[j]);

                // Calculate joint probabilities
                let n11 = ((ones_count[i] + ones_count[j] - xor_count) / 2) as f64;
                let n10 = ones_count[i] as f64 - n11;
                let n01 = ones_count[j] as f64 - n11;
                let n00 = self.population_size as f64 - n01 - n10 - n11;

                let p00 = n00 / self.population_size as f64;
                let p01 = n01 / self.population_size as f64;
                let p10 = n10 / self.population_size as f64;
                let p11 = n11 / self.population_size as f64;

                let p0x = p00 + p01;
                let p1x = p10 + p11;
                let px0 = p00 + p10;
                let px1 = p01 + p11;

                // Two-edge linkage: separate for matching and differing values
                let mut linkage_same = 0.0;
                let mut linkage_diff = 0.0;

                if p00 > EPSILON {
                    linkage_same += p00 * (p00.ln() - px0.ln() - p0x.ln());
                }
                if p11 > EPSILON {
                    linkage_same += p11 * (p11.ln() - px1.ln() - p1x.ln());
                }
                if p01 > EPSILON {
                    linkage_diff += p01 * (p01.ln() - p0x.ln() - px1.ln());
                }
                if p10 > EPSILON {
                    linkage_diff += p10 * (p10.ln() - p1x.ln() - px0.ln());
                }

                ((i, j), (linkage_same, linkage_diff))
            })
            .collect();

        // Write results back to graph
        for ((i, j), values) in results {
            self.linkage_graph.write(i, j, values);
        }
    }

    /// Build graph_size using mutual information (for size checking, parallelized)
    pub fn build_graph_size(&mut self) {
        // Count ones for each gene (parallelized)
        let ones_count: Vec<usize> = (0..self.problem_size)
            .into_par_iter()
            .map(|i| self.fast_counting[i].count_ones())
            .collect();

        // Compute all (i,j) pairs
        let pairs: Vec<(usize, usize)> = (0..self.problem_size)
            .flat_map(|i| ((i + 1)..self.problem_size).map(move |j| (i, j)))
            .collect();

        // Compute pairwise MI in parallel
        let results: Vec<((usize, usize), (f64, f64))> = pairs
            .par_iter()
            .map(|&(i, j)| {
                let xor_count = self.fast_counting[i].count_xor(&self.fast_counting[j]);

                // Calculate joint probabilities
                let n11 = ((ones_count[i] + ones_count[j] - xor_count) / 2) as f64;
                let n10 = ones_count[i] as f64 - n11;
                let n01 = ones_count[j] as f64 - n11;
                let n00 = self.population_size as f64 - n01 - n10 - n11;

                let p00 = n00 / self.population_size as f64;
                let p01 = n01 / self.population_size as f64;
                let p10 = n10 / self.population_size as f64;
                let p11 = n11 / self.population_size as f64;

                let p0x = p00 + p01;
                let p1x = p10 + p11;
                let px0 = p00 + p10;
                let px1 = p01 + p11;

                // Compute mutual information (computeMI)
                let mut mi = 0.0;
                if p00 > EPSILON {
                    mi += p00 * p00.ln();
                }
                if p01 > EPSILON {
                    mi += p01 * p01.ln();
                }
                if p10 > EPSILON {
                    mi += p10 * p10.ln();
                }
                if p11 > EPSILON {
                    mi += p11 * p11.ln();
                }

                if p0x > EPSILON {
                    mi -= p0x * p0x.ln();
                }
                if p1x > EPSILON {
                    mi -= p1x * p1x.ln();
                }
                if px0 > EPSILON {
                    mi -= px0 * px0.ln();
                }
                if px1 > EPSILON {
                    mi -= px1 * px1.ln();
                }

                ((i, j), (mi, mi))
            })
            .collect();

        // Write results back to graph
        for ((i, j), values) in results {
            self.linkage_graph_size.write(i, j, values);
        }
    }

    /// Find linkage mask starting from a gene
    pub fn find_mask(
        &self,
        chromosome: &Chromosome,
        start_gene: usize,
        rng: &mut Mt,
    ) -> SmallVec<[usize; 64]> {
        let mut mask = SmallVec::new();
        let mut rest = SimpleSet::new(self.problem_size);

        // Generate random gene order
        let mut gene_order: Vec<usize> = (0..self.problem_size).collect();
        gene_order.shuffle(rng);

        // Add start gene to mask, others to rest (more idiomatic)
        mask.push(start_gene);
        for &gene in gene_order.iter().filter(|&&g| g != start_gene) {
            rest.insert(gene);
        }

        // Build connection scores
        let mut connections = vec![0.0; self.problem_size];

        for gene in rest.iter() {
            let (linkage_same, linkage_diff) = self.linkage_graph.read(start_gene, gene);
            let start_val = chromosome.get_gene(start_gene);
            let gene_val = chromosome.get_gene(gene);

            connections[gene] = if start_val == gene_val {
                linkage_same
            } else {
                linkage_diff
            };
        }

        // Greedily add genes with strongest linkage
        while !rest.is_empty() {
            // Find gene with max connection (idiomatic Rust)
            let best_gene = rest
                .iter()
                .max_by(|&a, &b| connections[a].partial_cmp(&connections[b]).unwrap())
                .expect("rest is not empty");

            rest.remove(best_gene);
            mask.push(best_gene);

            // Update connections for remaining genes
            for gene in rest.iter() {
                let (linkage_same, linkage_diff) = self.linkage_graph.read(best_gene, gene);
                let best_val = chromosome.get_gene(best_gene);
                let gene_val = chromosome.get_gene(gene);

                connections[gene] += if best_val == gene_val {
                    linkage_same
                } else {
                    linkage_diff
                };
            }
        }

        mask
    }

    /// Find size bound using size-check graph
    pub fn find_size_bound(
        &self,
        chromosome: &Chromosome,
        start_gene: usize,
        max_size: usize,
        _rng: &mut Mt,
    ) -> usize {
        // Match C++ findMask_size: build a NEW mask using linkage_graph_size
        // with a size bound, then call findSize on it

        let mut mask_size = SmallVec::<[usize; 64]>::new();
        let mut rest = SimpleSet::new(self.problem_size);

        // Add start_gene to mask, others to rest (more idiomatic)
        mask_size.push(start_gene);
        for gene in (0..self.problem_size).filter(|&g| g != start_gene) {
            rest.insert(gene);
        }

        let mut connections = vec![0.0; self.problem_size];

        // Initialize connections with linkage to start_gene
        for gene in rest.iter() {
            let (linkage_same, linkage_diff) = self.linkage_graph_size.read(start_gene, gene);
            let start_val = chromosome.get_gene(start_gene);
            let gene_val = chromosome.get_gene(gene);

            connections[gene] = if start_val == gene_val {
                linkage_same
            } else {
                linkage_diff
            };
        }

        // Greedily add genes up to max_size (bound)
        let mut bound = max_size.saturating_sub(1); // Already added start_gene
        while !rest.is_empty() && bound > 0 {
            bound -= 1;

            // Find gene with max connection (idiomatic Rust)
            let best_gene = rest
                .iter()
                .max_by(|&a, &b| connections[a].partial_cmp(&connections[b]).unwrap())
                .expect("rest is not empty");

            rest.remove(best_gene);
            mask_size.push(best_gene);

            // Update connections with linkage to newly added gene
            for gene in rest.iter() {
                let (linkage_same, linkage_diff) = self.linkage_graph_size.read(best_gene, gene);
                let best_val = chromosome.get_gene(best_gene);
                let gene_val = chromosome.get_gene(gene);

                connections[gene] += if best_val == gene_val {
                    linkage_same
                } else {
                    linkage_diff
                };
            }
        }

        // Now call calculate_mask_size on this new mask
        self.calculate_mask_size(chromosome, &mask_size)
    }

    /// Calculate mask size (number of genes that would filter out all population)
    pub fn calculate_mask_size(&self, chromosome: &Chromosome, mask: &[usize]) -> usize {
        let mut candidates = SimpleSet::new(self.population_size);
        for i in 0..self.population_size {
            candidates.insert(i);
        }

        let mut size = 0;
        for &gene in mask.iter() {
            let allele = chromosome.get_gene(gene);

            // Collect indices to remove (can't modify while iterating)
            let to_remove: Vec<usize> = candidates
                .iter()
                .filter(|&idx| self.population[idx].get_gene(gene) == allele)
                .collect();

            for idx in to_remove {
                candidates.remove(idx);
            }

            // C++: check isEmpty BEFORE incrementing size
            if candidates.is_empty() {
                break;
            }

            size += 1;
        }

        size
    }

    /// Restricted mixing: compute mixing result without modifying state
    fn compute_mixing(&self, chromosome_idx: usize, rng: &mut Mt) -> MixingResult {
        // Match C++ line 295: int startNode = myRand.uniformInt(0, ell - 1);
        let start_gene = rng.uniform_int(0, self.problem_size - 1);

        // Build mask
        let chromosome = self.population[chromosome_idx].clone();
        let mut mask = self.find_mask(&chromosome, start_gene, rng);

        // Calculate sizes
        let size = self.calculate_mask_size(&chromosome, &mask);
        let size_bound = self.find_size_bound(&chromosome, start_gene, size, rng);

        let final_size = size.min(size_bound);
        mask.truncate(final_size);

        // Compute mixing result
        self.compute_mixing_with_mask(chromosome_idx, &mask)
    }

    /// Compute mixing with a mask (returns result without modifying state)
    fn compute_mixing_with_mask(&self, chromosome_idx: usize, mask: &[usize]) -> MixingResult {
        let original_key = self.population[chromosome_idx].key();
        // Population should already have cached fitness from update_statistics()
        let original_fitness = self.population[chromosome_idx]
            .fitness()
            .expect("Chromosome should have cached fitness");
        let mut evaluations = 0;

        // C++ line 448: for (size_t ub = 1; ub <= mask.size(); ++ub)
        // Try each mask size from 1 to mask.len()
        for ub in 1..=mask.len() {
            // C++ line 451: trial = ch; (fresh copy for each ub)
            let mut trial = self.population[chromosome_idx].clone();

            // C++ line 455: flip first ub genes in mask
            for &gene in mask.iter().take(ub) {
                trial.flip_gene(gene, &self.zobrist);
            }

            // C++ line 478: if (isInP(trial)) break;
            if self.population_hash.contains_key(&trial.key()) {
                break;
            }

            // C++ line 484: evaluate and check acceptance
            evaluations += 1;
            let trial_fitness = trial.evaluate(self.fitness_fn);

            // C++ line 488: if (trial.getFitness() >= ch.getFitness() - EPSILON)
            if trial_fitness >= original_fitness - EPSILON {
                // Return success result
                return MixingResult {
                    chromosome_idx,
                    new_chromosome: Some(trial),
                    new_fitness: Some(trial_fitness),
                    old_key: Some(original_key),
                    evaluations,
                };
            }
        }

        // Return no-change result
        MixingResult {
            chromosome_idx,
            new_chromosome: None,
            new_fitness: None,
            old_key: None,
            evaluations,
        }
    }

    /// Perform mixing operations
    fn mixing(&mut self) {
        // Tournament selection
        self.tournament_selection();

        // Build statistics from selected population
        self.build_fast_counting();

        // Build linkage graphs
        self.build_graph();

        // Determine number of mixing rounds
        let rounds = if self.problem_size > 50 {
            self.problem_size / 50
        } else {
            1
        };

        // Apply restricted mixing
        for _round in 0..rounds {
            // Shuffle population order
            self.population_order.shuffle(&mut self.rng);

            // Generate seeds for parallel mixing
            let seeds: Vec<u32> = (0..self.population_order.len())
                .map(|_| self.rng.next_u32())
                .collect();

            // Parallel: compute all mixing results
            let results: Vec<MixingResult> = self
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
                        self.population_hash.remove(&old_key);
                    }

                    // Insert new key with fitness
                    if let Some(new_fitness) = result.new_fitness {
                        self.population_hash
                            .insert(new_chromosome.key(), new_fitness);
                    }

                    // Update population
                    self.population[result.chromosome_idx] = new_chromosome;
                }
            }
        }
    }

    // Getters
    pub fn generation(&self) -> usize {
        self.generation
    }

    pub fn num_evaluations(&self) -> usize {
        self.num_evaluations.load(Ordering::Relaxed)
    }

    pub fn population_size(&self) -> usize {
        self.population_size
    }

    pub fn seed(&self) -> Option<u64> {
        // Seed is not stored after initialization
        None
    }

    pub fn problem_size(&self) -> usize {
        self.problem_size
    }

    pub fn max_generations(&self) -> Option<usize> {
        self.max_generations
    }

    pub fn max_evaluations(&self) -> Option<usize> {
        self.max_evaluations
    }

    pub fn use_ghc(&self) -> bool {
        self.use_ghc
    }

    pub fn best_fitness(&self) -> f64 {
        self.statistics.max()
    }

    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }

    /// Get the learned linkage structure
    ///
    /// Returns edges in the linkage graph as (gene_i, gene_j, weight) tuples.
    pub fn linkage(&self) -> Vec<(usize, usize, f64)> {
        let mut edges = Vec::new();
        for i in 0..self.problem_size {
            for j in (i + 1)..self.problem_size {
                let (weight, _) = self.linkage_graph.read(i, j);
                if weight > 0.0 {
                    edges.push((i, j, weight));
                }
            }
        }
        // Sort by weight descending
        edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        edges
    }

    /// Get population fitness values for debugging
    /// Returns vector of (index, fitness) pairs
    pub fn get_population_fitness(&self) -> Vec<(usize, f64)> {
        self.population
            .iter()
            .enumerate()
            .filter_map(|(i, chr)| chr.fitness().map(|f| (i, f)))
            .collect()
    }

    /// Get specific chromosome's genes as bit string for debugging
    pub fn get_chromosome_bits(&self, index: usize) -> Option<String> {
        if index >= self.population.len() {
            return None;
        }

        let chr = &self.population[index];
        let mut bits = String::with_capacity(chr.length());
        for i in 0..chr.length() {
            bits.push(if chr.get_gene(i) { '1' } else { '0' });
        }
        Some(bits)
    }

    pub fn best_solution(&self) -> &Chromosome {
        &self.population[self.best_index]
    }

    /// Get linkage value for debugging
    pub fn get_linkage(&self, i: usize, j: usize) -> (f64, f64) {
        self.linkage_graph.read(i, j)
    }

    /// Get selection indices for debugging
    pub fn get_selection_indices(&self) -> &[usize] {
        &self.selection_indices
    }

    /// Manually call tournament selection for debugging
    pub fn debug_tournament_selection(&mut self) {
        self.tournament_selection();
    }
}
