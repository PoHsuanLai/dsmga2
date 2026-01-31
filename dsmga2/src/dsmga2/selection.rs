use crate::chromosome::Chromosome;
use crate::fitness::FitnessFunction;
use crate::structures::FastCounting;
use rand::seq::SliceRandom;
use rand_mt::Mt;

/// Working buffers for selection and mixing
pub(crate) struct SelectionBuffers {
    pub(crate) indices: Vec<usize>,
    pub(crate) population_order: Vec<usize>,
}

impl SelectionBuffers {
    pub(crate) fn new(population_size: usize) -> Self {
        Self {
            indices: vec![0; population_size],
            population_order: (0..population_size).collect(),
        }
    }
}

/// Tournament selection
/// Matches C++ implementation exactly (sequential, not parallel)
pub(crate) fn tournament_selection(
    population: &mut [Chromosome],
    selection_pressure: usize,
    fitness_fn: &dyn FitnessFunction,
    buffers: &mut SelectionBuffers,
    rng: &mut Mt,
) {
    let population_size = population.len();

    // Ensure all fitness values are cached
    for chromosome in population.iter_mut() {
        chromosome.evaluate(fitness_fn);
    }

    let mut candidates = vec![0; selection_pressure * population_size];

    // Generate random indices for tournament (without replacement per pressure level)
    for i in 0..selection_pressure {
        let start = i * population_size;
        // Create range [0, 1, ..., population_size-1] and shuffle
        let mut indices: Vec<usize> = (0..population_size).collect();
        indices.shuffle(rng);
        candidates[start..start + population_size].copy_from_slice(&indices);
    }

    // Run tournaments sequentially to match C++ exactly
    buffers.indices.clear();
    for i in 0..population_size {
        let mut winner = 0;
        let mut winner_fitness = f64::NEG_INFINITY;

        for j in 0..selection_pressure {
            let candidate = candidates[selection_pressure * i + j];
            let fitness = population[candidate].fitness().unwrap();

            if fitness > winner_fitness {
                winner = candidate;
                winner_fitness = fitness;
            }
        }

        buffers.indices.push(winner);
    }
}

/// Build fast counting structure from population (parallelized)
pub(crate) fn build_fast_counting(
    population: &[Chromosome],
    selection_indices: &[usize],
    fast_counting: &mut [FastCounting],
) {
    use rayon::prelude::*;

    let population_size = population.len();

    // Process each gene position in parallel
    fast_counting
        .par_iter_mut()
        .enumerate()
        .for_each(|(j, fast_count)| {
            for (i, &idx) in selection_indices.iter().enumerate().take(population_size) {
                let value = population[idx].get_gene(j);
                fast_count.set(i, value);
            }
        });
}
