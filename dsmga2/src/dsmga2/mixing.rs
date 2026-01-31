use crate::chromosome::Chromosome;
use crate::fitness::FitnessFunction;
use crate::structures::{SimpleSet, TriMatrix};
use crate::utils::ZobristKeys;
use dashmap::DashMap;
use rand::seq::SliceRandom;
use rand_mt::Mt;
use smallvec::SmallVec;

pub(crate) const EPSILON: f64 = 1e-8;

/// Result of a mixing operation
pub(crate) struct MixingResult {
    pub(crate) chromosome_idx: usize,
    pub(crate) new_chromosome: Option<Chromosome>,
    pub(crate) new_fitness: Option<f64>,
    pub(crate) old_key: Option<u64>,
    pub(crate) evaluations: usize,
}

/// Find linkage mask starting from a gene
pub(crate) fn find_mask(
    chromosome: &Chromosome,
    start_gene: usize,
    problem_size: usize,
    linkage_graph: &TriMatrix,
    rng: &mut Mt,
) -> SmallVec<[usize; 64]> {
    let mut mask = SmallVec::new();
    let mut rest = SimpleSet::new(problem_size);

    // Generate random gene order
    let mut gene_order: Vec<usize> = (0..problem_size).collect();
    gene_order.shuffle(rng);

    // Initialize mask with start gene, remaining genes go to rest set
    mask.push(start_gene);
    for &gene in gene_order.iter().filter(|&&g| g != start_gene) {
        rest.insert(gene);
    }

    // Build connection scores
    let mut connections = vec![0.0; problem_size];

    for gene in &rest {
        let (linkage_same, linkage_diff) = linkage_graph.read(start_gene, gene);
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
        // Select gene with strongest connection
        let best_gene = (&rest)
            .into_iter()
            .max_by(|&a, &b| connections[a].partial_cmp(&connections[b]).unwrap())
            .expect("rest is not empty");

        rest.remove(best_gene);
        mask.push(best_gene);

        // Update connections for remaining genes
        for gene in &rest {
            let (linkage_same, linkage_diff) = linkage_graph.read(best_gene, gene);
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
pub(crate) fn find_size_bound(
    chromosome: &Chromosome,
    start_gene: usize,
    max_size: usize,
    problem_size: usize,
    linkage_graph_size: &TriMatrix,
) -> usize {
    // Build a NEW mask using linkage_graph_size with a size bound
    let mut mask_size = SmallVec::<[usize; 64]>::new();
    let mut rest = SimpleSet::new(problem_size);

    // Initialize mask with start gene
    mask_size.push(start_gene);
    for gene in (0..problem_size).filter(|&g| g != start_gene) {
        rest.insert(gene);
    }

    let mut connections = vec![0.0; problem_size];

    // Initialize connections with linkage to start_gene
    for gene in &rest {
        let (linkage_same, linkage_diff) = linkage_graph_size.read(start_gene, gene);
        let start_val = chromosome.get_gene(start_gene);
        let gene_val = chromosome.get_gene(gene);

        connections[gene] = if start_val == gene_val {
            linkage_same
        } else {
            linkage_diff
        };
    }

    // Greedily add genes up to max_size
    let mut bound = max_size.saturating_sub(1);
    while !rest.is_empty() && bound > 0 {
        bound -= 1;

        // Select gene with strongest connection
        let best_gene = (&rest)
            .into_iter()
            .max_by(|&a, &b| connections[a].partial_cmp(&connections[b]).unwrap())
            .expect("rest is not empty");

        rest.remove(best_gene);
        mask_size.push(best_gene);

        // Update connections with linkage to newly added gene
        for gene in &rest {
            let (linkage_same, linkage_diff) = linkage_graph_size.read(best_gene, gene);
            let best_val = chromosome.get_gene(best_gene);
            let gene_val = chromosome.get_gene(gene);

            connections[gene] += if best_val == gene_val {
                linkage_same
            } else {
                linkage_diff
            };
        }
    }

    mask_size.len()
}

/// Calculate mask size (number of genes that would filter out all population)
pub(crate) fn calculate_mask_size(
    chromosome: &Chromosome,
    mask: &[usize],
    population: &[Chromosome],
) -> usize {
    let mut candidates = SimpleSet::new(population.len());
    for i in 0..population.len() {
        candidates.insert(i);
    }

    let mut size = 0;
    for &gene in mask.iter() {
        let allele = chromosome.get_gene(gene);

        // Collect indices to remove
        let to_remove: Vec<usize> = (&candidates)
            .into_iter()
            .filter(|&idx| population[idx].get_gene(gene) == allele)
            .collect();

        for idx in to_remove {
            candidates.remove(idx);
        }

        if candidates.is_empty() {
            break;
        }

        size += 1;
    }

    size
}

/// Compute mixing with a mask
pub(crate) fn compute_mixing_with_mask(
    chromosome_idx: usize,
    mask: &[usize],
    population: &[Chromosome],
    population_hash: &DashMap<u64, f64>,
    zobrist: &ZobristKeys,
    fitness_fn: &dyn FitnessFunction,
) -> MixingResult {
    let original_key = population[chromosome_idx].key();
    let original_fitness = population[chromosome_idx]
        .fitness()
        .expect("Chromosome should have cached fitness");
    let mut evaluations = 0;

    // Try each mask size from 1 to mask.len()
    for ub in 1..=mask.len() {
        let mut trial = population[chromosome_idx].clone();

        // Flip first ub genes in mask
        for &gene in mask.iter().take(ub) {
            trial.flip_gene(gene, zobrist);
        }

        // Skip if trial already exists in population
        if population_hash.contains_key(&trial.key()) {
            break;
        }

        // Evaluate and check for improvement
        evaluations += 1;
        let trial_fitness = trial.evaluate(fitness_fn);

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
