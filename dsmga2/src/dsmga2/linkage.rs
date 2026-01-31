use crate::structures::{FastCounting, TriMatrix};
use rayon::prelude::*;

const EPSILON: f64 = 1e-8;

/// Linkage learning structures
pub(crate) struct LinkageModel {
    pub(crate) graph: TriMatrix,
    pub(crate) graph_size: TriMatrix,
    pub(crate) fast_counting: Vec<FastCounting>,
}

impl LinkageModel {
    /// Create a new linkage model
    pub(crate) fn new(problem_size: usize, population_size: usize) -> Self {
        let graph = TriMatrix::new(problem_size);
        let graph_size = TriMatrix::new(problem_size);
        let fast_counting = (0..problem_size)
            .map(|_| FastCounting::new(population_size))
            .collect();

        Self {
            graph,
            graph_size,
            fast_counting,
        }
    }

    /// Build linkage graph using two-edge model (parallelized)
    pub(crate) fn build_graph(&mut self, population_size: usize, problem_size: usize) {
        // Count ones for each gene (parallelized)
        let ones_count: Vec<usize> = (0..problem_size)
            .into_par_iter()
            .map(|i| self.fast_counting[i].count_ones())
            .collect();

        // Compute all (i,j) pairs to process
        let pairs: Vec<(usize, usize)> = (0..problem_size)
            .flat_map(|i| ((i + 1)..problem_size).map(move |j| (i, j)))
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
                let n00 = population_size as f64 - n01 - n10 - n11;

                let p00 = n00 / population_size as f64;
                let p01 = n01 / population_size as f64;
                let p10 = n10 / population_size as f64;
                let p11 = n11 / population_size as f64;

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
            self.graph.write(i, j, values);
        }
    }

    /// Build graph_size using mutual information (parallelized)
    pub(crate) fn build_graph_size(&mut self, population_size: usize, problem_size: usize) {
        // Count ones for each gene (parallelized)
        let ones_count: Vec<usize> = (0..problem_size)
            .into_par_iter()
            .map(|i| self.fast_counting[i].count_ones())
            .collect();

        // Compute all (i,j) pairs
        let pairs: Vec<(usize, usize)> = (0..problem_size)
            .flat_map(|i| ((i + 1)..problem_size).map(move |j| (i, j)))
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
                let n00 = population_size as f64 - n01 - n10 - n11;

                let p00 = n00 / population_size as f64;
                let p01 = n01 / population_size as f64;
                let p10 = n10 / population_size as f64;
                let p11 = n11 / population_size as f64;

                let p0x = p00 + p01;
                let p1x = p10 + p11;
                let px0 = p00 + p10;
                let px1 = p01 + p11;

                // Compute mutual information
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
            self.graph_size.write(i, j, values);
        }
    }

    /// Get linkage edges for visualization
    pub(crate) fn get_linkage_edges(&self, problem_size: usize) -> Vec<(usize, usize, f64)> {
        let mut edges: Vec<(usize, usize, f64)> = (0..problem_size)
            .flat_map(|i| {
                ((i + 1)..problem_size).filter_map(move |j| {
                    let (weight, _) = self.graph.read(i, j);
                    (weight > 0.0).then_some((i, j, weight))
                })
            })
            .collect();
        // Sort by weight descending
        edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        edges
    }
}
