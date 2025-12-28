use crate::fitness::FitnessFunction;
use crate::utils::{RngExt, ZobristKeys};
use rand::seq::SliceRandom;
use rand_mt::Mt;

/// A chromosome represents a candidate solution as a bit string
#[derive(Debug, Clone)]
pub struct Chromosome {
    /// Bit-packed genes stored as u64 words
    genes: Vec<u64>,
    /// Number of genes (bits)
    length: usize,
    /// Number of u64 words needed
    length_words: usize,
    /// Cached fitness value
    fitness: Option<f64>,
    /// Zobrist hash key for fast equality checking
    key: u64,
}

impl Chromosome {
    /// Create a new chromosome with all genes set to 0
    pub fn new(length: usize, _zobrist: &ZobristKeys) -> Self {
        let length_words = length.div_ceil(64);
        Self {
            genes: vec![0; length_words],
            length,
            length_words,
            fitness: None,
            key: 0,
        }
    }

    /// Create a new chromosome with random gene values
    pub fn new_random(length: usize, zobrist: &ZobristKeys, rng: &mut Mt) -> Self {
        let mut chromosome = Self::new(length, zobrist);

        for i in 0..length {
            if rng.flip() {
                chromosome.set_gene(i, true, zobrist);
            }
        }

        chromosome
    }

    /// Create a chromosome from a bit vector (for checkpoint restore)
    pub fn from_bits(bits: &[bool], zobrist: &ZobristKeys) -> Self {
        let length = bits.len();
        let mut chromosome = Self::new(length, zobrist);

        bits.iter()
            .enumerate()
            .filter(|(_, &bit)| bit)
            .for_each(|(i, _)| chromosome.set_gene(i, true, zobrist));

        chromosome
    }

    /// Get gene value at index
    #[inline(always)]
    pub fn get_gene(&self, index: usize) -> bool {
        debug_assert!(index < self.length);
        let word = index >> 6; // Faster than / 64
        let bit = index & 63; // Faster than % 64
        (self.genes[word] & (1u64 << bit)) != 0
    }

    /// Set gene value at index
    #[inline(always)]
    pub fn set_gene(&mut self, index: usize, value: bool, zobrist: &ZobristKeys) {
        debug_assert!(index < self.length);

        let current = self.get_gene(index);
        if current == value {
            return; // No change needed
        }

        let word = index >> 6; // Faster than / 64
        let bit = index & 63; // Faster than % 64

        if value {
            self.genes[word] |= 1u64 << bit;
        } else {
            self.genes[word] &= !(1u64 << bit);
        }

        // Update Zobrist hash
        self.key ^= zobrist.get(index);
        // Invalidate cached fitness
        self.fitness = None;
    }

    /// Flip gene at index
    #[inline(always)]
    pub fn flip_gene(&mut self, index: usize, zobrist: &ZobristKeys) {
        debug_assert!(index < self.length);

        let word = index >> 6; // Faster than / 64
        let bit = index & 63; // Faster than % 64

        self.genes[word] ^= 1u64 << bit;
        self.key ^= zobrist.get(index);
        self.fitness = None;
    }

    /// Evaluate fitness using the provided function
    #[inline]
    pub fn evaluate(&mut self, fitness_fn: &dyn FitnessFunction) -> f64 {
        if let Some(f) = self.fitness {
            return f;
        }

        let f = fitness_fn.evaluate(self);
        self.fitness = Some(f);
        f
    }

    /// Get cached fitness (if available)
    #[inline(always)]
    pub fn fitness(&self) -> Option<f64> {
        self.fitness
    }

    /// Get the Zobrist hash key
    #[inline(always)]
    pub fn key(&self) -> u64 {
        self.key
    }

    /// Get the length (number of genes)
    pub fn length(&self) -> usize {
        self.length
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Count number of 1 bits
    /// Optimized for compiler auto-vectorization
    #[inline]
    pub fn count_ones(&self) -> usize {
        let mut count = 0;

        // Manual loop helps compiler auto-vectorize better
        for &word in &self.genes {
            count += word.count_ones() as usize;
        }

        count
    }

    /// Get raw gene data (for bulk operations like FastCounting)
    pub fn genes(&self) -> &[u64] {
        &self.genes
    }

    /// Get number of words
    pub fn length_words(&self) -> usize {
        self.length_words
    }

    /// Greedy Hill Climbing (GHC) - local search
    /// Tries flipping each bit in random order, keeps improvements
    ///
    /// # Arguments
    /// * `early_stop_patience` - Stop after this many consecutive non-improvements (None = try all bits)
    ///
    /// Returns (improved, num_evaluations)
    pub fn greedy_hill_climb(
        &mut self,
        zobrist: &ZobristKeys,
        fitness_fn: &dyn FitnessFunction,
        rng: &mut Mt,
        early_stop_patience: Option<usize>,
    ) -> (bool, usize) {
        // Create random order for trying bit flips
        let mut order: Vec<usize> = (0..self.length).collect();
        order.shuffle(rng);

        let mut improved = false;
        let mut num_evals = 0;
        let mut no_improvement_count = 0;

        for &index in &order {
            let (flipped, evals) = self.try_flipping(index, zobrist, fitness_fn);
            num_evals += evals;

            if flipped {
                improved = true;
                no_improvement_count = 0;
            } else {
                no_improvement_count += 1;

                // Early stopping if patience limit reached
                if let Some(patience) = early_stop_patience {
                    if no_improvement_count >= patience {
                        break;
                    }
                }
            }
        }

        (improved, num_evals)
    }

    /// Try flipping a bit - keep it if fitness improves
    /// Returns (flipped, num_evaluations)
    #[inline]
    fn try_flipping(
        &mut self,
        index: usize,
        zobrist: &ZobristKeys,
        fitness_fn: &dyn FitnessFunction,
    ) -> (bool, usize) {
        const EPSILON: f64 = 1e-8;

        let old_fitness = self.evaluate(fitness_fn);

        // Flip the bit
        self.flip_gene(index, zobrist);

        let new_fitness = self.evaluate(fitness_fn);

        // Keep flip if it improves fitness (1 evaluation)
        if new_fitness > old_fitness + EPSILON {
            (true, 1)
        } else {
            // Revert the flip (no additional evaluation - we restore cached fitness)
            self.flip_gene(index, zobrist);
            self.fitness = Some(old_fitness);
            (false, 1)
        }
    }

    /// Invalidate cached fitness
    pub fn invalidate_fitness(&mut self) {
        self.fitness = None;
    }
}

impl PartialEq for Chromosome {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Chromosome {}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyFitness;
    impl FitnessFunction for DummyFitness {
        fn evaluate(&self, chromosome: &Chromosome) -> f64 {
            chromosome.count_ones() as f64
        }

        fn optimum(&self, _length: usize) -> f64 {
            100.0
        }
    }

    #[test]
    fn test_chromosome_basic() {
        let zobrist = ZobristKeys::new(10, 42);
        let mut chromosome = Chromosome::new(10, &zobrist);

        assert_eq!(chromosome.length(), 10);
        assert_eq!(chromosome.count_ones(), 0);

        chromosome.set_gene(0, true, &zobrist);
        chromosome.set_gene(5, true, &zobrist);

        assert!(chromosome.get_gene(0));
        assert!(chromosome.get_gene(5));
        assert!(!chromosome.get_gene(1));
        assert_eq!(chromosome.count_ones(), 2);
    }

    #[test]
    fn test_chromosome_flip() {
        let zobrist = ZobristKeys::new(10, 42);
        let mut chromosome = Chromosome::new(10, &zobrist);

        chromosome.flip_gene(3, &zobrist);
        assert!(chromosome.get_gene(3));

        chromosome.flip_gene(3, &zobrist);
        assert!(!chromosome.get_gene(3));
    }

    #[test]
    fn test_chromosome_fitness() {
        let zobrist = ZobristKeys::new(10, 42);
        let mut chromosome = Chromosome::new(10, &zobrist);
        let fitness_fn = DummyFitness;

        chromosome.set_gene(0, true, &zobrist);
        chromosome.set_gene(5, true, &zobrist);

        let fitness = chromosome.evaluate(&fitness_fn);
        assert_eq!(fitness, 2.0);

        // Should be cached
        assert_eq!(chromosome.fitness(), Some(2.0));
    }
}
