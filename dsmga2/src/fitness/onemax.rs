use crate::chromosome::Chromosome;
use crate::fitness::FitnessFunction;

/// OneMax fitness function: counts the number of 1s in the bit string
#[derive(Debug, Clone, Copy)]
pub struct OneMax;

impl FitnessFunction for OneMax {
    fn evaluate(&self, chromosome: &Chromosome) -> f64 {
        chromosome.count_ones() as f64
    }

    fn optimum(&self, length: usize) -> f64 {
        length as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::ZobristKeys;

    #[test]
    fn test_onemax() {
        let zobrist = ZobristKeys::new(10, 42);
        let mut chromosome = Chromosome::new(10, &zobrist);
        let fitness_fn = OneMax;

        chromosome.set_gene(0, true, &zobrist);
        chromosome.set_gene(5, true, &zobrist);
        chromosome.set_gene(9, true, &zobrist);

        assert_eq!(fitness_fn.evaluate(&chromosome), 3.0);
        assert_eq!(fitness_fn.optimum(10), 10.0);
    }
}
