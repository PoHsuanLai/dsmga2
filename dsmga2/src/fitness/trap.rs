use crate::chromosome::Chromosome;
use crate::fitness::FitnessFunction;

/// MK-Trap function with configurable k (trap size)
#[derive(Debug, Clone)]
pub struct MkTrap {
    k: usize,
}

impl MkTrap {
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "Trap size k must be greater than 0");
        Self { k }
    }

    fn trap_function(&self, ones: usize) -> f64 {
        // Match C++ implementation: fHigh=1.0, fLow=0.8
        // trap(u, 1.0, 0.8, k) = 1.0 if u==k, else 0.8 - u*0.8/(k-1)
        const F_HIGH: f64 = 1.0;
        const F_LOW: f64 = 0.8;

        if ones == self.k {
            F_HIGH
        } else {
            F_LOW - (ones as f64) * F_LOW / ((self.k - 1) as f64)
        }
    }
}

impl FitnessFunction for MkTrap {
    fn evaluate(&self, chromosome: &Chromosome) -> f64 {
        let length = chromosome.length();
        assert_eq!(
            length % self.k,
            0,
            "Chromosome length must be divisible by trap size k"
        );

        let num_blocks = length / self.k;

        (0..num_blocks)
            .map(|block| {
                let start = block * self.k;
                let ones = (start..(start + self.k))
                    .filter(|&i| chromosome.get_gene(i))
                    .count();
                self.trap_function(ones)
            })
            .sum()
    }

    fn optimum(&self, length: usize) -> f64 {
        // Optimum is number of blocks (each contributing 1.0)
        (length / self.k) as f64
    }
}

/// Folded Trap function (6-bit blocks)
#[derive(Debug, Clone, Copy)]
pub struct FoldedTrap;

impl FoldedTrap {
    const K: usize = 6;

    fn trap_function(&self, ones: usize) -> f64 {
        if ones == Self::K {
            Self::K as f64
        } else {
            (Self::K - 1 - ones) as f64
        }
    }
}

impl FitnessFunction for FoldedTrap {
    fn evaluate(&self, chromosome: &Chromosome) -> f64 {
        let length = chromosome.length();
        assert_eq!(
            length % Self::K,
            0,
            "Chromosome length must be divisible by 6"
        );

        let num_blocks = length / Self::K;

        (0..num_blocks)
            .map(|block| {
                let start = block * Self::K;
                let ones = (start..(start + Self::K))
                    .filter(|&i| chromosome.get_gene(i))
                    .count();
                self.trap_function(ones)
            })
            .sum()
    }

    fn optimum(&self, length: usize) -> f64 {
        (length / Self::K) as f64 * Self::K as f64
    }
}

/// Cyclic Trap function with overlap
#[derive(Debug, Clone)]
pub struct CyclicTrap {
    k: usize,
}

impl CyclicTrap {
    pub fn new(k: usize) -> Self {
        assert!(k > 1, "Trap size k must be greater than 1");
        Self { k }
    }

    fn trap_function(&self, ones: usize) -> f64 {
        // Match C++ implementation: fHigh=1.0, fLow=0.8
        const F_HIGH: f64 = 1.0;
        const F_LOW: f64 = 0.8;

        if ones == self.k {
            F_HIGH
        } else {
            F_LOW - (ones as f64) * F_LOW / ((self.k - 1) as f64)
        }
    }
}

impl FitnessFunction for CyclicTrap {
    fn evaluate(&self, chromosome: &Chromosome) -> f64 {
        let length = chromosome.length();
        let step = self.k - 1; // Overlap by 1

        (0..length)
            .step_by(step)
            .map(|block_start| {
                let ones = (0..self.k)
                    .filter(|&offset| {
                        let index = (block_start + offset) % length;
                        chromosome.get_gene(index)
                    })
                    .count();
                self.trap_function(ones)
            })
            .sum()
    }

    fn optimum(&self, length: usize) -> f64 {
        // Optimum is number of blocks (each contributing 1.0)
        let step = self.k - 1;
        let num_blocks = length / step;
        num_blocks as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::ZobristKeys;

    #[test]
    fn test_mk_trap() {
        let zobrist = ZobristKeys::new(15, 42);
        let mut chromosome = Chromosome::new(15, &zobrist);
        let fitness_fn = MkTrap::new(5);

        // Set all bits in first block (optimal)
        for i in 0..5 {
            chromosome.set_gene(i, true, &zobrist);
        }

        // Set no bits in second block (worst)
        // Set 2 bits in third block (deceptive)
        chromosome.set_gene(10, true, &zobrist);
        chromosome.set_gene(11, true, &zobrist);

        let fitness = fitness_fn.evaluate(&chromosome);
        // Using trap(u, fHigh=1.0, fLow=0.8, k=5):
        // Block 1: 5 ones = 1.0
        // Block 2: 0 ones = 0.8
        // Block 3: 2 ones = 0.8 - 2*0.8/4 = 0.4
        assert_eq!(fitness, 2.2);
    }

    #[test]
    fn test_folded_trap() {
        let zobrist = ZobristKeys::new(6, 42);
        let mut chromosome = Chromosome::new(6, &zobrist);
        let fitness_fn = FoldedTrap;

        // All ones (optimal)
        for i in 0..6 {
            chromosome.set_gene(i, true, &zobrist);
        }

        assert_eq!(fitness_fn.evaluate(&chromosome), 6.0);
        assert_eq!(fitness_fn.optimum(6), 6.0);
    }
}
