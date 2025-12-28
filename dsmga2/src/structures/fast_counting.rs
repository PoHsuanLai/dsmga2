/// Efficient bit counting structure for population statistics
/// Stores genes as packed u64 words for fast XOR and bit counting operations
#[derive(Debug, Clone)]
pub struct FastCounting {
    genes: Vec<u64>,
    length: usize,       // Number of bits
    length_words: usize, // Number of u64 words
}

impl FastCounting {
    pub fn new(length: usize) -> Self {
        let length_words = length.div_ceil(64); // Ceiling division
        Self {
            genes: vec![0; length_words],
            length,
            length_words,
        }
    }

    /// Get bit value at index
    #[inline(always)]
    pub fn get(&self, index: usize) -> bool {
        debug_assert!(index < self.length);
        let word = index >> 6; // Faster than / 64
        let bit = index & 63; // Faster than % 64
        (self.genes[word] & (1u64 << bit)) != 0
    }

    /// Set bit value at index
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: bool) {
        debug_assert!(index < self.length);
        let word = index >> 6; // Faster than / 64
        let bit = index & 63; // Faster than % 64

        if value {
            self.genes[word] |= 1u64 << bit;
        } else {
            self.genes[word] &= !(1u64 << bit);
        }
    }

    /// Count number of 1 bits
    /// Optimized for compiler auto-vectorization
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.genes
            .iter()
            .map(|&word| word.count_ones() as usize)
            .sum()
    }

    /// Count XOR differences with another FastCounting
    /// Optimized for SIMD auto-vectorization
    #[inline]
    pub fn count_xor(&self, other: &Self) -> usize {
        debug_assert_eq!(self.length_words, other.length_words);

        // Rust's optimizer can auto-vectorize this safe code effectively
        self.genes
            .iter()
            .zip(other.genes.iter())
            .map(|(&a, &b)| (a ^ b).count_ones() as usize)
            .sum()
    }

    pub fn length(&self) -> usize {
        self.length
    }

    pub fn length_words(&self) -> usize {
        self.length_words
    }

    /// Get raw gene data
    pub fn genes(&self) -> &[u64] {
        &self.genes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_counting() {
        let mut fc = FastCounting::new(100);

        fc.set(0, true);
        fc.set(50, true);
        fc.set(99, true);

        assert!(fc.get(0));
        assert!(fc.get(50));
        assert!(fc.get(99));
        assert!(!fc.get(1));

        assert_eq!(fc.count_ones(), 3);
    }

    #[test]
    fn test_count_xor() {
        let mut fc1 = FastCounting::new(100);
        let mut fc2 = FastCounting::new(100);

        fc1.set(0, true);
        fc1.set(50, true);

        fc2.set(0, true);
        fc2.set(99, true);

        // Differ at positions 50 and 99
        assert_eq!(fc1.count_xor(&fc2), 2);
    }
}
