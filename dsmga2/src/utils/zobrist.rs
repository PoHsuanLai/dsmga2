use rand_mt::Mt;

/// Zobrist hashing keys for efficient chromosome hashing
/// Uses random 64-bit integers, one per gene position
pub struct ZobristKeys {
    keys: Vec<u64>,
}

impl ZobristKeys {
    /// Create new Zobrist keys for a given problem size
    /// Generates keys deterministically from the seed for reproducibility
    pub fn new(problem_size: usize, seed: u64) -> Self {
        let mut rng = Mt::new(seed as u32);

        // Generate deterministic random keys
        let keys = (0..problem_size)
            .map(|_| {
                // Combine two 32-bit random numbers to make a 64-bit key
                let high = (rng.next_u32() as u64) << 32;
                let low = rng.next_u32() as u64;
                high | low
            })
            .collect();

        Self { keys }
    }

    /// Get the key for a specific gene index
    #[inline(always)]
    pub fn get(&self, index: usize) -> u64 {
        self.keys[index]
    }

    /// Get the number of keys
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}
