use rand_mt::Mt;

/// Extension trait for Mt19937 to provide DSMGA2-specific RNG methods
pub trait RngExt {
    /// Generate uniform integer in range [a, b] (inclusive)
    fn uniform_int(&mut self, a: usize, b: usize) -> usize;

    /// Generate a random boolean (coin flip)
    fn flip(&mut self) -> bool;
}

impl RngExt for Mt {
    fn uniform_int(&mut self, a: usize, b: usize) -> usize {
        // Match C++ behavior: return (a + (int) (genrand_real2() * (b - a + 1)));
        // genrand_real2() returns [0,1) with 32-bit resolution
        let val = self.next_u32();
        let real = (val as f64) / 4294967296.0; // Convert to [0,1)
        a + (real * ((b - a + 1) as f64)) as usize
    }

    fn flip(&mut self) -> bool {
        // Simple coin flip using next_u32
        self.next_u32() & 1 == 0
    }
}
