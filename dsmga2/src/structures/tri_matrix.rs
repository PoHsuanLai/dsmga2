use std::cell::UnsafeCell;

/// Triangular matrix for storing pairwise linkage information
/// For the two-edge model, stores linkage_same and linkage_diff separately
/// Split storage improves cache locality
/// Uses UnsafeCell for lock-free parallel writes (safe because each index is written by only one thread)
#[derive(Debug)]
pub struct TriMatrix {
    size: usize,
    // Separate vectors for better cache locality
    // Linkage when genes have same value (00 or 11)
    linkage_same: Vec<UnsafeCell<f64>>,
    // Linkage when genes have different values (01 or 10)
    linkage_diff: Vec<UnsafeCell<f64>>,
}

// SAFETY: TriMatrix is Sync because:
// 1. In parallel writes (build_graph), each thread writes to a unique (i,j) pair
// 2. No two threads ever write to the same index simultaneously
// 3. Reads only happen after all writes are complete (in find_mask)
unsafe impl Sync for TriMatrix {}

impl Clone for TriMatrix {
    fn clone(&self) -> Self {
        let linkage_same: Vec<UnsafeCell<f64>> = self
            .linkage_same
            .iter()
            .map(|cell| {
                // SAFETY: We're cloning, so no other threads can access this data
                unsafe { UnsafeCell::new(*cell.get()) }
            })
            .collect();
        let linkage_diff: Vec<UnsafeCell<f64>> = self
            .linkage_diff
            .iter()
            .map(|cell| {
                // SAFETY: We're cloning, so no other threads can access this data
                unsafe { UnsafeCell::new(*cell.get()) }
            })
            .collect();
        Self {
            size: self.size,
            linkage_same,
            linkage_diff,
        }
    }
}

impl TriMatrix {
    pub fn new(size: usize) -> Self {
        let capacity = if size > 0 { (size * (size - 1)) / 2 } else { 0 };
        Self {
            size,
            linkage_same: (0..capacity).map(|_| UnsafeCell::new(0.0)).collect(),
            linkage_diff: (0..capacity).map(|_| UnsafeCell::new(0.0)).collect(),
        }
    }

    /// Convert (i, j) coordinates to linear index
    /// Assumes i > j (upper triangle stored as lower triangle)
    #[inline(always)]
    fn index(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.size && j < self.size);
        debug_assert!(i != j, "Cannot access diagonal elements");

        let (row, col) = if i > j { (i, j) } else { (j, i) };
        // Row starts at: 0 + 1 + 2 + ... + (row-1) = row*(row-1)/2
        // Use bit shift for division by 2
        ((row * (row - 1)) >> 1) + col
    }

    /// Write a value to position (i, j)
    /// Lock-free parallel writes - safe because each (i,j) pair is written by only one thread
    #[inline]
    pub fn write(&self, i: usize, j: usize, value: (f64, f64)) {
        if i == j {
            return; // Ignore diagonal
        }
        let idx = self.index(i, j);

        // SAFETY: In build_graph(), each thread processes unique (i,j) pairs
        // so there's no data race - only one thread ever writes to this index
        unsafe {
            *self.linkage_same[idx].get() = value.0;
            *self.linkage_diff[idx].get() = value.1;
        }
    }

    /// Read a value from position (i, j)
    /// Returns (1.0, 1.0) for diagonal elements
    #[inline]
    pub fn read(&self, i: usize, j: usize) -> (f64, f64) {
        if i == j {
            return (1.0, 1.0);
        }
        let idx = self.index(i, j);

        // SAFETY: Reads only happen after build_graph() completes
        // so all writes are done before any reads
        unsafe { (*self.linkage_same[idx].get(), *self.linkage_diff[idx].get()) }
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tri_matrix() {
        let matrix = TriMatrix::new(4);

        matrix.write(0, 1, (0.5, 0.3));
        matrix.write(2, 3, (0.8, 0.2));

        assert_eq!(matrix.read(0, 1), (0.5, 0.3));
        assert_eq!(matrix.read(1, 0), (0.5, 0.3)); // Symmetric
        assert_eq!(matrix.read(2, 3), (0.8, 0.2));
        assert_eq!(matrix.read(0, 0), (1.0, 1.0)); // Diagonal
    }
}
