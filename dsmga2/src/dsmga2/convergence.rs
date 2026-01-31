use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Convergence detection state
pub(crate) struct ConvergenceTracker {
    count: AtomicUsize,
    last_max: AtomicU64,
    last_mean: AtomicU64,
    last_min: AtomicU64,
}

impl ConvergenceTracker {
    pub(crate) fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
            last_max: AtomicU64::new(0),
            last_mean: AtomicU64::new(0),
            last_min: AtomicU64::new(0),
        }
    }

    /// Check if population has converged (matching C++ implementation)
    pub(crate) fn check(&self, current_max: f64, current_mean: f64, current_min: f64) -> bool {
        let last_max = f64::from_bits(self.last_max.load(Ordering::Relaxed));
        let last_mean = f64::from_bits(self.last_mean.load(Ordering::Relaxed));
        let last_min = f64::from_bits(self.last_min.load(Ordering::Relaxed));

        if current_max == last_max && current_mean == last_mean && current_min == last_min {
            self.count.fetch_add(1, Ordering::Relaxed);
        } else {
            self.count.store(0, Ordering::Relaxed);
        }

        self.last_max
            .store(current_max.to_bits(), Ordering::Relaxed);
        self.last_mean
            .store(current_mean.to_bits(), Ordering::Relaxed);
        self.last_min
            .store(current_min.to_bits(), Ordering::Relaxed);

        self.count.load(Ordering::Relaxed) > 300
    }
}

impl Default for ConvergenceTracker {
    fn default() -> Self {
        Self::new()
    }
}
