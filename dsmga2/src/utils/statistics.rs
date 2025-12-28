/// Tracks fitness statistics for a population
#[derive(Debug, Clone)]
pub struct Statistics {
    count: usize,
    sum: f64,
    min: f64,
    max: f64,
}

impl Statistics {
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }

    #[inline]
    pub fn record(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    #[inline(always)]
    pub fn count(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    #[inline(always)]
    pub fn min(&self) -> f64 {
        self.min
    }

    #[inline(always)]
    pub fn max(&self) -> f64 {
        self.max
    }
}

impl Default for Statistics {
    fn default() -> Self {
        Self::new()
    }
}
