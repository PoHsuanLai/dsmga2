//! Checkpoint functionality for saving and loading DSMGA2 state
//!
//! Saves complete GA state to allow resuming optimization, similar to PyGAD's save/load.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io;
use std::path::Path;

/// Checkpoint data that can be saved/loaded
///
/// Contains all state needed to resume optimization
#[derive(Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Problem size
    pub problem_size: usize,
    /// Population size
    pub population_size: usize,
    /// Current generation
    pub generation: usize,
    /// Number of evaluations
    pub num_evaluations: usize,
    /// Entire population (chromosomes as bit vectors)
    pub population: Vec<Vec<bool>>,
    /// Population fitness values
    pub fitness_values: Vec<f64>,
    /// Random seed (if any)
    pub seed: Option<u64>,
    /// Configuration
    pub max_generations: Option<usize>,
    pub max_evaluations: Option<usize>,
    pub use_ghc: bool,
}

impl Checkpoint {
    /// Save checkpoint to file using bincode serialization
    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = File::create(path)?;
        bincode::serialize_into(file, self).map_err(io::Error::other)
    }

    /// Load checkpoint from file
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        bincode::deserialize_from(file).map_err(io::Error::other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_checkpoint_save_load() {
        let population = vec![vec![true, false, true], vec![false, true, false]];
        let fitness = vec![2.0, 1.0];

        let checkpoint = Checkpoint {
            problem_size: 3,
            population_size: 2,
            generation: 42,
            num_evaluations: 1000,
            population,
            fitness_values: fitness,
            seed: Some(42),
            max_generations: Some(100),
            max_evaluations: None,
            use_ghc: true,
        };

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_checkpoint_full.bin");
        checkpoint.save(&path).unwrap();

        let loaded = Checkpoint::load(&path).unwrap();
        assert_eq!(loaded.problem_size, 3);
        assert_eq!(loaded.population_size, 2);
        assert_eq!(loaded.generation, 42);
        assert_eq!(loaded.num_evaluations, 1000);
        assert_eq!(loaded.seed, Some(42));
        assert_eq!(loaded.population.len(), 2);
        assert_eq!(loaded.fitness_values, vec![2.0, 1.0]);

        fs::remove_file(&path).ok();
    }
}
