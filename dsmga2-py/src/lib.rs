#![allow(non_local_definitions)]

use dsmga2::{Chromosome, FitnessFunction};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::sync::Arc;

/// Python wrapper for fitness functions
struct PyFitnessFunction {
    py_obj: PyObject,
}

impl PyFitnessFunction {
    fn new(py_obj: PyObject) -> Self {
        Self { py_obj }
    }
}

impl FitnessFunction for PyFitnessFunction {
    fn evaluate(&self, chromosome: &Chromosome) -> f64 {
        Python::with_gil(|py| {
            let genes: Vec<i32> = (0..chromosome.length())
                .map(|i| if chromosome.get_gene(i) { 1 } else { 0 })
                .collect();

            let array = PyArray1::from_vec(py, genes);

            let result = self
                .py_obj
                .call_method1(py, "evaluate", (array,))
                .expect("Failed to call evaluate method");

            result.extract(py).expect("evaluate must return a float")
        })
    }

    fn optimum(&self, length: usize) -> f64 {
        Python::with_gil(|py| {
            let result = self
                .py_obj
                .call_method1(py, "optimum", (length,))
                .expect("Failed to call optimum method");

            result.extract(py).expect("optimum must return a float")
        })
    }
}

/// OneMax fitness function - counts the number of 1s
#[pyclass]
#[derive(Clone)]
struct OneMax;

#[pymethods]
impl OneMax {
    #[new]
    fn new() -> Self {
        OneMax
    }

    fn evaluate(&self, genes: PyReadonlyArray1<i32>) -> f64 {
        genes.as_slice().unwrap().iter().sum::<i32>() as f64
    }

    fn optimum(&self, length: usize) -> f64 {
        length as f64
    }
}

/// m-k Trap fitness function
#[pyclass]
#[derive(Clone)]
struct MkTrap {
    k: usize,
}

#[pymethods]
impl MkTrap {
    #[new]
    fn new(k: usize) -> Self {
        MkTrap { k }
    }

    fn evaluate(&self, genes: PyReadonlyArray1<i32>) -> f64 {
        let genes_slice = genes.as_slice().unwrap();
        let n_blocks = genes_slice.len() / self.k;
        let mut fitness = 0.0;

        for block_idx in 0..n_blocks {
            let start = block_idx * self.k;
            let end = start + self.k;
            let u: i32 = genes_slice[start..end].iter().sum();

            if u == self.k as i32 {
                fitness += 1.0;
            } else {
                fitness += 0.8 - (u as f64) * 0.8 / ((self.k - 1) as f64);
            }
        }

        fitness
    }

    fn optimum(&self, length: usize) -> f64 {
        let rust_trap = dsmga2::fitness::MkTrap::new(self.k);
        rust_trap.optimum(length)
    }
}

/// Result of optimization
#[pyclass]
#[derive(Clone)]
struct OptimizationResult {
    #[pyo3(get)]
    best_fitness: f64,
    #[pyo3(get)]
    generation: usize,
    #[pyo3(get)]
    num_evaluations: usize,
    #[pyo3(get)]
    mean_fitness: f64,
    #[pyo3(get)]
    converged: bool,
}

#[pymethods]
impl OptimizationResult {
    fn __repr__(&self) -> String {
        format!(
            "OptimizationResult(best_fitness={}, generation={}, num_evaluations={}, converged={})",
            self.best_fitness, self.generation, self.num_evaluations, self.converged
        )
    }
}

/// DSMGA2 optimizer
#[pyclass]
struct Dsmga2 {
    problem_size: usize,
    fitness_fn: Arc<dyn FitnessFunction + Send + Sync>,
    population_size: usize,
    max_generations: i32,
    seed: u64,
    fitness_threshold: Option<f64>,
    patience: Option<usize>,
    ga: Option<dsmga2::Dsmga2<'static>>,
}

#[pymethods]
impl Dsmga2 {
    #[new]
    #[pyo3(signature = (problem_size, fitness_function))]
    fn new(problem_size: usize, fitness_function: &PyAny) -> PyResult<Self> {
        // Check if it's a built-in fitness function or custom Python object
        let fitness_fn: Arc<dyn FitnessFunction + Send + Sync> =
            if fitness_function.extract::<PyRef<OneMax>>().is_ok() {
                Arc::new(dsmga2::fitness::OneMax)
            } else if let Ok(trap) = fitness_function.extract::<PyRef<MkTrap>>() {
                Arc::new(dsmga2::fitness::MkTrap::new(trap.k))
            } else {
                // Custom Python fitness function
                Arc::new(PyFitnessFunction::new(fitness_function.into()))
            };

        Ok(Dsmga2 {
            problem_size,
            fitness_fn,
            population_size: problem_size,
            max_generations: -1,
            seed: 42,
            fitness_threshold: None,
            patience: None,
            ga: None,
        })
    }

    #[getter]
    fn population_size(&self) -> usize {
        self.population_size
    }

    #[setter]
    fn set_population_size(&mut self, size: usize) {
        self.population_size = size;
        self.ga = None; // Invalidate existing GA
    }

    #[getter]
    fn max_generations(&self) -> Option<usize> {
        if self.max_generations < 0 {
            None
        } else {
            Some(self.max_generations as usize)
        }
    }

    #[setter]
    fn set_max_generations(&mut self, gens: Option<usize>) {
        self.max_generations = gens.map(|g| g as i32).unwrap_or(-1);
        self.ga = None;
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    #[setter]
    fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.ga = None;
    }

    #[getter]
    fn fitness_threshold(&self) -> Option<f64> {
        self.fitness_threshold
    }

    #[setter]
    fn set_fitness_threshold(&mut self, threshold: Option<f64>) {
        self.fitness_threshold = threshold;
        self.ga = None;
    }

    #[getter]
    fn patience(&self) -> Option<usize> {
        self.patience
    }

    #[setter]
    fn set_patience(&mut self, patience: Option<usize>) {
        self.patience = patience;
        self.ga = None;
    }

    /// Run optimization until convergence or max generations
    fn run(&mut self, py: Python) -> PyResult<OptimizationResult> {
        py.allow_threads(|| {
            self.ensure_ga_initialized();
            let ga = self.ga.as_mut().unwrap();

            // Use the iterator-based step() API
            let mut last_state = None;
            while let Some(state) = ga.step() {
                last_state = Some(state);
            }

            let state = last_state.unwrap_or_else(|| ga.state());

            Ok(OptimizationResult {
                best_fitness: state.best_fitness,
                generation: state.generation,
                num_evaluations: state.num_evaluations,
                mean_fitness: state.mean_fitness,
                converged: true,
            })
        })
    }

    /// Run one generation step
    fn step(&mut self, py: Python) -> PyResult<bool> {
        py.allow_threads(|| {
            self.ensure_ga_initialized();
            let ga = self.ga.as_mut().unwrap();

            // Use the iterator-based step() API
            Ok(ga.step().is_some())
        })
    }

    /// Get current best fitness
    fn best_fitness(&mut self) -> PyResult<f64> {
        self.ensure_ga_initialized();
        Ok(self.ga.as_ref().unwrap().best_fitness())
    }

    /// Get current generation number
    fn generation(&mut self) -> PyResult<usize> {
        self.ensure_ga_initialized();
        Ok(self.ga.as_ref().unwrap().generation())
    }

    /// Get total number of fitness evaluations
    fn num_evaluations(&mut self) -> PyResult<usize> {
        self.ensure_ga_initialized();
        Ok(self.ga.as_ref().unwrap().num_evaluations())
    }

    /// Get learned linkage structure
    ///
    /// Returns a list of (gene_i, gene_j, weight) tuples representing
    /// dependencies between genes. Higher weights indicate stronger dependencies.
    fn linkage(&mut self) -> PyResult<Vec<(usize, usize, f64)>> {
        self.ensure_ga_initialized();
        Ok(self.ga.as_ref().unwrap().linkage())
    }

    /// Save current state to a checkpoint file
    fn save_checkpoint(&mut self, path: String) -> PyResult<()> {
        self.ensure_ga_initialized();
        self.ga
            .as_ref()
            .unwrap()
            .save_checkpoint(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Dsmga2(problem_size={}, population_size={}, max_generations={}, seed={})",
            self.problem_size, self.population_size, self.max_generations, self.seed
        )
    }
}

impl Dsmga2 {
    fn ensure_ga_initialized(&mut self) {
        if self.ga.is_none() {
            // We need to leak the fitness function to get a 'static reference
            // This is safe because the GA will never outlive the Dsmga2 wrapper
            let fitness_ref: &'static dyn FitnessFunction =
                unsafe { std::mem::transmute(self.fitness_fn.as_ref() as &dyn FitnessFunction) };

            let mut builder = dsmga2::Dsmga2::new(self.problem_size, fitness_ref)
                .population_size(self.population_size)
                .seed(self.seed);

            // Set max_generations if specified
            if self.max_generations >= 0 {
                builder = builder.max_generations(self.max_generations as usize);
            }

            // Set stop criteria if specified
            if let Some(threshold) = self.fitness_threshold {
                builder = builder.fitness_threshold(threshold);
            }

            if let Some(patience) = self.patience {
                builder = builder.patience(patience);
            }

            self.ga = Some(builder.build());
        }
    }
}

/// Python module
#[pymodule]
fn _dsmga2(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Dsmga2>()?;
    m.add_class::<OneMax>()?;
    m.add_class::<MkTrap>()?;
    m.add_class::<OptimizationResult>()?;
    Ok(())
}
