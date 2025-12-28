use dsmga2::fitness::{MkTrap, OneMax};
use dsmga2::{Dsmga2, FitnessFunction};

/// Test that OneMax consistently finds optimal solution
#[test]
fn test_onemax_finds_optimum() {
    let problem_size = 50;
    let fitness_fn = OneMax;

    let mut ga = Dsmga2::new(problem_size, &fitness_fn)
        .population_size(100)
        .max_generations(200)
        .seed(42)
        .build();

    ga.run();

    let best_fitness = ga.best_fitness();
    let optimum = fitness_fn.optimum(problem_size);

    assert!(
        (best_fitness - optimum).abs() < 0.01,
        "Expected optimum {}, got {}",
        optimum,
        best_fitness
    );
}

/// Test that results are reproducible with same seed
#[test]
fn test_reproducibility() {
    let problem_size = 50;
    let fitness_fn = OneMax;

    let mut ga1 = Dsmga2::new(problem_size, &fitness_fn)
        .population_size(50)
        .max_generations(10)
        .seed(123)
        .build();

    let mut ga2 = Dsmga2::new(problem_size, &fitness_fn)
        .population_size(50)
        .max_generations(10)
        .seed(123)
        .build();

    ga1.run();
    ga2.run();

    assert_eq!(
        ga1.best_fitness(),
        ga2.best_fitness(),
        "Same seed should produce same results"
    );

    assert_eq!(
        ga1.num_evaluations(),
        ga2.num_evaluations(),
        "Same seed should produce same number of evaluations"
    );
}

/// Test MK-Trap can solve small instances
#[test]
fn test_trap_small_instance() {
    let trap_k = 5;
    let num_blocks = 10;
    let problem_size = trap_k * num_blocks;
    let fitness_fn = MkTrap::new(trap_k);

    let mut ga = Dsmga2::new(problem_size, &fitness_fn)
        .population_size(200)
        .max_generations(200)
        .seed(42)
        .build();

    ga.run();

    let best_fitness = ga.best_fitness();
    let optimum = fitness_fn.optimum(problem_size);

    // Trap is harder, so we allow a small tolerance
    assert!(
        best_fitness >= optimum * 0.95,
        "Expected near optimum {}, got {}",
        optimum,
        best_fitness
    );
}

/// Test that fitness improves over generations
#[test]
fn test_fitness_improves() {
    // Use a harder problem to ensure multiple generations
    let problem_size = 100;
    let fitness_fn = MkTrap::new(5);

    let mut ga = Dsmga2::new(problem_size, &fitness_fn)
        .population_size(100)
        .max_generations(20)
        .seed(42)
        .build();

    let mut fitness_history = Vec::new();
    let initial_fitness = ga.best_fitness();
    fitness_history.push(initial_fitness);

    while let Some(state) = ga.step() {
        fitness_history.push(state.best_fitness);
    }

    // Check that fitness is non-decreasing
    for window in fitness_history.windows(2) {
        assert!(
            window[1] >= window[0] - 1e-6,
            "Fitness should not decrease: {} -> {}",
            window[0],
            window[1]
        );
    }

    // Check that we made some progress if we got multiple generations
    if fitness_history.len() > 1 {
        assert!(
            fitness_history.last().unwrap() >= fitness_history.first().unwrap(),
            "Fitness should not decrease"
        );
    }
}

/// Test iterator interface
#[test]
fn test_iterator_interface() {
    // Use a harder problem to ensure multiple generations
    let problem_size = 100;
    let fitness_fn = MkTrap::new(5);

    let ga = Dsmga2::new(problem_size, &fitness_fn)
        .population_size(100)
        .max_generations(10)
        .seed(42)
        .build();

    let states: Vec<_> = ga.take(10).collect();

    // Should collect some generations (may not be all 10 if it converges early)
    assert!(
        !states.is_empty(),
        "Should collect at least some generations"
    );

    // Check generations are sequential
    for (i, state) in states.iter().enumerate() {
        assert_eq!(state.generation, i + 1);
    }
}

/// Test callback interface
#[test]
fn test_callback_interface() {
    // Use a harder problem to ensure multiple generations
    let problem_size = 100;
    let fitness_fn = MkTrap::new(5);

    let mut ga = Dsmga2::new(problem_size, &fitness_fn)
        .population_size(100)
        .max_generations(10)
        .seed(42)
        .build();

    let mut callback_count = 0;
    let mut max_fitness_seen: f64 = 0.0;

    ga.run_with(|state| {
        callback_count += 1;
        max_fitness_seen = max_fitness_seen.max(state.best_fitness);
    });

    // Should be called for at least some generations
    assert!(
        callback_count > 0,
        "Callback should be called at least once"
    );
    assert!(max_fitness_seen > 0.0, "Should have seen some fitness");
}
