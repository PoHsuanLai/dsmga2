use dsmga2::{Chromosome, Dsmga2, FitnessFunction};

/// Example: Custom fitness function that counts leading ones
/// Fitness = number of consecutive 1s from the start
///
/// Example: 11110100 has fitness 4 (four leading ones)
///          01111111 has fitness 0 (starts with 0)
///          11111111 has fitness 8 (all ones)
struct LeadingOnes;

impl FitnessFunction for LeadingOnes {
    fn evaluate(&self, chromosome: &Chromosome) -> f64 {
        let mut count = 0.0;
        for i in 0..chromosome.length() {
            if chromosome.get_gene(i) {
                count += 1.0;
            } else {
                // Stop at first zero
                break;
            }
        }
        count
    }

    fn optimum(&self, length: usize) -> f64 {
        length as f64
    }
}

/// Example: Another custom fitness function with deceptive landscape
/// Maximizes parity: rewards all 0s or all 1s, but nothing in between
struct Parity;

impl FitnessFunction for Parity {
    fn evaluate(&self, chromosome: &Chromosome) -> f64 {
        let ones = chromosome.count_ones() as f64;
        let n = chromosome.length() as f64;

        // Reward extremes: all 0s or all 1s get maximum fitness
        // Everything in between gets penalized
        if ones == 0.0 || ones == n {
            n
        } else {
            // Deceptive: closer to middle = lower fitness
            let distance_from_extreme = (ones - n / 2.0).abs();
            n / 2.0 - distance_from_extreme
        }
    }

    fn optimum(&self, length: usize) -> f64 {
        length as f64
    }
}

fn main() {
    println!("DSMGA-II: Custom Fitness Functions\n");

    // Example 1: LeadingOnes
    println!("=== Example 1: LeadingOnes ===");
    println!("Goal: Maximize number of consecutive 1s from the start\n");

    let problem_size = 50;
    let fitness_fn = LeadingOnes;

    let mut ga = Dsmga2::new(problem_size, &fitness_fn)
        .population_size(200)
        .max_generations(100)
        .seed(42)
        .build();

    ga.run_with(|state| {
        if state.generation % 10 == 0 {
            println!(
                "Gen {:3} | Best: {:5.1} | Mean: {:5.1} | Evals: {}",
                state.generation, state.best_fitness, state.mean_fitness, state.num_evaluations
            );
        }
    });

    println!("\nResults:");
    println!("  Final generation:   {}", ga.generation());
    println!("  Best fitness:       {:.1}", ga.best_fitness());
    println!(
        "  Optimum:            {:.1}",
        fitness_fn.optimum(problem_size)
    );

    let solution = ga.best_solution();
    print!("  Solution (first 20 bits): ");
    for i in 0..20.min(solution.length()) {
        print!("{}", if solution.get_gene(i) { "1" } else { "0" });
    }
    println!("...\n");

    // Example 2: Parity (deceptive)
    println!("=== Example 2: Parity (Deceptive) ===");
    println!("Goal: All bits same (either all 0s or all 1s)\n");

    let problem_size = 30;
    let fitness_fn = Parity;

    let mut ga = Dsmga2::new(problem_size, &fitness_fn)
        .population_size(100)
        .max_generations(100)
        .seed(123)
        .build();

    ga.run_with(|state| {
        if state.generation % 10 == 0 {
            println!(
                "Gen {:3} | Best: {:5.1} | Mean: {:5.1} | Evals: {}",
                state.generation, state.best_fitness, state.mean_fitness, state.num_evaluations
            );
        }
    });

    println!("\nResults:");
    println!("  Final generation:   {}", ga.generation());
    println!("  Best fitness:       {:.1}", ga.best_fitness());
    println!(
        "  Optimum:            {:.1}",
        fitness_fn.optimum(problem_size)
    );

    let solution = ga.best_solution();
    print!("  Solution: ");
    for i in 0..solution.length() {
        print!("{}", if solution.get_gene(i) { "1" } else { "0" });
    }
    println!();

    let ones = solution.count_ones();
    println!("  (Ones: {}, Zeros: {})", ones, problem_size - ones);

    if ga.best_fitness() >= fitness_fn.optimum(problem_size) - 0.01 {
        println!("\nSuccess! Found optimal solution!");
    }
}
