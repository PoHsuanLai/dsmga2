use dsmga2::fitness::OneMax;
use dsmga2::Dsmga2;
use std::path::Path;

fn main() {
    let fitness_fn = OneMax;
    let checkpoint_path = "/tmp/dsmga2_checkpoint.bin";

    // First run: optimize for 10 generations and save checkpoint
    println!("=== First run: 10 generations ===");
    let mut ga = Dsmga2::new(100, &fitness_fn)
        .population_size(200)
        .max_generations(10)
        .seed(42)
        .build();

    ga.run();
    println!(
        "Generation {}: best_fitness = {:.2}",
        ga.generation(),
        ga.best_fitness()
    );

    // Save checkpoint
    ga.save_checkpoint(checkpoint_path)
        .expect("Failed to save checkpoint");
    println!("Checkpoint saved to {}", checkpoint_path);

    // Load and inspect checkpoint
    println!("\n=== Loading checkpoint ===");
    let checkpoint = Dsmga2::load_checkpoint(checkpoint_path).expect("Failed to load checkpoint");
    println!(
        "Loaded checkpoint from generation {}",
        checkpoint.generation
    );
    println!("Population size: {}", checkpoint.population_size);
    println!("Number of evaluations: {}", checkpoint.num_evaluations);
    let checkpoint_best = checkpoint
        .fitness_values
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    println!("Best fitness in checkpoint: {:.2}", checkpoint_best);

    // Resume optimization from checkpoint
    println!("\n=== Resuming from checkpoint ===");
    let mut ga_resumed = Dsmga2::from_checkpoint(checkpoint, &fitness_fn);

    // Verify state was restored correctly
    println!(
        "Resumed at generation {}, fitness {:.2}",
        ga_resumed.generation(),
        ga_resumed.best_fitness()
    );
    assert_eq!(ga_resumed.best_fitness(), checkpoint_best);

    // Continue optimization for 10 more generations
    let mut count = 0;
    while let Some(_state) = ga_resumed.step() {
        count += 1;
        if count >= 10 {
            break;
        }
    }

    println!(
        "\nAfter 10 more generations (total {}): best_fitness = {:.2}",
        ga_resumed.generation(),
        ga_resumed.best_fitness()
    );
    println!("Total evaluations: {}", ga_resumed.num_evaluations());

    // Show improvement
    let improvement = ga_resumed.best_fitness() - checkpoint_best;
    if improvement > 0.0 {
        println!("Fitness improved by {:.2} after resuming!", improvement);
    } else {
        println!("No improvement (problem may be solved)");
    }

    // Clean up
    if Path::new(checkpoint_path).exists() {
        std::fs::remove_file(checkpoint_path).ok();
        println!("\nCheckpoint file cleaned up");
    }
}
