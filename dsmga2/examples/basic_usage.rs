// Basic usage example covering core DSMGA2 functionality
use dsmga2::{fitness::OneMax, Dsmga2};

fn main() {
    // Create and run DSMGA2 with default settings
    let fitness_fn = OneMax;
    let mut ga = Dsmga2::new(50, &fitness_fn).build();

    ga.run();

    println!("Best fitness: {}", ga.best_fitness());
    println!("Generations: {}", ga.generation());
    println!("Evaluations: {}", ga.num_evaluations());

    // Print first 20 bits of solution
    let solution = ga.best_solution();
    print!("Solution: ");
    for i in 0..20.min(solution.length()) {
        print!("{}", if solution.get_gene(i) { "1" } else { "0" });
    }
    println!("...");
}
