// Example: Inspecting learned linkage structure
//
// DSMGA2 learns which genes are statistically dependent by building
// a linkage tree. This example shows how to extract and visualize
// the learned dependencies.

use dsmga2::{fitness::MkTrap, Dsmga2};

fn main() {
    println!("DSMGA-II: Linkage Learning Example\n");

    // Use Trap-5 with 25 bits (5 blocks of 5 bits each)
    // DSMGA2 should learn that genes 0-4, 5-9, 10-14, 15-19, 20-24 are linked
    let problem_size = 25;
    let fitness_fn = MkTrap::new(5);

    let mut ga = Dsmga2::new(problem_size, &fitness_fn)
        .population_size(100)
        .max_generations(5)
        .seed(42)
        .build();

    // Run for a few generations to learn linkage
    ga.run();

    println!("Results after {} generations:", ga.generation());
    println!("  Best fitness: {:.1}", ga.best_fitness());
    println!("  Evaluations: {}\n", ga.num_evaluations());

    // Get learned linkage structure
    let mut linkages = ga.linkage();

    // Sort by weight (strongest dependencies first)
    linkages.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    println!("Top 15 learned dependencies (gene_i, gene_j, weight):");
    println!("Expected: Strong links within blocks [0-4], [5-9], [10-14], [15-19], [20-24]\n");

    for (i, j, weight) in linkages.iter().take(15) {
        // Determine which block each gene belongs to
        let block_i = i / 5;
        let block_j = j / 5;

        let same_block = if block_i == block_j {
            format!(" block {}", block_i)
        } else {
            " cross-block".to_string()
        };

        println!(
            "  Gene {:2} - Gene {:2}:  weight = {:8.2}  ({})",
            i, j, weight, same_block
        );
    }

    println!(
        "\nTotal linkages discovered: {} (out of {} possible pairs)",
        linkages.len(),
        problem_size * (problem_size - 1) / 2
    );
}
