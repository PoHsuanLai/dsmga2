#!/usr/bin/env python3
"""
Example of using a custom fitness function with DSMGA2.
"""

from dsmga2 import Dsmga2

class LeadingOnes:
    """
    LeadingOnes problem: count consecutive 1s from the start.
    This is harder than OneMax because bits are dependent on each other.
    """

    def evaluate(self, genes):
        """Count consecutive 1s from the start"""
        count = 0.0
        for gene in genes:
            if gene:
                count += 1.0
            else:
                break
        return count

    def optimum(self, length):
        """The optimal fitness is all 1s"""
        return float(length)

# Create optimizer with custom fitness function
optimizer = Dsmga2(problem_size=50, fitness_function=LeadingOnes())
optimizer.population_size = 100
optimizer.max_generations = 200
optimizer.seed = 42

print("Running DSMGA2 with custom LeadingOnes fitness...")
result = optimizer.run()

print(f"\nResults:")
print(f"  Best fitness: {result.best_fitness}")
print(f"  Generations: {result.generation}")
print(f"  Evaluations: {result.num_evaluations}")
print(f"  Converged: {result.converged}")

# Check if optimum found
if result.best_fitness == 50.0:
    print("\n✓ Found optimal solution!")
else:
    print(f"\n  Solution is {result.best_fitness / 50.0 * 100:.1f}% optimal")
