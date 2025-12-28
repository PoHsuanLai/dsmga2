#!/usr/bin/env python3
"""
Benchmark: DSMGA2 vs PyGAD vs DEAP on MAX-SAT Problem

Compares NFE (Number of Function Evaluations) to find satisfying assignments.
This matches the methodology used in the DSMGA-II paper which showed 20.2%
improvement on MAX-SAT instances.

We use Uniform Random-3-SAT instances similar to SATLIB benchmarks.
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Set
import sys

# Check if dsmga2 is available
try:
    import dsmga2
    HAS_DSMGA2 = True
except ImportError:
    print("Warning: dsmga2 not installed. Install with: cd dsmga2-py && maturin develop --release")
    HAS_DSMGA2 = False

# Check if PyGAD is available
try:
    import pygad
    HAS_PYGAD = True
except ImportError:
    print("Warning: PyGAD not installed. Install with: pip install pygad")
    HAS_PYGAD = False

# Check if DEAP is available
try:
    import random
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    print("Warning: DEAP not installed. Install with: pip install deap")
    HAS_DEAP = False


class MaxSatProblem:
    """
    MAX-SAT Problem (3-SAT variant)

    Encoding: Each variable is a single bit (True/False)
    Fitness: Number of satisfied clauses (maximize)

    We generate Uniform Random-3-SAT instances with clause-to-variable ratio
    of 4.3, which is near the satisfiability threshold for 3-SAT.
    """

    def __init__(self, n_vars: int, clause_ratio: float = 4.3, seed: int = 42):
        self.n_vars = n_vars
        self.n_clauses = int(n_vars * clause_ratio)

        # Generate random 3-SAT instance
        np.random.seed(seed)
        random.seed(seed)

        self.clauses: List[Tuple[int, int, int]] = []

        for _ in range(self.n_clauses):
            # Pick 3 distinct variables
            vars_in_clause = np.random.choice(n_vars, size=3, replace=False)

            # Each literal can be positive or negated
            # Encode as: positive var = var_idx, negative var = -(var_idx+1)
            clause = tuple(
                (v + 1) if np.random.random() < 0.5 else -(v + 1)
                for v in vars_in_clause
            )
            self.clauses.append(clause)

        print(f"\nMAX-SAT Problem (Uniform Random 3-SAT):")
        print(f"  Variables: {n_vars}")
        print(f"  Clauses: {self.n_clauses}")
        print(f"  Clause/Variable ratio: {clause_ratio:.2f}")
        print(f"  Problem size (bits): {n_vars}")

    def evaluate_clause(self, assignment: np.ndarray, clause: Tuple[int, int, int]) -> bool:
        """Check if a clause is satisfied by the assignment"""
        for literal in clause:
            if literal > 0:
                # Positive literal: check if variable is True
                var_idx = literal - 1
                if assignment[var_idx]:
                    return True
            else:
                # Negative literal: check if variable is False
                var_idx = (-literal) - 1
                if not assignment[var_idx]:
                    return True
        return False

    def count_satisfied(self, assignment: np.ndarray) -> int:
        """Count number of satisfied clauses"""
        satisfied = sum(
            1 for clause in self.clauses
            if self.evaluate_clause(assignment, clause)
        )
        return satisfied

    def evaluate(self, genes: np.ndarray) -> float:
        """Fitness = number of satisfied clauses (higher is better)"""
        # Convert to boolean
        assignment = genes.astype(bool)
        return float(self.count_satisfied(assignment))

    def is_satisfiable(self, assignment: np.ndarray) -> bool:
        """Check if all clauses are satisfied"""
        return self.count_satisfied(assignment) == self.n_clauses

    def optimum(self) -> float:
        """Optimum fitness (all clauses satisfied)"""
        return float(self.n_clauses)


def benchmark_dsmga2(problem: MaxSatProblem, max_evals: int = 500000, seed: int = 42) -> Dict:
    """Benchmark DSMGA2 - measure NFE to find satisfying assignment"""
    if not HAS_DSMGA2:
        return None

    print("\n[DSMGA2] Running...")

    # Create fitness function class (required by Python bindings)
    class MaxSatFitness:
        def evaluate(self, genes):
            return problem.evaluate(genes)

        def optimum(self, length):
            return problem.optimum()

    fitness_func = MaxSatFitness()

    start_time = time.time()

    # Use adaptive population sizing (DSMGA2's strength)
    # Start with heuristic: sqrt(n_vars) * 10
    initial_pop = max(50, int(np.sqrt(problem.n_vars) * 10))

    ga = dsmga2.Dsmga2(problem.n_vars, fitness_func)
    ga.population_size = initial_pop
    ga.max_generations = 1000  # Large enough to not be the limiting factor
    ga.seed = seed

    opt_result = ga.run()

    elapsed = time.time() - start_time

    best_fitness = opt_result.best_fitness
    satisfied = int(best_fitness)
    found_solution = satisfied == problem.n_clauses

    result = {
        'name': 'DSMGA2',
        'time': elapsed,
        'fitness': best_fitness,
        'satisfied_clauses': satisfied,
        'total_clauses': problem.n_clauses,
        'found_solution': found_solution,
        'generations': opt_result.generation,
        'nfe': opt_result.num_evaluations,
        'population_size': initial_pop,
    }

    print(f"  Time: {elapsed:.3f}s")
    print(f"  Satisfied: {satisfied}/{problem.n_clauses} clauses")
    print(f"  NFE: {result['nfe']:,}")
    print(f"  Generations: {result['generations']}")
    print(f"  Population: {initial_pop}")
    print(f"  Solution found: {'YES' if found_solution else 'NO'}")

    return result


def benchmark_pygad(problem: MaxSatProblem, max_evals: int = 500000, seed: int = 42) -> Dict:
    """Benchmark PyGAD - measure NFE to find satisfying assignment"""
    if not HAS_PYGAD:
        return None

    print("\n[PyGAD] Running...")

    def fitness_func(ga_instance, solution, solution_idx):
        return problem.evaluate(solution)

    # Use same population size as DSMGA2 for fair comparison
    population_size = max(50, int(np.sqrt(problem.n_vars) * 10))
    max_gens = max_evals // population_size

    start_time = time.time()

    # Track when solution is found
    solution_found_gen = None

    def on_generation(ga_instance):
        nonlocal solution_found_gen
        if solution_found_gen is None:
            best_solution, best_fitness, _ = ga_instance.best_solution()
            if best_fitness >= problem.optimum() - 0.01:
                solution_found_gen = ga_instance.generations_completed

    ga_instance = pygad.GA(
        num_generations=max_gens,
        num_parents_mating=population_size // 2,
        fitness_func=fitness_func,
        sol_per_pop=population_size,
        num_genes=problem.n_vars,
        gene_type=int,
        gene_space=[0, 1],
        parent_selection_type="tournament",
        crossover_type="uniform",
        mutation_type="random",
        mutation_percent_genes=10,
        random_seed=seed,
        on_generation=on_generation,
    )

    ga_instance.run()

    elapsed = time.time() - start_time

    best_solution, best_fitness, _ = ga_instance.best_solution()
    satisfied = int(best_fitness)
    found_solution = satisfied == problem.n_clauses

    # Calculate NFE to solution (or total if not found)
    if solution_found_gen is not None:
        nfe = solution_found_gen * population_size
    else:
        nfe = ga_instance.generations_completed * population_size

    result = {
        'name': 'PyGAD',
        'time': elapsed,
        'fitness': best_fitness,
        'satisfied_clauses': satisfied,
        'total_clauses': problem.n_clauses,
        'found_solution': found_solution,
        'generations': ga_instance.generations_completed,
        'nfe': nfe,
        'population_size': population_size,
    }

    print(f"  Time: {elapsed:.3f}s")
    print(f"  Satisfied: {satisfied}/{problem.n_clauses} clauses")
    print(f"  NFE: {result['nfe']:,}")
    print(f"  Generations: {result['generations']}")
    print(f"  Population: {population_size}")
    print(f"  Solution found: {'YES' if found_solution else 'NO'}")

    return result


def benchmark_deap(problem: MaxSatProblem, max_evals: int = 500000, seed: int = 42) -> Dict:
    """Benchmark DEAP - measure NFE to find satisfying assignment"""
    if not HAS_DEAP:
        return None

    print("\n[DEAP] Running...")

    # Setup DEAP
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, n=problem.n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_maxsat(individual):
        return (problem.evaluate(np.array(individual)),)

    toolbox.register("evaluate", eval_maxsat)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(seed)

    population_size = max(50, int(np.sqrt(problem.n_vars) * 10))
    max_gens = max_evals // population_size

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)

    start_time = time.time()

    # Track when solution is found
    solution_found_gen = None
    evaluations = 0

    # Custom evolution loop to track NFE to solution
    for gen in range(max_gens):
        # Select and clone offspring
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate offspring with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            evaluations += 1

        # Replace population
        pop[:] = offspring
        hof.update(pop)

        # Check if solution found
        if solution_found_gen is None and hof[0].fitness.values[0] >= problem.optimum() - 0.01:
            solution_found_gen = gen + 1
            break

    elapsed = time.time() - start_time

    best_solution = np.array(hof[0])
    best_fitness = problem.evaluate(best_solution)
    satisfied = int(best_fitness)
    found_solution = satisfied == problem.n_clauses

    nfe = evaluations if solution_found_gen else max_gens * population_size

    result = {
        'name': 'DEAP',
        'time': elapsed,
        'fitness': best_fitness,
        'satisfied_clauses': satisfied,
        'total_clauses': problem.n_clauses,
        'found_solution': found_solution,
        'generations': solution_found_gen if solution_found_gen else max_gens,
        'nfe': nfe,
        'population_size': population_size,
    }

    print(f"  Time: {elapsed:.3f}s")
    print(f"  Satisfied: {satisfied}/{problem.n_clauses} clauses")
    print(f"  NFE: {result['nfe']:,}")
    print(f"  Generations: {result['generations']}")
    print(f"  Population: {population_size}")
    print(f"  Solution found: {'YES' if found_solution else 'NO'}")

    return result


def run_comparison(problem_sizes: List[int], n_runs: int = 5, max_evals: int = 500000):
    """Run comparison across different problem sizes"""

    results = {size: [] for size in problem_sizes}

    for size in problem_sizes:
        print(f"\n{'='*60}")
        print(f"Problem Size: {size} variables")
        print(f"{'='*60}")

        for run in range(n_runs):
            print(f"\n--- Run {run + 1}/{n_runs} ---")

            # Create MAX-SAT problem
            problem = MaxSatProblem(
                n_vars=size,
                clause_ratio=4.3,  # Near satisfiability threshold
                seed=42 + run
            )

            run_results = []

            # Benchmark each library
            for benchmark_func in [benchmark_dsmga2, benchmark_pygad, benchmark_deap]:
                result = benchmark_func(problem, max_evals=max_evals, seed=42 + run)
                if result:
                    run_results.append(result)

            results[size].append(run_results)

    return results


def print_summary(results: Dict):
    """Print summary statistics focusing on NFE to solution"""
    print(f"\n{'='*60}")
    print("SUMMARY - NFE to Find Satisfying Assignment")
    print(f"{'='*60}\n")

    for size in sorted(results.keys()):
        print(f"Problem Size: {size} variables")
        print("-" * 60)

        # Aggregate results by library
        by_library = {}
        for run_results in results[size]:
            for result in run_results:
                name = result['name']
                if name not in by_library:
                    by_library[name] = {
                        'nfe': [],
                        'time': [],
                        'solved': [],
                        'satisfied': []
                    }
                by_library[name]['nfe'].append(result['nfe'])
                by_library[name]['time'].append(result['time'])
                by_library[name]['solved'].append(result['found_solution'])
                by_library[name]['satisfied'].append(result['satisfied_clauses'])

        # Print statistics
        for name in ['DSMGA2', 'PyGAD', 'DEAP']:
            if name in by_library:
                nfe = np.array(by_library[name]['nfe'])
                times = np.array(by_library[name]['time'])
                solved_count = sum(by_library[name]['solved'])
                total_runs = len(by_library[name]['solved'])
                avg_satisfied = np.mean(by_library[name]['satisfied'])

                print(f"{name:10s}: NFE: {nfe.mean():8.0f} ± {nfe.std():6.0f}  "
                      f"| Time: {times.mean():6.2f}s ± {times.std():5.2f}s  "
                      f"| Solved: {solved_count}/{total_runs}  "
                      f"| Avg clauses: {avg_satisfied:.1f}")

        print()

    # Print NFE comparison (key metric from paper)
    print("\nNFE Comparison (Lower is Better):")
    print("-" * 60)
    for size in sorted(results.keys()):
        by_library = {}
        for run_results in results[size]:
            for result in run_results:
                name = result['name']
                if name not in by_library:
                    by_library[name] = {'nfe': []}
                by_library[name]['nfe'].append(result['nfe'])

        print(f"Size {size:3d}:", end="")
        dsmga2_nfe = None
        for name in ['DSMGA2', 'PyGAD', 'DEAP']:
            if name in by_library:
                nfe = np.mean(by_library[name]['nfe'])
                if name == 'DSMGA2':
                    dsmga2_nfe = nfe
                    print(f"  {name}: {nfe:8.0f}", end="")
                else:
                    improvement = ((nfe - dsmga2_nfe) / nfe * 100) if dsmga2_nfe else 0
                    print(f"  {name}: {nfe:8.0f} ({improvement:+.1f}%)", end="")
        print()


if __name__ == "__main__":
    print("MAX-SAT Problem: GA Library Comparison")
    print("=" * 60)
    print("Methodology: Measure NFE to find satisfying assignment")
    print("(Matches DSMGA-II paper evaluation approach)")

    # Check what's available
    available = []
    if HAS_DSMGA2:
        available.append("DSMGA2")
    if HAS_PYGAD:
        available.append("PyGAD")
    if HAS_DEAP:
        available.append("DEAP")

    if not available:
        print("\nError: No GA libraries available!")
        print("Please install at least one of: dsmga2, pygad, deap")
        sys.exit(1)

    print(f"\nAvailable libraries: {', '.join(available)}")

    # Run benchmarks on different sizes
    # Start small to verify it works, then scale up
    problem_sizes = [20, 30, 40]  # Variables (clauses will be 4.3x this)
    n_runs = 5
    max_evals = 500000

    print(f"\nRunning {n_runs} runs on problem sizes: {problem_sizes}")
    print(f"Max evaluations per run: {max_evals:,}")

    results = run_comparison(problem_sizes, n_runs, max_evals)
    print_summary(results)

    # Save results
    import json
    with open('benchmarks/maxsat_results.json', 'w') as f:
        serializable_results = {}
        for size, runs in results.items():
            serializable_results[str(size)] = []
            for run in runs:
                run_data = []
                for r in run:
                    run_data.append({
                        'name': r['name'],
                        'time': float(r['time']),
                        'fitness': float(r['fitness']),
                        'satisfied_clauses': int(r['satisfied_clauses']),
                        'total_clauses': int(r['total_clauses']),
                        'found_solution': bool(r['found_solution']),
                        'generations': int(r['generations']),
                        'nfe': int(r['nfe']),
                        'population_size': int(r['population_size']),
                    })
                serializable_results[str(size)].append(run_data)
        json.dump(serializable_results, f, indent=2)

    print("\nResults saved to benchmarks/maxsat_results.json")
