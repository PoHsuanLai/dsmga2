#!/usr/bin/env python3
"""
Example: Hyperparameter tuning with Optuna

Demonstrates how to use Optuna for automatic hyperparameter optimization
with DSMGA2. Optuna uses Bayesian optimization (TPE sampler) to efficiently
search the parameter space.

Install Optuna:
    pip install optuna optuna-dashboard

Run optimization:
    python optuna_tuning.py

View dashboard:
    optuna-dashboard sqlite:///optuna_study.db
"""

import optuna
import dsmga2


def objective_minimize_evaluations(trial):
    """
    Objective: Find population size that minimizes evaluations to solve Trap-5

    Optuna will suggest population sizes and learn which ones work best.
    """
    # Problem configuration
    problem_size = 50
    fitness_fn = dsmga2.MkTrap(5)  # Trap-5 function

    # Optuna suggests parameter
    pop_size = trial.suggest_int('population_size', 50, 300)

    # Run DSMGA2
    ga = dsmga2.Dsmga2(problem_size, fitness_fn)
    ga.population_size = pop_size
    ga.max_generations = 100

    result = ga.run()

    # Check if we found the optimum
    optimum = fitness_fn.optimum(problem_size)
    if result.best_fitness < optimum - 0.01:
        # Penalize failures heavily
        return 1_000_000

    # Return number of evaluations (minimize this)
    return result.num_evaluations


def objective_multi_param(trial):
    """
    Objective: Optimize multiple parameters simultaneously

    This shows how to tune population size and other parameters together.
    """
    problem_size = 50
    fitness_fn = dsmga2.MkTrap(5)

    # Suggest multiple parameters
    pop_size = trial.suggest_int('population_size', 50, 300)
    # Could add more parameters here if DSMGA2 exposed them
    # For example: mutation_rate, crossover_rate, etc.

    ga = dsmga2.Dsmga2(problem_size, fitness_fn)
    ga.population_size = pop_size
    ga.max_generations = 100

    result = ga.run()

    optimum = fitness_fn.optimum(problem_size)
    if result.best_fitness < optimum - 0.01:
        return 1_000_000

    return result.num_evaluations


def run_basic_optimization():
    """Run basic population size optimization"""
    print("="*60)
    print("Optuna Hyperparameter Optimization")
    print("="*60)
    print()
    print("Problem: Trap-5, size 50")
    print("Goal: Find population size that minimizes evaluations")
    print()

    # Create study with SQLite storage (enables dashboard)
    study = optuna.create_study(
        study_name="dsmga2_population_size",
        storage="sqlite:///optuna_study.db",
        direction='minimize',
        load_if_exists=True,
    )

    # Run optimization
    print("Running optimization (50 trials)...")
    print("This will take a few minutes...")
    print()

    study.optimize(
        objective_minimize_evaluations,
        n_trials=50,
        n_jobs=1,  # Set to -1 for parallel
        show_progress_bar=True,
    )

    # Print results
    print()
    print("="*60)
    print("Results")
    print("="*60)
    print()
    print(f"Best population size: {study.best_params['population_size']}")
    print(f"Best evaluations:     {study.best_value:.0f}")
    print()

    # Show top 5 trials
    print("Top 5 trials:")
    print("-"*60)
    trials = sorted(study.trials, key=lambda t: t.value if t.value else float('inf'))
    for i, trial in enumerate(trials[:5], 1):
        if trial.value:
            print(f"{i}. Pop size: {trial.params['population_size']:3d}  "
                  f"Evaluations: {trial.value:7.0f}")

    print()
    print("Visualization:")
    print("  optuna-dashboard sqlite:///optuna_study.db")
    print("  Then open http://localhost:8080 in your browser")


def run_with_visualization():
    """Run optimization and generate plots"""
    try:
        import plotly.io as pio
        has_plotly = True
    except ImportError:
        has_plotly = False
        print("Install plotly for visualizations: pip install plotly")
        return

    study = optuna.create_study(
        study_name="dsmga2_viz",
        direction='minimize',
    )

    print("Running quick optimization for visualization...")
    study.optimize(objective_minimize_evaluations, n_trials=20, show_progress_bar=True)

    # Generate visualization
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
    )

    # Optimization history
    fig = plot_optimization_history(study)
    fig.write_html("optuna_history.html")
    print("\nSaved: optuna_history.html")

    # Parameter importance (needs more trials to be meaningful)
    if len(study.trials) > 10:
        fig = plot_param_importances(study)
        fig.write_html("optuna_importance.html")
        print("Saved: optuna_importance.html")

    # Slice plot
    fig = plot_slice(study)
    fig.write_html("optuna_slice.html")
    print("Saved: optuna_slice.html")

    print()
    print("Open the HTML files in your browser to view the plots!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--viz":
        run_with_visualization()
    else:
        run_basic_optimization()
