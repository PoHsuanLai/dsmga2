/// Bisection sweep for finding optimal population size
///
/// Implements the same algorithm as the C++ sweep.cpp to find the minimum
/// population size that reliably solves a problem.
///
/// Algorithm:
/// 1. Test 3 initial population sizes (start, start+step, start+2*step)
/// 2. Binary search to find the minimum size that achieves numConvergence successes
/// 3. Report optimal population size and mean evaluations
use dsmga2::fitness::{CyclicTrap, FoldedTrap, MkTrap, OneMax};
use dsmga2::{Dsmga2, FitnessFunction};
use std::env;
use std::process;

const MAX_GEN: usize = 200;
const INITIAL_STEP: usize = 30;

#[derive(Debug, Clone)]
struct Record {
    n: usize,      // Population size
    nfe: f64,      // Mean number of fitness evaluations
    gen: f64,      // Mean generations
    success: bool, // Whether all trials succeeded
}

impl Record {
    fn new(n: usize) -> Self {
        Self {
            n,
            nfe: f64::INFINITY,
            gen: 0.0,
            success: false,
        }
    }
}

fn run_trials(
    problem_size: usize,
    pop_size: usize,
    num_trials: usize,
    fitness_fn: &dyn FitnessFunction,
    show_progress: bool,
) -> Record {
    let mut total_gen = 0;
    let mut total_nfe = 0;
    let mut all_success = true;

    if show_progress {
        print!("[{}]: ", pop_size);
    }

    for _ in 0..num_trials {
        let mut ga = Dsmga2::new(problem_size, fitness_fn)
            .population_size(pop_size)
            .max_generations(MAX_GEN)
            .build();

        ga.run();

        let optimum = fitness_fn.optimum(problem_size);
        let found_optimum = (ga.best_fitness() - optimum).abs() < 1e-6;

        if !found_optimum {
            all_success = false;
            if show_progress {
                print!("-");
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
            break;
        } else {
            total_gen += ga.generation();
            total_nfe += ga.num_evaluations();
            if show_progress {
                print!("+");
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
        }
    }

    let mut rec = Record::new(pop_size);
    if all_success {
        rec.gen = total_gen as f64 / num_trials as f64;
        rec.nfe = total_nfe as f64 / num_trials as f64;
        rec.success = true;
    }

    if show_progress {
        println!(" : {:.1}", rec.nfe);
    }

    rec
}

fn bisection_sweep(
    problem_size: usize,
    num_convergence: usize,
    fitness_fn: &dyn FitnessFunction,
    initial_pop: usize,
    show_bisection: bool,
) -> Record {
    let step = INITIAL_STEP;

    // Phase 1: Test 3 initial population sizes
    if show_bisection {
        println!("Bisection phase 1");
    }

    let mut rec = vec![
        Record::new(initial_pop),
        Record::new(initial_pop + step),
        Record::new(initial_pop + step + step),
    ];

    for item in rec.iter_mut().take(3) {
        let pop_size = item.n;
        *item = run_trials(
            problem_size,
            pop_size,
            num_convergence,
            fitness_fn,
            show_bisection,
        );
    }

    // Phase 2: Binary search to find optimal population
    if show_bisection {
        println!("\nBisection phase 2");
    }

    while rec[0].nfe < rec[1].nfe && ((rec[2].n - rec[0].n) * 20 > rec[1].n) {
        rec[2] = rec[1].clone();
        rec[1].n = (rec[0].n + rec[2].n) / 2;

        rec[1] = run_trials(
            problem_size,
            rec[1].n,
            num_convergence,
            fitness_fn,
            show_bisection,
        );
    }

    // Phase 3: Further refinement
    if show_bisection {
        println!("\nBisection phase 3");
    }

    while rec[1].nfe < rec[2].nfe && ((rec[2].n - rec[0].n) * 20 > rec[1].n) {
        rec[0] = rec[1].clone();
        rec[1].n = (rec[0].n + rec[2].n) / 2;

        rec[1] = run_trials(
            problem_size,
            rec[1].n,
            num_convergence,
            fitness_fn,
            show_bisection,
        );
    }

    // Return the best result (minimum NFE that succeeds)
    rec.into_iter()
        .filter(|r| r.success)
        .min_by(|a, b| a.nfe.partial_cmp(&b.nfe).unwrap())
        .unwrap_or(Record::new(initial_pop))
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 || args.len() > 5 {
        eprintln!("Usage: dsmga2_sweep ell numConvergence function [initial_pop]");
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  ell            - Problem size (number of bits)");
        eprintln!("  numConvergence - Number of successful runs required");
        eprintln!("  function       - Fitness function to use");
        eprintln!("  initial_pop    - Initial population size to start search (default: 10)");
        eprintln!();
        eprintln!("Functions:");
        eprintln!("  0 - ONEMAX:  Maximize number of 1s");
        eprintln!("  1 - MK:      m-k Trap function (k=5)");
        eprintln!("  2 - FTRAP:   Folded Trap function");
        eprintln!("  3 - CYC:     Cyclic Trap function");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  dsmga2_sweep 50 30 1    # Find optimal pop size for Trap-5, size 50");
        eprintln!("                          # Require 30 successful runs");
        process::exit(1);
    }

    let ell: usize = args[1].parse().expect("Invalid ell");
    let num_convergence: usize = args[2].parse().expect("Invalid numConvergence");
    let function: u32 = args[3].parse().expect("Invalid function");
    let initial_pop: usize = if args.len() > 4 {
        args[4].parse().expect("Invalid initial_pop")
    } else {
        10
    };

    if function > 3 {
        eprintln!("Function {} not yet implemented", function);
        eprintln!("Currently supported: ONEMAX (0), MK (1), FTRAP (2), CYC (3)");
        process::exit(1);
    }

    println!("DSMGA2 Bisection Sweep");
    println!("======================");
    println!("Problem size:       {}", ell);
    println!("Convergence trials: {}", num_convergence);
    println!("Initial pop size:   {}", initial_pop);
    println!();

    // Select fitness function
    let onemax;
    let mktrap;
    let ftrap;
    let cyctrap;

    let fitness_fn: &dyn FitnessFunction = match function {
        0 => {
            println!("Function:           ONEMAX");
            onemax = OneMax;
            &onemax
        }
        1 => {
            println!("Function:           MK-Trap (k=5)");
            mktrap = MkTrap::new(5);
            &mktrap
        }
        2 => {
            println!("Function:           Folded Trap");
            ftrap = FoldedTrap;
            &ftrap
        }
        3 => {
            println!("Function:           Cyclic Trap (k=5)");
            cyctrap = CyclicTrap::new(5);
            &cyctrap
        }
        _ => unreachable!(),
    };

    println!();

    // Run bisection sweep
    let result = bisection_sweep(ell, num_convergence, fitness_fn, initial_pop, true);

    println!();
    println!("======================");
    println!("Results:");
    println!("  Optimal population size: {}", result.n);
    println!("  Mean generations:        {:.1}", result.gen);
    println!("  Mean evaluations:        {:.1}", result.nfe);
    println!();

    if !result.success {
        println!("Warning: Could not find a population size that reliably solves the problem.");
        println!("Try increasing the initial population size or reducing the problem difficulty.");
        process::exit(1);
    }
}
