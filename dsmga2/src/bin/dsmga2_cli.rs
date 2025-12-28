use dsmga2::fitness::{CyclicTrap, FoldedTrap, MkTrap, OneMax};
use dsmga2::{Dsmga2, FitnessFunction};
use std::env;
use std::process;

#[cfg(feature = "cli")]
use indicatif::{ProgressBar, ProgressStyle};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 9 {
        eprintln!("DSMGA2 ell nInitial function maxGen maxFe repeat display rand_seed");
        eprintln!("function: ");
        eprintln!("     ONEMAX:  0");
        eprintln!("     MK    :  1");
        eprintln!("     FTRAP :  2");
        eprintln!("     CYC   :  3");
        eprintln!("     NK    :  4");
        eprintln!("     SPIN  :  5");
        eprintln!("     SAT   :  6");
        process::exit(1);
    }

    let ell: usize = args[1].parse().expect("Invalid ell");
    let n_initial: usize = args[2].parse().expect("Invalid nInitial");
    let function: u32 = args[3].parse().expect("Invalid function");
    let max_gen: isize = args[4].parse().expect("Invalid maxGen");
    let max_fe: isize = args[5].parse().expect("Invalid maxFe");
    let repeat: usize = args[6].parse().expect("Invalid repeat");
    let display: bool = args[7].parse::<u32>().expect("Invalid display") != 0;
    let rand_seed: i32 = args[8].parse().expect("Invalid rand_seed");

    // For now, we only support ONEMAX (0), MK (1), FTRAP (2), and CYC (3)
    if function > 3 {
        eprintln!("Function {} not yet implemented in Rust version", function);
        eprintln!("Currently supported: ONEMAX (0), MK (1), FTRAP (2), CYC (3)");
        process::exit(1);
    }

    let mut total_gen = 0.0;
    let mut total_fe = 0.0;
    let mut fail_num = 0;

    // Create progress bar if available
    #[cfg(feature = "cli")]
    let progress = ProgressBar::new(repeat as u64);
    #[cfg(feature = "cli")]
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} runs | Success: {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    for run in 0..repeat {
        let seed = if rand_seed == -1 {
            // Use time-based seed
            use std::time::{SystemTime, UNIX_EPOCH};
            let duration = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards");
            duration
                .as_secs()
                .wrapping_mul(1000000)
                .wrapping_add(duration.subsec_micros() as u64)
                .wrapping_add(run as u64)
        } else {
            (rand_seed as u64).wrapping_add(run as u64)
        };

        // Convert -1 to None for max values
        let max_gen_opt = if max_gen < 0 {
            None
        } else {
            Some(max_gen as usize)
        };
        let max_fe_opt = if max_fe < 0 {
            None
        } else {
            Some(max_fe as usize)
        };

        let (found_optima, used_gen, used_fe) = match function {
            0 => run_onemax(ell, n_initial, max_gen_opt, max_fe_opt, seed, display),
            1 => run_mktrap(ell, n_initial, max_gen_opt, max_fe_opt, seed, display, 5),
            2 => run_ftrap(ell, n_initial, max_gen_opt, max_fe_opt, seed, display, 5),
            3 => run_cyctrap(ell, n_initial, max_gen_opt, max_fe_opt, seed, display, 5),
            _ => unreachable!(),
        };

        if found_optima {
            total_gen += used_gen as f64;
            total_fe += used_fe as f64;
        } else {
            fail_num += 1;
        }

        #[cfg(feature = "cli")]
        {
            let success_count = run + 1 - fail_num;
            progress.set_message(format!("{}/{}", success_count, run + 1));
            progress.inc(1);
        }

        #[cfg(not(feature = "cli"))]
        {
            // Fallback to simple output
            print!("{}", if found_optima { "+" } else { "-" });
            if (run + 1) % 50 == 0 {
                println!();
            }
        }
    }

    #[cfg(feature = "cli")]
    progress.finish();

    #[cfg(not(feature = "cli"))]
    if repeat % 50 != 0 {
        println!();
    }

    let success_count = repeat - fail_num;
    let avg_gen = if success_count > 0 {
        total_gen / success_count as f64
    } else {
        0.0
    };
    let avg_fe = if success_count > 0 {
        total_fe / success_count as f64
    } else {
        0.0
    };

    // Output format matches C++: avg_gen avg_fe ls_fe fail_num
    // Note: Rust version doesn't track local search FE separately, so we use 0.0
    println!("{:.6}  {:.6}  {:.6} {}", avg_gen, avg_fe, 0.0, fail_num);
}

fn run_onemax(
    ell: usize,
    n_initial: usize,
    max_gen: Option<usize>,
    max_fe: Option<usize>,
    seed: u64,
    display: bool,
) -> (bool, usize, usize) {
    let fitness_fn = OneMax;
    run_ga(ell, n_initial, max_gen, max_fe, seed, display, &fitness_fn)
}

fn run_mktrap(
    ell: usize,
    n_initial: usize,
    max_gen: Option<usize>,
    max_fe: Option<usize>,
    seed: u64,
    display: bool,
    k: usize,
) -> (bool, usize, usize) {
    let fitness_fn = MkTrap::new(k);
    run_ga(ell, n_initial, max_gen, max_fe, seed, display, &fitness_fn)
}

fn run_ftrap(
    ell: usize,
    n_initial: usize,
    max_gen: Option<usize>,
    max_fe: Option<usize>,
    seed: u64,
    display: bool,
    _k: usize,
) -> (bool, usize, usize) {
    let fitness_fn = FoldedTrap;
    run_ga(ell, n_initial, max_gen, max_fe, seed, display, &fitness_fn)
}

fn run_cyctrap(
    ell: usize,
    n_initial: usize,
    max_gen: Option<usize>,
    max_fe: Option<usize>,
    seed: u64,
    display: bool,
    k: usize,
) -> (bool, usize, usize) {
    let fitness_fn = CyclicTrap::new(k);
    run_ga(ell, n_initial, max_gen, max_fe, seed, display, &fitness_fn)
}

fn run_ga<F: FitnessFunction>(
    ell: usize,
    n_initial: usize,
    max_gen: Option<usize>,
    max_fe: Option<usize>,
    seed: u64,
    display: bool,
    fitness_fn: &F,
) -> (bool, usize, usize) {
    let mut builder = Dsmga2::new(ell, fitness_fn)
        .population_size(n_initial)
        .seed(seed);

    if let Some(mg) = max_gen {
        builder = builder.max_generations(mg);
    }
    if let Some(mf) = max_fe {
        builder = builder.max_evaluations(mf);
    }

    let mut ga = builder.build();

    if display {
        ga.run_with(|state| {
            println!(
                "Gen {:4}: Best={:.2} Mean={:.2} NFE={}",
                state.generation, state.best_fitness, state.mean_fitness, state.num_evaluations
            );
        });
    } else {
        ga.run();
    }

    let optimum = fitness_fn.optimum(ell);
    let found_optima = (ga.best_fitness() - optimum).abs() < 1e-6;

    (found_optima, ga.generation(), ga.num_evaluations())
}
