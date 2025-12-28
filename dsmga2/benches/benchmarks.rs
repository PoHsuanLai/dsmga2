use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dsmga2::fitness::{MkTrap, OneMax};
use dsmga2::Dsmga2;

fn benchmark_onemax(c: &mut Criterion) {
    let mut group = c.benchmark_group("onemax");

    for problem_size in [50, 100, 200] {
        group.bench_with_input(
            BenchmarkId::new("rust", problem_size),
            &problem_size,
            |b, &size| {
                b.iter(|| {
                    let fitness_fn = OneMax;
                    let mut ga = Dsmga2::new(size, &fitness_fn)
                        .population_size(size)
                        .max_generations(50)
                        .seed(42)
                        .build();
                    ga.run();
                    black_box(ga.best_fitness())
                });
            },
        );
    }

    group.finish();
}

fn benchmark_trap(c: &mut Criterion) {
    let mut group = c.benchmark_group("trap");

    for num_blocks in [10, 20, 40] {
        group.bench_with_input(
            BenchmarkId::new("rust_k5", num_blocks),
            &num_blocks,
            |b, &blocks| {
                b.iter(|| {
                    let fitness_fn = MkTrap::new(5);
                    let mut ga = Dsmga2::new(5 * blocks, &fitness_fn)
                        .population_size(blocks * 10)
                        .max_generations(50)
                        .seed(42)
                        .build();
                    ga.run();
                    black_box(ga.best_fitness())
                });
            },
        );
    }

    group.finish();
}

fn benchmark_single_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_generation");

    let fitness_fn = OneMax;
    let mut ga = Dsmga2::new(100, &fitness_fn)
        .population_size(100)
        .max_generations(1)
        .seed(42)
        .build();

    group.bench_function("one_step", |b| {
        b.iter(|| black_box(ga.step()));
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_onemax,
    benchmark_trap,
    benchmark_single_generation
);
criterion_main!(benches);
