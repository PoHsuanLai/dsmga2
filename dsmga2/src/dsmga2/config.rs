/// Algorithm configuration (immutable after construction)
#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub(crate) problem_size: usize,
    pub(crate) max_generations: Option<usize>,
    pub(crate) max_evaluations: Option<usize>,
    pub(crate) selection_pressure: usize,
    pub(crate) use_ghc: bool,
}

impl Config {
    pub(crate) fn new(
        problem_size: usize,
        max_generations: Option<usize>,
        max_evaluations: Option<usize>,
        use_ghc: bool,
    ) -> Self {
        Self {
            problem_size,
            max_generations,
            max_evaluations,
            selection_pressure: 2,
            use_ghc,
        }
    }
}
