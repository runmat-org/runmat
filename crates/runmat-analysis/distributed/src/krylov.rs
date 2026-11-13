//! Settings and evidence of the distributed preconditioned conjugate gradient iteration.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KrylovSettings {
    /// Stop when the global residual 2-norm falls below this fraction of the right-hand side.
    pub relative_tolerance: f64,
    pub maximum_iterations: u32,
    /// Add the coarse (null-space) level to the overlapping Schwarz preconditioner.
    pub coarse_space: bool,
}

impl Default for KrylovSettings {
    fn default() -> Self {
        Self {
            relative_tolerance: 1e-10,
            maximum_iterations: 2_000,
            coarse_space: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IterationRecord {
    pub iteration: u32,
    pub relative_residual: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KrylovSummary {
    pub iterations: u32,
    pub relative_residual: f64,
    pub converged: bool,
    pub history: Vec<IterationRecord>,
}
