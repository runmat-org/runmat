//! Device-resident algebraic multigrid for reduced-precision preconditioning.
//!
//! A numerical domain (for example the shared sparse solver crate's smoothed-aggregation AMG)
//! builds a hierarchy on the host and uploads it once. Each application uploads one residual
//! and downloads one correction; everything in between (the V-cycle, or a fixed-budget inner
//! conjugate-gradient solve preconditioned by it) stays on the device. The caller's outer
//! Krylov loop keeps its own precision and true-residual certificate; this contract only
//! promises an approximate inverse, applied with the provider's native scalar.

use anyhow::ensure;

/// CSR operator with 32-bit indices and single-precision values.
#[derive(Clone, Copy, Debug)]
pub struct MultigridCsr<'a> {
    pub rows: u32,
    pub columns: u32,
    pub row_offsets: &'a [u32],
    pub column_indices: &'a [u32],
    pub values: &'a [f32],
}

impl MultigridCsr<'_> {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.row_offsets.len() == self.rows as usize + 1
                && self.row_offsets.first() == Some(&0)
                && self.row_offsets.windows(2).all(|w| w[0] <= w[1])
                && *self.row_offsets.last().unwrap() as usize == self.column_indices.len()
                && self.column_indices.len() == self.values.len()
                && self.column_indices.iter().all(|&c| c < self.columns),
            "multigrid CSR operator is malformed"
        );
        Ok(())
    }
}

/// One level: operator, Chebyshev smoother data and transfer operators.
#[derive(Clone, Debug)]
pub struct MultigridLevel<'a> {
    pub matrix: MultigridCsr<'a>,
    /// Unknowns per node.
    pub block: u32,
    /// Inverse diagonal blocks, column-major `block × block` per node.
    pub inverse_blocks: &'a [f32],
    /// Upper Chebyshev bound of `ρ(D⁻¹A)`.
    pub lambda_max: f32,
    /// `P` (this level × next) and `R` (next × this); absent on the coarsest level.
    pub prolongation: Option<MultigridCsr<'a>>,
    pub restriction: Option<MultigridCsr<'a>>,
}

/// A hierarchy to upload. The V-cycle is: zero iterate, Chebyshev pre-smoothing of
/// `smoother_degree`, restrict the residual, recurse, prolongate and add, Chebyshev
/// post-smoothing; the coarsest level multiplies by `coarse_inverse` (dense, row-major).
#[derive(Clone, Debug)]
pub struct SparseMultigridRequest<'a> {
    pub levels: Vec<MultigridLevel<'a>>,
    pub coarse_inverse: &'a [f32],
    pub smoother_degree: u32,
    pub chebyshev_ratio: f32,
}

impl SparseMultigridRequest<'_> {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(!self.levels.is_empty(), "multigrid hierarchy has no levels");
        ensure!(
            self.smoother_degree >= 1 && self.chebyshev_ratio > 1.0,
            "invalid smoother"
        );
        for (index, level) in self.levels.iter().enumerate() {
            level.matrix.validate()?;
            let n = level.matrix.rows;
            ensure!(
                level.matrix.columns == n && level.block >= 1 && n % level.block == 0,
                "level operator shape"
            );
            ensure!(
                level.inverse_blocks.len()
                    == (n as usize / level.block as usize) * (level.block * level.block) as usize,
                "inverse block extent"
            );
            ensure!(
                level.lambda_max.is_finite() && level.lambda_max > 0.0,
                "Chebyshev bound"
            );
            let last = index + 1 == self.levels.len();
            match (&level.prolongation, &level.restriction, last) {
                (None, None, true) => {
                    ensure!(
                        self.coarse_inverse.len() == (n as usize) * (n as usize),
                        "coarse inverse extent"
                    );
                }
                (Some(p), Some(r), false) => {
                    p.validate()?;
                    r.validate()?;
                    let next = self.levels[index + 1].matrix.rows;
                    ensure!(
                        p.rows == n && p.columns == next && r.rows == next && r.columns == n,
                        "transfer operator shapes"
                    );
                }
                _ => {
                    anyhow::bail!("transfer operators must exist exactly above the coarsest level")
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseMultigridHandle {
    pub device_id: u32,
    pub hierarchy_id: u64,
    pub rows: u32,
}

/// What one application computes from the uploaded residual.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseMultigridApply {
    /// One V-cycle (a fixed linear map up to rounding).
    VCycle,
    /// `iterations` steps of conjugate gradients on the finest operator, preconditioned by the
    /// V-cycle, from a zero initial guess, with no host synchronization in between (a variable
    /// preconditioner for a flexible outer method).
    Pcg { iterations: u32 },
    /// The finest operator, `A · rhs`, evaluated `repetitions` times into the same output
    /// (a device operator apply; repetitions isolate kernel time from transfers).
    Operator { repetitions: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct SparseMultigridSolveRequest<'a> {
    pub hierarchy: &'a SparseMultigridHandle,
    pub rhs: &'a [f32],
    pub apply: SparseMultigridApply,
}

#[derive(Clone, Debug)]
pub struct SparseMultigridSolveResult {
    pub solution: Vec<f32>,
    /// Compute dispatches recorded for the application (evidence for profiling).
    pub dispatches: u32,
}
