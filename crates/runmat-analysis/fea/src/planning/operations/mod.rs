//! Operations estimate of a run plan, by usage category, before the run starts.
//!
//! The estimate is an upper estimate of the work the usage ledger records for the run, so a host
//! can compare it with the remaining simulation allowance and refuse a run that cannot finish
//! before it charges anything. Each category follows the batches its kernels reserve: fixed
//! per-element counts for assembly and recovery, the symbolic factor's work envelope for a
//! factorization, and per-iteration work up to a stated iteration count for an iterative solve.

mod admission;
mod static_solid;

pub(crate) use admission::admit_planned_static_operations;

pub use static_solid::{
    estimate_static_solid_operations, estimate_static_solid_operations_from_counts,
    estimate_static_solid_operations_from_plan, StaticSolidOperationCounts, StaticSolveOperations,
    MULTIGRID_ITERATIONS_COVERED,
};

use runmat_simulation_usage::{CountOverflow, OperationCount};

/// The operations a planned run is estimated to charge, by the ledger's usage categories.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FeaOperationEstimate {
    pub meshing: OperationCount,
    pub assembly: OperationCount,
    pub factorization: OperationCount,
    pub solve: OperationCount,
    pub other: OperationCount,
    /// Preconditioned CG iterations the solve estimate covers; `None` for a direct solve.
    /// An iterative solve that needs more iterations charges more than the estimate.
    pub solve_iterations: Option<u32>,
}

impl FeaOperationEstimate {
    pub fn total(&self) -> Result<OperationCount, CountOverflow> {
        [
            self.meshing,
            self.assembly,
            self.factorization,
            self.solve,
            self.other,
        ]
        .into_iter()
        .try_fold(OperationCount::ZERO, OperationCount::checked_add)
    }

    /// The estimate of `runs` independent runs of the same plan (sweep or frequency cases).
    pub fn repeated(&self, runs: u64) -> Result<Self, CountOverflow> {
        Ok(Self {
            meshing: self.meshing.checked_mul(runs)?,
            assembly: self.assembly.checked_mul(runs)?,
            factorization: self.factorization.checked_mul(runs)?,
            solve: self.solve.checked_mul(runs)?,
            other: self.other.checked_mul(runs)?,
            solve_iterations: self.solve_iterations,
        })
    }
}

#[cfg(test)]
mod qualification_tests;
