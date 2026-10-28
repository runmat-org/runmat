//! Cancellation and progress hooks threaded through every long-running solver loop.

use crate::error::{SolverError, SolverResult};

/// Phase of a solver run reported to progress observers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressPhase {
    Ordering,
    SymbolicFactorization,
    NumericFactorization,
    TriangularSolve,
    Refinement,
    Krylov,
    MultigridSetup,
    Eigen,
    Completed,
    Failed,
    Cancelled,
}

/// One progress observation. `residual` is relative where meaningful.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Progress {
    pub phase: ProgressPhase,
    pub iteration: usize,
    pub residual: f64,
    /// Converged eigenpairs (eigen phases) or zero.
    pub converged: usize,
}

/// Borrowed hooks; both are optional. Callbacks run on the calling thread between parallel
/// regions, never concurrently with each other.
#[derive(Clone, Copy, Default)]
pub struct SolveControl<'a> {
    pub cancelled: Option<&'a (dyn Fn() -> bool + Sync)>,
    pub progress: Option<&'a (dyn Fn(Progress) + Sync)>,
    /// Report Krylov progress every `progress_stride` iterations (0 = only first and last).
    pub progress_stride: usize,
}

impl<'a> SolveControl<'a> {
    pub const NONE: SolveControl<'static> = SolveControl {
        cancelled: None,
        progress: None,
        progress_stride: 0,
    };

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.is_some_and(|cancelled| cancelled())
    }

    pub fn check(&self) -> SolverResult<()> {
        if self.is_cancelled() {
            Err(SolverError::Cancelled)
        } else {
            Ok(())
        }
    }

    pub fn report(&self, progress: Progress) {
        if let Some(callback) = self.progress {
            callback(progress);
        }
    }

    pub(crate) fn report_iteration(&self, phase: ProgressPhase, iteration: usize, residual: f64) {
        if self.progress.is_none() {
            return;
        }
        let stride = if self.progress_stride == 0 {
            256
        } else {
            self.progress_stride
        };
        if iteration <= 1 || iteration.is_multiple_of(stride) {
            self.report(Progress {
                phase,
                iteration,
                residual,
                converged: 0,
            });
        }
    }
}

impl std::fmt::Debug for SolveControl<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SolveControl")
            .field("cancelled", &self.cancelled.is_some())
            .field("progress", &self.progress.is_some())
            .field("progress_stride", &self.progress_stride)
            .finish()
    }
}
