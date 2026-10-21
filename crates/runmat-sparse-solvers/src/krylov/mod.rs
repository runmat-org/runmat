//! Krylov subspace methods: PCG, MINRES, GMRES(m)/FGMRES and BiCGStab, real and complex.
//!
//! Each method is written once as an `async fn` over [`KrylovOps`]. Operator and preconditioner
//! applications, and a per-iteration checkpoint, are awaited, so:
//!
//! - synchronous callers (FEA) use the blocking drivers ([`pcg`], [`minres`], [`gmres`],
//!   [`bicgstab`]) which run the method with ready futures on the calling thread;
//! - asynchronous callers (the runtime's function-handle operators on a cooperative executor)
//!   implement [`KrylovOps`] with genuinely asynchronous callbacks and await the method directly.
//!
//! All vectors live in one caller-visible workspace whose exact length is quoted up front
//! (`*_workspace_len`), so admission can charge it before any work starts. Stopping flags follow
//! MATLAB's convention (0 converged, 1 iteration limit, 2 preconditioner failure, 3 stagnation,
//! 4 breakdown).

use std::future::Future;

use crate::control::{ProgressPhase, SolveControl};
use crate::error::{SolverError, SolverResult};
use crate::operator::{LinearOperator, Preconditioner};
use crate::scalar::Scalar;

mod bicgstab;
mod gmres;
mod minres;
mod pcg;

pub use bicgstab::{bicgstab_async, bicgstab_workspace_len};
pub use gmres::{gmres_async, gmres_workspace_len, GmresOptions, PreconditionerSide};
pub use minres::{minres_async, minres_workspace_len};
pub use pcg::{pcg_async, pcg_workspace_len};

/// Outcome of a callback application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyStatus {
    Applied,
    /// The callback produced a numerically unusable result (e.g. a singular preconditioner).
    NumericalFailure,
}

/// Operator, preconditioner and checkpoint hooks awaited by every Krylov method.
#[allow(async_fn_in_trait)]
pub trait KrylovOps<T: Scalar> {
    /// `y = A x`.
    async fn apply_operator(&mut self, x: &[T], y: &mut [T]) -> SolverResult<ApplyStatus>;
    /// `z = M⁻¹ r`. Only called when [`KrylovOps::has_preconditioner`] is true.
    async fn apply_preconditioner(&mut self, r: &[T], z: &mut [T]) -> SolverResult<ApplyStatus>;
    fn has_preconditioner(&self) -> bool;
    /// Called once per iteration with the current (relative) residual estimate; returning an
    /// error (typically [`SolverError::Cancelled`]) stops the method with that error.
    async fn checkpoint(&mut self, iteration: usize, relative_residual: f64) -> SolverResult<()>;
}

/// Stopping flag (MATLAB numbering for 0–4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopFlag {
    Converged = 0,
    MaxIterations = 1,
    PreconditionerFailure = 2,
    Stagnation = 3,
    Breakdown = 4,
}

impl StopFlag {
    pub fn code(self) -> u8 {
        self as u8
    }
}

/// Stopping controls shared by every method.
#[derive(Debug, Clone, PartialEq)]
pub struct KrylovOptions {
    /// Relative tolerance on `‖b − A x‖ / ‖b‖`.
    pub rtol: f64,
    /// Absolute floor on `‖b − A x‖`.
    pub atol: f64,
    pub max_iterations: usize,
    /// Keep a copy of the minimum-residual iterate and return it on failure (MATLAB semantics).
    pub return_best_iterate: bool,
    /// Record the residual norm at every iteration.
    pub record_history: bool,
    /// Consecutive stagnant steps before flag 3.
    pub max_stagnation_steps: usize,
    /// Extra true-residual verification rounds after the recursive residual converges.
    pub max_verification_rounds: usize,
}

impl Default for KrylovOptions {
    fn default() -> Self {
        Self {
            rtol: 1.0e-8,
            atol: 0.0,
            max_iterations: 1000,
            return_best_iterate: false,
            record_history: false,
            max_stagnation_steps: 3,
            max_verification_rounds: 5,
        }
    }
}

/// Result of a Krylov solve. The solution is written into the caller's `x`.
#[derive(Debug, Clone, PartialEq)]
pub struct KrylovReport {
    pub flag: StopFlag,
    /// Iteration count belonging to the returned iterate (GMRES: total inner iterations).
    pub iterations: usize,
    /// `‖b − A x‖ / ‖b‖` of the returned iterate (`‖b − A x‖` when `b = 0`).
    pub relative_residual: f64,
    pub residual_norm: f64,
    pub rhs_norm: f64,
    /// Residual norms per iteration when requested (index 0 is the initial residual).
    pub residual_history: Vec<f64>,
    /// GMRES (outer, inner) position of the returned iterate.
    pub restart_position: Option<(usize, usize)>,
    pub operator_applications: usize,
    pub preconditioner_applications: usize,
}

impl KrylovReport {
    pub fn converged(&self) -> bool {
        self.flag == StopFlag::Converged
    }
}

pub(crate) struct Counters {
    pub operator: usize,
    pub preconditioner: usize,
}

/// Runs a future whose awaited operations are all synchronous to completion on this thread.
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = std::task::Waker::noop();
    let mut context = std::task::Context::from_waker(waker);
    loop {
        if let std::task::Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        // Synchronous operations never return Pending; an asynchronous `KrylovOps` must be driven
        // by its own executor instead of `block_on`.
        std::thread::yield_now();
    }
}

/// [`KrylovOps`] over synchronous operators with optional cancellation/progress control.
pub struct SyncOps<'a, T: Scalar> {
    pub operator: &'a dyn LinearOperator<T>,
    pub preconditioner: Option<&'a dyn Preconditioner<T>>,
    pub control: SolveControl<'a>,
}

impl<T: Scalar> KrylovOps<T> for SyncOps<'_, T> {
    async fn apply_operator(&mut self, x: &[T], y: &mut [T]) -> SolverResult<ApplyStatus> {
        self.operator.apply(x, y)?;
        Ok(ApplyStatus::Applied)
    }
    async fn apply_preconditioner(&mut self, r: &[T], z: &mut [T]) -> SolverResult<ApplyStatus> {
        match self.preconditioner {
            Some(m) => match m.apply(r, z) {
                Ok(()) => Ok(ApplyStatus::Applied),
                Err(SolverError::Singular { .. } | SolverError::Breakdown(_)) => {
                    Ok(ApplyStatus::NumericalFailure)
                }
                Err(error) => Err(error),
            },
            None => {
                z.copy_from_slice(r);
                Ok(ApplyStatus::Applied)
            }
        }
    }
    fn has_preconditioner(&self) -> bool {
        self.preconditioner.is_some()
    }
    async fn checkpoint(&mut self, iteration: usize, relative_residual: f64) -> SolverResult<()> {
        self.control.check()?;
        self.control
            .report_iteration(ProgressPhase::Krylov, iteration, relative_residual);
        Ok(())
    }
}

fn check_dims<T: Scalar>(op: &dyn LinearOperator<T>, b: &[T], x: &[T]) -> SolverResult<()> {
    if op.nrows() != op.ncols() || b.len() != op.nrows() || x.len() != op.ncols() {
        return Err(SolverError::Dimension(format!(
            "operator {}x{}, rhs {}, x {}",
            op.nrows(),
            op.ncols(),
            b.len(),
            x.len()
        )));
    }
    Ok(())
}

/// Blocking PCG for Hermitian positive definite `A` with an HPD preconditioner. `x` holds the
/// initial guess on entry and the solution on return.
pub fn pcg<T: Scalar>(
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    control: SolveControl<'_>,
) -> SolverResult<KrylovReport> {
    check_dims(operator, b, x)?;
    let mut work = crate::par::try_zeros::<T>(pcg_workspace_len(b.len(), options))?;
    let mut ops = SyncOps {
        operator,
        preconditioner,
        control,
    };
    block_on(pcg_async(&mut ops, b, x, options, &mut work))
}

/// Blocking MINRES for Hermitian (possibly indefinite) `A` with an HPD preconditioner.
pub fn minres<T: Scalar>(
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    control: SolveControl<'_>,
) -> SolverResult<KrylovReport> {
    check_dims(operator, b, x)?;
    let mut work = crate::par::try_zeros::<T>(minres_workspace_len(b.len(), options))?;
    let mut ops = SyncOps {
        operator,
        preconditioner,
        control,
    };
    block_on(minres_async(&mut ops, b, x, options, &mut work))
}

/// Blocking restarted GMRES / FGMRES for general `A`.
pub fn gmres<T: Scalar>(
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    gmres_options: &GmresOptions,
    control: SolveControl<'_>,
) -> SolverResult<KrylovReport> {
    check_dims(operator, b, x)?;
    let mut work =
        crate::par::try_zeros::<T>(gmres_workspace_len(b.len(), options, gmres_options))?;
    let mut ops = SyncOps {
        operator,
        preconditioner,
        control,
    };
    block_on(gmres_async(&mut ops, b, x, options, gmres_options, &mut work))
}

/// Blocking right-preconditioned BiCGStab for general `A`.
pub fn bicgstab<T: Scalar>(
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    control: SolveControl<'_>,
) -> SolverResult<KrylovReport> {
    check_dims(operator, b, x)?;
    let mut work = crate::par::try_zeros::<T>(bicgstab_workspace_len(b.len(), options))?;
    let mut ops = SyncOps {
        operator,
        preconditioner,
        control,
    };
    block_on(bicgstab_async(&mut ops, b, x, options, &mut work))
}

/// Shared helpers for the method implementations.
pub(crate) fn tolerance(options: &KrylovOptions, rhs_norm: f64) -> f64 {
    (options.rtol * rhs_norm).max(options.atol)
}

pub(crate) fn relative(residual: f64, rhs_norm: f64) -> f64 {
    if rhs_norm > 0.0 {
        residual / rhs_norm
    } else {
        residual
    }
}

pub(crate) fn split_work<T>(work: &mut [T], n: usize, count: usize) -> Vec<&mut [T]> {
    assert!(work.len() >= n * count, "Krylov workspace too small");
    let mut out = Vec::with_capacity(count);
    let mut rest = work;
    for _ in 0..count {
        let (head, tail) = rest.split_at_mut(n);
        out.push(head);
        rest = tail;
    }
    out
}

#[cfg(test)]
mod tests;
