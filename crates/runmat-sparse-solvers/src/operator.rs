//! Operator and preconditioner abstractions used by the synchronous Krylov drivers.
//!
//! Callers whose operators are asynchronous callbacks (the runtime's function-handle `pcg` /
//! `gmres`) use the reverse-communication state machines in [`crate::krylov`] instead and never
//! implement these traits.

use crate::csr::{CsrMatrix, CsrRef};
use crate::error::SolverResult;
use crate::scalar::Scalar;

mod metered_csr;
pub use metered_csr::MeteredCsrOperator;

/// `y = A x` for a square or rectangular operator.
pub trait LinearOperator<T: Scalar>: Sync {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn apply(&self, x: &[T], y: &mut [T]) -> SolverResult<()>;
    /// Exact measured reference work for one application, when the operator owns a
    /// bounded quote. Async callers reject an absent quote before executing it.
    fn reference_apply_work(
        &self,
    ) -> SolverResult<Option<runmat_simulation_usage::OperationCount>> {
        Ok(None)
    }
}

/// `z ≈ A⁻¹ r`.
pub trait Preconditioner<T: Scalar>: Sync {
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()>;
    fn reference_apply_work(
        &self,
    ) -> SolverResult<Option<runmat_simulation_usage::OperationCount>> {
        Ok(None)
    }
    /// Short label for solver provenance (e.g. `"amg"`, `"ilu0"`).
    fn label(&self) -> &'static str {
        "custom"
    }
    /// True when the preconditioner is a fixed symmetric positive definite linear map (required by
    /// PCG and MINRES). Variable preconditioners need FGMRES.
    fn is_symmetric_positive_definite(&self) -> bool {
        true
    }
    /// True when the apply is variable or nonlinear (an inner iterative solve, a reduced-precision
    /// cycle): the drivers then select flexible GMRES and flexible CG.
    fn is_variable(&self) -> bool {
        false
    }
}

impl<T: Scalar> LinearOperator<T> for CsrRef<'_, T> {
    fn nrows(&self) -> usize {
        CsrRef::nrows(self)
    }
    fn ncols(&self) -> usize {
        CsrRef::ncols(self)
    }
    fn apply(&self, x: &[T], y: &mut [T]) -> SolverResult<()> {
        self.spmv(x, y);
        Ok(())
    }
}

impl<T: Scalar> LinearOperator<T> for CsrMatrix<T> {
    fn nrows(&self) -> usize {
        CsrMatrix::nrows(self)
    }
    fn ncols(&self) -> usize {
        CsrMatrix::ncols(self)
    }
    fn apply(&self, x: &[T], y: &mut [T]) -> SolverResult<()> {
        self.as_ref().spmv(x, y);
        Ok(())
    }
}

/// Operator from a closure.
pub struct FnOperator<F> {
    pub nrows: usize,
    pub ncols: usize,
    pub apply: F,
}

impl<T: Scalar, F: Fn(&[T], &mut [T]) -> SolverResult<()> + Sync> LinearOperator<T>
    for FnOperator<F>
{
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn ncols(&self) -> usize {
        self.ncols
    }
    fn apply(&self, x: &[T], y: &mut [T]) -> SolverResult<()> {
        (self.apply)(x, y)
    }
}

/// `z = r`.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityPreconditioner;

impl<T: Scalar> Preconditioner<T> for IdentityPreconditioner {
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
        z.copy_from_slice(r);
        Ok(())
    }
    fn label(&self) -> &'static str {
        "none"
    }
}

impl<T: Scalar, P: Preconditioner<T> + ?Sized> Preconditioner<T> for Box<P> {
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
        (**self).apply(r, z)
    }
    fn label(&self) -> &'static str {
        (**self).label()
    }
    fn is_symmetric_positive_definite(&self) -> bool {
        (**self).is_symmetric_positive_definite()
    }
    fn is_variable(&self) -> bool {
        (**self).is_variable()
    }
}

impl<T: Scalar, P: Preconditioner<T> + ?Sized> Preconditioner<T> for &P {
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
        (**self).apply(r, z)
    }
    fn label(&self) -> &'static str {
        (**self).label()
    }
    fn is_symmetric_positive_definite(&self) -> bool {
        (**self).is_symmetric_positive_definite()
    }
    fn is_variable(&self) -> bool {
        (**self).is_variable()
    }
}

impl<T: Scalar, A: LinearOperator<T> + ?Sized> LinearOperator<T> for &A {
    fn nrows(&self) -> usize {
        (**self).nrows()
    }
    fn ncols(&self) -> usize {
        (**self).ncols()
    }
    fn apply(&self, x: &[T], y: &mut [T]) -> SolverResult<()> {
        (**self).apply(x, y)
    }
}
