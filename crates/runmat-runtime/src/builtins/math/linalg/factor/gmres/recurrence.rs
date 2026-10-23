//! Borrowed Runtime adapter to the one value-free shared recurrence.
use super::scalar::{GmresScalar, Native};
pub(super) use runmat_sparse_solvers::krylov::gmres::{
    GmresOperations, GmresOutcome, GmresParameters, GmresWork,
};
use runmat_sparse_solvers::krylov::KrylovFailure;

pub(super) async fn solve<T: GmresScalar, O: GmresOperations<T>>(
    rhs: &[T],
    initial: Option<&[T]>,
    parameters: GmresParameters,
    work: &mut GmresWork<'_, T>,
    operations: &mut O,
) -> Result<GmresOutcome, KrylovFailure<O::Error>> {
    runmat_sparse_solvers::krylov::gmres::solve::<T, Native<T>, O>(
        rhs, initial, parameters, work, operations,
    )
    .await
}

pub(super) fn into_runtime_error(error: KrylovFailure<crate::RuntimeError>) -> crate::RuntimeError {
    match error {
        KrylovFailure::Operation(original) => original,
        KrylovFailure::Invalid(message) => {
            crate::runtime_error::semantic_error("GmresRecurrence", message)
        }
    }
}

#[cfg(test)]
use super::super::iterative::PreconditionerStatus;
#[cfg(test)]
use num_complex::Complex64;
#[cfg(test)]
use runmat_sparse_solvers::krylov::gmres::rotation;
#[cfg(test)]
#[path = "recurrence/tests.rs"]
mod tests;
