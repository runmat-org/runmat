//! Typed numeric/control projection; no error-code deoptimization guesses.
use crate::{
    builtins::math::linalg::ops::{
        dense_factor::DenseFactorError, iterative_preconditioner::MatrixFactorError,
        linear_operator::CooperativeOperatorError, sparse_factor::ManagedSparseLuError,
        sparse_operator::SparseOperatorError, triangular_operator::TriangularOperatorError,
    },
    context::{host_allocation::HostAllocationError, RuntimeContext},
    RuntimeError,
};
pub(super) fn invalid(message: &str) -> RuntimeError {
    super::super::error_with_detail(
        format!("eigs: {message}"),
        &super::super::ERROR_INVALID_ARGUMENT,
    )
}
pub(super) fn unsupported(message: &str) -> RuntimeError {
    super::super::error_with_detail(format!("eigs: {message}"), &super::super::ERROR_UNSUPPORTED)
}
pub(super) fn admission(
    runtime: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> RuntimeError {
    let mut error =
        crate::builtins::math::linalg::factor::iterative::copy::admission(runtime, "eigs", source);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
pub(super) fn matrix_error(
    runtime: Option<&RuntimeContext>,
    source: SparseOperatorError,
) -> RuntimeError {
    match source {
        SparseOperatorError::Admission(source) => admission(runtime, source),
        source => {
            crate::build_runtime_error(format!("eigs: coefficient application failed: {source:?}"))
                .with_builtin("eigs")
                .with_identifier("RunMat:eigs:Operator")
                .with_source(MatrixFactorError::Operator(source))
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        }
    }
}
pub(super) fn operator_error(
    runtime: Option<&RuntimeContext>,
    source: CooperativeOperatorError,
) -> RuntimeError {
    let mut error = match source {
        CooperativeOperatorError::Operator(source) => matrix_error(runtime, source),
        CooperativeOperatorError::Execution(source) => {
            source.into_runtime_error_in_context(runtime)
        }
    };
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
pub(super) fn factor_error(
    runtime: Option<&RuntimeContext>,
    source: MatrixFactorError,
) -> RuntimeError {
    let mut error = match source {
        MatrixFactorError::ManagedSparse(ManagedSparseLuError::Execution(source))
        | MatrixFactorError::Execution(source)
        | MatrixFactorError::Dense(DenseFactorError::Execution(source))
        | MatrixFactorError::Triangular(TriangularOperatorError::Execution(source)) => {
            source.into_runtime_error_in_context(runtime)
        }
        MatrixFactorError::Admission(source)
        | MatrixFactorError::ManagedSparse(ManagedSparseLuError::Admission(source))
        | MatrixFactorError::Dense(DenseFactorError::Admission(source))
        | MatrixFactorError::Operator(SparseOperatorError::Admission(source))
        | MatrixFactorError::Triangular(TriangularOperatorError::Admission(source))
        | MatrixFactorError::Triangular(TriangularOperatorError::Operator(
            SparseOperatorError::Admission(source),
        )) => admission(runtime, source),
        source => {
            crate::build_runtime_error(format!("eigs: reusable inverse action failed: {source}"))
                .with_builtin("eigs")
                .with_identifier("RunMat:eigs:Factorization")
                .with_source(source)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        }
    };
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
