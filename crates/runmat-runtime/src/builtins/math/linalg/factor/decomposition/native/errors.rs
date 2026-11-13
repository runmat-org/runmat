//! Selected decomposition errors owner.
use super::*;
pub(in crate::builtins::math::linalg::factor::decomposition) fn selected_runtime(
    runtime: Option<RuntimeContext>,
) -> BuiltinResult<RuntimeContext> {
    if let Some(runtime) = runtime {
        return Ok(runtime);
    }
    if HostAllocationReceipt::policy(None).map_err(|e| {
        crate::context::host_allocation::host_allocation_runtime_error(
            None,
            "RunMat:decomposition:Allocation",
            "decomposition requires its selected context",
            e,
        )
    })? != crate::context::host_allocation::HostAllocationPolicy::Unbounded
    {
        return Err(invalid("decomposition has no selected managed context"));
    }
    Ok(RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    )))
}
pub(in crate::builtins::math::linalg::factor::decomposition) fn publish_value(
    spec: &DecompositionSpec,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    // The canonical recipe contains numeric matrices and scalar metadata only;
    // there is no callback/GC edge between the selected rooted publication and
    // this synchronous transfer into the builtin's returned Value.
    let root = publication::publish(spec, runtime)?;
    Ok(root.value().clone())
}

pub(in crate::builtins::math::linalg::factor::decomposition) fn terminal(
    mut error: RuntimeError,
) -> RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
pub(in crate::builtins::math::linalg::factor::decomposition) fn allocation(
    error: HostAllocationError,
    runtime: &RuntimeContext,
) -> RuntimeError {
    terminal(
        crate::context::host_allocation::host_allocation_runtime_error(
            Some(runtime),
            "RunMat:decomposition:Allocation",
            "decomposition native storage",
            error,
        ),
    )
}
pub(in crate::builtins::math::linalg::factor::decomposition) fn retained_error(
    error: RetainedFactorError,
    runtime: &RuntimeContext,
) -> RuntimeError {
    let result = match error {
        RetainedFactorError::Admission(e)
        | RetainedFactorError::Lu(ManagedSparseLuError::Admission(e))
        | RetainedFactorError::Cholesky(ManagedSparseCholeskyError::Admission(e))
        | RetainedFactorError::Orthogonal(SparseCompleteQrError::Admission(e)) => {
            allocation(e, runtime)
        }
        RetainedFactorError::Execution(e)
        | RetainedFactorError::Lu(ManagedSparseLuError::Execution(e))
        | RetainedFactorError::Cholesky(ManagedSparseCholeskyError::Execution(e))
        | RetainedFactorError::Orthogonal(SparseCompleteQrError::Execution(e)) => {
            e.into_runtime_error_in_context(Some(runtime))
        }
        other => build_runtime_error(format!("decomposition: {other}"))
            .with_builtin(NAME)
            .with_identifier(
                ERROR_INVALID_INPUT
                    .identifier
                    .unwrap_or("RunMat:decomposition:InvalidInput"),
            )
            .with_source(other)
            .build(),
    };
    terminal(result)
}
pub(in crate::builtins::math::linalg::factor::decomposition) fn dense_error(
    error: crate::builtins::math::linalg::ops::dense_factor::DenseFactorError,
    runtime: &RuntimeContext,
) -> RuntimeError {
    use crate::builtins::math::linalg::ops::dense_factor::DenseFactorError as E;
    terminal(match error {
        E::Admission(e)
        | E::Operator(
            crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError::Admission(e),
        ) => allocation(e, runtime),
        E::Execution(e) => e.into_runtime_error_in_context(Some(runtime)),
        E::NotPositiveDefinite => build_runtime_error(
            "decomposition: type 'chol' requires a positive definite coefficient matrix",
        )
        .with_builtin(NAME)
        .with_identifier(
            ERROR_INVALID_INPUT
                .identifier
                .unwrap_or("RunMat:decomposition:InvalidInput"),
        )
        .with_source(E::NotPositiveDefinite)
        .build(),
        other => build_runtime_error(format!("decomposition: {other}"))
            .with_builtin(NAME)
            .with_identifier(
                ERROR_INVALID_INPUT
                    .identifier
                    .unwrap_or("RunMat:decomposition:InvalidInput"),
            )
            .with_source(other)
            .build(),
    })
}
