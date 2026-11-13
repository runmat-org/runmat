//! Original callable selection is shared by residual and multiply roles.
use super::*;
pub(super) enum Derivatives {
    Numerical,
    Matrix,
    Opaque(runmat_execution::resource::ExactShared<RepeatableCallable>),
}
impl Derivatives {
    pub(super) fn supplied(&self) -> bool {
        !matches!(self, Self::Numerical)
    }
    pub(super) fn new(
        supplied: bool,
        multiply: Option<&Value>,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        if !supplied {
            return Ok(Self::Numerical);
        }
        let Some(value) = multiply else {
            return Ok(Self::Matrix);
        };
        if !value.is_function_handle() {
            return Err(lsq_error_with_detail(
                &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                "JacobianMultiplyFcn must be a single function handle or []",
            ));
        }
        let target = crate::context::with_original_scope(runtime, || {
            RootedValue::in_optional_context(value.clone(), runtime)
        })?;
        let function = select(target, runtime)?;
        if runtime.is_some_and(|runtime| runtime.cancellation().is_cancelled()) {
            return Err(crate::execution::ExecutionServiceError::Cancelled
                .into_runtime_error_in_context(runtime));
        }
        // This invocation already passed the solver's unbounded-workspace
        // admission. Only one immutable selected-callable header is allocated;
        // pending/accepted Jinfo generations clone its ownership metadata.
        runmat_execution::resource::ExactShared::try_new(function)
            .map(Self::Opaque)
            .map_err(|_| {
                crate::build_runtime_error("lsqnonlin: selected multiply owner allocation failed")
                    .with_source(runmat_value::ExactHostNumericAllocationError::AllocationFailed)
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
            })
    }
}
pub(super) fn select(
    rooted: RootedValue,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<RepeatableCallable> {
    crate::context::with_original_scope(runtime, || {
        match runtime.and_then(|runtime| runtime.selected_graph_program()) {
            Some(program) => program.select_repeatable_callable(
                runtime.expect("selected program has original runtime"),
                rooted,
            ),
            None => RepeatableCallable::select_deferred_rooted(runtime, rooted),
        }
    })
}
