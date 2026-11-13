//! Optimization-variable construction and original-definition references.
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::{BuiltinResult, RuntimeError};
use runmat_value::{ObjectInstance, Value};
mod construction;
mod indexing;
mod parse;
mod properties;
pub(crate) mod protocol;
mod reference;
mod registration;
#[cfg(test)]
mod tests;
pub(in crate::builtins::math::optim) use construction::original_definition_reference;
pub(in crate::builtins::math::optim) use properties::axis_name as reference_axis_name;
pub(in crate::builtins::math::optim) use reference::View;
pub(crate) const CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::OPTIMIZATION_VARIABLE;
fn invalid(message: impl std::fmt::Display) -> RuntimeError {
    crate::build_runtime_error(format!("optimvar: {message}"))
        .with_identifier("RunMat:optimvar:InvalidInput")
        .with_builtin("optimvar")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn gc_error(message: &'static str, source: runmat_gc::GcError) -> RuntimeError {
    crate::build_runtime_error(format!("optimvar: {message}"))
        .with_identifier("RunMat:optimvar:GcFailure")
        .with_builtin("optimvar")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
}
fn data_error(error: impl std::fmt::Debug) -> RuntimeError {
    invalid(format!("invalid variable storage ({error:?})"))
}
pub(in crate::builtins::math::optim) async fn cooperation(
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<super::common::ScalarCooperation> {
    let cooperation = construction_cooperation(runtime).await?;
    crate::context::host_allocation::require_unbounded_workspace(cooperation.runtime(),"optimvar",
        "variable protocol indexing, property presentation and edit workspace is not yet certified for finite host memory")?;
    Ok(cooperation)
}
async fn construction_cooperation(
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<super::common::ScalarCooperation> {
    let mut cooperation = super::common::ScalarCooperation::for_builtin(
        runmat_builtins::OPTIMVAR_CATALOG_ENTRY.identity,
        runtime,
    );
    cooperation.pin_allocation_epoch()?;
    cooperation.completed(0).await?;
    Ok(cooperation)
}
pub(crate) fn is_reference(object: &ObjectInstance) -> bool {
    object.class_name().is(CLASS)
        && object.selected_class().is_none()
        && matches!(
            object.optimization_variable_payload(),
            Some(runmat_value::optimization_variable::OptimizationVariablePayload::Reference(_))
        )
}
#[runmat_macros::runtime_builtin(
    name = "optimvar",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimvar"
)]
async fn registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::OPTIMVAR_DESCRIPTOR,
        &runmat_builtins::OPTIMVAR_ERROR_INVALID_INPUT,
        "optimvar",
        runtime.as_ref(),
    )?;
    let mut cooperation = construction_cooperation(runtime.as_ref()).await?;
    let registration = registration::Fence::prepare(&cooperation)?;
    let candidate = parse::constructor(args.view(), &mut cooperation).await?;
    registration.check(&cooperation)?;
    construction::create(candidate, registration, &mut cooperation).await
}
