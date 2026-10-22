//! Legacy omit-NaN maximum, sharing the admitted extrema parser and engine.
use super::*;
use runmat_builtins::{NANMAX_DESCRIPTOR, NANMAX_ERROR_TOO_MANY_OUTPUTS, NANMAX_INTEGER_EXTENSION};
#[runtime_builtin(
    name = "nanmax",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::reduction::max"
)]
pub(super) async fn nanmax_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &NANMAX_DESCRIPTOR,
        &NANMAX_ERROR_TOO_MANY_OUTPUTS,
        "nanmax",
    )
    .map_err(super::super::numerical_execution::terminal)?;
    reject_pairwise_multiple_outputs(rest.view())?;
    if is_real_typed_integer_value(&value) || rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &NANMAX_INTEGER_EXTENSION,
            "nanmax",
        )?;
    }
    let call = super::super::numerical_execution::NumericalCall::new(
        runmat_builtins::NumericalReductionKind::NonNanMaximum,
        runtime.as_ref(),
    )?;
    // nanmax deliberately publishes a host result; this is not a provider-failure
    // fallback or permission for other resident-result APIs to gather explicitly.
    call.host()?;
    let evaluated = evaluate_for_missing(
        runtime.as_ref(),
        value,
        rest.view(),
        builtin_demand(),
        Some(ReductionNaN::Omit),
    )
    .await?;
    call.finish_value(evaluation_to_value(evaluated))
}
fn is_real_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value,Value::Tensor(t) if t.integer_storage().is_some())
        || matches!(value,Value::GpuTensor(h) if runmat_accelerate_api::handle_integer_type(h).is_some())
}
