//! Batching is semantic admission, not a second provider dispatcher.
mod admission;
mod normalize;

use super::callback::Callable;
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_builtins::{arrayfun_gpu_callback, ArrayfunGpuCallback, ArrayfunGpuUnary};
use runmat_value::Value;

pub(super) async fn try_fast_path(
    context: Option<&crate::context::RuntimeContext>,
    callable: &Callable,
    inputs: &crate::call::BuiltinArguments,
    error_handler: Option<&Callable>,
) -> BuiltinResult<Option<RootedValue>> {
    if error_handler.is_some()
        || !inputs
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Ok(None);
    }
    let Some(shape) = admission::shape(inputs.view())? else {
        return Ok(None);
    };
    // Empty mapping never invokes the callback or its domain/extension gates.
    if shape.contains(&0) {
        return Ok(None);
    }
    let Some(identity) = callable.builtin_identity() else {
        return Ok(None);
    };
    let Some(operation) = arrayfun_gpu_callback(identity) else {
        return Ok(None);
    };
    if crate::user_functions::resolve_semantic_function_by_name(identity.name).is_some() {
        return Ok(None);
    }
    let Some(binding) =
        crate::builtins::common::spec::selected_builtin_binding(identity.name, None, context)
    else {
        return Ok(None);
    };
    if !admission::equivalent(operation, identity.name, inputs.view()) {
        return Ok(None);
    }
    if matches!(operation, ArrayfunGpuCallback::Unary(ArrayfunGpuUnary::Log))
        && !crate::builtins::math::elementwise::logarithms::log::mapped_batch_equivalent(&inputs[0])
            .await
            .map_err(callback_error)?
    {
        return Ok(None);
    }
    // Canonical dispatch owns preparation, class promotion, exact provider,
    // unsupported fallback and output admission. Never retry an execution error.
    let result = crate::dispatcher::call_selected_retained_builtin_async_with_outputs(
        binding,
        inputs.fork(),
        1,
        context.cloned(),
    )
    .await
    .map_err(callback_error)?;
    let result = crate::dispatcher::resolve_single_sequence(result).map_err(callback_error)?;
    normalize::uniform(result, &shape, inputs.view())
        .await
        .map(Some)
}

fn callback_error(error: crate::RuntimeError) -> crate::RuntimeError {
    super::error::arrayfun_flow_with_source(format!("arrayfun: {}", error.message()), error)
}
