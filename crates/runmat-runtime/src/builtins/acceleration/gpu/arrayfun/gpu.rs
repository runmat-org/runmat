//! Batching is semantic admission, not a second provider dispatcher.
mod admission;
mod normalize;

use super::callback::Callable;
use crate::BuiltinResult;
use runmat_builtins::{arrayfun_gpu_callback, ArrayfunGpuCallback, ArrayfunGpuUnary};
use runmat_value::Value;

pub(super) async fn try_fast_path(
    callable: &Callable,
    inputs: &[Value],
    error_handler: Option<&Callable>,
) -> BuiltinResult<Option<Value>> {
    if error_handler.is_some()
        || !inputs
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Ok(None);
    }
    let Some(shape) = admission::shape(inputs)? else {
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
    if crate::user_functions::resolve_semantic_function_by_name(identity.name).is_some()
        || crate::builtins::common::spec::selected_builtin_binding(identity.name, None, None)
            .is_none()
    {
        return Ok(None);
    }
    if !admission::equivalent(operation, identity.name, inputs) {
        return Ok(None);
    }
    if matches!(operation, ArrayfunGpuCallback::Unary(ArrayfunGpuUnary::Log))
        && !crate::builtins::math::elementwise::logarithms::log::mapped_batch_equivalent(&inputs[0])
            .await?
    {
        return Ok(None);
    }
    // Canonical dispatch owns preparation, class promotion, exact provider,
    // unsupported fallback and output admission. Never retry an execution error.
    let result = callable.call(inputs).await?;
    normalize::uniform(result, &shape, inputs).await.map(Some)
}
