use runmat_value::Value;

use super::ProductResult as BuiltinResult;
use crate::builtins::common::{gpu_helpers, provider_restore};

use super::{errors, host, NAME};

pub(super) fn contains(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

pub(super) async fn evaluate(
    context: Option<&crate::context::RuntimeContext>,
    lhs: &Value,
    rhs: &Value,
) -> crate::BuiltinResult<BuiltinResult<Value>> {
    gpu_helpers::require_resident_compute_values(&[lhs, rhs], false, || {
        crate::RuntimeError::new("mtimes: resident input owner was lost")
    })?;
    let restore_source = gpu_helpers::select_resident_output_source(
        [lhs, rhs].into_iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        NAME,
    )?;
    let lhs = crate::dispatcher::gather_if_needed_async(lhs).await?;
    let rhs = crate::dispatcher::gather_if_needed_async(rhs).await?;
    let result = product_try!(host::evaluate(context, lhs, rhs).await?);
    let Some(source) = restore_source else {
        return Ok(Ok(result));
    };
    let owner = product_try!(gpu_helpers::exact_provider_for_handle(&source)
        .ok_or_else(|| errors::internal("mtimes: resident input owner was lost")));
    provider_restore::upload_value_like(owner, result, NAME, &source)
        .map(Ok)
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
