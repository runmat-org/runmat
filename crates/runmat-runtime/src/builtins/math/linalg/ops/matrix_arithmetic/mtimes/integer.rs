use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, provider_restore};
use crate::BuiltinResult;

use super::{errors, host, NAME};

pub(super) fn contains(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

pub(super) async fn evaluate(lhs: &Value, rhs: &Value) -> BuiltinResult<Value> {
    let restore_source = gpu_helpers::select_resident_output_source(
        [lhs, rhs].into_iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        NAME,
    )
    .map_err(errors::map_control_flow)?;
    let lhs = crate::dispatcher::gather_if_needed_async(lhs)
        .await
        .map_err(errors::map_control_flow)?;
    let rhs = crate::dispatcher::gather_if_needed_async(rhs)
        .await
        .map_err(errors::map_control_flow)?;
    let result = host::evaluate(lhs, rhs).await?;
    let Some(source) = restore_source else {
        return Ok(result);
    };
    let owner = gpu_helpers::exact_provider_for_handle(&source)
        .ok_or_else(|| errors::internal("mtimes: resident input owner was lost"))?;
    provider_restore::upload_value_like(owner, result, NAME, &source)
        .map_err(errors::map_control_flow)
}
