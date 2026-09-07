use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, provider_restore, tensor};
use crate::builtins::math::elementwise::integer_arithmetic::{try_integer_binary, IntegerBinaryOp};
use crate::BuiltinResult;

use super::{errors, provider, NAME};

pub(super) async fn evaluate(base: &Value, exponent: &Value) -> BuiltinResult<Value> {
    let restore_source = gpu_helpers::select_resident_output_source(
        [base, exponent]
            .into_iter()
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
        NAME,
    )
    .map_err(errors::map_control_flow)?;
    let integer_base = contains_integer(base);
    if integer_base && is_scalar(base) && is_scalar(exponent) {
        let base_host = gather(base).await?;
        let exponent_host = gather(exponent).await?;
        if let Some(result) =
            try_integer_binary(&base_host, &exponent_host, IntegerBinaryOp::Power, NAME)
                .map_err(errors::invalid_argument)?
        {
            return restore(restore_source.as_ref(), result);
        }
    }
    if !integer_base {
        if let Some(result) = provider::try_power(base, exponent).await? {
            return Ok(result);
        }
    }

    let base_host = gather(base).await?;
    let exponent_host = gather(exponent).await?;
    let result = crate::builtins::common::elementwise::power_typed(&base_host, &exponent_host)
        .map_err(errors::map_host_power)?;
    restore(restore_source.as_ref(), result)
}

async fn gather(value: &Value) -> BuiltinResult<Value> {
    crate::dispatcher::gather_if_needed_async(value)
        .await
        .map_err(errors::map_control_flow)
}

fn restore(source: Option<&GpuTensorHandle>, result: Value) -> BuiltinResult<Value> {
    let Some(source) = source else {
        return Ok(result);
    };
    let owner = gpu_helpers::exact_provider_for_handle(source)
        .ok_or_else(|| errors::internal("mpower: resident input owner was lost"))?;
    provider_restore::upload_value_like(owner, result, NAME, source)
        .map_err(errors::map_control_flow)
}

fn contains_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

fn is_scalar(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
        Value::LogicalArray(logical) => logical.data.len() == 1,
        Value::GpuTensor(handle) => crate::builtins::common::shape::is_scalar_shape(&handle.shape),
        _ => false,
    }
}
