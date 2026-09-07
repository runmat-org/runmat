use super::*;
use crate::builtins::math::elementwise::binary_arithmetic::provider_support::{
    try_host_left, try_host_right, try_pair, ArithmeticProviderOperation,
};

const OPERATION: ArithmeticProviderOperation = ArithmeticProviderOperation::Add;

pub(super) async fn plus_gpu_pair(
    lhs: GpuTensorHandle,
    rhs: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if let Some(output) = try_pair(OPERATION, PLUS_CATALOG_ENTRY.identity, &lhs, &rhs).await? {
        return Ok(output);
    }
    let left = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let right = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    plus_host(left, right)
}

pub(super) async fn plus_gpu_host_left(lhs: GpuTensorHandle, rhs: Value) -> BuiltinResult<Value> {
    if is_real_integer_operand(&rhs) {
        let host_lhs = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs.clone()))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let host = plus_host(host_lhs, rhs)?;
        return gpu_helpers::restore_class_preserving_value(&lhs, host, BUILTIN_NAME);
    }
    if let Some(output) = try_host_right(OPERATION, &lhs, &rhs).await? {
        return Ok(output);
    }
    let host_lhs = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    plus_host(host_lhs, rhs)
}

pub(super) async fn plus_gpu_host_right(lhs: Value, rhs: GpuTensorHandle) -> BuiltinResult<Value> {
    if is_real_integer_operand(&lhs) {
        let host_rhs = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs.clone()))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let host = plus_host(lhs, host_rhs)?;
        return gpu_helpers::restore_class_preserving_value(&rhs, host, BUILTIN_NAME);
    }
    if let Some(output) = try_host_left(OPERATION, &lhs, &rhs).await? {
        return Ok(output);
    }
    let host_rhs = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    plus_host(lhs, host_rhs)
}
