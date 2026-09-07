use super::*;
use crate::builtins::math::elementwise::binary_arithmetic::provider_support::{
    try_host_left, try_host_right, try_pair, ArithmeticProviderOperation,
};

const OPERATION: ArithmeticProviderOperation = ArithmeticProviderOperation::Multiply;

pub(super) async fn times_gpu_pair(
    lhs: GpuTensorHandle,
    rhs: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if let Some(output) = try_pair(OPERATION, TIMES_CATALOG_ENTRY.identity, &lhs, &rhs).await? {
        return Ok(output);
    }
    let left = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let right = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    times_host(left, right)
}

pub(super) async fn times_gpu_host_left(lhs: GpuTensorHandle, rhs: Value) -> BuiltinResult<Value> {
    if let Some(output) = try_host_right(OPERATION, &lhs, &rhs).await? {
        return Ok(output);
    }
    let host_lhs = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    times_host(host_lhs, rhs)
}

pub(super) async fn times_gpu_host_right(lhs: Value, rhs: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(output) = try_host_left(OPERATION, &lhs, &rhs).await? {
        return Ok(output);
    }
    let host_rhs = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    times_host(lhs, host_rhs)
}
