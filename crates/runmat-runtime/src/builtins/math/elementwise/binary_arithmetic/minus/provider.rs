use super::*;
use crate::builtins::math::elementwise::binary_arithmetic::provider_support::{
    try_host_left, try_host_right, try_pair, ArithmeticProviderOperation,
};

const OPERATION: ArithmeticProviderOperation = ArithmeticProviderOperation::Subtract;

pub(super) async fn minus_gpu_pair(
    lhs: GpuTensorHandle,
    rhs: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if let Some(output) = try_pair(OPERATION, MINUS_CATALOG_ENTRY.identity, &lhs, &rhs).await? {
        return Ok(output);
    }
    let left = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs)).await?;
    let right = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs)).await?;
    minus_host(left, right)
}

pub(super) async fn minus_gpu_host_left(lhs: GpuTensorHandle, rhs: Value) -> BuiltinResult<Value> {
    if is_real_integer_operand(&rhs) {
        let host_lhs = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs)).await?;
        return minus_host(host_lhs, rhs);
    }
    if let Some(output) = try_host_right(OPERATION, &lhs, &rhs).await? {
        return Ok(output);
    }
    let host_lhs = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs)).await?;
    minus_host(host_lhs, rhs)
}

pub(super) async fn minus_gpu_host_right(lhs: Value, rhs: GpuTensorHandle) -> BuiltinResult<Value> {
    if is_real_integer_operand(&lhs) {
        let host_rhs = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs)).await?;
        return minus_host(lhs, host_rhs);
    }
    if let Some(output) = try_host_left(OPERATION, &lhs, &rhs).await? {
        return Ok(output);
    }
    let host_rhs = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs)).await?;
    minus_host(lhs, host_rhs)
}
