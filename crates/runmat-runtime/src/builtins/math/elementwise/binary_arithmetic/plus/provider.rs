use super::*;
use crate::builtins::math::elementwise::binary_arithmetic::provider_support::{
    try_host_left, try_host_right, try_pair, ArithmeticProviderOperation, HostFallbackAdmission,
};

const OPERATION: ArithmeticProviderOperation = ArithmeticProviderOperation::Add;

pub(super) async fn plus_gpu_pair(
    runtime: Option<&crate::context::RuntimeContext>,
    lhs: GpuTensorHandle,
    rhs: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if let Some(output) = try_pair(OPERATION, PLUS_CATALOG_ENTRY.identity, &lhs, &rhs).await? {
        return Ok(output);
    }
    let fallback = HostFallbackAdmission::new(
        runtime,
        &PLUS_CATALOG_ENTRY,
        &Value::GpuTensor(lhs.clone()),
        &Value::GpuTensor(rhs.clone()),
    )?;
    let left = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let right = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    fallback.finish(plus_host(left, right)?)
}

pub(super) async fn plus_gpu_host_left(
    runtime: Option<&crate::context::RuntimeContext>,
    lhs: GpuTensorHandle,
    rhs: Value,
) -> BuiltinResult<Value> {
    if is_real_integer_operand(&rhs) {
        let fallback = HostFallbackAdmission::new(
            runtime,
            &PLUS_CATALOG_ENTRY,
            &Value::GpuTensor(lhs.clone()),
            &rhs,
        )?;
        let host_lhs = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs.clone()))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let host = plus_host(host_lhs, rhs)?;
        let output = fallback.finish(host)?;
        return if runmat_accelerate_api::handle_is_explicit(&lhs) {
            Ok(output)
        } else {
            gpu_helpers::restore_class_preserving_value(&lhs, output, BUILTIN_NAME)
        };
    }
    if let Some(output) = try_host_right(OPERATION, &lhs, &rhs).await? {
        return Ok(output);
    }
    let fallback = HostFallbackAdmission::new(
        runtime,
        &PLUS_CATALOG_ENTRY,
        &Value::GpuTensor(lhs.clone()),
        &rhs,
    )?;
    let host_lhs = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    fallback.finish(plus_host(host_lhs, rhs)?)
}

pub(super) async fn plus_gpu_host_right(
    runtime: Option<&crate::context::RuntimeContext>,
    lhs: Value,
    rhs: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if is_real_integer_operand(&lhs) {
        let fallback = HostFallbackAdmission::new(
            runtime,
            &PLUS_CATALOG_ENTRY,
            &lhs,
            &Value::GpuTensor(rhs.clone()),
        )?;
        let host_rhs = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs.clone()))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let host = plus_host(lhs, host_rhs)?;
        let output = fallback.finish(host)?;
        return if runmat_accelerate_api::handle_is_explicit(&rhs) {
            Ok(output)
        } else {
            gpu_helpers::restore_class_preserving_value(&rhs, output, BUILTIN_NAME)
        };
    }
    if let Some(output) = try_host_left(OPERATION, &lhs, &rhs).await? {
        return Ok(output);
    }
    let fallback = HostFallbackAdmission::new(
        runtime,
        &PLUS_CATALOG_ENTRY,
        &lhs,
        &Value::GpuTensor(rhs.clone()),
    )?;
    let host_rhs = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    fallback.finish(plus_host(lhs, host_rhs)?)
}
