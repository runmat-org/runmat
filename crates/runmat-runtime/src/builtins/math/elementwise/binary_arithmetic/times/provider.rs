use super::*;
use crate::builtins::math::elementwise::binary_arithmetic::provider_support::{
    broadcast_repetitions, device_real_scalar, host_real_scalar, is_scalar_shape,
    resident_output_from_sources,
};

pub(super) async fn times_gpu_pair(
    lhs: GpuTensorHandle,
    rhs: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if lhs.shape == rhs.shape {
            if let Ok(handle) = provider.elem_mul(&lhs, &rhs).await {
                return Ok(resident_output_from_sources(handle, [&lhs, &rhs]));
            }
        }
        // Attempt N-D broadcast via repmat on device
        if let Some((out_shape, reps_l, reps_r)) = broadcast_repetitions(&lhs.shape, &rhs.shape) {
            let made_left = reps_l.iter().any(|&r| r != 1);
            let made_right = reps_r.iter().any(|&r| r != 1);
            let left_expanded = if made_left {
                provider
                    .repmat(&lhs, &reps_l)
                    .map_err(|e| builtin_error(format!("times: {e}")))?
            } else {
                lhs.clone()
            };
            let right_expanded = if made_right {
                provider
                    .repmat(&rhs, &reps_r)
                    .map_err(|e| builtin_error(format!("times: {e}")))?
            } else {
                rhs.clone()
            };
            let result = provider
                .elem_mul(&left_expanded, &right_expanded)
                .await
                .map_err(|e| builtin_error(format!("times: {e}")));
            if made_left {
                let _ = provider.free(&left_expanded);
            }
            if made_right {
                let _ = provider.free(&right_expanded);
            }
            if let Ok(handle) = result {
                if handle.shape == out_shape {
                    return Ok(resident_output_from_sources(handle, [&lhs, &rhs]));
                } else {
                    let _ = provider.free(&handle);
                }
            }
        }
        if is_scalar_shape(&lhs.shape) {
            if let Some(scalar) = device_real_scalar(TIMES_CATALOG_ENTRY.identity, &lhs).await? {
                if let Ok(handle) = provider.scalar_mul(&rhs, scalar) {
                    return Ok(resident_output_from_sources(handle, [&lhs, &rhs]));
                }
            }
        }
        if is_scalar_shape(&rhs.shape) {
            if let Some(scalar) = device_real_scalar(TIMES_CATALOG_ENTRY.identity, &rhs).await? {
                if let Ok(handle) = provider.scalar_mul(&lhs, scalar) {
                    return Ok(resident_output_from_sources(handle, [&lhs, &rhs]));
                }
            }
        }
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
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = host_real_scalar(&rhs) {
            if let Ok(handle) = provider.scalar_mul(&lhs, scalar) {
                return Ok(resident_output_from_sources(handle, [&lhs]));
            }
        }
    }
    let host_lhs = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    times_host(host_lhs, rhs)
}

pub(super) async fn times_gpu_host_right(lhs: Value, rhs: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = host_real_scalar(&lhs) {
            if let Ok(handle) = provider.scalar_mul(&rhs, scalar) {
                return Ok(resident_output_from_sources(handle, [&rhs]));
            }
        }
    }
    let host_rhs = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    times_host(lhs, host_rhs)
}
